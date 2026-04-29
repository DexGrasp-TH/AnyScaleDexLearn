import os
from os.path import join as pjoin
from glob import glob
import random
import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.util import load_json
from dexlearn.utils.human_hand import normalize_hand_pos_source
from dexlearn.utils.config import flatten_multidex_data_config
from .grasp_types import GRASP_TYPES
from scipy.spatial.transform import Rotation as sciR

# Fixed left hand pose for right-only grasps
FIXED_LEFT_HAND_TRANS = np.array([0.0, 0.0, -0.5])
FIXED_LEFT_HAND_ROT = np.eye(3)


class HumanMultiDexDataset(Dataset):
    """Dataset for human bimanual grasps with multiple grasp types."""

    _MIRROR = np.diag([-1.0, 1.0, 1.0])  # YZ-plane reflection matrix
    _MIRRORED_ROTVEC_SIGN = np.array([1.0, -1.0, -1.0], dtype=np.float32)

    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        flatten_multidex_data_config(config)
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode
        self.grasp_path_dict = {}
        self.pc_path_dict = {}
        self.object_pc_folder = pjoin(config.object_path, config.pc_path)
        self.hand_pos_source = normalize_hand_pos_source(getattr(config, "hand_pos_source", "wrist"))

        if mode == "test":
            self.grasp_type_lst = (
                config.grasp_type_lst
                if hasattr(config, "grasp_type_lst") and config.grasp_type_lst
                else [gt for gt in GRASP_TYPES if gt != "0_any"]
            )
            self.grasp_type_num = len(self.grasp_type_lst)

        if mode in ["train", "eval"]:
            self._init_train_eval(mode)
        elif mode == "test":
            self._init_test()

    def _init_train_eval(self, mode):

        split_name = "test" if mode == "eval" else "train"
        self.mano_pose_dim = 24 if "GRAB" in self.config.grasp_path else 45
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        print(f"Pre-indexing {mode} data paths...")
        indexed_obj_ids = []
        for obj_id in self.obj_id_lst:
            found_paths = glob(pjoin(self.config.grasp_path, obj_id, "**/**.npy"), recursive=True)
            if found_paths:
                self.grasp_path_dict[obj_id] = sorted(found_paths)
                indexed_obj_ids.append(obj_id)
        self.obj_id_lst = indexed_obj_ids

        self.data_num = sum(len(paths) for paths in self.grasp_path_dict.values())
        print(f"mode: {mode}, grasp data num: {self.data_num}")

    def _init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]

        scene_patterns = (
            [self.config.test_scene_cfg] if isinstance(self.config.test_scene_cfg, str) else self.config.test_scene_cfg
        )

        test_cfg_set = set()
        for obj_id in self.obj_id_lst:
            base_dir = pjoin(self.config.object_path, "scene_cfg", obj_id)
            for pattern in scene_patterns:
                test_cfg_set.update(glob(pjoin(base_dir, pattern), recursive=True))

        self.test_cfg_lst = sorted(test_cfg_set)
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst)  # TO BE CHECKED
        print(
            f"Test split: {split_name}, grasp type list: {self.grasp_type_lst}, "
            f"object cfg num: {len(self.test_cfg_lst)}"
        )

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode in ["train", "eval"]:
            rand_grasp_type, pc, mirrored, grasp_data = self._load_train_data(ret_dict)
        else:  # test
            rand_grasp_type, pc = self._load_test_data(id, ret_dict)
            mirrored = False
            grasp_data = None

        if self.config.pc_centering:
            pc = self._apply_pc_centering(pc, ret_dict, mirrored, grasp_data)

        if self.mode == "train":
            pc = self._apply_geometric_aug(pc, ret_dict)

        if self.mode == "train" and getattr(self.config, "pc_noise_aug", False):
            pc = self._apply_pc_noise_aug(pc)

        ret_dict["point_clouds"] = pc
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0])
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size
            ret_dict["feats"] = pc

        return ret_dict

    def _determine_grasp_type(self, grasp_data):
        """Determine grasp type and whether mirroring is needed."""
        l_contacts = grasp_data["hand"]["left"]["contacts"] if grasp_data["hand"]["left"] else [False] * 5
        r_contacts = grasp_data["hand"]["right"]["contacts"] if grasp_data["hand"]["right"] else [False] * 5
        has_l, has_r = any(l_contacts), any(r_contacts)
        l_count, r_count = sum(l_contacts), sum(r_contacts)

        if not (has_l or has_r):
            raise ValueError("Grasp data has no contact")

        if has_l and has_r:
            grasp_type = GRASP_TYPES[5] if (l_count > 3 or r_count > 3) else GRASP_TYPES[4]
            return grasp_type, False

        # Single hand grasp - determine type by finger count
        count = l_count if has_l else r_count
        if count <= 2:
            grasp_type = GRASP_TYPES[1]
        elif count == 3:
            grasp_type = GRASP_TYPES[2]
        else:
            grasp_type = GRASP_TYPES[3]

        # Mirror left-only grasps to right hand
        mirrored = has_l and not has_r
        return grasp_type, mirrored

    def _extract_hand_poses(self, grasp_data, mirrored, ret_dict):
        """Extract and process hand poses from grasp data."""
        if not mirrored:
            for side, is_active in grasp_data["hand"].items():
                if is_active:
                    ret_dict[f"{side}_hand_trans"] = self._get_hand_translation(grasp_data["hand"][side]).reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = (
                        sciR.from_rotvec(grasp_data["hand"][side]["rot"]).as_matrix().reshape(1, 1, 3, 3)
                    )
                else:
                    ret_dict[f"{side}_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)
                    if side == "left":
                        ret_dict["left_hand_fixed"] = True
        else:
            # Mirror left-only grasp to right hand
            l_trans = self._get_hand_translation(grasp_data["hand"]["left"])
            l_rot = sciR.from_rotvec(grasp_data["hand"]["left"]["rot"]).as_matrix()
            ret_dict["right_hand_trans"] = (self._MIRROR @ l_trans).reshape(1, 1, 3)
            ret_dict["right_hand_rot"] = (self._MIRROR @ l_rot @ self._MIRROR).reshape(1, 1, 3, 3)
            ret_dict["left_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
            ret_dict["left_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)
            ret_dict["left_hand_fixed"] = True

        ret_dict.setdefault("left_hand_fixed", False)

        if getattr(self.config, "load_mano_params", False):
            self._extract_mano_params(grasp_data, mirrored, ret_dict)

    def _get_hand_translation(self, hand_data):
        if self.hand_pos_source == "wrist":
            return np.asarray(hand_data["trans"], dtype=np.float32)

        if "index_mcp_pos" not in hand_data:
            raise KeyError(
                "Missing hand['index_mcp_pos'] while hand_pos_source='index_mcp'. "
                "Run the human_preprocess task first."
            )
        return np.asarray(hand_data["index_mcp_pos"], dtype=np.float32)

    def _extract_mano_params(self, grasp_data, mirrored, ret_dict):
        """Extract MANO parameters for visualization."""
        if not mirrored:
            for side, is_active in grasp_data["hand"].items():
                if is_active:
                    ret_dict[f"{side}_mano_pose"] = np.asarray(grasp_data["hand"][side]["mano_pose"]).flatten()
                    ret_dict[f"{side}_mano_betas"] = np.asarray(grasp_data["hand"][side]["mano_betas"]).flatten()
                else:
                    ret_dict[f"{side}_mano_pose"] = np.zeros(self.mano_pose_dim, dtype=np.float32)
                    ret_dict[f"{side}_mano_betas"] = np.zeros(10, dtype=np.float32)
        else:
            left_mano_pose = np.asarray(grasp_data["hand"]["left"]["mano_pose"]).flatten().astype(np.float32)
            ret_dict["right_mano_pose"] = self._mirror_mano_pose(left_mano_pose)
            ret_dict["right_mano_betas"] = np.asarray(grasp_data["hand"]["left"]["mano_betas"]).flatten()
            ret_dict["left_mano_pose"] = np.zeros(self.mano_pose_dim, dtype=np.float32)
            ret_dict["left_mano_betas"] = np.zeros(10, dtype=np.float32)

    def _mirror_mano_pose(self, mano_pose):
        """
        Mirror MANO axis-angle joints from left hand to right hand across the YZ plane.
        For reflection S=diag(-1,1,1), mirrored rotvec is det(S)*S*w = [1,-1,-1]*w.
        """
        pose = np.asarray(mano_pose).flatten().astype(np.float32)
        if pose.size == 45:
            return (pose.reshape(-1, 3) * self._MIRRORED_ROTVEC_SIGN[None, :]).reshape(-1)
        return pose

    def _load_pointcloud(self, obj_name, obj_scale, obj_pose, mirrored=False):
        """Load and transform object point cloud."""
        if obj_name not in self.pc_path_dict:
            self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
        if not self.pc_path_dict[obj_name]:
            raise FileNotFoundError(
                f"No point-cloud files found for object '{obj_name}' under "
                f"{pjoin(self.object_pc_folder, obj_name)}"
            )
        pc_path = random.choice(self.pc_path_dict[obj_name])

        raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
        scaled_pc = raw_pc[idx] * obj_scale

        R, t = obj_pose[:3, :3], obj_pose[:3, 3]
        pc = np.matmul(scaled_pc, R.T) + t

        if mirrored:
            pc = pc.copy()
            pc[:, 0] = -pc[:, 0]

        return pc, pc_path

    def _load_train_data(self, ret_dict):
        """Load training/eval data."""
        rand_obj_id = random.choice(self.obj_id_lst)
        grasp_path = random.choice(self.grasp_path_dict[rand_obj_id])
        grasp_data = np.load(grasp_path, allow_pickle=True).item()

        ret_dict["path"] = grasp_path

        grasp_type, mirrored = self._determine_grasp_type(grasp_data)
        self._extract_hand_poses(grasp_data, mirrored, ret_dict)

        obj_name = grasp_data["object"]["name"]
        obj_scale = grasp_data["object"]["rel_scale"]
        obj_pose = grasp_data["object"]["pose"]

        pc, _ = self._load_pointcloud(obj_name, obj_scale, obj_pose, mirrored)

        return grasp_type, pc, mirrored, grasp_data

    def _load_test_data(self, id, ret_dict):
        """Load test data."""
        grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
        scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]

        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        obj_name = scene_cfg["object"]["name"]
        obj_scale = scene_cfg["object"]["rel_scale"]
        obj_pose = scene_cfg["object"]["pose"]

        pc, pc_path = self._load_pointcloud(obj_name, obj_scale, obj_pose)

        ret_dict["save_path"] = pjoin(self.config.name, grasp_type, scene_cfg["scene_id"], os.path.basename(pc_path))
        ret_dict["scene_path"] = scene_path
        ret_dict["pc_path"] = pc_path

        return grasp_type, pc

    def _apply_pc_centering(self, pc, ret_dict, mirrored, grasp_data):
        """Center point cloud and adjust hand poses accordingly."""
        pc_centroid = np.mean(pc, axis=-2, keepdims=True)
        pc = pc - pc_centroid
        ret_dict["pc_centroid"] = pc_centroid.astype(np.float32)

        if self.mode != "test":
            if "right_hand_trans" in ret_dict:
                ret_dict["right_hand_trans"] -= pc_centroid[None, :, :]
            if "left_hand_trans" in ret_dict and not ret_dict.get("left_hand_fixed", False):
                ret_dict["left_hand_trans"] -= pc_centroid[None, :, :]

        return pc

    def _apply_geometric_aug(self, pc, ret_dict):
        """Apply train-time rotation and translation after point-cloud centering."""
        rot = self._sample_rotation_aug()
        trans = self._sample_translation_aug()

        if rot is not None:
            pc = pc @ rot.T
            for side in ["right", "left"]:
                if f"{side}_hand_trans" in ret_dict:
                    if side == "left" and ret_dict.get("left_hand_fixed", False):
                        continue
                    ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] @ rot.T
                    ret_dict[f"{side}_hand_rot"] = rot @ ret_dict[f"{side}_hand_rot"]

        if trans is not None:
            pc = pc + trans.reshape(1, 3)
            for side in ["right", "left"]:
                if f"{side}_hand_trans" in ret_dict:
                    if side == "left" and ret_dict.get("left_hand_fixed", False):
                        continue
                    ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] + trans.reshape(1, 1, 3)

        return pc

    def _sample_rotation_aug(self):
        z_rotation_enabled = bool(
            getattr(self.config, "z_rotation_aug", getattr(self.config, "rotation_aug", False))
        )
        xy_rotation_enabled = bool(getattr(self.config, "xy_rotation_aug", False))
        if not z_rotation_enabled and not xy_rotation_enabled:
            return None

        z_angle = np.random.uniform(-np.pi, np.pi) if z_rotation_enabled else 0.0
        xy_angle_limit = np.deg2rad(float(getattr(self.config, "xy_rotation_max_angle_deg", 0.0)))
        x_angle, y_angle = (
            np.random.uniform(-xy_angle_limit, xy_angle_limit, size=2) if xy_rotation_enabled else (0.0, 0.0)
        )

        # Compose all enabled axis rotations into one matrix so point clouds and hands share one rigid transform.
        return sciR.from_euler("xyz", [x_angle, y_angle, z_angle]).as_matrix().astype(np.float32)

    def _sample_translation_aug(self):
        if not bool(getattr(self.config, "translation_aug", False)):
            return None
        translation_range = float(getattr(self.config, "translation_range", 0.0))
        if translation_range <= 0.0:
            return None
        return np.random.uniform(-translation_range, translation_range, size=3).astype(np.float32)

    def _apply_pc_noise_aug(self, pc):
        """Add zero-mean Gaussian noise to object points during training."""
        noise_scale = float(getattr(self.config, "pc_noise_scale", 0.0))
        if noise_scale <= 0.0:
            return pc.astype(np.float32, copy=False)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=pc.shape).astype(np.float32)
        return pc.astype(np.float32, copy=False) + noise
