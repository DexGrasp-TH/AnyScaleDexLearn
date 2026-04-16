import os
from os.path import join as pjoin
from glob import glob
import random
import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.util import load_json
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
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode
        self.grasp_path_dict = {}
        self.pc_path_dict = {}
        self.object_pc_folder = pjoin(config.object_path, config.pc_path)
        self.mano_pose_dim = 24 if "GRAB" in config.grasp_path else 45

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
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        print(f"Pre-indexing {mode} data paths...")
        for obj_id in self.obj_id_lst:
            found_paths = glob(pjoin(self.config.grasp_path, obj_id, "**/**.npy"), recursive=True)
            if found_paths:
                self.grasp_path_dict[obj_id] = sorted(found_paths)

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
        self.data_num = self.grasp_type_num * len(self.test_cfg_lst) # TO BE CHECKED
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

        if self.mode == "train" and getattr(self.config, "rotation_aug", False):
            pc = self._apply_rotation_aug(pc, ret_dict)

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
                    ret_dict[f"{side}_hand_trans"] = np.asarray(grasp_data["hand"][side]["trans"]).reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = (
                        sciR.from_rotvec(grasp_data["hand"][side]["rot"]).as_matrix().reshape(1, 1, 3, 3)
                    )
                else:
                    ret_dict[f"{side}_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
                    ret_dict[f"{side}_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)
        else:
            # Mirror left-only grasp to right hand
            l_trans = np.asarray(grasp_data["hand"]["left"]["trans"])
            l_rot = sciR.from_rotvec(grasp_data["hand"]["left"]["rot"]).as_matrix()
            ret_dict["right_hand_trans"] = (self._MIRROR @ l_trans).reshape(1, 1, 3)
            ret_dict["right_hand_rot"] = (self._MIRROR @ l_rot @ self._MIRROR).reshape(1, 1, 3, 3)
            ret_dict["left_hand_trans"] = FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
            ret_dict["left_hand_rot"] = FIXED_LEFT_HAND_ROT.reshape(1, 1, 3, 3)

        if getattr(self.config, "load_mano_params", False):
            self._extract_mano_params(grasp_data, mirrored, ret_dict)

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
            if mirrored:
                ret_dict["right_hand_trans"] -= pc_centroid[None, :, :]
            else:
                for side, is_active in grasp_data["hand"].items():
                    if is_active:
                        ret_dict[f"{side}_hand_trans"] -= pc_centroid[None, :, :]

        return pc

    def _apply_rotation_aug(self, pc, ret_dict):
        """Apply random rotation around Z axis."""
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        pc = pc @ rot_z.T

        for side in ["right", "left"]:
            if f"{side}_hand_trans" in ret_dict:
                # Skip rotating fixed left hand pose
                if side == "left" and np.allclose(
                    ret_dict[f"{side}_hand_trans"], FIXED_LEFT_HAND_TRANS.reshape(1, 1, 3)
                ):
                    continue
                # in-place change
                ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] @ rot_z.T
                ret_dict[f"{side}_hand_rot"] = rot_z @ ret_dict[f"{side}_hand_rot"]

        return pc
