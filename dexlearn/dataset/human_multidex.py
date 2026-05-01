import os
from os.path import join as pjoin
from glob import glob
from collections import Counter, defaultdict
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
        self.type_grasp_path_dict = defaultdict(list)
        self.type_object_grasp_path_dict = defaultdict(lambda: defaultdict(list))
        self.object_type_counter = defaultdict(Counter)
        self.type_counter = Counter()
        self.object_pc_folder = pjoin(config.object_path, config.pc_path)
        self.hand_pos_source = normalize_hand_pos_source(getattr(config, "hand_pos_source", "wrist"))
        self.random_pc_across_sequences = bool(getattr(config, "random_pc_across_sequences", True))
        self.type_balancing_enabled = self.mode == "train" and bool(getattr(config, "type_balancing_enabled", False))
        self.type_sampler_enabled = self.mode == "train" and bool(getattr(config, "type_sampler_enabled", False))
        self.type_sampler_alpha = float(getattr(config, "type_sampler_alpha", 1.0))
        self.type_sampler_object_uniform = bool(getattr(config, "type_sampler_object_uniform", True))

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
        self._index_grasp_type_distribution()
        print(f"mode: {mode}, grasp data num: {self.data_num}")

    def _index_grasp_type_distribution(self):
        """Index train/eval grasp paths by grasp type and object.

        Args:
            None.

        Returns:
            None. The per-type path dictionaries and counters are stored on
            this dataset instance.
        """
        self.type_grasp_path_dict = defaultdict(list)
        self.type_object_grasp_path_dict = defaultdict(lambda: defaultdict(list))
        self.object_type_counter = defaultdict(Counter)
        self.type_counter = Counter()

        for obj_id, grasp_paths in self.grasp_path_dict.items():
            for grasp_path in grasp_paths:
                try:
                    grasp_data = np.load(grasp_path, allow_pickle=True).item()
                    grasp_type, _ = self._determine_grasp_type(grasp_data)
                except Exception as exc:
                    raise RuntimeError(f"Failed to index grasp type for {grasp_path}") from exc

                grasp_type_id = int(grasp_type.split("_", 1)[0])
                self.type_grasp_path_dict[grasp_type_id].append(grasp_path)
                self.type_object_grasp_path_dict[grasp_type_id][obj_id].append(grasp_path)
                self.object_type_counter[obj_id][grasp_type_id] += 1
                self.type_counter[grasp_type_id] += 1

    def get_distribution_analysis(self):
        """Return the indexed grasp-type distribution for this dataset.

        Args:
            None.

        Returns:
            Dictionary containing global type counts, per-object type counts,
            and the dataset sampling configuration.
        """
        type_counts = {str(type_id): int(self.type_counter.get(type_id, 0)) for type_id in range(len(GRASP_TYPES))}
        object_counts = {
            str(obj_id): {str(type_id): int(count) for type_id, count in sorted(counter.items())}
            for obj_id, counter in sorted(self.object_type_counter.items())
        }
        return {
            "mode": self.mode,
            "data_num": int(self.data_num),
            "object_num": int(len(self.obj_id_lst)) if hasattr(self, "obj_id_lst") else 0,
            "type_counts": type_counts,
            "object_type_counts": object_counts,
            "sampler": {
                "type_balancing_enabled": bool(self.type_balancing_enabled),
                "type_sampler_enabled": bool(self.type_sampler_enabled),
                "type_sampler_alpha": float(self.type_sampler_alpha),
                "type_sampler_object_uniform": bool(self.type_sampler_object_uniform),
            },
        }

    def _sample_balanced_grasp_path(self):
        """Sample one grasp path with tempered type balancing.

        Args:
            None.

        Returns:
            Path to one selected grasp file.
        """
        if self.type_sampler_alpha < 0.0:
            raise ValueError(f"type_sampler_alpha must be non-negative, got {self.type_sampler_alpha}")

        active_type_ids = sorted(type_id for type_id, paths in self.type_grasp_path_dict.items() if paths)
        if not active_type_ids:
            rand_obj_id = random.choice(self.obj_id_lst)
            return random.choice(self.grasp_path_dict[rand_obj_id])

        type_weights = [len(self.type_grasp_path_dict[type_id]) ** self.type_sampler_alpha for type_id in active_type_ids]
        sampled_type_id = random.choices(active_type_ids, weights=type_weights, k=1)[0]

        if self.type_sampler_object_uniform:
            object_path_dict = self.type_object_grasp_path_dict[sampled_type_id]
            obj_id = random.choice(sorted(object_path_dict.keys()))
            return random.choice(object_path_dict[obj_id])
        return random.choice(self.type_grasp_path_dict[sampled_type_id])

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
            pc = self._apply_scale_aug(pc, ret_dict)
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

    def _get_grasp_sequence_id(self, grasp_data):
        """Read the sequence id associated with one grasp record.

        Args:
            grasp_data: Loaded human grasp dictionary.

        Returns:
            Sequence id string such as ``seq_0`` when available, otherwise
            ``None``.
        """
        object_data = grasp_data.get("object", {})
        sequence_id = object_data.get("sequence_id")
        if sequence_id is not None:
            return str(sequence_id)

        source_scene = object_data.get("source_scene")
        if source_scene is not None and "_seq_" in str(source_scene):
            return f"seq_{str(source_scene).rsplit('_seq_', 1)[1]}"
        return None

    def _get_scene_sequence_id(self, scene_cfg):
        """Read the sequence id associated with one test scene config.

        Args:
            scene_cfg: Loaded scene config dictionary.

        Returns:
            Sequence id string such as ``seq_0`` when available, otherwise
            ``None``.
        """
        object_data = scene_cfg.get("object", {})
        sequence_id = object_data.get("sequence_id")
        if sequence_id is not None:
            return str(sequence_id)

        scene_id = scene_cfg.get("scene_id")
        if scene_id is not None and "/" in str(scene_id):
            return str(scene_id).split("/", 1)[1]
        return None

    def _pointcloud_matches_sequence(self, pc_path, sequence_id):
        """Check whether a point-cloud path belongs to the requested sequence.

        Args:
            pc_path: Candidate point-cloud path.
            sequence_id: Required sequence id, for example ``seq_0``.

        Returns:
            ``True`` when the path name or parent directories identify the same
            sequence.
        """
        if sequence_id is None:
            return False
        sequence_id = str(sequence_id)
        base_name = os.path.splitext(os.path.basename(pc_path))[0]
        path_parts = pc_path.split(os.sep)
        return (
            base_name == sequence_id
            or base_name.startswith(f"{sequence_id}_")
            or sequence_id in path_parts
        )

    def _select_pointcloud_path(self, obj_name, sequence_id=None):
        """Select a point-cloud path for one object and optional sequence.

        Args:
            obj_name: Canonical object id used under ``object_pc_folder``.
            sequence_id: Sequence id that must be matched when cross-sequence
                point-cloud sampling is disabled.

        Returns:
            Path to one selected point-cloud file.
        """
        if obj_name not in self.pc_path_dict:
            self.pc_path_dict[obj_name] = sorted(glob(pjoin(self.object_pc_folder, obj_name, "**.npy")))
        if not self.pc_path_dict[obj_name]:
            raise FileNotFoundError(
                f"No point-cloud files found for object '{obj_name}' under "
                f"{pjoin(self.object_pc_folder, obj_name)}"
            )

        pc_candidates = self.pc_path_dict[obj_name]
        if not self.random_pc_across_sequences:
            pc_candidates = [
                pc_path
                for pc_path in pc_candidates
                if self._pointcloud_matches_sequence(pc_path, sequence_id)
            ]
            if not pc_candidates:
                raise FileNotFoundError(
                    f"No point-cloud file for object '{obj_name}' matches sequence_id='{sequence_id}' "
                    f"under {pjoin(self.object_pc_folder, obj_name)}"
                )
        return random.choice(pc_candidates)

    def _load_pointcloud(self, obj_name, obj_scale, obj_pose, mirrored=False, sequence_id=None):
        """Load and transform object point cloud.

        Args:
            obj_name: Canonical object id used under ``object_pc_folder``.
            obj_scale: Scalar or per-axis object scale.
            obj_pose: Object pose matrix from object frame to world frame.
            mirrored: Whether to mirror the point cloud across the YZ plane.
            sequence_id: Sequence id to use when same-sequence point-cloud
                sampling is required.

        Returns:
            Tuple of transformed point cloud and selected point-cloud path.
        """
        pc_path = self._select_pointcloud_path(obj_name, sequence_id=sequence_id)

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
        if self.type_balancing_enabled and self.type_sampler_enabled:
            grasp_path = self._sample_balanced_grasp_path()
        else:
            rand_obj_id = random.choice(self.obj_id_lst)
            grasp_path = random.choice(self.grasp_path_dict[rand_obj_id])
        grasp_data = np.load(grasp_path, allow_pickle=True).item()

        ret_dict["path"] = grasp_path

        grasp_type, mirrored = self._determine_grasp_type(grasp_data)
        self._extract_hand_poses(grasp_data, mirrored, ret_dict)

        obj_name = grasp_data["object"]["name"]
        obj_scale = grasp_data["object"]["rel_scale"]
        obj_pose = grasp_data["object"]["pose"]
        sequence_id = self._get_grasp_sequence_id(grasp_data)

        pc, _ = self._load_pointcloud(obj_name, obj_scale, obj_pose, mirrored, sequence_id=sequence_id)

        return grasp_type, pc, mirrored, grasp_data

    def _load_test_data(self, id, ret_dict):
        """Load test data."""
        grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
        scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]

        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        obj_name = scene_cfg["object"]["name"]
        obj_scale = scene_cfg["object"]["rel_scale"]
        obj_pose = scene_cfg["object"]["pose"]
        sequence_id = self._get_scene_sequence_id(scene_cfg)

        pc, pc_path = self._load_pointcloud(obj_name, obj_scale, obj_pose, sequence_id=sequence_id)

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

    def _apply_scale_aug(self, pc, ret_dict):
        """Apply train-time uniform scale augmentation to object and active hands.

        Args:
            pc: Centered object point cloud with shape ``(N, 3)``.
            ret_dict: Sample dictionary containing active hand translation targets.

        Returns:
            The scaled point cloud. Hand translations in ``ret_dict`` are updated
            in-place with the same factor.
        """
        scale = self._sample_scale_aug()
        ret_dict["scale_aug_factor"] = np.float32(scale)
        if scale == 1.0:
            return pc.astype(np.float32, copy=False)

        pc = pc * np.float32(scale)
        for side in ["right", "left"]:
            if f"{side}_hand_trans" not in ret_dict:
                continue
            if side == "left" and ret_dict.get("left_hand_fixed", False):
                continue
            ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] * np.float32(scale)
        return pc.astype(np.float32, copy=False)

    def _sample_scale_aug(self):
        """Sample a train-time uniform scale augmentation factor.

        Args:
            None.

        Returns:
            A positive scalar scale factor. ``1.0`` means no scaling.
        """
        if not bool(getattr(self.config, "scale_aug", False)):
            return 1.0
        scale_min = float(getattr(self.config, "scale_min", 1.0))
        scale_max = float(getattr(self.config, "scale_max", 1.0))
        if scale_min <= 0.0 or scale_max <= 0.0:
            raise ValueError("Scale augmentation bounds must be positive")
        if scale_min > scale_max:
            raise ValueError(f"scale_min must be <= scale_max, got {scale_min} > {scale_max}")
        if scale_min == scale_max:
            return scale_min
        return float(np.random.uniform(scale_min, scale_max))

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
