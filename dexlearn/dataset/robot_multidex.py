import os
from os.path import join as pjoin
from glob import glob
import random
import time
import numpy as np
from torch.utils.data import Dataset

from dexlearn.utils.util import load_json
from dexlearn.utils.rot import numpy_quaternion_to_matrix
from .grasp_types import GRASP_TYPES

# Fixed left hand pose for right-only grasps
FIXED_LEFT_HAND_TRANS = np.array([0.0, 0.0, -0.5])
FIXED_LEFT_HAND_ROT = np.eye(3)


class RobotMultiDexDataset(Dataset):
    """Dataset for robot bimanual grasps with multiple grasp types."""

    def __init__(self, config: dict, mode: str, sc_voxel_size: float = None):
        self.config = config
        self.sc_voxel_size = sc_voxel_size
        self.mode = mode
        self.grasp_path_dict = {}
        self.grasp_scene_id_dict = {}
        self.pc_path_dict = {}
        self.pc_data_dict = {}
        self.object_pc_folder = pjoin(config.object_path, config.pc_path)
        self.preload_point_clouds = getattr(config, "preload_point_clouds", False)

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
        obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        print(f"Pre-indexing {mode} data paths...")
        self.obj_id_lst = []
        scene_ids = set()
        for obj_id in obj_id_lst:
            found_paths = glob(pjoin(self.config.grasp_path, "*", obj_id, "**/**.npy"), recursive=True)
            if found_paths:
                sorted_paths = sorted(found_paths)
                self.grasp_path_dict[obj_id] = sorted_paths
                for grasp_path in sorted_paths:
                    scene_id = self._scene_id_from_grasp_path(grasp_path, obj_id)
                    self.grasp_scene_id_dict[grasp_path] = scene_id
                    scene_ids.add(scene_id)
                self.obj_id_lst.append(obj_id)
        self._index_point_cloud_paths(scene_ids)

        self.data_num = sum(len(paths) for paths in self.grasp_path_dict.values())
        print(f"mode: {mode}, grasp data num: {self.data_num}")

    def _init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = load_json(pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json"))

        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:10]

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

    def _scene_id_from_grasp_path(self, grasp_path, obj_id):
        relative_path = os.path.relpath(grasp_path, self.config.grasp_path)
        path_parts = relative_path.split(os.sep)
        obj_idx = path_parts.index(obj_id)
        return os.path.splitext(os.path.join(*path_parts[obj_idx:]))[0]

    def _scene_id_from_scene_cfg(self, scene_cfg, scene_path):
        scene_id = scene_cfg.get("scene_id")
        if scene_id is None and "scene" in scene_cfg:
            scene = scene_cfg["scene"]
            scene_id = scene.get("id", scene.get("scene_id"))
        if scene_id is None:
            raise KeyError(f"Could not find scene id in scene config: {scene_path}")
        return scene_id

    def _index_point_cloud_paths(self, scene_ids):
        print("------------------------------------------------")
        print(f"Pre-indexing point cloud paths from {self.object_pc_folder} for {len(scene_ids)} scenes...")
        t_start = time.perf_counter()
        pc_file_count = 0
        for scene_id in sorted(scene_ids):
            pc_paths = sorted(glob(pjoin(self.object_pc_folder, scene_id, "partial_pc**.npy")))
            if pc_paths:
                self.pc_path_dict[scene_id] = pc_paths
                pc_file_count += len(pc_paths)
        elapsed = time.perf_counter() - t_start
        print(
            f"Finished pre-indexing point cloud paths: "
            f"{len(self.pc_path_dict)} scenes, {pc_file_count} point cloud files "
            f"in {elapsed:.3f}s"
        )
        if self.preload_point_clouds:
            preload_start = time.perf_counter()
            print("Pre-loading point clouds into memory...")
            for pc_paths in self.pc_path_dict.values():
                for pc_path in pc_paths:
                    self.pc_data_dict[pc_path] = np.load(pc_path, allow_pickle=True)
            preload_elapsed = time.perf_counter() - preload_start
            print(f"Finished pre-loading {len(self.pc_data_dict)} point cloud files in {preload_elapsed:.3f}s")
        print("------------------------------------------------")

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode in ["train", "eval"]:
            rand_grasp_type, pc, grasp_data = self._load_train_data(ret_dict)
        else:  # test
            rand_grasp_type, pc = self._load_test_data(id, ret_dict)
            grasp_data = None

        if self.config.pc_centering:
            pc = self._apply_pc_centering(pc, ret_dict, grasp_data)

        if self.mode == "train" and getattr(self.config, "rotation_aug", False):
            pc = self._apply_rotation_aug(pc, ret_dict)

        ret_dict["point_clouds"] = pc
        ret_dict["grasp_type_id"] = int(rand_grasp_type.split("_")[0])
        if self.sc_voxel_size is not None:
            ret_dict["coors"] = pc / self.sc_voxel_size
            ret_dict["feats"] = pc

        return ret_dict

    def _load_train_data(self, ret_dict):
        """Load training/eval data."""
        rand_obj_id = random.choice(self.obj_id_lst)
        grasp_path = random.choice(self.grasp_path_dict[rand_obj_id])
        grasp_data = np.load(grasp_path, allow_pickle=True).item()

        robot_global_pose = np.stack(
            [
                grasp_data["pregrasp_global_pose"],
                grasp_data["grasp_global_pose"],
                grasp_data["squeeze_global_pose"],
            ],
            axis=-2,
        )
        robot_joint_pos = np.stack(
            [
                grasp_data["pregrasp_joint_pos"],
                grasp_data["grasp_joint_pos"],
                grasp_data["squeeze_joint_pos"],
            ],
            axis=-2,
        )
        if len(robot_global_pose.shape) == 3:
            rand_pose_id = np.random.randint(robot_global_pose.shape[0])
            robot_global_pose = robot_global_pose[rand_pose_id : rand_pose_id + 1]  # 1, 3, x
            robot_joint_pos = robot_joint_pos[rand_pose_id : rand_pose_id + 1]  # 1, 3, x
        else:
            raise NotImplementedError

        scene_id = self.grasp_scene_id_dict[grasp_path]
        raw_grasp_type = str(grasp_data["grasp_type"][0])
        grasp_type = next((gt for gt in GRASP_TYPES if gt.endswith(raw_grasp_type)), GRASP_TYPES[0])

        # read point cloud
        pc_path_lst = self.pc_path_dict[scene_id]
        pc_path = random.choice(pc_path_lst)
        if self.preload_point_clouds:
            raw_pc = self.pc_data_dict[pc_path]
        else:
            raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
        pc = np.asarray(raw_pc[idx], dtype=np.float32)
        if "scene_scale" in grasp_data:
            pc *= grasp_data["scene_scale"][rand_pose_id]

        joint_num = robot_joint_pos.shape[-1] // 2 if "both" in grasp_type else robot_joint_pos.shape[-1]

        ret_dict["right_hand_trans"] = robot_global_pose[:, :, :3]
        ret_dict["right_hand_rot"] = numpy_quaternion_to_matrix(robot_global_pose[:, :, 3:7])
        ret_dict["right_hand_joint"] = robot_joint_pos[:, :, :joint_num]

        if "both" in grasp_type:
            ret_dict["left_hand_trans"] = robot_global_pose[:, :, 7:10]
            ret_dict["left_hand_rot"] = numpy_quaternion_to_matrix(robot_global_pose[:, :, 10:14])
            ret_dict["left_hand_joint"] = robot_joint_pos[:, :, joint_num:]
            ret_dict["left_hand_fixed"] = False
        else:
            ret_dict["left_hand_trans"] = np.tile(FIXED_LEFT_HAND_TRANS, (1, 3, 1))
            ret_dict["left_hand_rot"] = np.tile(FIXED_LEFT_HAND_ROT, (1, 3, 1, 1))
            ret_dict["left_hand_joint"] = np.zeros((1, 3, joint_num), dtype=np.float32)
            ret_dict["left_hand_fixed"] = True

        ret_dict["path"] = grasp_path
        ret_dict["rand_pose_id"] = rand_pose_id

        return grasp_type, pc, grasp_data

    def _load_test_data(self, id, ret_dict):
        """Load test data."""
        grasp_type = self.grasp_type_lst[id // len(self.test_cfg_lst)]
        scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]
        scene_cfg = np.load(scene_path, allow_pickle=True).item()
        scene_id = self._scene_id_from_scene_cfg(scene_cfg, scene_path)

        if scene_id not in self.pc_path_dict:
            self.pc_path_dict[scene_id] = sorted(glob(pjoin(self.object_pc_folder, scene_id, "partial_pc**.npy")))
            if self.preload_point_clouds:
                for pc_path in self.pc_path_dict[scene_id]:
                    self.pc_data_dict[pc_path] = np.load(pc_path, allow_pickle=True)
        pc_path = random.choice(self.pc_path_dict[scene_id])

        if self.preload_point_clouds:
            raw_pc = self.pc_data_dict[pc_path]
        else:
            raw_pc = np.load(pc_path, allow_pickle=True)
        idx = np.random.choice(raw_pc.shape[0], self.config.num_points, replace=True)
        pc = np.asarray(raw_pc[idx], dtype=np.float32)

        ret_dict["save_path"] = pjoin(self.config.name, grasp_type, scene_id, os.path.basename(pc_path))
        ret_dict["scene_path"] = scene_path
        ret_dict["pc_path"] = pc_path

        return grasp_type, pc

    def _apply_pc_centering(self, pc, ret_dict, grasp_data):
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

    def _apply_rotation_aug(self, pc, ret_dict):
        """Apply random rotation around Z axis."""
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        pc = pc @ rot_z.T

        for side in ["right", "left"]:
            if f"{side}_hand_trans" in ret_dict:
                # Skip rotating the synthetic fixed left hand pose used by single-hand grasps.
                if side == "left" and ret_dict.get("left_hand_fixed", False):
                    continue
                ret_dict[f"{side}_hand_trans"] = ret_dict[f"{side}_hand_trans"] @ rot_z.T
                ret_dict[f"{side}_hand_rot"] = rot_z @ ret_dict[f"{side}_hand_rot"]

        return pc
