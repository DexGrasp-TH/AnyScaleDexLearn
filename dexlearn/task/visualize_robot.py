import os
import sys
import json
import random
import re
from glob import glob

import hydra
import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from omegaconf import DictConfig
from pytorch3d import transforms as pttf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.util import set_seed
from dexlearn.utils.logger import Logger
from dexlearn.dataset import GRASP_TYPES
from mr_utils.robot.pk_helper import PytorchKinematicsHelper
from mr_utils.robot.pk_visualizer import Visualizer


def resolve_dataset_path(path: str, object_root: str) -> str:
    if os.path.exists(path):
        return path

    if "/AnyScaleGrasp/" in path:
        relative_path = path.split("/AnyScaleGrasp/", 1)[1]
        dataset_root = os.environ.get("AnyScaleGraspDataset")
        if dataset_root:
            resolved = os.path.join(dataset_root, relative_path)
            if os.path.exists(resolved):
                return resolved

    if "/object/" in path and "/object/" in object_root:
        suffix = path.split("/object/", 1)[1]
        dataset_root = object_root.split("/object/", 1)[0]
        resolved = os.path.join(dataset_root, "object", suffix)
        if os.path.exists(resolved):
            return resolved

    return path


def get_output_dir(config: DictConfig) -> str:
    return os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )


def get_group_dirs(output_dir: str, include_groups):
    if include_groups is not None:
        return include_groups
    return sorted([name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))])


def get_balanced_samples(root_dir: str, include_groups, num_samples_per_group: int):
    sample_files = []
    for group in include_groups:
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        group_files = sorted(glob(os.path.join(group_path, "**/*.npy"), recursive=True))
        if not group_files:
            continue
        sample_files.extend(random.sample(group_files, k=min(len(group_files), num_samples_per_group)))
    return sample_files


def build_robot_components(config: DictConfig):
    urdf_path = config.test_data.robot_urdf_path
    mesh_dir_path = config.test_data.robot_mesh_dir_path

    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=config.device)
    robot_helper = PytorchKinematicsHelper(
        chain,
        base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=config.device,
    )
    robot_visualizer = Visualizer(urdf_path, mesh_dir_path, device=config.device)

    metadata_path = os.path.join(config.data.grasp_path, config.test_data.metadata_group, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Joint metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        joint_metadata = json.load(f)

    joint_names = joint_metadata["joint_names"]
    wrist_link_names = joint_metadata["wrist_body_names"]
    right_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("ra_")]
    left_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("la_")]
    right_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("rh_")]
    left_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("lh_")]

    ik_solver_kwargs = dict(
        pos_tolerance=1e-3,
        rot_tolerance=1e-2,
        max_iterations=50,
        retry_configs=None,
        num_retries=20,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="any",
        debug=False,
        lr=0.2,
        regularlization=1e-3,
    )
    for wrist_link_name in wrist_link_names:
        robot_helper.create_serial_chain(wrist_link_name)
        robot_helper.create_ik_solver(wrist_link_name, **ik_solver_kwargs)

    return {
        "joint_names": joint_names,
        "wrist_link_names": wrist_link_names,
        "right_arm_indices": right_arm_indices,
        "left_arm_indices": left_arm_indices,
        "right_hand_indices": right_hand_indices,
        "left_hand_indices": left_hand_indices,
        "robot_helper": robot_helper,
        "robot_visualizer": robot_visualizer,
    }


def recover_pc_path(sample_file: str, scene_cfg: dict, config: DictConfig) -> str:
    original_name = re.sub(r"_\d+\.npy$", ".npy", os.path.basename(sample_file))
    return os.path.join(config.test_data.object_path, config.test_data.pc_path, scene_cfg["scene_id"], original_name)


def load_visualization_pc(sample_file: str, scene_cfg: dict, config: DictConfig):
    pc_path = recover_pc_path(sample_file, scene_cfg, config)
    raw_pc = np.load(pc_path, allow_pickle=True)
    idx = np.random.choice(raw_pc.shape[0], config.test_data.num_points, replace=True)
    pc = raw_pc[idx]
    if config.test_data.pc_centering:
        pc = pc - np.mean(pc, axis=-2, keepdims=True)
    return pc


def solve_stage_pose(stage_qpos: np.ndarray, robot_assets: dict, device: str):
    half_dim = stage_qpos.shape[0] // 2
    right_qpos = torch.from_numpy(stage_qpos[:half_dim]).to(device=device, dtype=torch.float32).unsqueeze(0)
    left_qpos = torch.from_numpy(stage_qpos[half_dim:]).to(device=device, dtype=torch.float32).unsqueeze(0)

    right_trans, right_quat, right_joint = right_qpos[:, :3], right_qpos[:, 3:7], right_qpos[:, 7:]
    left_trans, left_quat, left_joint = left_qpos[:, :3], left_qpos[:, 3:7], left_qpos[:, 7:]

    right_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    right_matrix[:, :3, :3] = pttf.quaternion_to_matrix(right_quat)
    right_matrix[:, :3, 3] = right_trans

    left_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    left_matrix[:, :3, :3] = pttf.quaternion_to_matrix(left_quat)
    left_matrix[:, :3, 3] = left_trans

    right_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][0], right_matrix)
    left_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][1], left_matrix)

    combined_q = torch.zeros((1, len(robot_assets["joint_names"])), device=device, dtype=torch.float32)
    right_q = right_sol["q"].to(device=device, dtype=combined_q.dtype)
    left_q = left_sol["q"].to(device=device, dtype=combined_q.dtype)
    combined_q[:, robot_assets["right_arm_indices"]] = right_q[:, robot_assets["right_arm_indices"]]
    combined_q[:, robot_assets["left_arm_indices"]] = left_q[:, robot_assets["left_arm_indices"]]
    combined_q[:, robot_assets["right_hand_indices"]] = right_joint
    combined_q[:, robot_assets["left_hand_indices"]] = left_joint

    robot_pose = torch.zeros((1, 7 + combined_q.shape[-1]), device=device, dtype=torch.float32)
    robot_pose[:, 3] = 1.0
    robot_pose[:, 7:] = combined_q
    return robot_pose, right_sol["success"][0].item(), left_sol["success"][0].item()


def task_visualize_robot(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)  # necessary for get the ckpt path from config

    output_dir = get_output_dir(config)
    include_groups = get_group_dirs(output_dir, config.task.include_groups)
    sample_files = get_balanced_samples(output_dir, include_groups, config.task.num_samples_per_group)
    if len(sample_files) == 0:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    robot_assets = build_robot_components(config)
    stage_names = ["grasp_qpos"]
    stage_colors = [(255, 120, 120, 180)]

    for sample_file in sample_files:
        print(f"Processing {sample_file}")

        data = np.load(sample_file, allow_pickle=True).item()
        scene_path = resolve_dataset_path(data["scene_path"], config.test_data.object_path)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene config not found: {scene_path}")
        scene_cfg = np.load(scene_path, allow_pickle=True).item()

        pc = load_visualization_pc(sample_file, scene_cfg, config)
        scene_elements = [
            trimesh.points.PointCloud(pc, colors=[255, 165, 0, 255]),
            trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3),
        ]

        ik_status = []
        for stage_name, color in zip(stage_names, stage_colors):
            robot_pose, right_ok, left_ok = solve_stage_pose(data[stage_name], robot_assets, config.device)
            robot_assets["robot_visualizer"].set_robot_parameters(robot_pose, joint_names=robot_assets["joint_names"])
            scene_elements.append(robot_assets["robot_visualizer"].get_robot_trimesh_data(i=0, color=color))
            ik_status.append(f"{stage_name}: R={int(right_ok)} L={int(left_ok)}")

        caption = f"{os.path.basename(sample_file)} | err={float(data['grasp_error']):.4f}"
        if "grasp_type_id" in data:
            grasp_type_id = int(data["grasp_type_id"])
            caption = f"{caption} | grasp_type_id={GRASP_TYPES[grasp_type_id]}"
        if "pred_grasp_type_id" in data:
            pred_grasp_type_id = int(data["pred_grasp_type_id"])
            caption = f"{caption} | pred_grasp_type_id={GRASP_TYPES[pred_grasp_type_id]}"
        if "pred_grasp_type_prob" in data:
            pred_grasp_type_prob = np.asarray(data["pred_grasp_type_prob"]).reshape(-1)
            prob_str = ", ".join([f"{p:.3f}" for p in pred_grasp_type_prob.tolist()])
            caption = f"{caption} | pred_grasp_type_prob=[{prob_str}]"
        if config.task.show_ik_status:
            caption = f"{caption} | {' ; '.join(ik_status)}"

        scene = trimesh.Scene(geometry=scene_elements)
        scene.set_camera(
            angles=(np.deg2rad(60.0), 0.0, np.deg2rad(45.0)),
            distance=config.task.camera_distance,
            center=scene.centroid,
        )
        scene.show(caption=caption, smooth=False)


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    task_visualize_robot(config)


if __name__ == "__main__":
    main()
