import sys
import os
import json

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import trange
from torch.utils.data import DataLoader
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_quaternion
import pytorch_kinematics as pk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.network.models import *

from dexlearn.dataset import create_dataset, minkowski_collate_fn, InfLoader
from dexlearn.dataset.grasp_types import GRASP_TYPES
from manopth.manolayer import ManoLayer
from mr_utils.robot.pk_helper import PytorchKinematicsHelper
from mr_utils.robot.pk_visualizer import Visualizer


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)

    train_dataset = create_dataset(config, mode="train")

    config.algo.batch_size = 4
    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=0,
            shuffle=False,
            collate_fn=minkowski_collate_fn,
        ),
        config.device,
    )

    # Load URDF
    urdf_path = "../BimanBODex/src/curobo/content/assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf"
    mesh_dir_path = "../BimanBODex/src/curobo/content/assets/robot/shadow_hand"

    # Create robot model
    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=config.device)
    robot_helper = PytorchKinematicsHelper(
        chain,
        base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=config.device,
    )
    robot_visualizer = Visualizer(urdf_path, mesh_dir_path, device=config.device)

    metadata = os.path.join(config.data.grasp_path, "both_full/metadata.json")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Joint metadata file not found: {metadata}")
    with open(metadata, "r") as f:
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

    # Create serial chains and IK solvers for each wrist.
    for wrist_link_name in wrist_link_names:
        robot_helper.create_serial_chain(wrist_link_name)
        robot_helper.create_ik_solver(wrist_link_name, **ik_solver_kwargs)

    for it in trange(0, config.algo.max_iter):
        data = train_loader.get()

        for i in range(config.algo.batch_size):
            grasp_type_id = int(data["grasp_type_id"][i])
            grasp_type_name = GRASP_TYPES[grasp_type_id]
            print(f"path: {data['path'][i]}  |  grasp_type: {grasp_type_name}")

            scene_elements = []

            # Extract hand poses
            right_trans = data["right_hand_trans"][i, 0, 1, ...]
            right_rot = data["right_hand_rot"][i, 0, 1, ...]
            left_trans = data["left_hand_trans"][i, 0, 1, ...]
            left_rot = data["left_hand_rot"][i, 0, 1, ...]
            right_joint = data["right_hand_joint"][i, 0, 1, ...]
            left_joint = data["left_hand_joint"][i, 0, 1, ...]

            # Build 4x4 transformation matrices
            right_matrix = torch.eye(4, device=config.device)
            right_matrix[:3, :3] = right_rot
            right_matrix[:3, 3] = right_trans

            left_matrix = torch.eye(4, device=config.device)
            left_matrix[:3, :3] = left_rot
            left_matrix[:3, 3] = left_trans

            # Solve IK for each wrist pose.
            right_sol = robot_helper.solve_ik_batch(wrist_link_names[0], right_matrix.unsqueeze(0))
            left_sol = robot_helper.solve_ik_batch(wrist_link_names[1], left_matrix.unsqueeze(0))

            print(f"Right IK converged: {right_sol['success']}, Left IK converged: {left_sol['success']}")

            if right_joint.shape[-1] != len(right_hand_indices):
                raise ValueError(
                    f"Right hand joint dim mismatch: expected {len(right_hand_indices)}, got {right_joint.shape[-1]}"
                )
            if left_joint.shape[-1] != len(left_hand_indices):
                raise ValueError(
                    f"Left hand joint dim mismatch: expected {len(left_hand_indices)}, got {left_joint.shape[-1]}"
                )

            combined_q = torch.zeros((1, len(joint_names)), device=config.device, dtype=right_sol["q"].dtype)
            combined_q[:, right_arm_indices] = right_sol["q"][:, right_arm_indices]
            combined_q[:, left_arm_indices] = left_sol["q"][:, left_arm_indices]
            combined_q[:, right_hand_indices] = right_joint.unsqueeze(0).to(
                device=config.device, dtype=combined_q.dtype
            )
            combined_q[:, left_hand_indices] = left_joint.unsqueeze(0).to(device=config.device, dtype=combined_q.dtype)

            robot_pose = torch.zeros((1, 7 + combined_q.shape[-1]), device=config.device, dtype=torch.float32)
            robot_pose[:, 3] = 1.0
            robot_pose[:, 7:] = combined_q.to(dtype=torch.float32)
            robot_visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)
            scene_elements.append(robot_visualizer.get_robot_trimesh_data(i=0, color=(180, 180, 220, 255)))

            ############# Object pointcloud ##############
            pc_np = data["point_clouds"][i, ...].cpu().numpy()
            points = trimesh.points.PointCloud(pc_np, colors=[255, 165, 0, 255])  # Orange
            scene_elements.append(points)

            world_axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3)
            scene_elements.append(world_axis)

            scene = trimesh.Scene(geometry=scene_elements)
            scene.set_camera(
                angles=(np.deg2rad(60.0), 0.0, np.deg2rad(45.0)),
                distance=1.0,
                center=scene.centroid,
            )
            scene.show(caption=grasp_type_name)

            a = 1

    return


if __name__ == "__main__":
    main()
