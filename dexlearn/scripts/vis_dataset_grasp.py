"""Visualize one saved dataset grasp by rendering the robot hand pose and object mesh in trimesh."""

import numpy as np
import torch
import trimesh as tm
import yaml
import os
import json

import sys
from pathlib import Path

# Get parent of parent
parent_parent = Path(__file__).resolve().parents[2]
sys.path.append(str(parent_parent))

from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW
from mr_utils.robot.pk_visualizer import Visualizer


if __name__ == "__main__":
    robot_urdf_path = "../BimanBODex/src/curobo/content/assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf"
    mesh_dir_path = "../BimanBODex/src/curobo/content/assets/robot/shadow_hand"
    visualizer = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    grasp_file_path = "/data/dataset/AnyScaleGrasp/BimanBODex/shadow/both_full/core_bottle_47ebee26ca51177ac3fdab075b98c5d8/tabletop_ur10e/scale028_pose002_0.npy"
    pose_idx = 0

    grasp_data = np.load(grasp_file_path, allow_pickle=True).item()
    scene_path = grasp_data["scene_path"]
    scene_data = np.load(scene_path, allow_pickle=True).item()

    joint_names = grasp_data["joint_names"][0].tolist()

    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    grasp_qpos = torch.tensor(grasp_data["grasp_qpos"][pose_idx], dtype=torch.float32)
    if "dummy_arm" not in robot_urdf_path:
        robot_pose = grasp_qpos.unsqueeze(0)
    else:
        # Concatenate an identity global pose to obtain the full robot pose expected by the visualizer.
        global_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        robot_pose = torch.cat([global_pose, grasp_qpos], dim=0).unsqueeze(0)
    visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)
    robot_mesh = visualizer.get_robot_trimesh_data(i=0, color=[255, 0, 0])

    axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
    scene = tm.Scene(geometry=[robot_mesh, obj_mesh, axis])
    scene.show(smooth=False)
