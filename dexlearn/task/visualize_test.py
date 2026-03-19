import sys
import os
from os.path import join as pjoin
import argparse
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle
from glob import glob
import numpy as np
import random
import re
from collections import defaultdict
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.network.models import *
from dexlearn.dataset import GRASP_TYPES

from manopth.manolayer import ManoLayer
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW


def resolve_dataset_path(path):
    """Replace /AnyScaleGrasp/ prefix with AnyScaleGraspDataset env var."""
    if "/AnyScaleGrasp/" in path:
        path = path.split("/AnyScaleGrasp/", 1)[1]
        dataset_root = os.environ.get("AnyScaleGraspDataset")
        if not dataset_root:
            raise ValueError("AnyScaleGraspDataset environment variable not set")
        return os.path.join(dataset_root, path)
    return path


def visualize_with_trimesh(verts, faces, joints=None, color=[200, 200, 250, 255]):
    """
    verts: (778, 3) numpy array
    faces: (1538, 3) numpy array
    joints: (21, 3) numpy array (optional)
    """
    # 1. Create the mesh
    # Note: MANO faces are typically (1538, 3)
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # 2. Aesthetic updates
    hand_mesh.visual.face_colors = color

    scene_contents = [hand_mesh]

    # 3. Add joints as small spheres (optional)
    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)  # 5mm radius
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red
            sphere.apply_translation(joint)
            scene_contents.append(sphere)

    return scene_contents


def get_balanced_samples(root_dir, search_pattern, include_groups, num_samples=20):
    """
    从指定的 grasp_type 文件夹中均匀采样。

    Args:
        root_dir: 基础目录 (output_dir)。
        search_pattern: 匹配模式 (如 "**/*.npy")。
        include_groups: 指定的文件夹列表，如 ["0_right", "1_left", "2_both"]。
        num_samples: 总采样数量。
    """

    samples = []
    for group in include_groups:
        group_path = os.path.join(root_dir, group)
        if not os.path.exists(group_path):
            raise ValueError(f"指定的分组目录不存在: {group_path}")
        group_files = glob(os.path.join(group_path, search_pattern), recursive=True)
        if not group_files:
            raise ValueError(f"指定的分组目录没有找到文件: {group_path}")

        random_sample_files = random.sample(group_files, k=min(len(group_files), num_samples))
        samples.extend(random_sample_files)

    # random.shuffle(samples)
    return samples


def task_visualize_test(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)

    mano_layers = {}
    for side in ["left", "right"]:
        mano_layers[side] = ManoLayer(
            center_idx=0,
            mano_root="third_party/manopth/mano/models",
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)

    ############### Find and sample grasp files ###############
    output_dir = os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )
    search_pattern = "**/*.npy"  # choose pattern

    random_sample_files = get_balanced_samples(
        root_dir=output_dir,
        search_pattern=search_pattern,
        include_groups=config.test_data.grasp_type_lst,
        num_samples=config.task.num_samples_per_group,
    )

    for sample_file in random_sample_files:
        print(f"Processing {sample_file}")

        data = np.load(sample_file, allow_pickle=True).item()
        grasp_pose = data["grasp_pose"]
        scene_path = resolve_dataset_path(data["scene_path"])
        scene_cfg = np.load(scene_path, allow_pickle=True).item()

        # Get grasp type info if available
        grasp_type_str = ""
        if "grasp_type_id" in data:
            gt_type_id = int(data["grasp_type_id"])
            grasp_type_str += f" | Given: {GRASP_TYPES[gt_type_id]}"
        if "pred_grasp_type_id" in data:
            pred_type_id = int(data["pred_grasp_type_id"])
            grasp_type_str += f" | Pred: {GRASP_TYPES[pred_type_id]}"

        ########### Process object pointcloud ###########
        if config.test_data.human:
            obj_name = scene_cfg["object"]["name"]
            obj_scale = scene_cfg["object"]["rel_scale"]
            obj_pose = scene_cfg["object"]["pose"]
            pc_path = resolve_dataset_path(data["pc_path"])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            scaled_pc = raw_pc[idx] * obj_scale  # re-scale the raw mesh (pointcloud) to the actual scale
            R = obj_pose[:3, :3]  # (3, 3)
            t = obj_pose[:3, 3]  # (3,)
            transformed_pc = np.matmul(scaled_pc, R.T) + t  # transform the object pointcloud based on the object pose
            pc = transformed_pc
        else:
            # obj_name = scene_cfg["task"]["obj_name"]
            # obj_scale = scene_cfg["scene"][obj_name]["scale"]
            # assert obj_scale[0] == obj_scale[1] == obj_scale[2], "Non-uniform scale not supported."
            # obj_scale = obj_scale[0]
            # obj_pose7d = scene_cfg["scene"][obj_name]["pose"]
            # obj_pose = posQuat2Isometry3d(obj_pose7d[:3], quatWXYZ2XYZW(obj_pose7d[3:7]))

            # DGN 数据集中的物体已经是正确的尺度和位姿，无需额外处理
            pc_path = data["pc_path"]
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            pc = raw_pc[idx]

        # Center point cloud
        if config.data.pc_centering:
            pc_centroid = np.mean(pc, axis=-2, keepdims=True)
            pc = pc - pc_centroid  # normalization

        ############### Visualization ###############
        scene_elements = []

        ############# Hand base pose ##############
        # 1. 自动切分位姿：每 7 维一组 (pos:3, quat:4)
        # 如果是 7 维则得到 1 组，14 维则得到 2 组
        poses = np.split(grasp_pose, len(grasp_pose) // 7)
        side_names = ["right", "left"]
        colors = [
            [180, 200, 255, 220],
            [210, 190, 250, 220],
        ]

        for i, p in enumerate(poses):
            # 屏蔽判定：如果位置和四元数全为 0，则跳过
            if np.allclose(p, 0, atol=1e-3):
                continue

            hand_pose_np = posQuat2Isometry3d(p[:3], quatWXYZ2XYZW(p[3:7]))
            side = side_names[i]
            m_layer = mano_layers[side]

            # --- 渲染坐标轴 ---
            scene_elements.append(trimesh.creation.axis(transform=hand_pose_np, origin_size=0.01))

            # --- MANO 推理 ---
            hand_trans = torch.from_numpy(hand_pose_np[:3, 3]).to(config.device).float()
            hand_rot_mat = torch.from_numpy(hand_pose_np[:3, :3]).to(config.device).unsqueeze(0).float()

            # 构造参数：3维轴角 + 45维姿态
            mano_params = torch.cat(
                [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=config.device)], dim=-1
            )
            verts, joints = m_layer(mano_params, th_betas=torch.zeros((1, 10), device=config.device))

            # 变换坐标 (mm -> m) 并平移
            v_np = ((verts[0] / 1000.0) + hand_trans).cpu().numpy()
            j_np = ((joints[0] / 1000.0) + hand_trans).cpu().numpy()
            f_np = m_layer.th_faces.cpu().numpy()

            # --- 添加手部 Mesh ---
            scene_elements.extend(visualize_with_trimesh(v_np, f_np, None, color=colors[i]))

        ############# Object pointcloud & World Axis ##############
        points = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 255])
        scene_elements.append(points)
        scene_elements.append(trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3))

        # 显示场景
        scene = trimesh.Scene(scene_elements)
        scene.show(caption=f"{os.path.basename(sample_file)}{grasp_type_str}")

    return
