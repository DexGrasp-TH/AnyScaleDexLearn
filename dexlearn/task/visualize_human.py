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
from dexlearn.utils.human_hand import (
    MANO_INDEX_MCP_IDX,
    get_wrist_translation_from_target,
    normalize_hand_pos_source,
)
from dexlearn.utils.util import set_seed
from dexlearn.utils.rot import numpy_quaternion_to_matrix
from dexlearn.network.models import *
from dexlearn.dataset import GRASP_TYPES

from manopth.manolayer import ManoLayer
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW


def get_output_dir(config: DictConfig) -> str:
    return os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )


def get_group_dirs(output_dir: str, include_groups):
    if include_groups is not None:
        return include_groups
    return sorted([name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))])


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


def reorder_samples_by_pred_grasp_type(sample_files, target_grasp_type_ids):
    grouped_files = {grasp_type_id: [] for grasp_type_id in target_grasp_type_ids}
    remaining_files = []

    for sample_file in sample_files:
        data = np.load(sample_file, allow_pickle=True).item()
        grasp_type_id = data.get("pred_grasp_type_id")
        if grasp_type_id is None:
            remaining_files.append(sample_file)
            continue
        grasp_type_id = int(grasp_type_id)
        if grasp_type_id in grouped_files:
            grouped_files[grasp_type_id].append(sample_file)
        else:
            remaining_files.append(sample_file)

    ordered_files = []
    while all(grouped_files[grasp_type_id] for grasp_type_id in target_grasp_type_ids):
        for grasp_type_id in target_grasp_type_ids:
            ordered_files.append(grouped_files[grasp_type_id].pop(0))

    for grasp_type_id in target_grasp_type_ids:
        ordered_files.extend(grouped_files[grasp_type_id])
    ordered_files.extend(remaining_files)
    return ordered_files


def infer_pc_source_from_sample_file(sample_file):
    """Infer point-cloud source of a saved sample result file."""
    base_name = os.path.basename(sample_file)
    if base_name.startswith("complete_point_cloud"):
        return "complete"
    if base_name.startswith("partial_pc"):
        return "partial"

    # Fallback for legacy naming: inspect saved metadata in the sample file.
    try:
        sample_data = np.load(sample_file, allow_pickle=True).item()
    except Exception:
        return "unknown"
    pc_path = str(sample_data.get("pc_path", ""))
    pc_base = os.path.basename(pc_path)
    if "processed_data" in pc_path or pc_base == "complete_point_cloud.npy":
        return "complete"
    if "vision_data" in pc_path or pc_base.startswith("partial_pc"):
        return "partial"
    return "unknown"


def normalize_object_scale(obj_scale, scene_path):
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Unsupported object scale shape {scale.shape} in scene config: {scene_path}")


def object_pose_to_rt(obj_pose, scene_path):
    pose = np.asarray(obj_pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]

    pose = pose.reshape(-1)
    if pose.size == 7:
        # DGN stores pose as [x, y, z, qw, qx, qy, qz].
        rot = numpy_quaternion_to_matrix(pose[3:7].reshape(1, 4))[0].astype(np.float32)
        return rot, pose[:3]

    raise ValueError(f"Unsupported object pose shape {pose.shape} in scene config: {scene_path}")


def extract_object_meta(scene_cfg, scene_path):
    """Extract object name, scale and pose transform from scene config."""
    if "object" in scene_cfg:
        obj_data = scene_cfg["object"]
        obj_name = obj_data.get("name")
        obj_scale = obj_data.get("rel_scale", obj_data.get("scale"))
        obj_pose = obj_data.get("pose")
    else:
        scene = scene_cfg.get("scene")
        if scene is None:
            raise KeyError(f"Could not find 'scene' in scene config: {scene_path}")

        obj_name = scene_cfg.get("task", {}).get("obj_name")
        if obj_name is None:
            candidates = [
                name
                for name, entry in scene.items()
                if isinstance(entry, dict) and "scale" in entry and "pose" in entry and name != "table"
            ]
            if not candidates:
                raise KeyError(f"Could not infer object name from scene config: {scene_path}")
            obj_name = candidates[0]

        if obj_name not in scene:
            raise KeyError(f"Object '{obj_name}' not found in scene config: {scene_path}")
        obj_data = scene[obj_name]
        obj_scale = obj_data.get("scale")
        obj_pose = obj_data.get("pose")

    if obj_name is None:
        raise KeyError(f"Could not find object name in scene config: {scene_path}")
    if obj_scale is None:
        raise KeyError(f"Could not find object scale for '{obj_name}' in scene config: {scene_path}")
    if obj_pose is None:
        raise KeyError(f"Could not find object pose for '{obj_name}' in scene config: {scene_path}")

    obj_scale_xyz = normalize_object_scale(obj_scale, scene_path)
    obj_rot, obj_trans = object_pose_to_rt(obj_pose, scene_path)
    return obj_name, obj_scale_xyz, obj_rot, obj_trans


def transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans):
    # complete_point_cloud.npy is normalized and must be transformed to world coordinates.
    scaled_pc = pc * obj_scale_xyz[None, :]
    return np.matmul(scaled_pc, obj_rot.T) + obj_trans[None, :]


def task_visualize_human(config: DictConfig) -> None:
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
    output_dir = get_output_dir(config)
    include_groups = get_group_dirs(output_dir, getattr(config.task, "include_groups", None))

    pc_source = str(getattr(config.test_data, "pc_source", "partial")).lower()
    if pc_source not in {"partial", "complete"}:
        raise ValueError(f"Unsupported pc_source={pc_source}. Expected one of ['partial', 'complete'].")

    sample_files = get_balanced_samples(output_dir, include_groups, config.task.num_samples_per_group)
    if len(sample_files) == 0:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")
    # Reorder saved grasps so visualization cycles through a fixed grasp-type sequence
    # such as 1, 2, 3, 4, 5 whenever those grasp types are available.
    sample_files = reorder_samples_by_pred_grasp_type(sample_files, [1, 2, 3, 4, 5])

    for sample_file in sample_files:
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
        if "pred_grasp_type_prob" in data:
            pred_grasp_type_prob = np.asarray(data["pred_grasp_type_prob"]).reshape(-1)
            prob_str = ", ".join([f"{p:.3f}" for p in pred_grasp_type_prob.tolist()])
            grasp_type_str += f" | pred_grasp_type_prob=[{prob_str}]"
        grasp_pos_source = normalize_hand_pos_source(data.get("grasp_pos_source", "wrist"))
        grasp_type_str += f" | Pos: {grasp_pos_source}"
        grasp_type_str += f" | PC: {infer_pc_source_from_sample_file(sample_file)}"

        ########### Process object pointcloud ###########
        if config.test_data.human:
            _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
            pc_path = resolve_dataset_path(data["pc_path"])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            pc = transform_complete_pc(raw_pc[idx], obj_scale_xyz, obj_rot, obj_trans)
        else:
            pc_path = resolve_dataset_path(data["pc_path"])
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            pc = raw_pc[idx]
            if pc_source == "complete":
                _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
                pc = transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans)

        # # Center point cloud
        # if config.data.pc_centering: # the saved results have already been decentered.
        #     pc_centroid = np.mean(pc, axis=-2, keepdims=True)
        #     pc = pc - pc_centroid  # normalization

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

            # --- MANO 推理 ---
            hand_target_pos = torch.from_numpy(p[:3]).to(config.device).float()
            hand_rot_mat = torch.from_numpy(hand_pose_np[:3, :3]).to(config.device).unsqueeze(0).float()

            # 构造参数：3维轴角 + 45维姿态
            mano_params = torch.cat(
                [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=config.device)], dim=-1
            )
            verts, joints = m_layer(mano_params, th_betas=torch.zeros((1, 10), device=config.device))
            wrist_trans = get_wrist_translation_from_target(hand_target_pos, joints[0], grasp_pos_source)

            # --- 渲染坐标轴 ---
            hand_pose_np[:3, 3] = wrist_trans.cpu().numpy()
            scene_elements.append(trimesh.creation.axis(transform=hand_pose_np, origin_size=0.01))

            # 变换坐标 (mm -> m) 并平移
            v_np = ((verts[0] / 1000.0) + wrist_trans).cpu().numpy()
            j_np = ((joints[0] / 1000.0) + wrist_trans).cpu().numpy()
            f_np = m_layer.th_faces.cpu().numpy()

            # --- 添加手部 Mesh ---
            scene_elements.extend(visualize_with_trimesh(v_np, f_np, None, color=colors[i]))
            if grasp_pos_source == "index_mcp":
                # Visualize the NN-predicted MCP frame directly.
                mcp_pose = np.eye(4)
                mcp_pose[:3, :3] = hand_rot_mat[0].cpu().numpy()
                mcp_pose[:3, 3] = hand_target_pos.cpu().numpy()
                scene_elements.append(
                    trimesh.creation.axis(
                        transform=mcp_pose,
                        origin_size=0.008,
                        axis_radius=0.003,
                        axis_length=0.08,
                    )
                )

        ############# Object pointcloud & World Axis ##############
        points = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 255])
        scene_elements.append(points)
        scene_elements.append(trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3))

        # 显示场景
        scene = trimesh.Scene(scene_elements)
        scene.show(caption=f"{os.path.basename(sample_file)}{grasp_type_str}")

    return
