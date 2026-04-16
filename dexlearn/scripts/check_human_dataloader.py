import sys
import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import trange
from torch.utils.data import DataLoader
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.network.models import *

from dexlearn.dataset import create_dataset, minkowski_collate_fn, InfLoader
from dexlearn.dataset.grasp_types import GRASP_TYPES
from manopth.manolayer import ManoLayer


class ManoConfig:
    def __init__(self, dataset_name):
        if dataset_name == "ContactPose":
            use_pca = True
            flat_hand_mean = False
            ncomps = 15
        elif dataset_name == "HOGraspNet":
            use_pca = False
            flat_hand_mean = True
            ncomps = 45
        elif dataset_name == "GRAB":
            use_pca = True
            flat_hand_mean = True
            ncomps = 24
        elif dataset_name == "OurHumanGraspFormat":
            use_pca = False
            flat_hand_mean = True
            ncomps = 45
        else:
            raise NotImplementedError()

        self.use_pca = use_pca
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps


def visualize_with_trimesh(verts, faces, joints=None, color=(200, 200, 250, 255)):
    """
    verts: (778, 3) numpy array
    faces: (1538, 3) numpy array
    joints: (21, 3) numpy array (optional)
    color: RGBA tuple for the mesh face color
    """
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    hand_mesh.visual.face_colors = color

    scene_contents = [hand_mesh]

    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red
            sphere.apply_translation(joint)
            scene_contents.append(sphere)

    return scene_contents


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)

    grasp_path = os.path.normpath(str(config.data.grasp_path))
    path_parts = grasp_path.split(os.sep)
    if "grasp" not in path_parts:
        raise ValueError(f"Invalid grasp_path (missing 'grasp' folder): {grasp_path}")
    grasp_idx = path_parts.index("grasp")
    if grasp_idx == 0:
        raise ValueError(f"Invalid grasp_path (cannot infer dataset name): {grasp_path}")
    dataset_name = path_parts[grasp_idx - 1]
    mano_cfg = ManoConfig(dataset_name)

    mano_layers = {
        side: ManoLayer(
            center_idx=0,
            mano_root="third_party/manopth/mano/models",
            side=side,
            use_pca=mano_cfg.use_pca,
            flat_hand_mean=mano_cfg.flat_hand_mean,
            ncomps=mano_cfg.ncomps,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)
        for side in ["right", "left"]
    }

    hand_colors = {
        "right": (200, 200, 250, 255),  # light blue
        "left": (250, 200, 200, 255),  # light red
    }

    train_dataset = create_dataset(config, mode="train")
    train_dataset.config.load_mano_params = True  # enable for visualization

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

    for it in trange(0, config.algo.max_iter):
        data = train_loader.get()

        for i in range(config.algo.batch_size):
            grasp_type_id = int(data["grasp_type_id"][i])
            grasp_type_name = GRASP_TYPES[grasp_type_id]
            scene_caption = f"path: {data['path'][i]}  |  grasp_type: {grasp_type_name}"

            scene_elements = []

            for side in ["right", "left"]:
                hand_trans = data[f"{side}_hand_trans"][i, 0, 0, :]  # (3,)
                hand_rot_mat = data[f"{side}_hand_rot"][i, 0, 0, ...]  # (3, 3)

                # Skip inactive hands (all-zero pose)
                if not hand_rot_mat.any():
                    continue

                ############# Hand base pose axis ##############
                hand_pose = torch.eye(4)
                hand_pose[:3, :3] = hand_rot_mat
                hand_pose[:3, 3] = hand_trans
                axis = trimesh.creation.axis(
                    transform=hand_pose.cpu().numpy(),
                    origin_size=0.01,
                    axis_radius=0.005,
                    axis_length=0.1,
                )
                scene_elements.append(axis)

                ############# MANO ##############
                hand_rot_aa = matrix_to_axis_angle(hand_rot_mat.unsqueeze(0))
                if f"{side}_mano_pose" in data:
                    mano_pose = data[f"{side}_mano_pose"][i].unsqueeze(0).to(config.device)  # (1, 24)
                    mano_betas = data[f"{side}_mano_betas"][i].unsqueeze(0).to(config.device)  # (1, 10)
                else:
                    mano_pose = torch.zeros((1, 24), device=config.device)
                    mano_betas = torch.zeros((1, 10), device=config.device)
                mano_params = torch.cat([hand_rot_aa, mano_pose], dim=-1)  # (1, 27)

                verts, joints = mano_layers[side](mano_params, th_betas=mano_betas)
                verts = (verts / 1000.0) + hand_trans.unsqueeze(0)
                joints = (joints / 1000.0) + hand_trans.unsqueeze(0)

                v_np = verts[0].cpu().detach().numpy()
                j_np = joints[0].cpu().detach().numpy()
                f_np = mano_layers[side].th_faces.cpu().numpy()

                scene_elements.extend(visualize_with_trimesh(v_np, f_np, j_np, color=hand_colors[side]))

            ############# Object pointcloud ##############
            pc_np = data["point_clouds"][i, ...].cpu().numpy()
            points = trimesh.points.PointCloud(pc_np, colors=[255, 165, 0, 255])  # Orange
            scene_elements.append(points)

            world_axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3)
            scene_elements.append(world_axis)

            scene = trimesh.Scene(scene_elements)
            # Start from a Z-up friendly oblique view for tabletop grasp inspection.
            scene.set_camera(
                angles=(np.deg2rad(80.0), 0.0, np.deg2rad(45.0)),
                distance=0.8,
                center=scene.centroid,
            )
            scene.show(caption=scene_caption)

    return


if __name__ == "__main__":
    main()
