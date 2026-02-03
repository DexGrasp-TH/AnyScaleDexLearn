import sys
import os

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
from manopth.manolayer import ManoLayer


def visualize_with_trimesh(verts, faces, joints=None):
    """
    verts: (778, 3) numpy array
    faces: (1538, 3) numpy array
    joints: (21, 3) numpy array (optional)
    """
    # 1. Create the mesh
    # Note: MANO faces are typically (1538, 3)
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # 2. Aesthetic updates
    hand_mesh.visual.face_colors = [200, 200, 250, 255]  # Light blue/grey

    scene_contents = [hand_mesh]

    # 3. Add joints as small spheres (optional)
    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)  # 5mm radius
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red
            sphere.apply_translation(joint)
            scene_contents.append(sphere)

    return scene_contents


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)

    mano_layer = ManoLayer(
        center_idx=0,
        mano_root="third_party/manopth/mano/models",
        side="right",
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
    ).to(config.device)

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

    for it in trange(0, config.algo.max_iter):
        data = train_loader.get()

        for i in range(config.algo.batch_size):
            print("path: ", data["path"][i])

            scene_elements = []

            ############# Hand base pose ##############
            hand_trans = data["hand_trans"][i, 0, 0, :]
            hand_pose = torch.eye(4)
            hand_pose[:3, :3] = data["hand_rot"][i, 0, 0, ...]
            hand_pose[:3, 3] = hand_trans
            hand_pose_np = hand_pose.cpu().numpy()
            # Create the coordinate axis at the hand_pose
            axis = trimesh.creation.axis(
                transform=hand_pose_np,
                origin_size=0.01,
                axis_radius=0.005,
                axis_length=0.1,
            )
            scene_elements.append(axis)

            ############# MANO ##############
            hand_rot = matrix_to_axis_angle(data["hand_rot"][i, 0, 0, ...].unsqueeze(0))
            mano_pose = torch.zeros((1, 45), device=config.device)
            mano_params = torch.cat([hand_rot, mano_pose], dim=-1)
            mano_betas = torch.zeros((1, 10), device=config.device)

            verts, joints = mano_layer(mano_params, th_betas=mano_betas)
            verts = (verts / 1000.0) + hand_trans.unsqueeze(0)
            joints = (joints / 1000.0) + hand_trans.unsqueeze(0)

            v_np = verts[0].cpu().detach().numpy()
            j_np = joints[0].cpu().detach().numpy()
            f_np = mano_layer.th_faces.cpu().numpy()

            scene_elements.extend(visualize_with_trimesh(v_np, f_np, j_np))

            ############# Object pointcloud ##############
            pc_np = data["point_clouds"][i, ...].cpu().numpy()
            points = trimesh.points.PointCloud(
                pc_np, colors=[255, 0, 0, 255]
            )  # Red points
            scene_elements.append(points)

            world_axis = trimesh.creation.axis(
                origin_size=0.01, axis_radius=0.001, axis_length=0.3
            )
            scene_elements.append(world_axis)

            # Create a scene to hold both objects
            scene = trimesh.Scene(scene_elements)
            scene.show()

    return


if __name__ == "__main__":
    main()
