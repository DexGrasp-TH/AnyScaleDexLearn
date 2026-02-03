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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.network.models import *

from manopth.manolayer import ManoLayer
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW


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

    object_pc_folder = pjoin(config.data.object_path, config.data.pc_path)

    # Search all grasp sample files
    output_dir = config.ckpt.replace("ckpts", "tests").replace(".pth", "")
    search_pattern = os.path.join(output_dir, "**", "*.npy")  # choose pattern
    sample_files = sorted(glob(search_pattern, recursive=True))

    # 随机采样 10 个
    num_samples = min(len(sample_files), 10)
    random_sample_files = random.sample(sample_files, k=num_samples)

    for sample_file in random_sample_files:
        print(f"Processing {sample_file}")

        data = np.load(sample_file, allow_pickle=True).item()
        grasp_pose = data["grasp_pose"]
        scene_path = data["scene_path"]
        scene_cfg = np.load(scene_path, allow_pickle=True).item()

        ########### Process object pointcloud ###########
        if config.data.human:
            obj_name = scene_cfg["object"]["name"]
            obj_scale = scene_cfg["object"]["rel_scale"]
            obj_pose = scene_cfg["object"]["pose"]
            pc_path = data["pc_path"]
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            scaled_pc = raw_pc[idx] * obj_scale  # re-scale the raw mesh (pointcloud) to the actual scale
            R = obj_pose[:3, :3]  # (3, 3)
            t = obj_pose[:3, 3]  # (3,)
            transformed_pc = np.matmul(scaled_pc, R.T) + t  # transform the object pointcloud based on the object pose
            pc = transformed_pc
        else:
            pc_path = data["pc_path"]
            raw_pc = np.load(pc_path, allow_pickle=True)
            idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
            pc = raw_pc[idx]

        if config.data.pc_centering:
            pc_centroid = np.mean(pc, axis=-2, keepdims=True)
            pc = pc - pc_centroid  # normalization

        ############### Visualization ###############

        scene_elements = []

        ############# Hand base pose ##############
        hand_pose_np = posQuat2Isometry3d(grasp_pose[:3], quatWXYZ2XYZW(grasp_pose[3:]))
        # Create the coordinate axis at the hand_pose
        axis = trimesh.creation.axis(
            transform=hand_pose_np,
            origin_size=0.01,
            axis_radius=0.005,
            axis_length=0.1,
        )
        scene_elements.append(axis)

        ############# MANO ##############
        hand_trans = torch.tensor(hand_pose_np[:3, 3], device=config.device).float()
        hand_rot = matrix_to_axis_angle(torch.tensor(hand_pose_np[:3, :3], device=config.device).unsqueeze(0)).float()
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
        pc_np = pc
        points = trimesh.points.PointCloud(pc_np, colors=[255, 0, 0, 255])  # Red points
        scene_elements.append(points)

        world_axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3)
        scene_elements.append(world_axis)

        # Create a scene to hold both objects
        scene = trimesh.Scene(scene_elements)
        scene.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        required=True,
        help="experiment folder, e.g. shadow_tabletop_debug",
    )
    args, unknown = parser.parse_known_args()

    sys.argv = sys.argv[:1] + unknown + list(OmegaConf.load(f"output/{args.exp_name}/.hydra/overrides.yaml"))

    # remove duplicated args. Note: cmd has the priority!
    check_dict = {}
    for argv in sys.argv[1:]:
        arg_key = argv.split("=")[0]
        if arg_key not in check_dict:
            check_dict[arg_key] = True
        else:
            sys.argv.remove(argv)

    ################### hydra.main call ###################
    main()
