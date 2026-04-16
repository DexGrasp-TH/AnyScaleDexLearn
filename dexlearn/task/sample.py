import sys
import os
import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.human_hand import normalize_hand_pos_source
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_test_dataloader
from dexlearn.network.models import *


def _expand_centroid(pc_centroid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand centroid to (B, 1, ..., 1, 3) matching target dims before the xyz axis."""
    centroid = pc_centroid.to(target.device, dtype=target.dtype)

    # Accept centroid shaped (B, 3) or (B, 1, 3); collapse extra singleton dims if present.
    if centroid.ndim == 3 and centroid.shape[1] == 1:
        centroid = centroid[:, 0, :]
    elif centroid.ndim > 2:
        centroid = centroid.reshape(centroid.shape[0], -1, 3).mean(dim=1)
    elif centroid.ndim == 1:
        centroid = centroid.unsqueeze(0)

    if centroid.ndim != 2 or centroid.shape[-1] != 3:
        raise ValueError(f"Unexpected pc_centroid shape: {tuple(pc_centroid.shape)}")

    view_shape = [centroid.shape[0]] + [1] * (target.ndim - 2) + [3]
    return centroid.reshape(*view_shape)


def _decenter_human_pose(
    robot_pose: torch.Tensor, pc_centroid: torch.Tensor, grasp_type_id: torch.Tensor = None
) -> torch.Tensor:
    # robot_pose: (B, K, P, D), D is 7*n_hands (pos3 + quat4 per hand)
    if robot_pose.shape[-1] < 3:
        return robot_pose
    out = robot_pose.clone()
    centroid = _expand_centroid(pc_centroid, out)
    out[..., :3] = out[..., :3] + centroid

    # Add centroid to left-hand translation only for bimanual grasp types (ids 4,5).
    if grasp_type_id is None or out.shape[-1] < 14:
        return out

    both_mask = grasp_type_id >= 4
    if not torch.any(both_mask):
        return out

    both_mask = both_mask[..., None, None]
    out[..., 7:10] = out[..., 7:10] + centroid * both_mask.to(dtype=out.dtype)
    return out


def _decenter_robot_pose(
    robot_pose: torch.Tensor, pc_centroid: torch.Tensor, grasp_type_id: torch.Tensor = None
) -> torch.Tensor:
    # robot_pose: (B, K, P, D), right translation at [0:3]
    # If bimanual, layout is right_wrist(7) + right_joint + left_wrist(7) + left_joint.
    if robot_pose.shape[-1] < 3:
        return robot_pose
    out = robot_pose.clone()
    centroid = _expand_centroid(pc_centroid, out)
    out[..., :3] = out[..., :3] + centroid

    # Add centroid to left-hand translation only for bimanual grasp types (ids 4,5).
    if grasp_type_id is None or out.shape[-1] < 14:
        return out

    both_mask = grasp_type_id >= 4
    if not torch.any(both_mask):
        return out

    pose_dim = out.shape[-1]
    if pose_dim < 14:
        return out
    right_joint_dim = (pose_dim - 14) // 2
    if 14 + 2 * right_joint_dim != pose_dim:
        return out
    left_offset = 7 + right_joint_dim

    both_mask = both_mask[..., None, None]
    out[..., left_offset : left_offset + 3] = out[..., left_offset : left_offset + 3] + centroid * both_mask.to(
        dtype=out.dtype
    )
    return out


def task_sample(config: DictConfig):
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    logger = Logger(config)
    test_loader = create_test_dataloader(config)

    model = eval(config.algo.model.name)(config.algo.model)

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ckpt_iter = ckpt["iter"]
        print("loaded ckpt from", config.ckpt)
    else:
        print("Find no ckpt!")
        exit(1)

    # training
    model.to(config.device)
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Sampling grasps (inference)"):
            result = model.sample(data, config.algo.test_grasp_num)

            pred_grasp_type_prob = None
            if len(result) == 4:
                robot_pose, pred_grasp_type, pred_grasp_type_prob, log_prob = result
            elif len(result) == 3:
                robot_pose, pred_grasp_type, log_prob = result
            else:
                robot_pose, log_prob = result
                pred_grasp_type = None

            # select top k predictions with higher log_prob
            topk_indices = torch.topk(log_prob, config.algo.test_topk, dim=1).indices
            batch_indices = torch.arange(robot_pose.size(0)).unsqueeze(1).expand(-1, config.algo.test_topk)
            robot_pose = robot_pose[batch_indices, topk_indices]
            log_prob = log_prob[batch_indices, topk_indices]
            if pred_grasp_type is not None:
                pred_grasp_type = pred_grasp_type[batch_indices, topk_indices]
            if pred_grasp_type_prob is not None:
                pred_grasp_type_prob = pred_grasp_type_prob[batch_indices, topk_indices]

            if "pc_centroid" in data:
                grasp_type_for_decenter = pred_grasp_type if pred_grasp_type is not None else data["grasp_type_id"]
                if grasp_type_for_decenter.ndim == 1:
                    grasp_type_for_decenter = grasp_type_for_decenter.unsqueeze(1).expand(-1, robot_pose.shape[1])
                if config.algo.human:
                    robot_pose = _decenter_human_pose(robot_pose, data["pc_centroid"], grasp_type_for_decenter)
                else:
                    robot_pose = _decenter_robot_pose(robot_pose, data["pc_centroid"], grasp_type_for_decenter)

            if config.algo.human:
                grasp_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
                save_dict = {
                    "grasp_pose": robot_pose[..., 0, :],
                    "grasp_error": -log_prob,
                    "scene_path": data["scene_path"],
                    "pc_path": data["pc_path"],
                    "grasp_pos_source": grasp_pos_source,
                }
            else:
                save_dict = {
                    "pregrasp_qpos": robot_pose[..., 0, :],
                    "grasp_qpos": robot_pose[..., 1, :],
                    "squeeze_qpos": robot_pose[..., 2, :],
                    "grasp_error": -log_prob,
                    "scene_path": data["scene_path"],
                }

            if "grasp_type_id" in data:
                save_dict["grasp_type_id"] = data["grasp_type_id"]
            if pred_grasp_type is not None:
                save_dict["pred_grasp_type_id"] = pred_grasp_type
            if pred_grasp_type_prob is not None:
                save_dict["pred_grasp_type_prob"] = pred_grasp_type_prob

            logger.save_samples(save_dict, ckpt_iter, data["save_path"])

    return
