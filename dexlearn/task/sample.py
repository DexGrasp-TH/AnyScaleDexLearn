import sys
import os
import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.config import resolve_type_supervision_config
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


def _sample_selection_config(config: DictConfig) -> tuple[str, float, float]:
    """Read candidate selection hyperparameters from the algo config.

    Args:
        config: Full Hydra config used by ``task=sample``.

    Returns:
        Tuple ``(mode, translation_scale_m, rotation_weight)``. ``mode`` is one
        of ``prob``, ``random``, or ``pose_diversity``.
    """
    selection_cfg = getattr(config.algo, "sample_selection", None)
    mode = str(getattr(selection_cfg, "mode", "prob")).strip().lower() if selection_cfg is not None else "prob"
    valid_modes = {"prob", "random", "pose_diversity"}
    if mode not in valid_modes:
        raise ValueError(f"sample_selection.mode must be one of {sorted(valid_modes)}, got {mode!r}")
    translation_scale_m = float(getattr(selection_cfg, "translation_scale_m", 0.05)) if selection_cfg is not None else 0.05
    rotation_weight = float(getattr(selection_cfg, "rotation_weight", 1.0)) if selection_cfg is not None else 1.0
    if translation_scale_m <= 0.0:
        raise ValueError("sample_selection.translation_scale_m must be positive")
    if rotation_weight < 0.0:
        raise ValueError("sample_selection.rotation_weight must be non-negative")
    return mode, translation_scale_m, rotation_weight


def _pose_diversity_feature(
    robot_pose: torch.Tensor,
    translation_scale_m: float,
    rotation_weight: float,
) -> torch.Tensor:
    """Convert generated poses into features used for diversity selection.

    Args:
        robot_pose: Candidate pose tensor shaped ``(B, N, ..., D)``.
        translation_scale_m: Meter scale used to normalize position channels.
        rotation_weight: Multiplicative weight for normalized quaternion
            channels.

    Returns:
        Feature tensor shaped ``(B, N, F)`` for farthest-point selection.
    """
    if robot_pose.ndim < 3:
        raise ValueError(f"Expected robot_pose with at least 3 dims, got {tuple(robot_pose.shape)}")
    batch_size, candidate_num = robot_pose.shape[:2]
    pose = robot_pose.reshape(batch_size, candidate_num, -1, robot_pose.shape[-1])
    pose_dim = pose.shape[-1]

    if pose_dim % 7 != 0:
        feature = robot_pose.reshape(batch_size, candidate_num, -1)
        feature_std = feature.std(dim=1, keepdim=True).clamp_min(1e-6)
        return feature / feature_std

    hand_pose = pose.reshape(batch_size, candidate_num, -1, 7)
    pos = hand_pose[..., :3] / float(translation_scale_m)
    quat = F.normalize(hand_pose[..., 3:7], dim=-1, eps=1e-8)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat) * float(rotation_weight)
    return torch.cat([pos, quat], dim=-1).reshape(batch_size, candidate_num, -1)


def _pose_diversity_indices(
    robot_pose: torch.Tensor,
    log_prob: torch.Tensor,
    topk: int,
    translation_scale_m: float,
    rotation_weight: float,
) -> torch.Tensor:
    """Select diverse candidates with greedy farthest-point sampling.

    Args:
        robot_pose: Candidate pose tensor shaped ``(B, N, ..., D)``.
        log_prob: Candidate log probabilities shaped ``(B, N)``.
        topk: Number of candidates to keep.
        translation_scale_m: Meter scale used to normalize position channels.
        rotation_weight: Multiplicative weight for normalized quaternion
            channels.

    Returns:
        Long index tensor shaped ``(B, topk)``. The first selected candidate is
        the highest-probability candidate; subsequent candidates maximize their
        minimum feature distance to the already selected set.
    """
    feature = _pose_diversity_feature(robot_pose, translation_scale_m, rotation_weight)
    batch_size, candidate_num = feature.shape[:2]
    selected = torch.empty((batch_size, topk), dtype=torch.long, device=robot_pose.device)

    for batch_index in range(batch_size):
        batch_feature = feature[batch_index]
        first_index = int(torch.argmax(log_prob[batch_index]).item())
        selected[batch_index, 0] = first_index
        chosen_mask = torch.zeros(candidate_num, dtype=torch.bool, device=robot_pose.device)
        chosen_mask[first_index] = True
        min_distance = torch.cdist(batch_feature[first_index : first_index + 1], batch_feature).squeeze(0)

        for out_index in range(1, topk):
            score = min_distance.masked_fill(chosen_mask, -1.0)
            next_index = int(torch.argmax(score).item())
            selected[batch_index, out_index] = next_index
            chosen_mask[next_index] = True
            next_distance = torch.cdist(batch_feature[next_index : next_index + 1], batch_feature).squeeze(0)
            min_distance = torch.minimum(min_distance, next_distance)

    return selected


def _candidate_selection_indices(robot_pose: torch.Tensor, log_prob: torch.Tensor, config: DictConfig) -> torch.Tensor:
    """Choose which sampled candidates should be saved.

    Args:
        robot_pose: Candidate pose tensor shaped ``(B, N, ..., D)``.
        log_prob: Candidate log probabilities shaped ``(B, N)``.
        config: Full Hydra config with ``algo.test_topk`` and optional
            ``algo.sample_selection`` fields.

    Returns:
        Long index tensor shaped ``(B, K)`` selecting candidates along dim 1.
    """
    mode, translation_scale_m, rotation_weight = _sample_selection_config(config)
    candidate_num = int(log_prob.shape[1])
    topk = int(config.algo.test_topk)
    if topk <= 0:
        raise ValueError("algo.test_topk must be positive")
    if topk > candidate_num:
        raise ValueError(f"algo.test_topk={topk} exceeds generated candidate count {candidate_num}")

    if mode == "prob":
        return torch.topk(log_prob, topk, dim=1).indices
    if mode == "random":
        random_scores = torch.rand(log_prob.shape, device=log_prob.device, dtype=log_prob.dtype)
        return torch.topk(random_scores, topk, dim=1).indices
    return _pose_diversity_indices(robot_pose, log_prob, topk, translation_scale_m, rotation_weight)


def _gather_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one tensor on the candidate axis.

    Args:
        value: Tensor whose first two dimensions are batch and candidate.
        indices: Long index tensor shaped ``(B, K)``.

    Returns:
        Tensor with candidate dimension reduced to ``K``.
    """
    gather_index = indices
    for _ in range(value.ndim - indices.ndim):
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*indices.shape, *value.shape[2:])
    return torch.gather(value, dim=1, index=gather_index)


def task_sample(config: DictConfig):
    resolve_type_supervision_config(config)
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

            if isinstance(result, dict):
                pred_grasp_type_prob = result.get("pred_grasp_type_prob")
                pred_grasp_type = result.get("pred_grasp_type_id")
                if pred_grasp_type_prob is None:
                    raise KeyError("Score-only sample result must contain pred_grasp_type_prob")
                if pred_grasp_type_prob.ndim == 2:
                    pred_grasp_type_prob = pred_grasp_type_prob.unsqueeze(1)
                if pred_grasp_type is not None and pred_grasp_type.ndim == 1:
                    pred_grasp_type = pred_grasp_type.unsqueeze(1)

                save_dict = {
                    "scene_path": data["scene_path"],
                    "pred_grasp_type_prob": pred_grasp_type_prob,
                }
                if "pc_path" in data:
                    save_dict["pc_path"] = data["pc_path"]
                if "grasp_type_id" in data:
                    save_dict["grasp_type_id"] = data["grasp_type_id"]
                if pred_grasp_type is not None:
                    save_dict["pred_grasp_type_id"] = pred_grasp_type

                logger.save_samples(save_dict, ckpt_iter, data["save_path"])
                continue

            pred_grasp_type_prob = None
            if len(result) == 4:
                robot_pose, pred_grasp_type, pred_grasp_type_prob, log_prob = result
            elif len(result) == 3:
                robot_pose, pred_grasp_type, log_prob = result
            else:
                robot_pose, log_prob = result
                pred_grasp_type = None

            topk_indices = _candidate_selection_indices(robot_pose, log_prob, config)
            robot_pose = _gather_candidates(robot_pose, topk_indices)
            log_prob = _gather_candidates(log_prob, topk_indices)
            if pred_grasp_type is not None:
                pred_grasp_type = _gather_candidates(pred_grasp_type, topk_indices)
            if pred_grasp_type_prob is not None:
                pred_grasp_type_prob = _gather_candidates(pred_grasp_type_prob, topk_indices)

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
