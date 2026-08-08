import copy
import hashlib
import json
import os
import sys
from glob import glob
from os.path import join as pjoin
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.dataset import GRASP_TYPES, create_test_dataloader, get_sparse_tensor
from dexlearn.network.models import *  # noqa: F401,F403
from dexlearn.task.sample import (
    _candidate_selection_indices,
    _decenter_human_pose,
    _gather_candidates,
    _sample_selection_config,
)
from dexlearn.utils.config import resolve_type_supervision_config
from dexlearn.utils.human_hand import normalize_hand_pos_source
from dexlearn.utils.util import load_json, set_seed


REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_GRASP_TYPE_NAMES = tuple(GRASP_TYPES[type_id] for type_id in REAL_GRASP_TYPE_IDS)
REVERSE_FACTORIZATION = "reverse_T_to_C"
PROPOSED_FACTORIZATION = "proposed_C_to_T"


def factorization_from_config(config: DictConfig) -> str:
    """Return the Human Prior factorization recorded in exported artifacts."""
    factorization = str(getattr(config.algo, "factorization", PROPOSED_FACTORIZATION))
    valid_factorizations = {PROPOSED_FACTORIZATION, REVERSE_FACTORIZATION}
    if factorization not in valid_factorizations:
        raise ValueError(
            f"Unsupported Human Prior factorization={factorization!r}; "
            f"expected one of {sorted(valid_factorizations)}"
        )
    return factorization


def is_reverse_factorization(config: DictConfig) -> bool:
    """Check whether the config selects the Reverse T-to-C pipeline."""
    return factorization_from_config(config) == REVERSE_FACTORIZATION


def reverse_sampling_config(config: DictConfig) -> dict:
    """Resolve and validate Reverse shared-pool sampling options.

    Args:
        config: Full Hydra config containing ``algo.reverse_sampling``.

    Returns:
        Plain dictionary of validated Reverse sampling parameters.
    """
    sampling_cfg = getattr(config.algo, "reverse_sampling", None)
    if sampling_cfg is None:
        raise ValueError("Reverse export requires algo.reverse_sampling")
    pool_size = int(getattr(sampling_cfg, "marginal_pool_size", 500))
    conditional_num = int(getattr(sampling_cfg, "conditional_candidate_num", config.algo.test_grasp_num))
    policy = str(getattr(sampling_cfg, "resampling_policy", "weighted_without_replacement"))
    seed = int(getattr(sampling_cfg, "resampling_seed", config.seed))
    include_raw_pool = bool(getattr(sampling_cfg, "include_raw_pool", True))
    ess_warning_threshold = float(getattr(sampling_cfg, "ess_warning_threshold", config.algo.test_topk))
    if pool_size <= 0 or conditional_num <= 0:
        raise ValueError("Reverse pool and conditional candidate counts must be positive")
    if conditional_num > pool_size:
        raise ValueError("Reverse conditional_candidate_num must not exceed marginal_pool_size")
    if policy != "weighted_without_replacement":
        raise ValueError("Reverse export currently requires weighted_without_replacement")
    if int(config.algo.test_topk) > conditional_num:
        raise ValueError("algo.test_topk must not exceed Reverse conditional_candidate_num")
    if ess_warning_threshold < 0.0:
        raise ValueError("Reverse ess_warning_threshold must be non-negative")
    return {
        "marginal_pool_size": pool_size,
        "conditional_candidate_num": conditional_num,
        "resampling_policy": policy,
        "resampling_seed": seed,
        "include_raw_pool": include_raw_pool,
        "ess_warning_threshold": ess_warning_threshold,
    }


def _as_list(value: Any) -> list:
    """Convert a scalar or config list into a plain Python list.

    Args:
        value: Scalar, list-like config value, or ``None``.

    Returns:
        Plain list. ``None`` returns an empty list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _json_default(value: Any) -> Any:
    """Convert numpy values to JSON-compatible Python objects.

    Args:
        value: Object passed by ``json.dump`` when the default encoder fails.

    Returns:
        JSON-compatible representation of numpy arrays and scalar values.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_robot_name(config: DictConfig) -> str:
    """Read and validate the robot name used to namespace export files.

    Args:
        config: Full Hydra config containing ``task.robot_name``.

    Returns:
        Robot name that is safe to use as one path component.
    """
    robot_name = str(getattr(config.task, "robot_name", "shadow_hand")).strip()
    if not robot_name:
        raise ValueError("task.robot_name must be a non-empty string")
    invalid_parts = {os.sep}
    if os.altsep is not None:
        invalid_parts.add(os.altsep)
    if any(part in robot_name for part in invalid_parts):
        raise ValueError(f"task.robot_name must be one path component, got {robot_name!r}")
    return robot_name


def export_robot_size(config: DictConfig) -> float:
    """Read and validate the robot hand size ratio used for inference.

    Args:
        config: Full Hydra config containing ``task.robot_size``.

    Returns:
        Positive hand-size ratio relative to a human hand.
    """
    robot_size = float(getattr(config.task, "robot_size", 1.0))
    if robot_size <= 0.0:
        raise ValueError(f"task.robot_size must be positive, got {robot_size}")
    return robot_size


def export_pc_runtime_scale(config: DictConfig) -> float:
    """Compute the point-cloud runtime scale for robot-specific inference.

    Args:
        config: Full Hydra config containing ``task.robot_size``.

    Returns:
        Scale applied to centered test point clouds before backbone inference.
    """
    return 1.0 / export_robot_size(config)


def _checkpoint_name(ckpt_value: Any) -> str:
    """Build a checkpoint filename from a Hydra ``ckpt`` value.

    Args:
        ckpt_value: Checkpoint override such as ``010000`` or ``step_010000.pth``.

    Returns:
        File name that should exist under the run's ``ckpts`` directory.
    """
    ckpt_text = str(ckpt_value)
    if ckpt_text.endswith(".pth"):
        return os.path.basename(ckpt_text)
    if ckpt_text.startswith("step_"):
        return f"{ckpt_text}.pth"
    return f"step_{ckpt_text.zfill(6) if ckpt_text.isdigit() else ckpt_text}.pth"


def _checkpoint_step_label(ckpt_value: Any, checkpoint_iter: int | None) -> str:
    """Build a stable step label for output directory names.

    Args:
        ckpt_value: Checkpoint override, path, or ``None``.
        checkpoint_iter: Training iteration stored in the checkpoint.

    Returns:
        Six-digit step text when possible, otherwise the checkpoint stem.
    """
    if checkpoint_iter is not None:
        return f"{checkpoint_iter:06d}"
    if ckpt_value is None:
        return "unknown"
    checkpoint_stem = os.path.splitext(_checkpoint_name(ckpt_value))[0]
    if checkpoint_stem.startswith("step_"):
        checkpoint_stem = checkpoint_stem[len("step_") :]
    return checkpoint_stem


def resolve_checkpoint_path(config: DictConfig) -> str:
    """Resolve the checkpoint path without creating a wandb Logger.

    Args:
        config: Full Hydra config containing ``ckpt``, ``output_folder`` and
            ``wandb.id``.

    Returns:
        Absolute checkpoint path.
    """
    if config.ckpt is None:
        raise ValueError("task=obj_human_prior_export requires ckpt to be set")

    ckpt_text = str(config.ckpt)
    candidates = [ckpt_text, to_absolute_path(ckpt_text)]
    candidates.append(
        to_absolute_path(pjoin(str(config.output_folder), str(config.wandb.id), "ckpts", _checkpoint_name(ckpt_text)))
    )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint from ckpt={config.ckpt}. Tried: {candidates}")


def clone_config_for_checkpoint(
    config: DictConfig,
    exp_name: str | None,
    ckpt: str | None,
) -> DictConfig:
    """Clone the Hydra config and optionally point it at another run checkpoint.

    Args:
        config: Base export config.
        exp_name: Optional experiment name whose ``output/<wandb.id>/ckpts`` should be used.
        ckpt: Optional checkpoint step or path for the cloned config.

    Returns:
        Config clone with ``exp_name``, ``wandb.id`` and ``ckpt`` updated when
        overrides are provided.
    """
    branch_config = copy.deepcopy(config)
    with open_dict(branch_config):
        if exp_name is not None:
            branch_config.exp_name = str(exp_name)
            branch_config.wandb.id = f"{branch_config.data_name}_{branch_config.algo_name}_{branch_config.exp_name}"
        if ckpt is not None:
            branch_config.ckpt = str(ckpt)
    return branch_config


def clone_config_for_model_checkpoint(
    config: DictConfig,
    model_config: DictConfig,
    exp_name: str | None,
    ckpt: str | None,
) -> DictConfig:
    """Clone an export config for one heterogeneous model checkpoint."""
    branch_config = clone_config_for_checkpoint(config, exp_name, ckpt)
    with open_dict(branch_config.algo):
        branch_config.algo.model = copy.deepcopy(model_config)
    resolve_type_supervision_config(branch_config)
    return branch_config


def checkpoint_sha256(checkpoint_path: str) -> str:
    """Compute the SHA-256 digest of one checkpoint for export provenance."""
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as checkpoint_handle:
        for chunk in iter(lambda: checkpoint_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_export_model(config: DictConfig) -> tuple[torch.nn.Module, str, int | None]:
    """Instantiate the human prior model and load the requested checkpoint.

    Args:
        config: Full Hydra config containing the model definition and checkpoint.

    Returns:
        Tuple ``(model, checkpoint_path, checkpoint_iter)``.
    """
    model = eval(config.algo.model.name)(config.algo.model)
    checkpoint_path = resolve_checkpoint_path(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    checkpoint_iter = ckpt.get("iter")
    if checkpoint_iter is not None:
        checkpoint_iter = int(checkpoint_iter)
    model.to(config.device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model, checkpoint_path, checkpoint_iter


def load_reverse_export_models(config: DictConfig) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Load the independent posterior and marginal models for Reverse export."""
    score_exp_name = getattr(config.task, "score_exp_name", None) or f"{config.exp_name}_type_posterior"
    pose_exp_name = getattr(config.task, "pose_exp_name", None) or f"{config.exp_name}_pose_marginal"
    score_ckpt = getattr(config.task, "score_ckpt", None)
    pose_ckpt = getattr(config.task, "pose_ckpt", None)
    if score_ckpt is None or pose_ckpt is None:
        raise ValueError("Reverse export requires task.score_ckpt and task.pose_ckpt")

    posterior_cfg = OmegaConf.select(config, "algo.models.type_posterior")
    marginal_cfg = OmegaConf.select(config, "algo.models.pose_marginal")
    if posterior_cfg is None or marginal_cfg is None:
        raise ValueError("Reverse export requires algo.models.type_posterior and algo.models.pose_marginal")
    score_config = clone_config_for_model_checkpoint(config, posterior_cfg, score_exp_name, score_ckpt)
    pose_config = clone_config_for_model_checkpoint(config, marginal_cfg, pose_exp_name, pose_ckpt)
    posterior_model, posterior_path, posterior_iter = load_export_model(score_config)
    marginal_model, marginal_path, marginal_iter = load_export_model(pose_config)
    return posterior_model, marginal_model, {
        "checkpoint_path": marginal_path,
        "checkpoint_iter": marginal_iter,
        "score_checkpoint_path": posterior_path,
        "score_checkpoint_iter": posterior_iter,
        "score_checkpoint_sha256": checkpoint_sha256(posterior_path),
        "score_ckpt": score_config.ckpt,
        "pose_checkpoint_path": marginal_path,
        "pose_checkpoint_iter": marginal_iter,
        "pose_checkpoint_sha256": checkpoint_sha256(marginal_path),
        "pose_ckpt": pose_config.ckpt,
        "uses_independent_models": True,
        "model_roles": {
            "score": "pose_conditioned_type_posterior",
            "pose": "marginal_pose_generator",
        },
        "score_model_config": OmegaConf.to_container(score_config.algo.model, resolve=True),
        "pose_model_config": OmegaConf.to_container(pose_config.algo.model, resolve=True),
    }


def load_export_models(config: DictConfig) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Load the score and pose models used by object human-prior export.

    Args:
        config: Full Hydra config. ``task.score_*`` and ``task.pose_*`` may
            point to separate independent-network checkpoints.

    Returns:
        Tuple ``(score_model, pose_model, checkpoint_meta)``. The two models are
        the same object when no independent checkpoint override is configured.
    """
    if is_reverse_factorization(config):
        return load_reverse_export_models(config)

    score_exp_name = getattr(config.task, "score_exp_name", None)
    score_ckpt = getattr(config.task, "score_ckpt", None)
    pose_exp_name = getattr(config.task, "pose_exp_name", None)
    pose_ckpt = getattr(config.task, "pose_ckpt", None)
    use_independent_models = any(value is not None for value in (score_exp_name, score_ckpt, pose_exp_name, pose_ckpt))

    if not use_independent_models:
        model, checkpoint_path, checkpoint_iter = load_export_model(config)
        return model, model, {
            "checkpoint_path": checkpoint_path,
            "checkpoint_iter": checkpoint_iter,
            "score_checkpoint_path": checkpoint_path,
            "score_checkpoint_iter": checkpoint_iter,
            "pose_checkpoint_path": checkpoint_path,
            "pose_checkpoint_iter": checkpoint_iter,
            "uses_independent_models": False,
        }

    score_config = clone_config_for_checkpoint(config, score_exp_name, score_ckpt)
    pose_config = clone_config_for_checkpoint(config, pose_exp_name, pose_ckpt)
    score_model, score_checkpoint_path, score_checkpoint_iter = load_export_model(score_config)
    pose_model, pose_checkpoint_path, pose_checkpoint_iter = load_export_model(pose_config)
    return score_model, pose_model, {
        "checkpoint_path": pose_checkpoint_path,
        "checkpoint_iter": pose_checkpoint_iter,
        "score_checkpoint_path": score_checkpoint_path,
        "score_checkpoint_iter": score_checkpoint_iter,
        "score_ckpt": score_config.ckpt,
        "pose_checkpoint_path": pose_checkpoint_path,
        "pose_checkpoint_iter": pose_checkpoint_iter,
        "pose_ckpt": pose_config.ckpt,
        "uses_independent_models": True,
    }


def clone_config_with_grasp_types(config: DictConfig, grasp_types: list[str], split_name: str) -> DictConfig:
    """Clone config and override the test split plus grasp-type list.

    Args:
        config: Full Hydra config.
        grasp_types: Test-time grasp type names for this pass.
        split_name: Object split name, such as ``train`` or ``test``.

    Returns:
        Independent config clone for one dataloader pass.
    """
    pass_config = copy.deepcopy(config)
    with open_dict(pass_config.test_data):
        pass_config.test_data.grasp_type_lst = list(grasp_types)
        pass_config.test_data.test_split = str(split_name)
        pass_config.test_data.display_grasp_type_lst = list(REAL_GRASP_TYPE_NAMES)
        pass_config.test_data.pc_runtime_scale = export_pc_runtime_scale(config)
    return pass_config


def build_object_split_lookup(test_data_cfg: DictConfig, split_names: list[str]) -> dict[str, str]:
    """Read object split files for scene metadata.

    Args:
        test_data_cfg: Test-data config containing ``object_path`` and
            ``split_path``.
        split_names: Split names that should be indexed.

    Returns:
        Mapping from object id to split name.
    """
    split_root = to_absolute_path(pjoin(str(test_data_cfg.object_path), str(test_data_cfg.split_path)))
    split_lookup: dict[str, str] = {}
    for split_name in split_names:
        split_json = pjoin(split_root, f"{split_name}.json")
        if not os.path.isfile(split_json):
            continue
        for object_id in load_json(split_json):
            split_lookup[str(object_id)] = str(split_name)
    return split_lookup


def read_scene_metadata(scene_path: str) -> dict[str, str]:
    """Read scene id and object id from a scene config file.

    Args:
        scene_path: Path to a ``.npy`` scene config.

    Returns:
        Dictionary with ``scene_id`` and ``object_id``.
    """
    scene_cfg = np.load(scene_path, allow_pickle=True).item()
    scene_id = scene_cfg.get("scene_id")
    if scene_id is None and "scene" in scene_cfg:
        scene = scene_cfg["scene"]
        scene_id = scene.get("id", scene.get("scene_id"))
    if scene_id is None:
        raise KeyError(f"Could not find scene id in scene config: {scene_path}")

    object_id = str(scene_id).split("/")[0]
    if "object" in scene_cfg and isinstance(scene_cfg["object"], dict):
        object_id = str(scene_cfg["object"].get("name", object_id))
    elif isinstance(scene_cfg.get("task"), dict) and scene_cfg["task"].get("obj_name") is not None:
        object_id = str(scene_cfg["task"]["obj_name"])
    return {"scene_id": str(scene_id), "object_id": object_id}


def get_batch_value(data: dict, key: str, index: int) -> Any:
    """Return one sample value from a collated dataloader batch.

    Args:
        data: Batch dictionary returned by ``create_test_dataloader``.
        key: Field name to index.
        index: Batch row index.

    Returns:
        Single value from the requested batch field.
    """
    value = data[key]
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value):
        return value[index]
    return value[index]


def filter_batch_data(data: dict, keep_indices: list[int], config: DictConfig) -> dict:
    """Filter one collated batch to the selected row indices.

    Args:
        data: Batch dictionary returned by the finite test dataloader.
        keep_indices: Batch row indices to keep.
        config: Full Hydra config used to rebuild sparse tensor fields.

    Returns:
        Batch dictionary containing only selected rows. Sparse MinkowskiEngine
        fields are rebuilt from ``point_clouds`` because they encode batch ids.
    """
    if len(keep_indices) == len(data["scene_path"]):
        return data

    sparse_keys = {"coors", "feats", "original2quantize", "quantize2original"}
    filtered = {}
    for key, value in data.items():
        if key in sparse_keys:
            continue
        if isinstance(value, (list, tuple)):
            filtered[key] = [value[index] for index in keep_indices]
        elif torch.is_tensor(value):
            index_tensor = torch.as_tensor(keep_indices, device=value.device, dtype=torch.long)
            filtered[key] = value.index_select(0, index_tensor)
        elif isinstance(value, np.ndarray):
            filtered[key] = value[keep_indices]
        else:
            filtered[key] = value

    if "point_clouds" in filtered and "MinkUNet" in str(config.algo.model.backbone.name):
        filtered.update(get_sparse_tensor(filtered["point_clouds"], float(config.algo.model.backbone.voxel_size)))
    return filtered


def export_scene_dir(output_dir: str, config: DictConfig) -> str:
    """Build the root directory for per-scene export files.

    Args:
        output_dir: Resolved task output directory.
        config: Full Hydra config containing ``test_data.object_path`` and
            robot-specific export settings.

    Returns:
        Directory named after the object asset path and robot name.
    """
    object_path = to_absolute_path(str(config.test_data.object_path)).rstrip(os.sep)
    asset_name = os.path.basename(object_path)
    if not asset_name:
        raise ValueError(f"Could not infer asset name from object_path={config.test_data.object_path}")
    return pjoin(output_dir, asset_name, export_robot_name(config))


def scene_file_path(scene_dir: str, scene_id: str) -> str:
    """Build the per-scene export file path for a scene id.

    Args:
        scene_dir: Directory that stores per-scene ``.npy`` files.
        scene_id: Source scene id.

    Returns:
        Path to the per-scene export file.
    """
    relative_scene_id = str(scene_id).strip("/")
    scene_parts = [part for part in relative_scene_id.split("/") if part]
    if not scene_parts:
        raise ValueError(f"Cannot build export path from empty scene_id={scene_id!r}")
    if any(part in (".", "..") for part in scene_parts):
        raise ValueError(f"Cannot build export path from unsafe scene_id={scene_id!r}")
    return pjoin(scene_dir, *scene_parts) + ".npy"


def extract_real_type_scores(pred_grasp_type_prob: torch.Tensor | np.ndarray) -> np.ndarray:
    """Extract real grasp-type scores from model output.

    Args:
        pred_grasp_type_prob: Tensor shaped ``(B, 1, 5)``, ``(B, 5)``, or legacy
            ``(..., 6)`` including ``0_any``.

    Returns:
        Float32 array shaped ``(B, 5)`` aligned with ``GRASP_TYPES[1:]``.
    """
    if torch.is_tensor(pred_grasp_type_prob):
        scores = pred_grasp_type_prob.detach().cpu().numpy()
    else:
        scores = np.asarray(pred_grasp_type_prob)

    if scores.ndim == 3:
        scores = scores[:, 0, :]
    if scores.ndim != 2:
        raise ValueError(f"Expected score tensor with 2 or 3 dims, got shape {scores.shape}")
    if scores.shape[-1] == len(GRASP_TYPES):
        scores = scores[:, 1:]
    if scores.shape[-1] != len(REAL_GRASP_TYPE_IDS):
        raise ValueError(f"Expected 5 real-type scores, got shape {scores.shape}")
    return scores.astype(np.float32, copy=False)


def score_semantics_from_config(config: DictConfig) -> str:
    """Describe the numeric meaning of the exported type scores.

    Args:
        config: Full Hydra config containing ``algo.model.type_objective``.

    Returns:
        Human-readable score semantics for the manifest.
    """
    if is_reverse_factorization(config):
        return "monte_carlo_pose_posterior_probability"
    objective = str(getattr(config.algo.model, "type_objective", "ce")).lower()
    if objective != "ce":
        raise ValueError("Human prior export only supports CE type scores")
    return "softmax_probability"


def scene_split_for_record(object_id: str, fallback_split: str, split_lookup: dict[str, str]) -> str:
    """Choose the split label stored in exported scene metadata.

    Args:
        object_id: Object id parsed from the scene config.
        fallback_split: Split currently being iterated by the dataloader.
        split_lookup: Object id to split mapping built from split JSON files.

    Returns:
        Split label for this scene.
    """
    return split_lookup.get(str(object_id), str(fallback_split))


def sample_scene_budget_scores(
    config: DictConfig,
    model: torch.nn.Module,
    split_lookup: dict[str, str],
    skip_scene_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Run the score pass and collect one 5-type score vector per scene.

    Args:
        config: Full Hydra config.
        model: Loaded human prior model.
        split_lookup: Object id to split mapping for metadata.
        skip_scene_ids: Scene ids that already have a complete export.

    Returns:
        Mapping from scene id to score metadata and budget scores.
    """
    score_records: dict[str, dict] = {}
    skip_scene_ids = skip_scene_ids or set()
    score_grasp_types = _as_list(getattr(config.task, "score_grasp_types", ["0_any"]))
    for split_name in _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])):
        pass_config = clone_config_with_grasp_types(config, score_grasp_types, str(split_name))
        test_loader = create_test_dataloader(pass_config)
        desc = f"obj prior score [{split_name}]"
        for data in tqdm(test_loader, desc=desc):
            batch_metadata = [read_scene_metadata(scene_path) for scene_path in data["scene_path"]]
            keep_indices = [
                batch_idx for batch_idx, metadata in enumerate(batch_metadata) if metadata["scene_id"] not in skip_scene_ids
            ]
            if not keep_indices:
                continue
            data = filter_batch_data(data, keep_indices, config)
            batch_metadata = [batch_metadata[index] for index in keep_indices]

            result = model.sample(data, 1)
            if isinstance(result, dict):
                pred_grasp_type_prob = result["pred_grasp_type_prob"]
            elif len(result) >= 4:
                pred_grasp_type_prob = result[2]
            else:
                raise ValueError("Score pass requires model.sample to return pred_grasp_type_prob")

            scores = extract_real_type_scores(pred_grasp_type_prob)
            for batch_idx, metadata in enumerate(batch_metadata):
                scene_path = data["scene_path"][batch_idx]
                scene_id = metadata["scene_id"]
                if scene_id in score_records:
                    raise ValueError(f"Duplicate score record for scene_id={scene_id}")
                object_id = metadata["object_id"]
                score_records[scene_id] = {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "split": scene_split_for_record(object_id, str(split_name), split_lookup),
                    "scene_path": scene_path,
                    "pc_path": get_batch_value(data, "pc_path", batch_idx),
                    "budget_scores": scores[batch_idx],
                }
    return score_records


def pose_tensor_to_grasp_pose(robot_pose: torch.Tensor) -> torch.Tensor:
    """Convert model pose output into ``(B, K, D)`` grasp-pose tensor.

    Args:
        robot_pose: Model output, usually ``(B, K, 1, 14)``.

    Returns:
        Tensor shaped ``(B, K, D)`` where ``D`` is divisible by seven.
    """
    if robot_pose.ndim == 4:
        return robot_pose[..., 0, :]
    if robot_pose.ndim == 3:
        return robot_pose
    raise ValueError(f"Unsupported robot_pose shape: {tuple(robot_pose.shape)}")


def split_grasp_pose_samples(grasp_pose: np.ndarray) -> np.ndarray:
    """Split flat grasp poses into per-hand pose blocks.

    Args:
        grasp_pose: Array shaped ``(K, 7 * H)``.

    Returns:
        Array shaped ``(K, H, 7)``.
    """
    pose = np.asarray(grasp_pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[-1] % 7 != 0:
        raise ValueError(f"Expected grasp_pose shape (K, 7*H), got {pose.shape}")
    return pose.reshape(pose.shape[0], pose.shape[-1] // 7, 7)


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternions with a small zero-norm guard.

    Args:
        quat: Quaternion array with final dimension four.

    Returns:
        Normalized float32 quaternion array.
    """
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return (quat / norm).astype(np.float32, copy=False)


def position_key_for_source(grasp_pos_source: str) -> str:
    """Return the exported position field for the configured target source.

    Args:
        grasp_pos_source: Position target source used during human training.

    Returns:
        Export field name that stores the model position target.
    """
    grasp_pos_source = normalize_hand_pos_source(grasp_pos_source)
    if grasp_pos_source == "wrist":
        return "wrist_pos"
    return "index_mcp_pos"


def position_key_from_scene_data(scene_data: dict) -> str:
    """Infer the primary position field stored in one scene export.

    Args:
        scene_data: Per-scene export dictionary.

    Returns:
        Name of the primary position array field.
    """
    if scene_data.get("export_position_key") is not None:
        return str(scene_data["export_position_key"])
    if "wrist_pos" in scene_data:
        return "wrist_pos"
    if "index_mcp_pos" in scene_data:
        return "index_mcp_pos"
    raise KeyError("Scene export contains neither wrist_pos nor index_mcp_pos")


def convert_target_pose_to_export_pose(
    grasp_pose: np.ndarray,
    grasp_pos_source: str,
) -> dict[str, np.ndarray]:
    """Convert model target poses into the export position and wrist quaternion fields.

    Args:
        grasp_pose: Array shaped ``(K, 14)`` for right and left hand poses.
        grasp_pos_source: Position target source used during human training.

    Returns:
        Dictionary containing the configured position field and ``wrist_quat``.
    """
    hand_pose_samples = split_grasp_pose_samples(grasp_pose)
    wrist_quat = normalize_quaternions(hand_pose_samples[..., 3:7])
    grasp_pos_source = normalize_hand_pos_source(grasp_pos_source)
    position_key = position_key_for_source(grasp_pos_source)
    return {
        position_key: hand_pose_samples[..., :3].astype(np.float32, copy=False),
        "wrist_quat": wrist_quat,
    }


def build_active_hand_mask(grasp_type_id: int, sample_num: int, hand_num: int = 2) -> np.ndarray:
    """Build the active-hand mask for one grasp type.

    Args:
        grasp_type_id: Numeric grasp type id in ``[1, 5]``.
        sample_num: Number of pose samples for this type.
        hand_num: Number of hands in the exported pose tensor.

    Returns:
        Boolean array shaped ``(sample_num, hand_num)``.
    """
    if hand_num < 1:
        raise ValueError(f"hand_num must be positive, got {hand_num}")
    mask = np.zeros((sample_num, hand_num), dtype=bool)
    mask[:, 0] = True
    if grasp_type_id >= 4 and hand_num > 1:
        mask[:, 1] = True
    return mask


def unpack_pose_sample_result(result: tuple) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Unpack model.sample output for explicit grasp-type pose generation.

    Args:
        result: Tuple returned by ``model.sample``.

    Returns:
        Tuple ``(robot_pose, pred_grasp_type, log_prob)``.
    """
    if not isinstance(result, tuple):
        raise ValueError("Pose pass with explicit grasp types must return a tuple, not a score-only dict")
    if len(result) == 4:
        robot_pose, pred_grasp_type, _, log_prob = result
        return robot_pose, pred_grasp_type, log_prob
    if len(result) == 3:
        robot_pose, pred_grasp_type, log_prob = result
        return robot_pose, pred_grasp_type, log_prob
    if len(result) == 2:
        robot_pose, log_prob = result
        return robot_pose, None, log_prob
    raise ValueError(f"Unsupported pose sample result length: {len(result)}")


def rescale_human_pose_translations_for_export(robot_pose: torch.Tensor, robot_size: float) -> torch.Tensor:
    """Map generated human pose translations back to physical scene scale.

    Args:
        robot_pose: Pose tensor whose last dimension stores right-hand
            ``xyz+quat`` followed optionally by left-hand ``xyz+quat``.
        robot_size: Robot hand size ratio used to shrink inference point clouds.

    Returns:
        Pose tensor with translation channels multiplied by ``robot_size``.
        Quaternion channels are unchanged.
    """
    if robot_size == 1.0:
        return robot_pose
    if robot_pose.shape[-1] < 3:
        return robot_pose
    scaled_pose = robot_pose.clone()
    scaled_pose[..., :3] = scaled_pose[..., :3] * float(robot_size)
    if scaled_pose.shape[-1] >= 14:
        scaled_pose[..., 7:10] = scaled_pose[..., 7:10] * float(robot_size)
    return scaled_pose


def export_pose_candidate_num(config: DictConfig) -> int:
    """Read how many pose candidates should be generated before selection.

    Args:
        config: Full Hydra config containing ``algo.test_grasp_num``.

    Returns:
        Positive candidate count used before ``sample_selection`` filtering.
    """
    if is_reverse_factorization(config):
        candidate_num = reverse_sampling_config(config)["conditional_candidate_num"]
    else:
        candidate_num = int(getattr(config.algo, "test_grasp_num", getattr(config.task, "samples_per_type", 20)))
    if candidate_num <= 0:
        raise ValueError("algo.test_grasp_num must be positive for obj_human_prior_export")
    return candidate_num


def export_samples_per_type(config: DictConfig) -> int:
    """Read how many selected pose samples should be saved per grasp type.

    Args:
        config: Full Hydra config containing ``algo.test_topk`` and the legacy
            ``task.samples_per_type`` alias.

    Returns:
        Positive selected sample count. The value must match
        ``algo.test_topk`` so export uses the same semantics as ``sample.py``.
    """
    topk = int(getattr(config.algo, "test_topk", getattr(config.task, "samples_per_type", 20)))
    samples_per_type = int(getattr(config.task, "samples_per_type", topk))
    if topk <= 0:
        raise ValueError("algo.test_topk must be positive for obj_human_prior_export")
    if samples_per_type != topk:
        raise ValueError(
            "task.samples_per_type must match algo.test_topk. "
            f"Got task.samples_per_type={samples_per_type}, algo.test_topk={topk}."
        )
    return topk


def export_sample_selection_metadata(config: DictConfig) -> dict:
    """Build JSON/NumPy-friendly metadata for pose candidate selection.

    Args:
        config: Full Hydra config containing ``algo.sample_selection``.

    Returns:
        Dictionary describing the generated candidate count, selected count,
        and active selection strategy.
    """
    enabled, scope, mode, translation_scale_m, rotation_weight, intermediate_topk = _sample_selection_config(config)
    return {
        "pose_candidate_num": export_pose_candidate_num(config),
        "samples_per_type": export_samples_per_type(config),
        "sample_selection_enabled": bool(enabled),
        "sample_selection_scope": scope,
        "sample_selection_mode": mode,
        "sample_selection_intermediate_topk": intermediate_topk,
        "sample_selection_translation_scale_m": float(translation_scale_m),
        "sample_selection_rotation_weight": float(rotation_weight),
    }


def select_export_pose_candidates(
    robot_pose: torch.Tensor,
    log_prob: torch.Tensor,
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select final export candidates along the generated candidate dimension.

    Args:
        robot_pose: Candidate pose tensor shaped ``(B, N, ..., D)``.
        log_prob: Candidate log probabilities shaped ``(B, N)``.
        config: Full Hydra config containing ``algo.test_topk`` and
            ``algo.sample_selection``.

    Returns:
        Tuple ``(robot_pose, log_prob)`` after candidate selection. If all
        generated candidates are kept, the original tensors are returned
        directly to avoid unnecessary random/top-k indexing work.
    """
    candidate_num = int(log_prob.shape[1])
    topk = int(config.algo.test_topk)
    if topk == candidate_num:
        return robot_pose, log_prob
    selection_indices = _candidate_selection_indices(robot_pose, log_prob, config)
    return _gather_candidates(robot_pose, selection_indices), _gather_candidates(log_prob, selection_indices)


def sample_fixed_types_from_features(
    config: DictConfig,
    model: torch.nn.Module,
    data: dict,
    global_feature: torch.Tensor,
    samples_per_type: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample all explicit grasp types from one shared scene feature batch.

    Args:
        config: Full Hydra config containing task sampling options.
        model: Loaded hierarchical human prior model.
        data: Collated dataloader batch used for centering metadata.
        global_feature: Backbone output shaped ``(B, C)``.
        samples_per_type: Number of selected pose samples per explicit grasp type.

    Returns:
        Tuple ``(grasp_pose, log_prob, grasp_type_ids)`` where ``grasp_pose`` is
        shaped ``(B, T, S, D)``, ``log_prob`` is shaped ``(B, T, S)``, and
        ``grasp_type_ids`` is shaped ``(T,)``.
    """
    if not all(hasattr(model, attr) for attr in ("grasp_type_emb", "output_head")):
        raise TypeError("Shared-feature fixed-type sampling requires a hierarchical type-conditioned model")

    batch_size = int(global_feature.shape[0])
    type_ids = torch.as_tensor(REAL_GRASP_TYPE_IDS, device=global_feature.device, dtype=torch.long)
    type_num = int(type_ids.numel())
    feature_dim = int(global_feature.shape[-1])
    candidate_num = export_pose_candidate_num(config)
    if candidate_num < samples_per_type:
        raise ValueError(
            f"algo.test_grasp_num={candidate_num} is smaller than selected samples_per_type={samples_per_type}"
        )
    global_feature_expanded = (
        global_feature[:, None, None, :]
        .expand(batch_size, type_num, candidate_num, feature_dim)
        .reshape(batch_size * type_num * candidate_num, feature_dim)
    )
    type_ids_flat = (
        type_ids[None, :, None]
        .expand(batch_size, type_num, candidate_num)
        .reshape(batch_size * type_num * candidate_num)
    )
    cond_feat = torch.cat([global_feature_expanded, model.grasp_type_emb(type_ids_flat)], dim=-1)
    robot_pose, log_prob = model.output_head.sample(cond_feat, type_ids_flat, 1)

    robot_pose = robot_pose.reshape(batch_size, type_num, candidate_num, *robot_pose.shape[1:])
    robot_pose = robot_pose[:, :, :, 0]
    robot_pose = robot_pose.reshape(batch_size * type_num, candidate_num, *robot_pose.shape[3:])
    log_prob = log_prob.reshape(batch_size * type_num, candidate_num, -1)
    if log_prob.shape[-1] != 1:
        raise ValueError(f"Expected one log_prob per generated pose, got shape {tuple(log_prob.shape)}")
    log_prob = log_prob[..., 0]

    robot_pose, log_prob = select_export_pose_candidates(robot_pose, log_prob, config)

    robot_pose = robot_pose.reshape(batch_size, type_num, samples_per_type, *robot_pose.shape[2:])
    robot_pose = robot_pose.reshape(batch_size, type_num * samples_per_type, *robot_pose.shape[3:])
    robot_pose = rescale_human_pose_translations_for_export(robot_pose, export_robot_size(config))
    if "pc_centroid" in data:
        grasp_type_for_decenter = (
            type_ids[None, :, None]
            .expand(batch_size, type_num, samples_per_type)
            .reshape(batch_size, type_num * samples_per_type)
        )
        robot_pose = _decenter_human_pose(robot_pose, data["pc_centroid"], grasp_type_for_decenter)
    grasp_pose = pose_tensor_to_grasp_pose(robot_pose)
    grasp_pose = grasp_pose.reshape(batch_size, type_num, samples_per_type, grasp_pose.shape[-1])

    log_prob = log_prob.reshape(batch_size, type_num, samples_per_type)
    return grasp_pose, log_prob, type_ids


def _stable_reverse_seed(base_seed: int, scene_id: str, type_id: int) -> int:
    """Derive a batching-independent RNG seed for one scene and mode."""
    payload = f"{int(base_seed)}:{scene_id}:{int(type_id)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return value % np.iinfo(np.int64).max


def reverse_weighted_resample_indices(
    posterior_probs: torch.Tensor,
    scene_ids: list[str],
    candidate_num: int,
    base_seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Draw unique per-mode candidates from one shared marginal pool.

    Args:
        posterior_probs: Tensor shaped ``(B, M, 5)``.
        scene_ids: Stable scene ids aligned with the batch.
        candidate_num: Number of unique candidates to draw per mode.
        base_seed: Experiment-level resampling seed.

    Returns:
        Pool indices shaped ``(B, 5, C)`` and derived seeds ``(B, 5)``.
    """
    if posterior_probs.ndim != 3 or posterior_probs.shape[-1] != len(REAL_GRASP_TYPE_IDS):
        raise ValueError(f"Expected posterior_probs shape (B,M,5), got {tuple(posterior_probs.shape)}")
    batch_size, pool_size, _ = posterior_probs.shape
    if len(scene_ids) != batch_size:
        raise ValueError("scene_ids must match posterior batch size")
    if candidate_num > pool_size:
        raise ValueError("candidate_num must not exceed marginal pool size")

    probabilities = posterior_probs.detach().cpu().numpy().astype(np.float64)
    indices = np.empty((batch_size, len(REAL_GRASP_TYPE_IDS), candidate_num), dtype=np.int64)
    seeds = np.empty((batch_size, len(REAL_GRASP_TYPE_IDS)), dtype=np.int64)
    for batch_index, scene_id in enumerate(scene_ids):
        for type_index, type_id in enumerate(REAL_GRASP_TYPE_IDS):
            derived_seed = _stable_reverse_seed(base_seed, scene_id, type_id)
            seeds[batch_index, type_index] = derived_seed
            weights = np.maximum(probabilities[batch_index, :, type_index], 0.0)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0.0:
                weights = np.full((pool_size,), 1.0 / pool_size, dtype=np.float64)
            else:
                # Float32 softmax values may underflow to exact zero. Keep every
                # pool item technically sampleable so unique resampling remains
                # defined when candidate_num exceeds the non-zero support.
                weights = np.maximum(weights, 1e-12)
                weights = weights / float(weights.sum())
            rng = np.random.default_rng(derived_seed)
            indices[batch_index, type_index] = rng.choice(
                pool_size,
                size=candidate_num,
                replace=False,
                p=weights,
            )
    return torch.as_tensor(indices, device=posterior_probs.device, dtype=torch.long), seeds


def reverse_raw_joint_type_ids(
    posterior_probs: torch.Tensor,
    scene_ids: list[str],
    base_seed: int,
) -> np.ndarray:
    """Sample raw joint contact modes after the marginal T pool exists."""
    probabilities = posterior_probs.detach().cpu().numpy().astype(np.float64)
    batch_size, pool_size, type_num = probabilities.shape
    sampled = np.empty((batch_size, pool_size), dtype=np.int64)
    for batch_index, scene_id in enumerate(scene_ids):
        rng = np.random.default_rng(_stable_reverse_seed(base_seed, scene_id, 0))
        weights = np.maximum(probabilities[batch_index], 0.0)
        totals = weights.sum(axis=-1, keepdims=True)
        valid_rows = np.isfinite(totals[:, 0]) & (totals[:, 0] > 0.0) & np.isfinite(weights).all(axis=-1)
        normalized = np.full((pool_size, type_num), 1.0 / type_num, dtype=np.float64)
        normalized[valid_rows] = weights[valid_rows] / totals[valid_rows]
        cumulative = np.cumsum(normalized, axis=-1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random(pool_size)
        sampled[batch_index] = np.sum(uniforms[:, None] > cumulative, axis=-1, dtype=np.int64) + 1
    return sampled


def _gather_reverse_pool(value: torch.Tensor, pool_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(B,M,...)`` values into ``(B,T,C,...)`` by pool index."""
    if value.shape[0] != pool_indices.shape[0]:
        raise ValueError("Pool value and index batch dimensions must match")
    batch_index = torch.arange(value.shape[0], device=value.device)[:, None, None]
    return value[batch_index, pool_indices]


def validate_reverse_shared_pool(
    canonical_t24: torch.Tensor,
    robot_pose: torch.Tensor,
    marginal_log_prob: torch.Tensor,
    posterior_probs: torch.Tensor,
    expected_pool_size: int,
) -> None:
    """Validate the model-facing Reverse shared-pool contract before adapting it."""
    if canonical_t24.ndim != 3 or canonical_t24.shape[1:] != (expected_pool_size, 24):
        raise ValueError(
            f"Expected Reverse canonical T24 shape (B,{expected_pool_size},24), "
            f"got {tuple(canonical_t24.shape)}"
        )
    batch_size = canonical_t24.shape[0]
    expected_pose_shape = (batch_size, expected_pool_size, 1, 14)
    if tuple(robot_pose.shape) != expected_pose_shape:
        raise ValueError(f"Expected Reverse pose shape {expected_pose_shape}, got {tuple(robot_pose.shape)}")
    expected_score_shape = (batch_size, expected_pool_size)
    if tuple(marginal_log_prob.shape) != expected_score_shape:
        raise ValueError(
            f"Expected Reverse marginal log-probability shape {expected_score_shape}, "
            f"got {tuple(marginal_log_prob.shape)}"
        )
    expected_posterior_shape = (batch_size, expected_pool_size, len(REAL_GRASP_TYPE_IDS))
    if tuple(posterior_probs.shape) != expected_posterior_shape:
        raise ValueError(
            f"Expected Reverse posterior shape {expected_posterior_shape}, got {tuple(posterior_probs.shape)}"
        )
    for name, value in (
        ("canonical T24", canonical_t24),
        ("pose", robot_pose),
        ("marginal log probability", marginal_log_prob),
        ("posterior probability", posterior_probs),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"Reverse {name} contains non-finite values")
    if torch.any((posterior_probs < 0.0) | (posterior_probs > 1.0)):
        raise ValueError("Reverse posterior probabilities must be in [0, 1]")
    if not torch.allclose(
        posterior_probs.sum(dim=-1),
        torch.ones_like(posterior_probs[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Reverse posterior rows must sum to 1")


def build_reverse_conditional_adapter(
    robot_pose: torch.Tensor,
    marginal_log_prob: torch.Tensor,
    posterior_probs: torch.Tensor,
    scene_ids: list[str],
    config: DictConfig,
) -> dict:
    """Build fixed per-mode candidates from one Reverse marginal pool."""
    sampling = reverse_sampling_config(config)
    if robot_pose.shape[:2] != posterior_probs.shape[:2]:
        raise ValueError("Reverse pose and posterior pool dimensions must match")
    if marginal_log_prob.shape != posterior_probs.shape[:2]:
        raise ValueError("Reverse marginal score and posterior pool dimensions must match")
    resampled_pool_indices, resampling_seeds = reverse_weighted_resample_indices(
        posterior_probs,
        scene_ids,
        sampling["conditional_candidate_num"],
        sampling["resampling_seed"],
    )
    candidate_pose = _gather_reverse_pool(robot_pose, resampled_pool_indices)
    candidate_marginal_log_prob = _gather_reverse_pool(marginal_log_prob, resampled_pool_indices)
    posterior_by_type = posterior_probs.permute(0, 2, 1)
    candidate_posterior = torch.gather(posterior_by_type, 2, resampled_pool_indices)
    conditional_score = candidate_marginal_log_prob + torch.log(candidate_posterior.clamp_min(1e-12))

    batch_size, type_num, candidate_num = conditional_score.shape
    flat_pose = candidate_pose.reshape(batch_size * type_num, candidate_num, *candidate_pose.shape[3:])
    flat_score = conditional_score.reshape(batch_size * type_num, candidate_num)
    selected_resample_indices = _candidate_selection_indices(flat_pose, flat_score, config)
    selected_resample_indices = selected_resample_indices.reshape(batch_size, type_num, -1)
    selected_pose = _gather_candidates(flat_pose, selected_resample_indices.reshape(batch_size * type_num, -1))
    selected_pose = selected_pose.reshape(batch_size, type_num, -1, *candidate_pose.shape[3:])
    selected_pool_indices = torch.gather(resampled_pool_indices, 2, selected_resample_indices)
    selected_marginal_log_prob = torch.gather(candidate_marginal_log_prob, 2, selected_resample_indices)
    selected_posterior = torch.gather(candidate_posterior, 2, selected_resample_indices)
    selected_conditional_score = torch.gather(conditional_score, 2, selected_resample_indices)

    posterior_sum = posterior_probs.sum(dim=1)
    ess = posterior_sum.square() / posterior_probs.square().sum(dim=1).clamp_min(1e-12)
    threshold = sampling["ess_warning_threshold"]
    if threshold > 0.0:
        low_ess = torch.nonzero(ess < threshold, as_tuple=False)
        for batch_index, type_index in low_ess.detach().cpu().tolist():
            print(
                f"Warning: Reverse low ESS scene={scene_ids[batch_index]} "
                f"type={REAL_GRASP_TYPE_NAMES[type_index]} ess={float(ess[batch_index, type_index]):.3f}"
            )

    return {
        "robot_pose": selected_pose,
        "marginal_log_prob": selected_marginal_log_prob,
        "posterior_probability": selected_posterior,
        "conditional_score": selected_conditional_score,
        "resampled_pool_indices": resampled_pool_indices,
        "selected_resample_indices": selected_resample_indices,
        "selected_pool_indices": selected_pool_indices,
        "importance_ess": ess,
        "resampling_seeds": resampling_seeds,
    }


def sample_reverse_scene_scores_and_poses(
    config: DictConfig,
    posterior_model: torch.nn.Module,
    marginal_model: torch.nn.Module,
    split_lookup: dict[str, str],
    scene_dir: str,
    checkpoint_meta: dict,
    skip_scene_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Export Reverse T-to-C priors from one shared marginal pose pool.

    Args:
        config: Full Reverse export config.
        posterior_model: Independent ``q(c|T,o)`` network.
        marginal_model: Independent ``p(T|o)`` network.
        split_lookup: Object id to split mapping.
        scene_dir: Destination root for per-scene files.
        checkpoint_meta: Loaded checkpoint paths, iterations, and hashes.
        skip_scene_ids: Existing complete scenes that should not be regenerated.

    Returns:
        JSON-friendly scene summaries and index rows for newly saved scenes.
    """
    if not hasattr(marginal_model, "sample_with_t24"):
        raise TypeError("Reverse pose model must implement sample_with_t24")
    if not hasattr(posterior_model, "posterior_probabilities"):
        raise TypeError("Reverse posterior model must implement posterior_probabilities")

    score_lines: list[dict] = []
    scene_index: list[dict] = []
    saved_scene_ids: set[str] = set()
    skip_scene_ids = skip_scene_ids or set()
    sampling = reverse_sampling_config(config)
    samples_per_type = export_samples_per_type(config)
    grasp_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    include_log_prob = bool(getattr(config.task, "include_log_prob", True))
    include_grasp_pose = bool(getattr(config.task, "include_grasp_pose", False))
    scene_pass_grasp_types = _as_list(getattr(config.task, "score_grasp_types", ["0_any"]))[:1] or ["0_any"]

    for split_name in _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])):
        pass_config = clone_config_with_grasp_types(config, scene_pass_grasp_types, str(split_name))
        test_loader = create_test_dataloader(pass_config)
        for data in tqdm(test_loader, desc=f"reverse obj prior scene [{split_name}]"):
            batch_metadata = [read_scene_metadata(scene_path) for scene_path in data["scene_path"]]
            keep_indices = [
                batch_idx
                for batch_idx, metadata in enumerate(batch_metadata)
                if metadata["scene_id"] not in skip_scene_ids
            ]
            if not keep_indices:
                continue
            data = filter_batch_data(data, keep_indices, config)
            batch_metadata = [batch_metadata[index] for index in keep_indices]
            scene_ids = [metadata["scene_id"] for metadata in batch_metadata]

            canonical_t24, raw_robot_pose, marginal_log_prob = marginal_model.sample_with_t24(
                data,
                sampling["marginal_pool_size"],
            )
            posterior_probs = posterior_model.posterior_probabilities(data, canonical_t24)
            validate_reverse_shared_pool(
                canonical_t24,
                raw_robot_pose,
                marginal_log_prob,
                posterior_probs,
                sampling["marginal_pool_size"],
            )
            budget_scores = posterior_probs.mean(dim=1)
            adapter = build_reverse_conditional_adapter(
                raw_robot_pose,
                marginal_log_prob,
                posterior_probs,
                scene_ids,
                config,
            )
            selected_centered_t24 = _gather_reverse_pool(canonical_t24, adapter["selected_pool_indices"])

            batch_size = int(canonical_t24.shape[0])
            type_num = len(REAL_GRASP_TYPE_IDS)
            selected_pose = adapter["robot_pose"].reshape(
                batch_size,
                type_num * samples_per_type,
                *adapter["robot_pose"].shape[3:],
            )
            selected_pose = rescale_human_pose_translations_for_export(selected_pose, export_robot_size(config))
            type_ids = torch.as_tensor(REAL_GRASP_TYPE_IDS, device=selected_pose.device, dtype=torch.long)
            type_ids_for_decenter = (
                type_ids[None, :, None]
                .expand(batch_size, type_num, samples_per_type)
                .reshape(batch_size, type_num * samples_per_type)
            )
            if "pc_centroid" in data:
                selected_pose = _decenter_human_pose(selected_pose, data["pc_centroid"], type_ids_for_decenter)
            grasp_pose = pose_tensor_to_grasp_pose(selected_pose).reshape(batch_size, type_num, samples_per_type, -1)

            grasp_pose_np = grasp_pose.detach().cpu().numpy().astype(np.float32)
            budget_scores_np = budget_scores.detach().cpu().numpy().astype(np.float32)
            if sampling["include_raw_pool"]:
                raw_t24_np = canonical_t24.detach().cpu().numpy().astype(np.float32)
                raw_log_prob_np = marginal_log_prob.detach().cpu().numpy().astype(np.float32)
                raw_posterior_np = posterior_probs.detach().cpu().numpy().astype(np.float32)
                raw_joint_type_ids = reverse_raw_joint_type_ids(
                    posterior_probs,
                    scene_ids,
                    sampling["resampling_seed"],
                )
            adapter_np = {
                key: value.detach().cpu().numpy()
                for key, value in adapter.items()
                if torch.is_tensor(value)
            }

            for batch_idx, metadata in enumerate(batch_metadata):
                scene_path = data["scene_path"][batch_idx]
                scene_id = metadata["scene_id"]
                if scene_id in saved_scene_ids:
                    raise ValueError(f"Duplicate Reverse export record for scene_id={scene_id}")
                object_id = metadata["object_id"]
                split = scene_split_for_record(object_id, str(split_name), split_lookup)
                score_record = {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "split": split,
                    "scene_path": scene_path,
                    "pc_path": get_batch_value(data, "pc_path", batch_idx),
                    "budget_scores": budget_scores_np[batch_idx],
                }
                pose_record_by_type = {}
                for type_index, grasp_type_id in enumerate(REAL_GRASP_TYPE_IDS):
                    pose_record = convert_target_pose_to_export_pose(
                        grasp_pose_np[batch_idx, type_index],
                        grasp_pos_source,
                    )
                    hand_num = pose_record["wrist_quat"].shape[1]
                    pose_record.update(
                        {
                            "scene_id": scene_id,
                            "object_id": object_id,
                            "split": split,
                            "scene_path": scene_path,
                            "pc_path": score_record["pc_path"],
                            "grasp_type_id": grasp_type_id,
                            "grasp_type_name": GRASP_TYPES[grasp_type_id],
                            "active_hand_mask": build_active_hand_mask(
                                grasp_type_id,
                                samples_per_type,
                                hand_num,
                            ),
                        }
                    )
                    if include_log_prob:
                        pose_record["log_prob"] = adapter_np["conditional_score"][batch_idx, type_index].astype(
                            np.float32
                        )
                    if include_grasp_pose:
                        pose_record["grasp_pose"] = grasp_pose_np[batch_idx, type_index]
                    pose_record_by_type[grasp_type_id] = pose_record

                scene_data = build_scene_export_record(score_record, pose_record_by_type, config)
                scene_data.update(
                    {
                        "reverse_sampling_order": np.asarray(
                            ["marginal_T", "posterior_C", "budget_integral", "resample", "select", "export"]
                        ),
                        "reverse_marginal_pool_size": np.int64(sampling["marginal_pool_size"]),
                        "reverse_conditional_candidate_num": np.int64(sampling["conditional_candidate_num"]),
                        "reverse_resampling_policy": sampling["resampling_policy"],
                        "reverse_resampling_base_seed": np.int64(sampling["resampling_seed"]),
                        "reverse_resampling_seeds": adapter["resampling_seeds"][batch_idx].astype(np.int64),
                        "reverse_importance_ess": adapter_np["importance_ess"][batch_idx].astype(np.float32),
                        "reverse_resampled_pool_indices": adapter_np["resampled_pool_indices"][batch_idx].astype(
                            np.int64
                        ),
                        "reverse_selected_resample_indices": adapter_np["selected_resample_indices"][batch_idx].astype(
                            np.int64
                        ),
                        "reverse_selected_pool_indices": adapter_np["selected_pool_indices"][batch_idx].astype(
                            np.int64
                        ),
                        "reverse_selected_centered_t24": selected_centered_t24[batch_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "reverse_selected_marginal_log_prob": adapter_np["marginal_log_prob"][batch_idx].astype(
                            np.float32
                        ),
                        "reverse_selected_posterior_probability": adapter_np["posterior_probability"][
                            batch_idx
                        ].astype(np.float32),
                        "reverse_selected_conditional_score": adapter_np["conditional_score"][batch_idx].astype(
                            np.float32
                        ),
                        "reverse_raw_pool_included": np.bool_(sampling["include_raw_pool"]),
                        "reverse_score_checkpoint_sha256": checkpoint_meta["score_checkpoint_sha256"],
                        "reverse_pose_checkpoint_sha256": checkpoint_meta["pose_checkpoint_sha256"],
                    }
                )
                if sampling["include_raw_pool"]:
                    scene_data.update(
                        {
                            "reverse_raw_centered_t24": raw_t24_np[batch_idx],
                            "reverse_raw_marginal_log_prob": raw_log_prob_np[batch_idx],
                            "reverse_raw_posterior_probability": raw_posterior_np[batch_idx],
                            "reverse_raw_sampled_type_ids": raw_joint_type_ids[batch_idx],
                        }
                    )
                validate_reverse_scene_export(scene_data, config, checkpoint_meta)
                scene_file = scene_file_path(scene_dir, scene_id)
                os.makedirs(os.path.dirname(scene_file), exist_ok=True)
                np.save(scene_file, scene_data)
                summary = scene_summary_from_data(scene_data, scene_file)
                score_lines.append(summary)
                scene_index.append(
                    {
                        "scene_id": summary["scene_id"],
                        "object_id": summary["object_id"],
                        "split": summary["split"],
                        "robot_name": summary["robot_name"],
                        "robot_size": summary["robot_size"],
                        "pc_runtime_scale": summary["pc_runtime_scale"],
                        "factorization": summary["factorization"],
                        "scene_file": summary["scene_file"],
                    }
                )
                saved_scene_ids.add(scene_id)
    return score_lines, scene_index


def sample_scene_scores_and_fixed_type_poses(
    config: DictConfig,
    score_model: torch.nn.Module,
    pose_model: torch.nn.Module,
    split_lookup: dict[str, str],
    scene_dir: str,
    skip_scene_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run one scene pass and save each generated batch immediately.

    Args:
        config: Full Hydra config.
        score_model: Loaded model used for grasp-type budget scores.
        pose_model: Loaded model used for fixed-type hand-position proposals.
        split_lookup: Object id to split mapping for metadata.
        scene_dir: Root directory where per-scene files are saved.
        skip_scene_ids: Scene ids that already have a complete export.

    Returns:
        Tuple of score summary rows and scene-index rows for newly saved scenes.
    """
    score_lines: list[dict] = []
    scene_index: list[dict] = []
    saved_scene_ids: set[str] = set()
    skip_scene_ids = skip_scene_ids or set()
    samples_per_type = export_samples_per_type(config)
    grasp_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    include_log_prob = bool(getattr(config.task, "include_log_prob", True))
    include_grasp_pose = bool(getattr(config.task, "include_grasp_pose", False))
    scene_pass_grasp_types = _as_list(getattr(config.task, "score_grasp_types", ["0_any"]))[:1] or ["0_any"]

    for split_name in _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])):
        pass_config = clone_config_with_grasp_types(config, scene_pass_grasp_types, str(split_name))
        test_loader = create_test_dataloader(pass_config)
        desc = f"obj prior scene [{split_name}]"
        for data in tqdm(test_loader, desc=desc):
            batch_metadata = [read_scene_metadata(scene_path) for scene_path in data["scene_path"]]
            keep_indices = [
                batch_idx for batch_idx, metadata in enumerate(batch_metadata) if metadata["scene_id"] not in skip_scene_ids
            ]
            if not keep_indices:
                continue
            data = filter_batch_data(data, keep_indices, config)
            batch_metadata = [batch_metadata[index] for index in keep_indices]

            score_global_feature, _ = score_model.backbone(data)
            _, type_scores = score_model._compute_type_scores(score_global_feature)
            scores = extract_real_type_scores(type_scores)
            if pose_model is score_model:
                pose_global_feature = score_global_feature
            else:
                pose_global_feature, _ = pose_model.backbone(data)
            grasp_pose_tensor, log_prob_tensor, sampled_type_ids = sample_fixed_types_from_features(
                config,
                pose_model,
                data,
                pose_global_feature,
                samples_per_type,
            )
            grasp_pose_np = grasp_pose_tensor.detach().cpu().numpy().astype(np.float32)
            log_prob_np = log_prob_tensor.detach().cpu().numpy().astype(np.float32)
            sampled_type_ids_np = sampled_type_ids.detach().cpu().numpy().astype(np.int64)

            for batch_idx, metadata in enumerate(batch_metadata):
                scene_path = data["scene_path"][batch_idx]
                scene_id = metadata["scene_id"]
                if scene_id in saved_scene_ids:
                    raise ValueError(f"Duplicate score record for scene_id={scene_id}")
                object_id = metadata["object_id"]
                split = scene_split_for_record(object_id, str(split_name), split_lookup)
                score_record = {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "split": split,
                    "scene_path": scene_path,
                    "pc_path": get_batch_value(data, "pc_path", batch_idx),
                    "budget_scores": scores[batch_idx],
                }
                pose_record_by_type = {}
                for type_idx, grasp_type_id_raw in enumerate(sampled_type_ids_np):
                    grasp_type_id = int(grasp_type_id_raw)
                    pose_record = convert_target_pose_to_export_pose(
                        grasp_pose_np[batch_idx, type_idx],
                        grasp_pos_source,
                    )
                    hand_num = pose_record["wrist_quat"].shape[1]
                    pose_record.update(
                        {
                            "scene_id": scene_id,
                            "object_id": object_id,
                            "split": split,
                            "scene_path": scene_path,
                            "pc_path": get_batch_value(data, "pc_path", batch_idx),
                            "grasp_type_id": grasp_type_id,
                            "grasp_type_name": GRASP_TYPES[grasp_type_id],
                            "active_hand_mask": build_active_hand_mask(grasp_type_id, samples_per_type, hand_num),
                        }
                    )
                    if include_log_prob:
                        pose_record["log_prob"] = log_prob_np[batch_idx, type_idx]
                    if include_grasp_pose:
                        pose_record["grasp_pose"] = grasp_pose_np[batch_idx, type_idx]
                    if grasp_type_id in pose_record_by_type:
                        raise ValueError(f"Duplicate pose record for scene_id={scene_id}, grasp_type_id={grasp_type_id}")
                    pose_record_by_type[grasp_type_id] = pose_record

                scene_data = build_scene_export_record(score_record, pose_record_by_type, config)
                scene_file = scene_file_path(scene_dir, scene_id)
                os.makedirs(os.path.dirname(scene_file), exist_ok=True)
                np.save(scene_file, scene_data)
                summary = scene_summary_from_data(scene_data, scene_file)
                score_lines.append(summary)
                scene_index.append(
                    {
                        "scene_id": summary["scene_id"],
                        "object_id": summary["object_id"],
                        "split": summary["split"],
                        "robot_name": summary["robot_name"],
                        "robot_size": summary["robot_size"],
                        "pc_runtime_scale": summary["pc_runtime_scale"],
                        "scene_file": summary["scene_file"],
                    }
                )
                saved_scene_ids.add(scene_id)
    return score_lines, scene_index


def sample_fixed_type_wrist_poses(
    config: DictConfig,
    model: torch.nn.Module,
    split_lookup: dict[str, str],
    skip_scene_ids: set[str] | None = None,
) -> dict[str, dict[int, dict]]:
    """Run fixed-type pose passes and collect unsorted wrist samples.

    Args:
        config: Full Hydra config.
        model: Loaded human prior model.
        split_lookup: Object id to split mapping for metadata.
        skip_scene_ids: Scene ids that already have a complete export.

    Returns:
        Nested mapping ``scene_id -> grasp_type_id -> pose record``.
    """
    pose_records: dict[str, dict[int, dict]] = {}
    skip_scene_ids = skip_scene_ids or set()
    samples_per_type = export_samples_per_type(config)
    pose_grasp_types = _as_list(getattr(config.task, "pose_grasp_types", list(REAL_GRASP_TYPE_NAMES)))
    grasp_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    include_log_prob = bool(getattr(config.task, "include_log_prob", True))
    include_grasp_pose = bool(getattr(config.task, "include_grasp_pose", False))

    for split_name in _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])):
        pass_config = clone_config_with_grasp_types(config, pose_grasp_types, str(split_name))
        test_loader = create_test_dataloader(pass_config)
        desc = f"obj prior pose [{split_name}]"
        for data in tqdm(test_loader, desc=desc):
            batch_metadata = [read_scene_metadata(scene_path) for scene_path in data["scene_path"]]
            keep_indices = [
                batch_idx for batch_idx, metadata in enumerate(batch_metadata) if metadata["scene_id"] not in skip_scene_ids
            ]
            if not keep_indices:
                continue
            data = filter_batch_data(data, keep_indices, config)
            batch_metadata = [batch_metadata[index] for index in keep_indices]

            result = model.sample(data, export_pose_candidate_num(config))
            robot_pose, pred_grasp_type, log_prob = unpack_pose_sample_result(result)
            if int(log_prob.shape[1]) != int(config.algo.test_topk):
                selection_indices = _candidate_selection_indices(robot_pose, log_prob, config)
                robot_pose = _gather_candidates(robot_pose, selection_indices)
                log_prob = _gather_candidates(log_prob, selection_indices)
                if pred_grasp_type is not None:
                    pred_grasp_type = _gather_candidates(pred_grasp_type, selection_indices)

            grasp_type_for_decenter = pred_grasp_type if pred_grasp_type is not None else data["grasp_type_id"]
            if grasp_type_for_decenter.ndim == 1:
                grasp_type_for_decenter = grasp_type_for_decenter.unsqueeze(1).expand(-1, robot_pose.shape[1])
            robot_pose = rescale_human_pose_translations_for_export(robot_pose, export_robot_size(config))
            if "pc_centroid" in data:
                robot_pose = _decenter_human_pose(robot_pose, data["pc_centroid"], grasp_type_for_decenter)

            grasp_pose = pose_tensor_to_grasp_pose(robot_pose).detach().cpu().numpy().astype(np.float32)
            log_prob_np = log_prob.detach().cpu().numpy().astype(np.float32)
            grasp_type_ids = data["grasp_type_id"].detach().cpu().numpy().astype(np.int64)

            for batch_idx, metadata in enumerate(batch_metadata):
                scene_path = data["scene_path"][batch_idx]
                scene_id = metadata["scene_id"]
                object_id = metadata["object_id"]
                grasp_type_id = int(grasp_type_ids[batch_idx])
                wrist_record = convert_target_pose_to_export_pose(
                    grasp_pose[batch_idx],
                    grasp_pos_source,
                )
                hand_num = wrist_record["wrist_quat"].shape[1]
                wrist_record.update(
                    {
                        "scene_id": scene_id,
                        "object_id": object_id,
                        "split": scene_split_for_record(object_id, str(split_name), split_lookup),
                        "scene_path": scene_path,
                        "pc_path": get_batch_value(data, "pc_path", batch_idx),
                        "grasp_type_id": grasp_type_id,
                        "grasp_type_name": GRASP_TYPES[grasp_type_id],
                        "active_hand_mask": build_active_hand_mask(grasp_type_id, samples_per_type, hand_num),
                    }
                )
                if include_log_prob:
                    wrist_record["log_prob"] = log_prob_np[batch_idx]
                if include_grasp_pose:
                    wrist_record["grasp_pose"] = grasp_pose[batch_idx]
                pose_records.setdefault(scene_id, {})
                if grasp_type_id in pose_records[scene_id]:
                    raise ValueError(f"Duplicate pose record for scene_id={scene_id}, grasp_type_id={grasp_type_id}")
                pose_records[scene_id][grasp_type_id] = wrist_record
    return pose_records


def resolve_output_dir(config: DictConfig, checkpoint_meta: dict) -> str:
    """Resolve the export output directory.

    Args:
        config: Full Hydra config.
        checkpoint_meta: Checkpoint metadata returned by ``load_export_models``.

    Returns:
        Absolute output directory path.
    """
    configured_output = getattr(config.task, "output_dir", None)
    if configured_output:
        return to_absolute_path(str(configured_output))
    if bool(checkpoint_meta["uses_independent_models"]):
        pose_step = _checkpoint_step_label(checkpoint_meta.get("pose_ckpt"), checkpoint_meta.get("pose_checkpoint_iter"))
        score_step = _checkpoint_step_label(checkpoint_meta.get("score_ckpt"), checkpoint_meta.get("score_checkpoint_iter"))
        step_name = f"step_{pose_step}_{score_step}"
    elif checkpoint_meta["checkpoint_iter"] is None:
        step_name = os.path.splitext(_checkpoint_name(config.ckpt))[0]
    else:
        step_name = f"step_{checkpoint_meta['checkpoint_iter']:06d}"
    return to_absolute_path(pjoin(str(config.output_folder), str(config.wandb.id), "obj_human_prior", step_name))


def validate_scene_export(scene_data: dict, quat_norm_tol: float) -> None:
    """Validate one merged per-scene export record before saving.

    Args:
        scene_data: Per-scene export dictionary.
        quat_norm_tol: Maximum allowed quaternion norm error for active hands.

    Returns:
        None. Raises an exception if validation fails.
    """
    budget_scores = np.asarray(scene_data["budget_scores"])
    position_key = position_key_from_scene_data(scene_data)
    position = np.asarray(scene_data[position_key])
    wrist_quat = np.asarray(scene_data["wrist_quat"])
    active_hand_mask = np.asarray(scene_data["active_hand_mask"])
    if budget_scores.shape != (len(REAL_GRASP_TYPE_IDS),):
        raise ValueError(f"Invalid budget_scores shape for {scene_data['scene_id']}: {budget_scores.shape}")
    if position.shape[:3] != active_hand_mask.shape:
        raise ValueError(f"{position_key} and active_hand_mask shape mismatch for {scene_data['scene_id']}")
    if wrist_quat.shape[:3] != active_hand_mask.shape:
        raise ValueError(f"wrist_quat and active_hand_mask shape mismatch for {scene_data['scene_id']}")
    if not np.isfinite(budget_scores).all() or not np.isfinite(position).all() or not np.isfinite(wrist_quat).all():
        raise ValueError(f"Non-finite value found in scene export: {scene_data['scene_id']}")
    active_quat = wrist_quat[active_hand_mask]
    quat_norm = np.linalg.norm(active_quat, axis=-1)
    if active_quat.size and np.max(np.abs(quat_norm - 1.0)) > quat_norm_tol:
        raise ValueError(f"Quaternion norm validation failed for {scene_data['scene_id']}")
    if not os.path.exists(scene_data["scene_path"]):
        raise FileNotFoundError(f"scene_path does not exist: {scene_data['scene_path']}")
    if not os.path.exists(scene_data["pc_path"]):
        raise FileNotFoundError(f"pc_path does not exist: {scene_data['pc_path']}")


def validate_reverse_scene_export(
    scene_data: dict,
    config: DictConfig,
    checkpoint_meta: dict | None = None,
) -> None:
    """Validate Reverse-specific provenance and shared-pool shapes."""
    validate_scene_export(scene_data, float(getattr(config.task, "quat_norm_tol", 1e-3)))
    if str(scene_data.get("factorization")) != REVERSE_FACTORIZATION:
        raise ValueError("Reverse scene export has an invalid factorization tag")
    if str(scene_data.get("score_semantics")) != score_semantics_from_config(config):
        raise ValueError("Reverse scene export has invalid score semantics")
    sampling = reverse_sampling_config(config)
    type_num = len(REAL_GRASP_TYPE_IDS)
    samples_per_type = export_samples_per_type(config)
    expected_shapes = {
        "reverse_importance_ess": (type_num,),
        "reverse_resampling_seeds": (type_num,),
        "reverse_resampled_pool_indices": (type_num, sampling["conditional_candidate_num"]),
        "reverse_selected_resample_indices": (type_num, samples_per_type),
        "reverse_selected_pool_indices": (type_num, samples_per_type),
        "reverse_selected_centered_t24": (type_num, samples_per_type, 24),
        "reverse_selected_marginal_log_prob": (type_num, samples_per_type),
        "reverse_selected_posterior_probability": (type_num, samples_per_type),
        "reverse_selected_conditional_score": (type_num, samples_per_type),
    }
    expected_order = ["marginal_T", "posterior_C", "budget_integral", "resample", "select", "export"]
    if np.asarray(scene_data.get("reverse_sampling_order", [])).astype(str).tolist() != expected_order:
        raise ValueError("Reverse scene export has an invalid sampling order")
    if int(scene_data.get("reverse_resampling_base_seed", -1)) != sampling["resampling_seed"]:
        raise ValueError("Reverse scene export has an invalid resampling base seed")
    expected_resampling_seeds = np.asarray(
        [
            _stable_reverse_seed(sampling["resampling_seed"], str(scene_data["scene_id"]), type_id)
            for type_id in REAL_GRASP_TYPE_IDS
        ],
        dtype=np.int64,
    )
    if not np.array_equal(np.asarray(scene_data.get("reverse_resampling_seeds")), expected_resampling_seeds):
        raise ValueError("Reverse scene export has invalid per-mode resampling seeds")
    for key in ("reverse_score_checkpoint_sha256", "reverse_pose_checkpoint_sha256"):
        value = str(scene_data.get(key, ""))
        if len(value) != 64:
            raise ValueError(f"Reverse scene export has an invalid {key}")
    if checkpoint_meta is not None:
        if str(scene_data["reverse_score_checkpoint_sha256"]) != str(checkpoint_meta["score_checkpoint_sha256"]):
            raise ValueError("Reverse scene export uses a different posterior checkpoint")
        if str(scene_data["reverse_pose_checkpoint_sha256"]) != str(checkpoint_meta["pose_checkpoint_sha256"]):
            raise ValueError("Reverse scene export uses a different marginal checkpoint")
    for key, expected_shape in expected_shapes.items():
        if key not in scene_data:
            raise KeyError(f"Reverse scene export is missing {key}")
        if np.asarray(scene_data[key]).shape != expected_shape:
            raise ValueError(f"Invalid {key} shape: {np.asarray(scene_data[key]).shape}, expected {expected_shape}")
    finite_fields = (
        "reverse_importance_ess",
        "reverse_selected_centered_t24",
        "reverse_selected_marginal_log_prob",
        "reverse_selected_posterior_probability",
        "reverse_selected_conditional_score",
    )
    for key in finite_fields:
        if not np.isfinite(np.asarray(scene_data[key])).all():
            raise ValueError(f"Reverse scene export contains non-finite values in {key}")
    selected_posterior = np.asarray(scene_data["reverse_selected_posterior_probability"])
    if ((selected_posterior < 0.0) | (selected_posterior > 1.0)).any():
        raise ValueError("Reverse selected posterior probabilities must be in [0, 1]")
    expected_active_hand_mask = np.stack(
        [build_active_hand_mask(type_id, samples_per_type, hand_num=2) for type_id in REAL_GRASP_TYPE_IDS],
        axis=0,
    )
    if not np.array_equal(np.asarray(scene_data["active_hand_mask"], dtype=bool), expected_active_hand_mask):
        raise ValueError("Reverse active-hand mask does not match the exported grasp types")
    if int(scene_data["reverse_marginal_pool_size"]) != sampling["marginal_pool_size"]:
        raise ValueError("Reverse marginal pool size does not match config")
    if int(scene_data["reverse_conditional_candidate_num"]) != sampling["conditional_candidate_num"]:
        raise ValueError("Reverse conditional candidate count does not match config")
    if str(scene_data["reverse_resampling_policy"]) != sampling["resampling_policy"]:
        raise ValueError("Reverse resampling policy does not match config")
    resampled_pool_indices = np.asarray(scene_data["reverse_resampled_pool_indices"])
    if (resampled_pool_indices < 0).any() or (resampled_pool_indices >= sampling["marginal_pool_size"]).any():
        raise ValueError("Reverse resampled pool indices are out of range")
    for type_indices in resampled_pool_indices:
        if np.unique(type_indices).size != type_indices.size:
            raise ValueError("Reverse weighted-without-replacement indices contain duplicates within a mode")
    selected_resample_indices = np.asarray(scene_data["reverse_selected_resample_indices"])
    if (selected_resample_indices < 0).any() or (
        selected_resample_indices >= sampling["conditional_candidate_num"]
    ).any():
        raise ValueError("Reverse selected resample indices are out of range")
    selected_pool_indices = np.asarray(scene_data["reverse_selected_pool_indices"])
    if (selected_pool_indices < 0).any() or (selected_pool_indices >= sampling["marginal_pool_size"]).any():
        raise ValueError("Reverse selected pool indices are out of range")
    expected_selected_pool_indices = np.take_along_axis(
        resampled_pool_indices,
        selected_resample_indices,
        axis=1,
    )
    if not np.array_equal(selected_pool_indices, expected_selected_pool_indices):
        raise ValueError("Reverse selected pool indices do not match resampling provenance")
    raw_included = bool(scene_data.get("reverse_raw_pool_included", False))
    if raw_included != sampling["include_raw_pool"]:
        raise ValueError("Reverse raw-pool inclusion does not match config")
    if raw_included:
        raw_shapes = {
            "reverse_raw_centered_t24": (sampling["marginal_pool_size"], 24),
            "reverse_raw_marginal_log_prob": (sampling["marginal_pool_size"],),
            "reverse_raw_posterior_probability": (sampling["marginal_pool_size"], type_num),
            "reverse_raw_sampled_type_ids": (sampling["marginal_pool_size"],),
        }
        for key, expected_shape in raw_shapes.items():
            if key not in scene_data or np.asarray(scene_data[key]).shape != expected_shape:
                raise ValueError(f"Invalid or missing Reverse raw field {key}")
        raw_type_ids = np.asarray(scene_data["reverse_raw_sampled_type_ids"])
        if ((raw_type_ids < 1) | (raw_type_ids > type_num)).any():
            raise ValueError("Reverse raw sampled type ids must be in [1, 5]")
        for key in (
            "reverse_raw_centered_t24",
            "reverse_raw_marginal_log_prob",
            "reverse_raw_posterior_probability",
        ):
            if not np.isfinite(np.asarray(scene_data[key])).all():
                raise ValueError(f"Reverse scene export contains non-finite values in {key}")
        raw_posterior = np.asarray(scene_data["reverse_raw_posterior_probability"])
        if ((raw_posterior < 0.0) | (raw_posterior > 1.0)).any():
            raise ValueError("Reverse raw posterior probabilities must be in [0, 1]")
        if not np.allclose(raw_posterior.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("Reverse raw posterior rows must sum to 1")
        if not np.allclose(np.asarray(scene_data["budget_scores"]), raw_posterior.mean(axis=0), atol=1e-6):
            raise ValueError("Reverse budget scores must be computed from the unfiltered raw posterior pool")
        expected_ess = raw_posterior.sum(axis=0) ** 2 / np.maximum(
            np.square(raw_posterior).sum(axis=0),
            1e-12,
        )
        if not np.allclose(scene_data["reverse_importance_ess"], expected_ess, atol=1e-5):
            raise ValueError("Reverse importance ESS does not match the raw posterior pool")

        raw_t24 = np.asarray(scene_data["reverse_raw_centered_t24"])
        raw_log_prob = np.asarray(scene_data["reverse_raw_marginal_log_prob"])
        type_indices = np.arange(type_num)[:, None]
        expected_selected_t24 = raw_t24[selected_pool_indices]
        expected_selected_log_prob = raw_log_prob[selected_pool_indices]
        expected_selected_posterior = raw_posterior[selected_pool_indices, type_indices]
        expected_selected_score = expected_selected_log_prob + np.log(
            np.maximum(expected_selected_posterior, 1e-12)
        )
        if not np.allclose(scene_data["reverse_selected_centered_t24"], expected_selected_t24, atol=1e-6):
            raise ValueError("Reverse selected T24 does not match raw-pool provenance")
        if not np.allclose(scene_data["reverse_selected_marginal_log_prob"], expected_selected_log_prob, atol=1e-6):
            raise ValueError("Reverse selected marginal scores do not match raw-pool provenance")
        if not np.allclose(
            scene_data["reverse_selected_posterior_probability"],
            expected_selected_posterior,
            atol=1e-6,
        ):
            raise ValueError("Reverse selected posterior does not match raw-pool provenance")
        if not np.allclose(scene_data["reverse_selected_conditional_score"], expected_selected_score, atol=1e-6):
            raise ValueError("Reverse conditional scores do not match raw-pool provenance")


def validate_scene_export_completeness(
    scene_data: dict,
    config: DictConfig,
    checkpoint_meta: dict | None = None,
) -> None:
    """Validate that an existing scene file satisfies the current export config.

    Args:
        scene_data: Loaded per-scene export dictionary.
        config: Full Hydra config containing current export options.

    Returns:
        None. Raises an exception if the file should not be reused.
    """
    samples_per_type = export_samples_per_type(config)
    selection_metadata = export_sample_selection_metadata(config)
    required_keys = {
        "scene_id",
        "object_id",
        "split",
        "robot_name",
        "robot_size",
        "pc_runtime_scale",
        "scene_path",
        "pc_path",
        "grasp_type_ids",
        "grasp_type_names",
        "budget_scores",
        "wrist_quat",
        "active_hand_mask",
        "grasp_pos_source",
        "pose_candidate_num",
        "sample_selection_enabled",
        "sample_selection_scope",
        "sample_selection_mode",
        "sample_selection_intermediate_topk",
        "sample_selection_translation_scale_m",
        "sample_selection_rotation_weight",
    }
    expected_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    scene_source = normalize_hand_pos_source(str(scene_data.get("grasp_pos_source", expected_source)))
    if scene_source != expected_source:
        raise ValueError(f"Existing export uses grasp_pos_source={scene_source}, expected {expected_source}")
    position_key = position_key_for_source(expected_source)
    required_keys.add(position_key)
    if str(scene_data.get("export_position_key", position_key)) != position_key:
        raise ValueError("Existing export uses a different primary position field")
    missing_keys = sorted(required_keys.difference(scene_data.keys()))
    if missing_keys:
        raise KeyError(f"Missing required export keys: {missing_keys}")
    if bool(getattr(config.task, "include_log_prob", True)) and "log_prob" not in scene_data:
        raise KeyError("Existing export does not contain required log_prob")
    if bool(getattr(config.task, "include_grasp_pose", False)) and "grasp_pose" not in scene_data:
        raise KeyError("Existing export does not contain required grasp_pose")

    if is_reverse_factorization(config):
        validate_reverse_scene_export(scene_data, config, checkpoint_meta)

    position = np.asarray(scene_data[position_key])
    wrist_quat = np.asarray(scene_data["wrist_quat"])
    active_hand_mask = np.asarray(scene_data["active_hand_mask"])
    expected_prefix = (len(REAL_GRASP_TYPE_IDS), samples_per_type, 2)
    if position.shape[:3] != expected_prefix:
        raise ValueError(f"Expected {position_key} prefix {expected_prefix}, got {position.shape}")
    if wrist_quat.shape[:3] != expected_prefix:
        raise ValueError(f"Expected wrist_quat prefix {expected_prefix}, got {wrist_quat.shape}")
    if active_hand_mask.shape != expected_prefix:
        raise ValueError(f"Expected active_hand_mask shape {expected_prefix}, got {active_hand_mask.shape}")
    if np.asarray(scene_data["grasp_type_ids"]).astype(int).tolist() != list(REAL_GRASP_TYPE_IDS):
        raise ValueError("Existing export uses a different grasp type id order")
    if int(np.asarray(scene_data.get("samples_per_type", samples_per_type)).item()) != samples_per_type:
        raise ValueError("Existing export uses a different samples_per_type value")
    if str(scene_data.get("robot_name")) != export_robot_name(config):
        raise ValueError("Existing export uses a different robot_name value")
    if not np.isclose(float(scene_data.get("robot_size")), export_robot_size(config)):
        raise ValueError("Existing export uses a different robot_size value")
    if not np.isclose(float(scene_data.get("pc_runtime_scale")), export_pc_runtime_scale(config)):
        raise ValueError("Existing export uses a different pc_runtime_scale value")
    if int(np.asarray(scene_data.get("pose_candidate_num", -1)).item()) != selection_metadata["pose_candidate_num"]:
        raise ValueError("Existing export uses a different pose_candidate_num value")
    if bool(scene_data.get("sample_selection_enabled")) != selection_metadata["sample_selection_enabled"]:
        raise ValueError("Existing export uses a different sample_selection_enabled value")
    if str(scene_data.get("sample_selection_scope")) != selection_metadata["sample_selection_scope"]:
        raise ValueError("Existing export uses a different sample_selection_scope value")
    if str(scene_data.get("sample_selection_mode")) != selection_metadata["sample_selection_mode"]:
        raise ValueError("Existing export uses a different sample_selection_mode value")
    if scene_data.get("sample_selection_intermediate_topk") != selection_metadata["sample_selection_intermediate_topk"]:
        raise ValueError("Existing export uses a different sample_selection_intermediate_topk value")
    if not np.isclose(
        float(scene_data.get("sample_selection_translation_scale_m")),
        selection_metadata["sample_selection_translation_scale_m"],
    ):
        raise ValueError("Existing export uses a different sample_selection_translation_scale_m value")
    if not np.isclose(
        float(scene_data.get("sample_selection_rotation_weight")),
        selection_metadata["sample_selection_rotation_weight"],
    ):
        raise ValueError("Existing export uses a different sample_selection_rotation_weight value")
    validate_scene_export(scene_data, float(getattr(config.task, "quat_norm_tol", 1e-3)))


def load_complete_scene_export(
    scene_file: str,
    config: DictConfig,
    checkpoint_meta: dict | None = None,
) -> dict | None:
    """Load an existing per-scene export if it is complete for this run.

    Args:
        scene_file: Path to an existing per-scene ``.npy`` file.
        config: Full Hydra config containing current export options.

    Returns:
        Loaded scene dictionary, or ``None`` when the file should be regenerated.
    """
    try:
        scene_data = np.load(scene_file, allow_pickle=True).item()
        validate_scene_export_completeness(scene_data, config, checkpoint_meta)
    except Exception as exc:
        print(f"Will regenerate incomplete existing scene export {scene_file}: {exc}")
        return None
    return scene_data


def collect_complete_scene_ids(
    output_dir: str,
    config: DictConfig,
    checkpoint_meta: dict | None = None,
) -> set[str]:
    """Collect scene ids that can be safely skipped.

    Args:
        output_dir: Export output directory.
        config: Full Hydra config containing current export options.

    Returns:
        Set of scene ids with complete per-scene export files.
    """
    return set(collect_complete_scene_exports(output_dir, config, checkpoint_meta))


def collect_complete_scene_exports(
    output_dir: str,
    config: DictConfig,
    checkpoint_meta: dict | None = None,
) -> dict[str, tuple[dict, str]]:
    """Load reusable scene records together with their source file paths.

    Args:
        output_dir: Export output directory.
        config: Full Hydra config containing current export options.
        checkpoint_meta: Optional checkpoint hashes used by Reverse validation.

    Returns:
        Mapping from scene id to ``(scene_data, scene_file)``.
    """
    scene_dir = export_scene_dir(output_dir, config)
    if not os.path.isdir(scene_dir):
        return {}
    complete_exports: dict[str, tuple[dict, str]] = {}
    scene_files = sorted(glob(pjoin(scene_dir, "**", "*.npy"), recursive=True))
    for scene_file in tqdm(scene_files, desc="scan existing obj prior", leave=False):
        scene_data = load_complete_scene_export(scene_file, config, checkpoint_meta)
        if scene_data is None:
            continue
        scene_id = str(scene_data["scene_id"])
        if scene_id in complete_exports:
            raise ValueError(f"Duplicate complete scene export for scene_id={scene_id}")
        complete_exports[scene_id] = (scene_data, scene_file)
    return complete_exports


def scene_summary_from_data(scene_data: dict, scene_file: str) -> dict:
    """Build manifest/index metadata from one per-scene export.

    Args:
        scene_data: Loaded per-scene export dictionary.
        scene_file: Path to the per-scene export file.

    Returns:
        JSON-serializable summary row.
    """
    return {
        "scene_id": str(scene_data["scene_id"]),
        "object_id": str(scene_data["object_id"]),
        "split": str(scene_data["split"]),
        "robot_name": str(scene_data["robot_name"]),
        "robot_size": float(scene_data["robot_size"]),
        "pc_runtime_scale": float(scene_data["pc_runtime_scale"]),
        "scene_path": str(scene_data["scene_path"]),
        "pc_path": str(scene_data["pc_path"]),
        "scene_file": scene_file,
        "grasp_type_ids": np.asarray(scene_data["grasp_type_ids"]).astype(int),
        "grasp_type_names": np.asarray(scene_data["grasp_type_names"]).astype(str),
        "budget_scores": np.asarray(scene_data["budget_scores"], dtype=np.float32),
        "score_semantics": str(scene_data["score_semantics"]),
        "factorization": str(scene_data.get("factorization", PROPOSED_FACTORIZATION)),
    }


def scene_index_from_summary(summary: dict) -> dict:
    """Build the compact scene-index row shared by new and reused exports."""
    return {
        "scene_id": summary["scene_id"],
        "object_id": summary["object_id"],
        "split": summary["split"],
        "robot_name": summary["robot_name"],
        "robot_size": summary["robot_size"],
        "pc_runtime_scale": summary["pc_runtime_scale"],
        "factorization": summary["factorization"],
        "scene_file": summary["scene_file"],
    }


def build_scene_export_record(score_record: dict, pose_records: dict[int, dict], config: DictConfig) -> dict:
    """Merge score and fixed-type pose records for one scene.

    Args:
        score_record: Score-pass record for one scene.
        pose_records: Mapping from grasp type id to pose-pass record.
        config: Full Hydra config containing task export options.

    Returns:
        Per-scene export dictionary ready to save with ``np.save``.
    """
    samples_per_type = export_samples_per_type(config)
    missing_types = [type_id for type_id in REAL_GRASP_TYPE_IDS if type_id not in pose_records]
    if missing_types:
        raise KeyError(f"Missing pose records for scene_id={score_record['scene_id']}, type_ids={missing_types}")

    ordered_pose_records = [pose_records[type_id] for type_id in REAL_GRASP_TYPE_IDS]
    position_key = position_key_for_source(getattr(config.data, "hand_pos_source", "wrist"))
    position = np.stack([record[position_key] for record in ordered_pose_records], axis=0).astype(np.float32)
    wrist_quat = np.stack([record["wrist_quat"] for record in ordered_pose_records], axis=0).astype(np.float32)
    active_hand_mask = np.stack([record["active_hand_mask"] for record in ordered_pose_records], axis=0).astype(bool)
    selection_metadata = export_sample_selection_metadata(config)

    scene_data = {
        "scene_id": score_record["scene_id"],
        "object_id": score_record["object_id"],
        "split": score_record["split"],
        "robot_name": export_robot_name(config),
        "robot_size": np.float32(export_robot_size(config)),
        "pc_runtime_scale": np.float32(export_pc_runtime_scale(config)),
        "scene_path": score_record["scene_path"],
        "pc_path": score_record["pc_path"],
        "grasp_type_ids": np.asarray(REAL_GRASP_TYPE_IDS, dtype=np.int64),
        "grasp_type_names": np.asarray(REAL_GRASP_TYPE_NAMES),
        "samples_per_type": np.int64(samples_per_type),
        "pose_candidate_num": np.int64(selection_metadata["pose_candidate_num"]),
        "sample_selection_enabled": np.bool_(selection_metadata["sample_selection_enabled"]),
        "sample_selection_scope": selection_metadata["sample_selection_scope"],
        "sample_selection_mode": selection_metadata["sample_selection_mode"],
        "sample_selection_intermediate_topk": selection_metadata["sample_selection_intermediate_topk"],
        "sample_selection_translation_scale_m": np.float32(selection_metadata["sample_selection_translation_scale_m"]),
        "sample_selection_rotation_weight": np.float32(selection_metadata["sample_selection_rotation_weight"]),
        "score_semantics": score_semantics_from_config(config),
        "factorization": factorization_from_config(config),
        "budget_scores": np.asarray(score_record["budget_scores"], dtype=np.float32),
        position_key: position,
        "wrist_quat": wrist_quat,
        "active_hand_mask": active_hand_mask,
        "grasp_pos_source": normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist")),
        "export_position_key": position_key,
    }

    if all("log_prob" in record for record in ordered_pose_records):
        scene_data["log_prob"] = np.stack([record["log_prob"] for record in ordered_pose_records], axis=0).astype(
            np.float32
        )
    if all("grasp_pose" in record for record in ordered_pose_records):
        scene_data["grasp_pose"] = np.stack([record["grasp_pose"] for record in ordered_pose_records], axis=0).astype(
            np.float32
        )
    validate_scene_export(scene_data, float(getattr(config.task, "quat_norm_tol", 1e-3)))
    return scene_data


def expected_scene_count(config: DictConfig) -> int | None:
    """Count expected scene config files for the configured object splits.

    Args:
        config: Full Hydra config with ``test_data`` and ``task.object_splits``.

    Returns:
        Expected scene count, or ``None`` if split files cannot be read.
    """
    if bool(getattr(config.test_data, "mini_test", False)):
        return None
    if int(getattr(config.test_data, "test_object_num", 0)) > 0:
        return None
    if int(getattr(config.test_data, "test_scene_num", 0)) > 0:
        return None

    split_root = to_absolute_path(pjoin(str(config.test_data.object_path), str(config.test_data.split_path)))
    scene_count = 0
    for split_name in _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])):
        split_json = pjoin(split_root, f"{split_name}.json")
        if not os.path.isfile(split_json):
            return None
        object_ids = load_json(split_json)
        scene_patterns = _as_list(config.test_data.test_scene_cfg)
        for object_id in object_ids:
            base_dir = to_absolute_path(pjoin(str(config.test_data.object_path), "scene_cfg", str(object_id)))
            for pattern in scene_patterns:
                scene_count += len(glob(pjoin(base_dir, str(pattern)), recursive=True))
    return scene_count


def write_obj_human_prior_export(
    score_lines: list[dict],
    scene_index: list[dict],
    output_dir: str,
    manifest: dict,
    config: DictConfig,
) -> dict:
    """Write score JSONL, scene index and manifest after batch scene saves.

    Args:
        score_lines: Per-scene score summaries for all current scenes.
        scene_index: Per-scene index rows for all current scenes.
        output_dir: Destination output directory.
        manifest: Manifest fields built by the caller.
        config: Full Hydra config.

    Returns:
        Dictionary with output paths and scene count.
    """
    scene_dir = export_scene_dir(output_dir, config)
    os.makedirs(scene_dir, exist_ok=True)

    score_jsonl_path = pjoin(output_dir, "scene_budget_scores.jsonl")
    scene_index_path = pjoin(output_dir, "scene_index.json")
    manifest_path = pjoin(output_dir, "manifest.json")

    score_lines = sorted(score_lines, key=lambda row: row["scene_id"])
    scene_index = sorted(scene_index, key=lambda row: row["scene_id"])

    with open(score_jsonl_path, "w", encoding="utf-8") as score_handle:
        for row in score_lines:
            score_line = {
                "scene_id": row["scene_id"],
                "object_id": row["object_id"],
                "split": row["split"],
                "robot_name": row["robot_name"],
                "robot_size": row["robot_size"],
                "pc_runtime_scale": row["pc_runtime_scale"],
                "scene_path": row["scene_path"],
                "pc_path": row["pc_path"],
                "grasp_type_ids": row["grasp_type_ids"],
                "grasp_type_names": row["grasp_type_names"],
                "budget_scores": row["budget_scores"],
                "score_semantics": row["score_semantics"],
                "factorization": row["factorization"],
            }
            score_handle.write(json.dumps(score_line, default=_json_default, ensure_ascii=False) + "\n")

    manifest = dict(manifest)
    manifest.update(
        {
            "scene_count": len(scene_index),
            "scene_dir": scene_dir,
            "score_jsonl": score_jsonl_path,
            "scene_index": scene_index_path,
        }
    )
    with open(scene_index_path, "w", encoding="utf-8") as index_handle:
        json.dump(scene_index, index_handle, indent=2, ensure_ascii=False)
    with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
        json.dump(manifest, manifest_handle, indent=2, ensure_ascii=False, default=_json_default)

    return {
        "output_dir": output_dir,
        "manifest": manifest_path,
        "scene_index": scene_index_path,
        "score_jsonl": score_jsonl_path,
        "scene_count": len(scene_index),
    }


def build_manifest(config: DictConfig, checkpoint_meta: dict) -> dict:
    """Build export manifest metadata.

    Args:
        config: Full Hydra config.
        checkpoint_meta: Checkpoint metadata returned by ``load_export_models``.

    Returns:
        JSON-serializable manifest dictionary.
    """
    selection_metadata = export_sample_selection_metadata(config)
    return {
        "task_name": "obj_human_prior_export",
        "factorization": factorization_from_config(config),
        "checkpoint_path": checkpoint_meta["checkpoint_path"],
        "checkpoint_iter": checkpoint_meta["checkpoint_iter"],
        "uses_independent_models": bool(checkpoint_meta["uses_independent_models"]),
        "robot_name": export_robot_name(config),
        "robot_size": export_robot_size(config),
        "pc_runtime_scale": export_pc_runtime_scale(config),
        "score_checkpoint_path": checkpoint_meta["score_checkpoint_path"],
        "score_checkpoint_iter": checkpoint_meta["score_checkpoint_iter"],
        "score_checkpoint_sha256": checkpoint_meta.get("score_checkpoint_sha256"),
        "score_ckpt": checkpoint_meta.get("score_ckpt"),
        "pose_checkpoint_path": checkpoint_meta["pose_checkpoint_path"],
        "pose_checkpoint_iter": checkpoint_meta["pose_checkpoint_iter"],
        "pose_checkpoint_sha256": checkpoint_meta.get("pose_checkpoint_sha256"),
        "pose_ckpt": checkpoint_meta.get("pose_ckpt"),
        "model_roles": checkpoint_meta.get("model_roles"),
        "score_model_config": checkpoint_meta.get("score_model_config"),
        "pose_model_config": checkpoint_meta.get("pose_model_config"),
        "model_name": str(config.algo.model.name),
        "type_objective": str(getattr(config.algo.model, "type_objective", "ce")),
        "score_semantics": score_semantics_from_config(config),
        "samples_per_type": export_samples_per_type(config),
        "pose_candidate_num": selection_metadata["pose_candidate_num"],
        "sample_selection": {
            "enabled": selection_metadata["sample_selection_enabled"],
            "scope": selection_metadata["sample_selection_scope"],
            "mode": selection_metadata["sample_selection_mode"],
            "intermediate_topk": selection_metadata["sample_selection_intermediate_topk"],
            "translation_scale_m": selection_metadata["sample_selection_translation_scale_m"],
            "rotation_weight": selection_metadata["sample_selection_rotation_weight"],
        },
        "object_splits": _as_list(getattr(config.task, "object_splits", [config.test_data.test_split])),
        "score_grasp_types": _as_list(getattr(config.task, "score_grasp_types", ["0_any"])),
        "pose_grasp_types": _as_list(getattr(config.task, "pose_grasp_types", list(REAL_GRASP_TYPE_NAMES))),
        "include_log_prob": bool(getattr(config.task, "include_log_prob", True)),
        "include_grasp_pose": bool(getattr(config.task, "include_grasp_pose", False)),
        "export_position_key": position_key_for_source(getattr(config.data, "hand_pos_source", "wrist")),
        "grasp_type_ids": list(REAL_GRASP_TYPE_IDS),
        "grasp_type_names": list(REAL_GRASP_TYPE_NAMES),
        "test_data": OmegaConf.to_container(config.test_data, resolve=True),
        "data_hand_pos_source": normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist")),
        "seed": int(config.seed),
        "reverse_sampling": reverse_sampling_config(config) if is_reverse_factorization(config) else None,
        "posterior_contract": (
            OmegaConf.to_container(config.algo.posterior_contract, resolve=True)
            if is_reverse_factorization(config)
            else None
        ),
    }


def task_obj_human_prior_export(config: DictConfig) -> None:
    """Export object-scene human prior scores and hand-position proposals.

    Args:
        config: Full Hydra config. The task uses ``test_data`` as the object
            asset source and ``task.object_splits`` to choose object splits.

    Returns:
        None. Files are written under ``obj_human_prior``.
    """
    resolve_type_supervision_config(config)
    set_seed(config.seed)
    config.wandb.mode = "disabled"
    if not bool(config.algo.human):
        raise ValueError("task=obj_human_prior_export expects a human prior model with algo.human=True")
    export_robot_name(config)
    export_robot_size(config)

    score_model, pose_model, checkpoint_meta = load_export_models(config)
    output_dir = resolve_output_dir(config, checkpoint_meta)
    skip_existing = bool(getattr(config.task, "skip_existing", True))
    complete_exports = (
        collect_complete_scene_exports(output_dir, config, checkpoint_meta)
        if skip_existing
        else {}
    )
    skip_scene_ids = set(complete_exports)
    if skip_scene_ids:
        print(f"Skipping {len(skip_scene_ids)} existing complete scene exports from {output_dir}")

    split_names = _as_list(getattr(config.task, "object_splits", [config.test_data.test_split]))
    split_lookup = build_object_split_lookup(config.test_data, split_names)
    scene_dir = export_scene_dir(output_dir, config)

    with torch.no_grad():
        if is_reverse_factorization(config):
            new_score_lines, new_scene_index = sample_reverse_scene_scores_and_poses(
                config,
                score_model,
                pose_model,
                split_lookup,
                scene_dir,
                checkpoint_meta,
                skip_scene_ids=skip_scene_ids,
            )
        else:
            new_score_lines, new_scene_index = sample_scene_scores_and_fixed_type_poses(
                config,
                score_model,
                pose_model,
                split_lookup,
                scene_dir,
                skip_scene_ids=skip_scene_ids,
            )

    expected_count = expected_scene_count(config)
    if expected_count is not None and not skip_existing and expected_count != len(new_score_lines):
        raise RuntimeError(f"Expected {expected_count} scenes, but exported scores for {len(new_score_lines)} scenes")

    existing_score_lines = []
    existing_scene_index = []
    for scene_data, scene_file in complete_exports.values():
        summary = scene_summary_from_data(scene_data, scene_file)
        existing_score_lines.append(summary)
        existing_scene_index.append(scene_index_from_summary(summary))
    score_lines = existing_score_lines + new_score_lines
    scene_index = existing_scene_index + new_scene_index
    scene_ids = [row["scene_id"] for row in scene_index]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Duplicate scene ids found while merging reused and newly exported scenes")

    manifest = build_manifest(config, checkpoint_meta)
    if expected_count is not None:
        manifest["expected_scene_count"] = expected_count
    manifest["skip_existing"] = skip_existing
    manifest["skipped_complete_scene_count"] = len(skip_scene_ids)
    manifest["new_scene_count"] = len(new_score_lines)
    output_paths = write_obj_human_prior_export(
        score_lines,
        scene_index,
        output_dir,
        manifest,
        config,
    )
    if expected_count is not None and skip_existing and len(score_lines) < expected_count:
        print(
            f"Warning: expected {expected_count} scenes, but found {len(skip_scene_ids)} skipped and "
            f"{len(new_score_lines)} newly exported scenes for this config."
        )
    print(f"Saved object human prior export to {output_paths['output_dir']}")
    print(f"Exported {output_paths['scene_count']} scenes")
