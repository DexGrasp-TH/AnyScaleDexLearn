import copy
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
from dexlearn.task.sample import _decenter_human_pose
from dexlearn.utils.config import resolve_type_supervision_config
from dexlearn.utils.human_hand import normalize_hand_pos_source
from dexlearn.utils.util import load_json, set_seed


REAL_GRASP_TYPE_IDS = tuple(range(1, len(GRASP_TYPES)))
REAL_GRASP_TYPE_NAMES = tuple(GRASP_TYPES[type_id] for type_id in REAL_GRASP_TYPE_IDS)


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
        config: Full Hydra config containing ``test_data.object_path``.

    Returns:
        Directory named after the final component of the object asset path.
    """
    object_path = to_absolute_path(str(config.test_data.object_path)).rstrip(os.sep)
    asset_name = os.path.basename(object_path)
    if not asset_name:
        raise ValueError(f"Could not infer asset name from object_path={config.test_data.object_path}")
    return pjoin(output_dir, asset_name)


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
    objective = str(getattr(config.algo.model, "type_objective", "ce")).lower()
    if objective == "ce":
        return "softmax_probability"
    if objective == "object_bce":
        return "sigmoid_compatibility"
    if objective == "scene_ranking":
        return "ranking_compatibility"
    return f"model_score:{objective}"


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
        samples_per_type: Number of pose samples per explicit grasp type.

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
    global_feature_expanded = (
        global_feature[:, None, None, :]
        .expand(batch_size, type_num, samples_per_type, feature_dim)
        .reshape(batch_size * type_num * samples_per_type, feature_dim)
    )
    type_ids_flat = (
        type_ids[None, :, None]
        .expand(batch_size, type_num, samples_per_type)
        .reshape(batch_size * type_num * samples_per_type)
    )
    cond_feat = torch.cat([global_feature_expanded, model.grasp_type_emb(type_ids_flat)], dim=-1)
    robot_pose, log_prob = model.output_head.sample(cond_feat, type_ids_flat, 1)

    robot_pose = robot_pose.reshape(batch_size, type_num, samples_per_type, *robot_pose.shape[1:])
    robot_pose = robot_pose[:, :, :, 0]
    robot_pose = robot_pose.reshape(batch_size, type_num * samples_per_type, *robot_pose.shape[3:])
    if "pc_centroid" in data:
        grasp_type_for_decenter = (
            type_ids[None, :, None]
            .expand(batch_size, type_num, samples_per_type)
            .reshape(batch_size, type_num * samples_per_type)
        )
        robot_pose = _decenter_human_pose(robot_pose, data["pc_centroid"], grasp_type_for_decenter)
    grasp_pose = pose_tensor_to_grasp_pose(robot_pose)
    grasp_pose = grasp_pose.reshape(batch_size, type_num, samples_per_type, grasp_pose.shape[-1])

    log_prob = log_prob.reshape(batch_size, type_num, samples_per_type, -1)
    if log_prob.shape[-1] != 1:
        raise ValueError(f"Expected one log_prob per generated pose, got shape {tuple(log_prob.shape)}")
    log_prob = log_prob[..., 0]
    return grasp_pose, log_prob, type_ids


def sample_scene_scores_and_fixed_type_poses(
    config: DictConfig,
    model: torch.nn.Module,
    split_lookup: dict[str, str],
    scene_dir: str,
    skip_scene_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run one scene pass and save each generated batch immediately.

    Args:
        config: Full Hydra config.
        model: Loaded human prior model.
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
    samples_per_type = int(getattr(config.task, "samples_per_type", 20))
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

            global_feature, _ = model.backbone(data)
            _, type_scores = model._compute_type_scores(global_feature)
            scores = extract_real_type_scores(type_scores)
            grasp_pose_tensor, log_prob_tensor, sampled_type_ids = sample_fixed_types_from_features(
                config,
                model,
                data,
                global_feature,
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
    samples_per_type = int(getattr(config.task, "samples_per_type", 20))
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

            result = model.sample(data, samples_per_type)
            robot_pose, pred_grasp_type, log_prob = unpack_pose_sample_result(result)

            grasp_type_for_decenter = pred_grasp_type if pred_grasp_type is not None else data["grasp_type_id"]
            if grasp_type_for_decenter.ndim == 1:
                grasp_type_for_decenter = grasp_type_for_decenter.unsqueeze(1).expand(-1, robot_pose.shape[1])
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


def resolve_output_dir(config: DictConfig, checkpoint_iter: int | None) -> str:
    """Resolve the export output directory.

    Args:
        config: Full Hydra config.
        checkpoint_iter: Training iteration stored in the checkpoint.

    Returns:
        Absolute output directory path.
    """
    configured_output = getattr(config.task, "output_dir", None)
    if configured_output:
        return to_absolute_path(str(configured_output))
    if checkpoint_iter is None:
        step_name = os.path.splitext(_checkpoint_name(config.ckpt))[0]
    else:
        step_name = f"step_{checkpoint_iter:06d}"
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


def validate_scene_export_completeness(scene_data: dict, config: DictConfig) -> None:
    """Validate that an existing scene file satisfies the current export config.

    Args:
        scene_data: Loaded per-scene export dictionary.
        config: Full Hydra config containing current export options.

    Returns:
        None. Raises an exception if the file should not be reused.
    """
    samples_per_type = int(getattr(config.task, "samples_per_type", 20))
    required_keys = {
        "scene_id",
        "object_id",
        "split",
        "scene_path",
        "pc_path",
        "grasp_type_ids",
        "grasp_type_names",
        "budget_scores",
        "wrist_quat",
        "active_hand_mask",
        "grasp_pos_source",
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
    validate_scene_export(scene_data, float(getattr(config.task, "quat_norm_tol", 1e-3)))


def load_complete_scene_export(scene_file: str, config: DictConfig) -> dict | None:
    """Load an existing per-scene export if it is complete for this run.

    Args:
        scene_file: Path to an existing per-scene ``.npy`` file.
        config: Full Hydra config containing current export options.

    Returns:
        Loaded scene dictionary, or ``None`` when the file should be regenerated.
    """
    try:
        scene_data = np.load(scene_file, allow_pickle=True).item()
        validate_scene_export_completeness(scene_data, config)
    except Exception as exc:
        print(f"Will regenerate incomplete existing scene export {scene_file}: {exc}")
        return None
    return scene_data


def collect_complete_scene_ids(output_dir: str, config: DictConfig) -> set[str]:
    """Collect scene ids that can be safely skipped.

    Args:
        output_dir: Export output directory.
        config: Full Hydra config containing current export options.

    Returns:
        Set of scene ids with complete per-scene export files.
    """
    scene_dir = export_scene_dir(output_dir, config)
    if not os.path.isdir(scene_dir):
        return set()
    complete_scene_ids = set()
    scene_files = sorted(glob(pjoin(scene_dir, "**", "*.npy"), recursive=True))
    for scene_file in tqdm(scene_files, desc="scan existing obj prior", leave=False):
        scene_data = load_complete_scene_export(scene_file, config)
        if scene_data is not None:
            complete_scene_ids.add(str(scene_data["scene_id"]))
    return complete_scene_ids


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
        "scene_path": str(scene_data["scene_path"]),
        "pc_path": str(scene_data["pc_path"]),
        "scene_file": scene_file,
        "grasp_type_ids": np.asarray(scene_data["grasp_type_ids"]).astype(int),
        "grasp_type_names": np.asarray(scene_data["grasp_type_names"]).astype(str),
        "budget_scores": np.asarray(scene_data["budget_scores"], dtype=np.float32),
        "score_semantics": str(scene_data["score_semantics"]),
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
    samples_per_type = int(getattr(config.task, "samples_per_type", 20))
    missing_types = [type_id for type_id in REAL_GRASP_TYPE_IDS if type_id not in pose_records]
    if missing_types:
        raise KeyError(f"Missing pose records for scene_id={score_record['scene_id']}, type_ids={missing_types}")

    ordered_pose_records = [pose_records[type_id] for type_id in REAL_GRASP_TYPE_IDS]
    position_key = position_key_for_source(getattr(config.data, "hand_pos_source", "wrist"))
    position = np.stack([record[position_key] for record in ordered_pose_records], axis=0).astype(np.float32)
    wrist_quat = np.stack([record["wrist_quat"] for record in ordered_pose_records], axis=0).astype(np.float32)
    active_hand_mask = np.stack([record["active_hand_mask"] for record in ordered_pose_records], axis=0).astype(bool)

    scene_data = {
        "scene_id": score_record["scene_id"],
        "object_id": score_record["object_id"],
        "split": score_record["split"],
        "scene_path": score_record["scene_path"],
        "pc_path": score_record["pc_path"],
        "grasp_type_ids": np.asarray(REAL_GRASP_TYPE_IDS, dtype=np.int64),
        "grasp_type_names": np.asarray(REAL_GRASP_TYPE_NAMES),
        "samples_per_type": np.int64(samples_per_type),
        "score_semantics": score_semantics_from_config(config),
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
        score_lines: Per-scene score summaries for newly saved scenes.
        scene_index: Per-scene index rows for newly saved scenes.
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
                "scene_path": row["scene_path"],
                "pc_path": row["pc_path"],
                "grasp_type_ids": row["grasp_type_ids"],
                "grasp_type_names": row["grasp_type_names"],
                "budget_scores": row["budget_scores"],
                "score_semantics": row["score_semantics"],
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


def build_manifest(config: DictConfig, checkpoint_path: str, checkpoint_iter: int | None) -> dict:
    """Build export manifest metadata.

    Args:
        config: Full Hydra config.
        checkpoint_path: Resolved checkpoint path.
        checkpoint_iter: Iteration stored in the checkpoint, if available.

    Returns:
        JSON-serializable manifest dictionary.
    """
    return {
        "task_name": "obj_human_prior_export",
        "checkpoint_path": checkpoint_path,
        "checkpoint_iter": checkpoint_iter,
        "model_name": str(config.algo.model.name),
        "type_objective": str(getattr(config.algo.model, "type_objective", "ce")),
        "score_semantics": score_semantics_from_config(config),
        "samples_per_type": int(getattr(config.task, "samples_per_type", 20)),
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

    model, checkpoint_path, checkpoint_iter = load_export_model(config)
    output_dir = resolve_output_dir(config, checkpoint_iter)
    skip_existing = bool(getattr(config.task, "skip_existing", True))
    skip_scene_ids = collect_complete_scene_ids(output_dir, config) if skip_existing else set()
    if skip_scene_ids:
        print(f"Skipping {len(skip_scene_ids)} existing complete scene exports from {output_dir}")

    split_names = _as_list(getattr(config.task, "object_splits", [config.test_data.test_split]))
    split_lookup = build_object_split_lookup(config.test_data, split_names)
    scene_dir = export_scene_dir(output_dir, config)

    with torch.no_grad():
        score_lines, scene_index = sample_scene_scores_and_fixed_type_poses(
            config,
            model,
            split_lookup,
            scene_dir,
            skip_scene_ids=skip_scene_ids,
        )

    expected_count = expected_scene_count(config)
    if expected_count is not None and not skip_existing and expected_count != len(score_lines):
        raise RuntimeError(f"Expected {expected_count} scenes, but exported scores for {len(score_lines)} scenes")

    manifest = build_manifest(config, checkpoint_path, checkpoint_iter)
    if expected_count is not None:
        manifest["expected_scene_count"] = expected_count
    manifest["skip_existing"] = skip_existing
    manifest["skipped_complete_scene_count"] = len(skip_scene_ids)
    manifest["new_scene_count"] = len(score_lines)
    output_paths = write_obj_human_prior_export(
        score_lines,
        scene_index,
        output_dir,
        manifest,
        config,
    )
    if expected_count is not None and skip_existing and len(skip_scene_ids) + len(score_lines) < expected_count:
        print(
            f"Warning: expected {expected_count} scenes, but found {len(skip_scene_ids)} skipped and "
            f"{len(score_lines)} newly exported scenes for this config."
        )
    print(f"Saved object human prior export to {output_paths['output_dir']}")
    print(f"Exported {output_paths['scene_count']} scenes")
