import csv
import json
import os
import re
from copy import deepcopy
from glob import glob

import numpy as np
from scipy.spatial.transform import Rotation as SciR

try:
    from dexlearn.dataset.grasp_types import GRASP_TYPES
    from dexlearn.utils.config import cfg_get, flatten_multidex_data_config, resolve_type_supervision_config
    from omegaconf import ListConfig
except ModuleNotFoundError:
    GRASP_TYPES = [
        "0_any",
        "1_right_two",
        "2_right_three",
        "3_right_full",
        "4_both_three",
        "5_both_full",
    ]
    ListConfig = list

    def cfg_get(config, *keys, default=None):
        """Read the first available flat or dotted key without OmegaConf.

        Args:
            config: Object or dictionary-like config.
            *keys: Candidate keys, including dotted nested keys.
            default: Value returned when no key exists.

        Returns:
            First matched value or ``default``.
        """
        for key in keys:
            current = config
            found = True
            for part in str(key).split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    found = False
                    break
            if found:
                return current
        return default

    def flatten_multidex_data_config(config):
        """No-op fallback used only when OmegaConf is unavailable.

        Args:
            config: Data config object.

        Returns:
            The original config.
        """
        return config

    def resolve_type_supervision_config(config):
        """No-op fallback used only when OmegaConf is unavailable."""
        return config


MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
FIXED_LEFT_HAND_TRANS = np.array([0.0, 0.0, -0.5], dtype=np.float32)
FIXED_LEFT_HAND_ROT = np.eye(3, dtype=np.float32)
VALID_HAND_POS_SOURCES = {"wrist", "index_mcp"}
TARGET_GRASP_TYPE_IDS = (1, 2, 3, 4, 5)
DEXLEARN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(DEXLEARN_ROOT)
FIXED_TYPE_SAMPLE_LIST = ",".join(GRASP_TYPES[1:])


def resolve_existing_path(path: str):
    """Resolve a possibly relative saved metadata path.

    Args:
        path: Path saved in a sample file.

    Returns:
        Existing filesystem path, or the original path when no candidate exists.
    """
    if not path:
        return path
    path = str(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidates = [
        os.path.join(REPO_ROOT, path),
        os.path.join(os.getcwd(), path),
        os.path.abspath(path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def load_json(path: str):
    """Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r") as f:
        return json.load(f)


def normalize_hand_pos_source(value, default: str = "wrist") -> str:
    """Normalize the configured human hand position source.

    Args:
        value: Raw config value.
        default: Fallback source when ``value`` is unset.

    Returns:
        ``wrist`` or ``index_mcp``.
    """
    hand_pos_source = str(value if value is not None else default).lower()
    if hand_pos_source not in VALID_HAND_POS_SOURCES:
        raise ValueError(
            f"Unsupported hand_pos_source={hand_pos_source}. "
            f"Expected one of {sorted(VALID_HAND_POS_SOURCES)}."
        )
    return hand_pos_source


def resolve_ckpt_path(config) -> str:
    """Resolve the checkpoint path used to derive the sampled-output directory.

    Args:
        config: Hydra config with ``ckpt``, ``output_folder``, and ``wandb.id`` fields.

    Returns:
        Absolute or relative checkpoint path that exists or is expected under the run
        checkpoint directory.
    """
    if config.ckpt is not None and os.path.exists(str(config.ckpt)):
        return str(config.ckpt)

    ckpt_dir = os.path.join(config.output_folder, config.wandb.id, "ckpts")
    if config.ckpt is None:
        all_ckpts = sorted(glob(os.path.join(ckpt_dir, "step_**.pth")))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        return all_ckpts[-1]

    return os.path.join(ckpt_dir, f"step_{config.ckpt}.pth")


def get_sample_output_dir(config) -> str:
    """Build the directory containing saved sampled grasp ``.npy`` files.

    Args:
        config: Hydra config with checkpoint and ``test_data.name`` fields.

    Returns:
        Sample output directory corresponding to the resolved checkpoint.
    """
    ckpt_path = resolve_ckpt_path(config)
    return os.path.join(ckpt_path.replace("ckpts", "tests").replace(".pth", ""), config.test_data.name)


def iter_data_configs(data_config):
    """Iterate over one or more data configs after applying flat compatibility keys.

    Args:
        data_config: Hydra data config. ``object_path`` may be a string or a list.

    Returns:
        Generator yielding one normalized config per object root.
    """
    flatten_multidex_data_config(data_config)
    object_path = cfg_get(data_config, "object_path", "paths.object_path")
    if isinstance(object_path, ListConfig):
        for path in object_path:
            single_config = deepcopy(data_config)
            single_config.object_path = path
            flatten_multidex_data_config(single_config)
            yield single_config
    else:
        yield data_config


def canonical_object_id(object_name: str) -> str:
    """Infer a stable object id by removing common sequence suffixes.

    Args:
        object_name: Object or sequence name such as ``obj_0_seq_3`` or
            ``obj_0/seq_3``.

    Returns:
        Canonical object id, for example ``obj_0`` for ``obj_0_seq_3``.
    """
    normalized_name = str(object_name).strip().replace("\\", "/").strip("/")
    parts = [part for part in normalized_name.split("/") if part]
    if len(parts) >= 2 and re.match(r"^seq[_-]?\d+.*$", parts[-1]):
        base_name = parts[-2]
    else:
        base_name = os.path.basename(normalized_name)
    return re.sub(r"([_-]seq[_-]?\d+.*)$", "", base_name)


def normalize_sequence_id(sequence_id) -> str:
    """Normalize a sequence id into a stable string.

    Args:
        sequence_id: Raw sequence id from scene or grasp metadata.

    Returns:
        Normalized sequence id, or an empty string when unavailable.
    """
    if sequence_id is None:
        return ""
    return str(sequence_id).strip().replace("\\", "/").strip("/")


def scene_id_from_object_sequence(object_id: str, sequence_id: str) -> str:
    """Build the canonical human scene id used by formatted human data.

    Args:
        object_id: Canonical object id.
        sequence_id: Sequence id such as ``seq_3``.

    Returns:
        Scene id in ``object_id/sequence_id`` form when a sequence is available.
    """
    sequence_id = normalize_sequence_id(sequence_id)
    if not sequence_id:
        return str(object_id)
    return f"{object_id}/{sequence_id}"


def scene_key(object_id: str, scene_id: str) -> str:
    """Build a stable key for joining saved scores with human scene labels.

    Args:
        object_id: Canonical object id.
        scene_id: Scene id, optionally including the object id.

    Returns:
        Join key combining object and sequence-level scene identity.
    """
    object_id = canonical_object_id(object_id)
    normalized_scene = normalize_sequence_id(scene_id)
    if normalized_scene.startswith(f"{object_id}/"):
        return normalized_scene
    if normalized_scene:
        return scene_id_from_object_sequence(object_id, os.path.basename(normalized_scene))
    return object_id


def scene_id_from_grasp_data(grasp_data: dict, grasp_path: str = "") -> str:
    """Infer a scene id from formatted human grasp metadata.

    Args:
        grasp_data: Loaded human grasp dictionary.
        grasp_path: Optional path used as a fallback for old data.

    Returns:
        Scene id such as ``obj_173/seq_3``.
    """
    object_data = grasp_data.get("object", {})
    object_id = canonical_object_id(object_data.get("name", ""))
    sequence_id = normalize_sequence_id(object_data.get("sequence_id", ""))
    if not sequence_id and object_data.get("source_scene") is not None:
        source_scene = str(object_data.get("source_scene"))
        match = re.search(r"(seq[_-]?\d+.*)$", source_scene)
        sequence_id = normalize_sequence_id(match.group(1)) if match else ""
    if not sequence_id and grasp_path:
        sequence_id = os.path.splitext(os.path.basename(grasp_path))[0]
    return scene_id_from_object_sequence(object_id, sequence_id)


def scene_id_from_scene_cfg(scene_cfg: dict, scene_path: str = "") -> str:
    """Infer a scene id from a saved scene config.

    Args:
        scene_cfg: Loaded scene config dictionary.
        scene_path: Optional path used as a fallback.

    Returns:
        Scene id string.
    """
    if scene_cfg.get("scene_id") is not None:
        return normalize_sequence_id(scene_cfg.get("scene_id"))
    if "object" in scene_cfg:
        object_data = scene_cfg["object"]
        object_id = canonical_object_id(object_data.get("name", ""))
        sequence_id = normalize_sequence_id(object_data.get("sequence_id", ""))
        if sequence_id:
            return scene_id_from_object_sequence(object_id, sequence_id)
    if scene_path:
        return os.path.splitext(os.path.basename(scene_path))[0]
    return ""


def human_grasp_type_id(grasp_data: dict) -> int:
    """Match ``HumanMultiDexDataset`` grasp-type inference for one raw grasp.

    Args:
        grasp_data: Raw human grasp dictionary loaded from a training ``.npy`` file.

    Returns:
        Grasp type id in ``1..5``.
    """
    l_contacts = grasp_data["hand"]["left"]["contacts"] if grasp_data["hand"]["left"] else [False] * 5
    r_contacts = grasp_data["hand"]["right"]["contacts"] if grasp_data["hand"]["right"] else [False] * 5
    has_l, has_r = any(l_contacts), any(r_contacts)
    l_count, r_count = sum(l_contacts), sum(r_contacts)

    if not (has_l or has_r):
        raise ValueError("Grasp data has no active contacts.")
    if has_l and has_r:
        return 5 if (l_count > 3 or r_count > 3) else 4
    count = l_count if has_l else r_count
    if count <= 2:
        return 1
    if count == 3:
        return 2
    return 3


def hand_translation(hand_data: dict, hand_pos_source: str) -> np.ndarray:
    """Read the hand target position used by the trained human model.

    Args:
        hand_data: Per-hand dictionary from a raw training grasp.
        hand_pos_source: ``wrist`` or ``index_mcp``.

    Returns:
        Three-dimensional hand target position.
    """
    if hand_pos_source == "wrist":
        return np.asarray(hand_data["trans"], dtype=np.float32)
    if "index_mcp_pos" not in hand_data:
        raise KeyError("Missing hand['index_mcp_pos']; run human_preprocess or use hand_pos_source=wrist.")
    return np.asarray(hand_data["index_mcp_pos"], dtype=np.float32)


def hand_rotation_matrix(hand_data: dict) -> np.ndarray:
    """Read one hand axis-angle rotation as a single ``3x3`` matrix."""
    rotvec = np.asarray(hand_data["rot"], dtype=np.float32).reshape(-1, 3)[0]
    return SciR.from_rotvec(rotvec).as_matrix().astype(np.float32)


def training_pose_record(grasp_path: str, hand_pos_source: str) -> dict:
    """Convert one raw training grasp into the pose representation used for comparison.

    Args:
        grasp_path: Path to one training grasp ``.npy`` file.
        hand_pos_source: ``wrist`` or ``index_mcp`` position convention.

    Returns:
        Dictionary containing object ids, grasp type, right/left translations, and
        right/left rotation matrices.
    """
    grasp_data = np.load(grasp_path, allow_pickle=True).item()
    grasp_type_id = human_grasp_type_id(grasp_data)
    object_name = str(grasp_data["object"]["name"])
    object_id = canonical_object_id(object_name)
    grasp_scene_id = scene_id_from_grasp_data(grasp_data, grasp_path)
    left_data = grasp_data["hand"]["left"]
    right_data = grasp_data["hand"]["right"]
    has_left = bool(left_data) and any(left_data["contacts"])
    has_right = bool(right_data) and any(right_data["contacts"])

    if has_left and not has_right:
        left_trans = hand_translation(left_data, hand_pos_source)
        left_rot = hand_rotation_matrix(left_data)
        right_trans = MIRROR @ left_trans
        right_rot = MIRROR @ left_rot @ MIRROR
        left_trans = FIXED_LEFT_HAND_TRANS.astype(np.float32)
        left_rot = FIXED_LEFT_HAND_ROT.astype(np.float32)
    else:
        if has_right:
            right_trans = hand_translation(right_data, hand_pos_source)
            right_rot = hand_rotation_matrix(right_data)
        else:
            right_trans = np.zeros(3, dtype=np.float32)
            right_rot = np.eye(3, dtype=np.float32)

        if has_left:
            left_trans = hand_translation(left_data, hand_pos_source)
            left_rot = hand_rotation_matrix(left_data)
        else:
            left_trans = FIXED_LEFT_HAND_TRANS.astype(np.float32)
            left_rot = FIXED_LEFT_HAND_ROT.astype(np.float32)

    return {
        "path": grasp_path,
        "object_name": object_name,
        "canonical_object_id": object_id,
        "scene_id": grasp_scene_id,
        "scene_key": scene_key(object_id, grasp_scene_id),
        "grasp_type_id": grasp_type_id,
        "right_trans": np.asarray(right_trans, dtype=np.float32),
        "right_rot": np.asarray(right_rot, dtype=np.float32),
        "left_trans": np.asarray(left_trans, dtype=np.float32),
        "left_rot": np.asarray(left_rot, dtype=np.float32),
    }


def load_training_records(config) -> list:
    """Load all human training grasp pose records from configured train splits.

    Args:
        config: Hydra config with ``data`` pointing to ``HumanMultiDexDataset`` roots.

    Returns:
        List of comparable training pose records.
    """
    records = []
    hand_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    for data_config in iter_data_configs(config.data):
        if data_config.dataset_type != "HumanMultiDexDataset":
            raise NotImplementedError(f"Only HumanMultiDexDataset is supported, got {data_config.dataset_type}")
        split_path = os.path.join(data_config.object_path, data_config.split_path, "train.json")
        for obj_id in load_json(split_path):
            grasp_paths = sorted(glob(os.path.join(data_config.grasp_path, obj_id, "**/*.npy"), recursive=True))
            for grasp_path in grasp_paths:
                records.append(training_pose_record(grasp_path, hand_pos_source))
    if not records:
        raise RuntimeError("No training grasp records found.")
    return records


def resolve_sample_object_name(sample_data: dict) -> str:
    """Resolve the object name associated with a saved sampled result.

    Args:
        sample_data: Saved sampled grasp dictionary from ``task=sample``.

    Returns:
        Object name from the sample scene config, falling back to saved paths when
        scene metadata is unavailable.
    """
    scene_path = str(sample_data.get("scene_path", ""))
    if scene_path:
        try:
            scene_cfg = np.load(resolve_existing_path(scene_path), allow_pickle=True).item()
            if "object" in scene_cfg and "name" in scene_cfg["object"]:
                return str(scene_cfg["object"]["name"])
            task_obj_name = scene_cfg.get("task", {}).get("obj_name")
            if task_obj_name is not None:
                return str(task_obj_name)
        except Exception as exc:
            print(f"[evaluate] Could not load scene config {scene_path}: {exc}")

    pc_path = str(sample_data.get("pc_path", ""))
    return os.path.basename(os.path.dirname(pc_path)) if pc_path else ""


def resolve_sample_scene_metadata(sample_data: dict) -> dict:
    """Resolve object and scene identifiers from a saved sample.

    Args:
        sample_data: Saved sample dictionary from ``task=sample``.

    Returns:
        Dictionary with ``object_name``, ``canonical_object_id``, ``scene_id``,
        and ``scene_key``.
    """
    scene_id = ""
    object_name = ""
    scene_path = str(sample_data.get("scene_path", ""))
    if scene_path:
        try:
            scene_cfg = np.load(resolve_existing_path(scene_path), allow_pickle=True).item()
            scene_id = scene_id_from_scene_cfg(scene_cfg, scene_path)
            if "object" in scene_cfg and "name" in scene_cfg["object"]:
                object_name = str(scene_cfg["object"]["name"])
            else:
                task_obj_name = scene_cfg.get("task", {}).get("obj_name")
                if task_obj_name is not None:
                    object_name = str(task_obj_name)
        except Exception as exc:
            print(f"[evaluate] Could not load scene metadata {scene_path}: {exc}")

    if not object_name:
        object_name = resolve_sample_object_name(sample_data)
    object_id = canonical_object_id(object_name)
    return {
        "object_name": object_name,
        "canonical_object_id": object_id,
        "scene_id": scene_id,
        "scene_key": scene_key(object_id, scene_id),
    }


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to a rotation matrix.

    Args:
        quat: Quaternion in PyTorch3D / saved sample order ``[w, x, y, z]``.

    Returns:
        A ``3x3`` rotation matrix.
    """
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    quat = quat / norm
    return SciR.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix().astype(np.float32)


def sample_pose_records(sample_path: str) -> list:
    """Convert one saved sampled result file into comparable pose records.

    Args:
        sample_path: Path to one saved sample ``.npy`` file.

    Returns:
        List with one record for the file; empty if the file has no human
        ``grasp_pose`` field.
    """
    sample_data = np.load(sample_path, allow_pickle=True).item()
    if "grasp_pose" not in sample_data:
        return []

    pose = np.asarray(sample_data["grasp_pose"], dtype=np.float32).reshape(-1)
    if pose.size < 7:
        return []
    if pose.size < 14:
        pose = np.concatenate([pose[:7], np.zeros(7, dtype=np.float32)], axis=0)

    grasp_type_source = sample_data.get("pred_grasp_type_id", sample_data.get("grasp_type_id", 0))
    grasp_type_id = int(np.asarray(grasp_type_source).reshape(-1)[0])
    metadata = resolve_sample_scene_metadata(sample_data)
    return [
        {
            "path": sample_path,
            "object_name": metadata["object_name"],
            "canonical_object_id": metadata["canonical_object_id"],
            "scene_id": metadata["scene_id"],
            "scene_key": metadata["scene_key"],
            "grasp_type_id": grasp_type_id,
            "right_trans": pose[0:3],
            "right_rot": quat_wxyz_to_matrix(pose[3:7]),
            "left_trans": pose[7:10],
            "left_rot": quat_wxyz_to_matrix(pose[10:14]),
        }
    ]


def load_sample_records(sample_files: list) -> list:
    """Load all comparable sampled pose records from saved sample files.

    Args:
        sample_files: List of saved sample ``.npy`` paths.

    Returns:
        List of comparable sampled pose records. Files without ``grasp_pose`` are
        skipped because this task is specific to human sampled grasps.
    """
    records = []
    for sample_file in sample_files:
        records.extend(sample_pose_records(sample_file))
    return records


def active_sides(grasp_type_id: int) -> tuple:
    """Return sides that should contribute to pose distance.

    Args:
        grasp_type_id: Human multi-grasp type id.

    Returns:
        Tuple of active side names. Types 4 and 5 use both hands; types 1..3 use
        the right hand only.
    """
    return ("right", "left") if int(grasp_type_id) >= 4 else ("right",)


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Compute geodesic rotation distance in degrees.

    Args:
        rot_a: First ``3x3`` rotation matrix.
        rot_b: Second ``3x3`` rotation matrix.

    Returns:
        Angular distance in degrees.
    """
    rel = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
    cos_angle = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_distance(sample_record: dict, train_record: dict, trans_ref_m: float, rot_ref_deg: float) -> dict:
    """Measure pose distance between one sampled grasp and one training grasp.

    Args:
        sample_record: Comparable sampled pose record.
        train_record: Comparable training pose record.
        trans_ref_m: Translation scale used to normalize nearest-neighbor score.
        rot_ref_deg: Rotation scale used to normalize nearest-neighbor score.

    Returns:
        Distance dictionary with translation, rotation, normalized score, and active
        side list.
    """
    sides = active_sides(sample_record["grasp_type_id"])
    trans_distances = []
    rot_distances = []
    for side in sides:
        trans_distances.append(
            float(np.linalg.norm(sample_record[f"{side}_trans"] - train_record[f"{side}_trans"]))
        )
        rot_distances.append(rotation_distance_deg(sample_record[f"{side}_rot"], train_record[f"{side}_rot"]))

    trans_m = float(np.mean(trans_distances))
    rot_deg = float(np.mean(rot_distances))
    score = trans_m / max(trans_ref_m, 1e-8) + rot_deg / max(rot_ref_deg, 1e-8)
    return {"trans_m": trans_m, "rot_deg": rot_deg, "score": score, "sides": "+".join(sides)}


def nearest_train_record(sample_record: dict, candidates: list, trans_ref_m: float, rot_ref_deg: float) -> dict:
    """Find the closest training pose for one sampled pose.

    Args:
        sample_record: Comparable sampled pose record.
        candidates: Training pose records to search.
        trans_ref_m: Translation scale used to normalize nearest-neighbor score.
        rot_ref_deg: Rotation scale used to normalize nearest-neighbor score.

    Returns:
        Joined sample/training nearest-neighbor result row.
    """
    best_record = None
    best_distance = None
    for train_record in candidates:
        distance = pose_distance(sample_record, train_record, trans_ref_m, rot_ref_deg)
        if best_distance is None or distance["score"] < best_distance["score"]:
            best_distance = distance
            best_record = train_record

    return {
        "sample_path": sample_record["path"],
        "sample_object_name": sample_record["object_name"],
        "sample_canonical_object_id": sample_record["canonical_object_id"],
        "sample_grasp_type_id": sample_record["grasp_type_id"],
        "nearest_train_path": best_record["path"],
        "nearest_train_object_name": best_record["object_name"],
        "nearest_train_canonical_object_id": best_record["canonical_object_id"],
        "nearest_train_grasp_type_id": best_record["grasp_type_id"],
        "active_sides": best_distance["sides"],
        "trans_m": best_distance["trans_m"],
        "rot_deg": best_distance["rot_deg"],
        "score": best_distance["score"],
    }


def records_by_canonical_object(records: list) -> dict:
    """Group comparable records by canonical object id.

    Args:
        records: Training or sampled pose records.

    Returns:
        Mapping from canonical object id to records.
    """
    groups = {}
    for record in records:
        groups.setdefault(record["canonical_object_id"], []).append(record)
    return groups


def count_grasp_types_by_object(records: list) -> dict:
    """Count grasp types for each canonical object.

    Args:
        records: Comparable human records with ``canonical_object_id`` and
            ``grasp_type_id`` fields.

    Returns:
        Mapping ``canonical_object_id -> dict-like counter``.
    """
    grouped_counts = {}
    for record in records:
        object_id = record["canonical_object_id"]
        grasp_type_id = int(record["grasp_type_id"])
        object_counts = grouped_counts.setdefault(object_id, {})
        object_counts[grasp_type_id] = object_counts.get(grasp_type_id, 0) + 1
    return grouped_counts


def parse_object_id_list(value) -> list:
    """Normalize object-id config values into a list.

    Args:
        value: String, list-like value, or ``None`` from Hydra config.

    Returns:
        List of non-empty canonical object ids.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    return [canonical_object_id(item) for item in raw_values if str(item).strip()]


def select_object_analysis_ids(train_records: list, sample_records: list, task_cfg) -> list:
    """Select canonical object ids for object-specific NN analysis.

    Args:
        train_records: Comparable training records.
        sample_records: Comparable sampled records.
        task_cfg: Hydra task config with optional object-analysis settings.

    Returns:
        Canonical object ids to analyze. Explicit config values are used first;
        otherwise the objects with the most sampled records are selected.
    """
    object_ids = []
    object_ids.extend(parse_object_id_list(getattr(task_cfg, "object_analysis_object_id", "")))
    object_ids.extend(parse_object_id_list(getattr(task_cfg, "object_analysis_object_ids", [])))
    if object_ids:
        return list(dict.fromkeys(object_ids))

    train_groups = records_by_canonical_object(train_records)
    sample_groups = records_by_canonical_object(sample_records)
    common_ids = sorted(set(train_groups) & set(sample_groups))
    top_k = int(getattr(task_cfg, "object_analysis_top_k", 5))
    ranked_ids = sorted(
        common_ids,
        key=lambda object_id: (len(sample_groups[object_id]), len(train_groups[object_id]), object_id),
        reverse=True,
    )
    return ranked_ids[: max(0, top_k)]


def nearest_training_distance_for_object(
    sample_record: dict,
    train_records: list,
    trans_ref_m: float,
    rot_ref_deg: float,
    same_type_only: bool,
) -> dict:
    """Find a sample's nearest training grasp inside one canonical object.

    Args:
        sample_record: One sampled pose record.
        train_records: Training pose records from the same canonical object.
        trans_ref_m: Translation scale used to normalize the score.
        rot_ref_deg: Rotation scale used to normalize the score.
        same_type_only: Whether to search only training grasps with the same
            grasp type as the sampled grasp.

    Returns:
        Joined nearest-neighbor row for object-specific analysis.
    """
    candidates = train_records
    if same_type_only:
        same_type_candidates = [
            record for record in train_records if int(record["grasp_type_id"]) == int(sample_record["grasp_type_id"])
        ]
        if same_type_candidates:
            candidates = same_type_candidates

    row = nearest_train_record(sample_record, candidates, trans_ref_m, rot_ref_deg)
    row["same_type_only"] = bool(same_type_only)
    row["used_same_type_candidates"] = bool(
        same_type_only and int(row["sample_grasp_type_id"]) == int(row["nearest_train_grasp_type_id"])
    )
    return row


def percentile_summary(values: np.ndarray) -> dict:
    """Compute compact percentiles for a one-dimensional numeric array.

    Args:
        values: Numeric values.

    Returns:
        Dictionary with mean and selected percentiles.
    """
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def print_object_specific_summary(
    object_id: str,
    object_rows: list,
    train_count: int,
    sample_count: int,
    near_copy_trans_m: float,
    near_copy_rot_deg: float,
    novel_trans_m: float,
    novel_rot_deg: float,
) -> None:
    """Print object-specific NN overfit and novelty statistics.

    Args:
        object_id: Canonical object id being analyzed.
        object_rows: Per-sample nearest-neighbor rows for this object.
        train_count: Number of training grasps for this object.
        sample_count: Number of sampled grasps for this object.
        near_copy_trans_m: Translation threshold for near-copy classification.
        near_copy_rot_deg: Rotation threshold for near-copy classification.
        novel_trans_m: Translation threshold for novel classification.
        novel_rot_deg: Rotation threshold for novel classification.

    Returns:
        None.
    """
    print(f"\nObject-specific NN analysis: {object_id}")
    print(f"  Training grasps: {train_count}")
    print(f"  Sampled grasps: {sample_count}")
    if not object_rows:
        print("  No comparable sampled rows for this object.")
        return

    trans_values = np.asarray([row["trans_m"] for row in object_rows], dtype=np.float64)
    rot_values = np.asarray([row["rot_deg"] for row in object_rows], dtype=np.float64)
    near_copy_rows = [
        row for row in object_rows if row["trans_m"] <= near_copy_trans_m and row["rot_deg"] <= near_copy_rot_deg
    ]
    novel_rows = [row for row in object_rows if row["trans_m"] >= novel_trans_m or row["rot_deg"] >= novel_rot_deg]

    print(
        f"  Near-copy rows: {len(near_copy_rows)}/{len(object_rows)} "
        f"({len(near_copy_rows) / len(object_rows) * 100.0:.1f}%) "
        f"with trans <= {near_copy_trans_m:.4f} m and rot <= {near_copy_rot_deg:.2f} deg"
    )
    print(
        f"  Novel rows: {len(novel_rows)}/{len(object_rows)} "
        f"({len(novel_rows) / len(object_rows) * 100.0:.1f}%) "
        f"with trans >= {novel_trans_m:.4f} m or rot >= {novel_rot_deg:.2f} deg"
    )

    trans_stats = percentile_summary(trans_values)
    rot_stats = percentile_summary(rot_values)
    print(
        "  Translation NN distance (m): "
        f"mean={trans_stats['mean']:.4f}, p05={trans_stats['p05']:.4f}, "
        f"p25={trans_stats['p25']:.4f}, p50={trans_stats['p50']:.4f}, "
        f"p75={trans_stats['p75']:.4f}, p95={trans_stats['p95']:.4f}"
    )
    print(
        "  Rotation NN distance (deg): "
        f"mean={rot_stats['mean']:.2f}, p05={rot_stats['p05']:.2f}, "
        f"p25={rot_stats['p25']:.2f}, p50={rot_stats['p50']:.2f}, "
        f"p75={rot_stats['p75']:.2f}, p95={rot_stats['p95']:.2f}"
    )


def write_object_analysis_csv(rows: list, csv_path: str) -> None:
    """Write object-specific nearest-neighbor rows to a CSV file.

    Args:
        rows: Per-sample object-specific nearest-neighbor rows.
        csv_path: Destination CSV path.

    Returns:
        None.
    """
    if not rows:
        print("[evaluate] No object-specific rows to write.")
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[evaluate] Wrote object-specific NN rows to {csv_path}")


def run_object_specific_nn_analysis(
    train_records: list,
    sample_records: list,
    task_cfg,
    sample_dir: str,
) -> list:
    """Analyze whether selected objects generate grasps different from training.

    Args:
        train_records: Comparable training pose records.
        sample_records: Comparable sampled pose records.
        task_cfg: Hydra task config containing object-analysis settings.
        sample_dir: Directory containing saved sample results.

    Returns:
        Object-specific nearest-neighbor rows for all analyzed objects.
    """
    object_ids = select_object_analysis_ids(train_records, sample_records, task_cfg)
    print("\nObject-specific NN overfit analysis:")
    if not object_ids:
        print("  No common train/sample objects selected.")
        return []

    train_groups = records_by_canonical_object(train_records)
    sample_groups = records_by_canonical_object(sample_records)
    trans_ref_m = float(getattr(task_cfg, "trans_ref_m", 0.02))
    rot_ref_deg = float(getattr(task_cfg, "rot_ref_deg", 10.0))
    near_copy_trans_m = float(getattr(task_cfg, "near_copy_trans_m", 0.01))
    near_copy_rot_deg = float(getattr(task_cfg, "near_copy_rot_deg", 5.0))
    novel_trans_m = float(getattr(task_cfg, "object_novel_trans_m", 0.03))
    novel_rot_deg = float(getattr(task_cfg, "object_novel_rot_deg", 15.0))
    same_type_only = bool(getattr(task_cfg, "object_analysis_same_type_only", False))
    rows = []

    print(f"  Analyze objects: {', '.join(object_ids)}")
    print(f"  Same-type-only NN search: {same_type_only}")
    for object_id in object_ids:
        object_train_records = train_groups.get(object_id, [])
        object_sample_records = sample_groups.get(object_id, [])
        object_rows = []
        if object_train_records and object_sample_records:
            for sample_record in object_sample_records:
                row = nearest_training_distance_for_object(
                    sample_record,
                    object_train_records,
                    trans_ref_m=trans_ref_m,
                    rot_ref_deg=rot_ref_deg,
                    same_type_only=same_type_only,
                )
                row["analysis_canonical_object_id"] = object_id
                row["is_near_copy"] = row["trans_m"] <= near_copy_trans_m and row["rot_deg"] <= near_copy_rot_deg
                row["is_novel"] = row["trans_m"] >= novel_trans_m or row["rot_deg"] >= novel_rot_deg
                object_rows.append(row)
        print_object_specific_summary(
            object_id,
            object_rows,
            train_count=len(object_train_records),
            sample_count=len(object_sample_records),
            near_copy_trans_m=near_copy_trans_m,
            near_copy_rot_deg=near_copy_rot_deg,
            novel_trans_m=novel_trans_m,
            novel_rot_deg=novel_rot_deg,
        )
        rows.extend(object_rows)

    output_csv_value = getattr(task_cfg, "object_analysis_output_csv", "")
    output_csv = "" if output_csv_value is None else str(output_csv_value)
    if not output_csv:
        output_csv = os.path.join(sample_dir, "evaluation_object_specific_nn.csv")
    write_object_analysis_csv(rows, output_csv)
    return rows


class MarkdownReport:
    """Small helper that writes an evaluation report after each completed stage."""

    def __init__(self, path: str, title: str):
        """Create an incremental markdown report.

        Args:
            path: Destination markdown path.
            title: Report title.

        Returns:
            None.
        """
        self.path = path
        self.title = title
        self.sections = []

    def add_section(self, title: str, lines: list) -> None:
        """Append a section and flush the report to disk.

        Args:
            title: Markdown section title.
            lines: Section body lines.

        Returns:
            None.
        """
        self.sections.append((title, lines))
        self.write()

    def write(self) -> None:
        """Write the current report content.

        Args:
            None.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write(f"# {self.title}\n\n")
            for title, lines in self.sections:
                f.write(f"## {title}\n\n")
                for line in lines:
                    f.write(f"{line}\n")
                f.write("\n")


def write_generic_csv(rows: list, csv_path: str) -> None:
    """Write arbitrary row dictionaries to CSV.

    Args:
        rows: Row dictionaries with a shared key set.
        csv_path: Destination CSV path.

    Returns:
        None.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def step_id_from_ckpt_path(ckpt_path: str) -> str:
    """Extract a Hydra-friendly checkpoint id from a resolved checkpoint path.

    Args:
        ckpt_path: Resolved checkpoint path.

    Returns:
        Checkpoint id such as ``010000`` when possible, otherwise the raw path.
    """
    basename = os.path.basename(str(ckpt_path))
    match = re.match(r"step_(\d+)\.pth$", basename)
    if match:
        return match.group(1)
    return str(ckpt_path)


def sample_command(config, grasp_type_list: str) -> str:
    """Build the command the user should run before evaluation.

    Args:
        config: Full Hydra config.
        grasp_type_list: Comma-separated grasp type list for Hydra override.

    Returns:
        Shell command string.
    """
    ckpt_value = step_id_from_ckpt_path(resolve_ckpt_path(config))
    return (
        "python dexlearn/main.py "
        f"task=sample data={config.data_name} algo={config.algo_name} "
        f"test_data={config.test_data_name} exp_name={config.exp_name} "
        f"ckpt={ckpt_value} test_data.grasp_type_lst='[{grasp_type_list}]'"
    )


def raise_missing_samples(config, sample_dir: str, reason: str, grasp_type_list: str) -> None:
    """Raise a sample-missing error with a concrete sample command.

    Args:
        config: Full Hydra config.
        sample_dir: Expected sample output directory.
        reason: Human-readable missing sample description.
        grasp_type_list: Grasp types needed by the sample command.

    Returns:
        None.
    """
    command = sample_command(config, grasp_type_list)
    raise RuntimeError(
        f"{reason}\nExpected sample directory: {sample_dir}\n"
        f"Evaluation does not run sampling. Please generate samples first:\n{command}"
    )


def collect_sample_files(config, sample_dir: str) -> list:
    """Collect saved sample files and enforce that sampling has already run.

    Args:
        config: Full Hydra config.
        sample_dir: Expected sample output directory.

    Returns:
        Sorted sample file paths.
    """
    task_cfg = config.task
    sample_files = sorted(glob(os.path.join(sample_dir, "**/*.npy"), recursive=True))
    if not sample_files:
        raise_missing_samples(config, sample_dir, "No saved sample files were found.", "0_any")

    max_sample_files = int(getattr(task_cfg, "max_sample_files", 0))
    if max_sample_files > 0:
        sample_files = sample_files[:max_sample_files]
    return sample_files


def inspect_sample_file_types(sample_files: list) -> dict:
    """Split sample files by the metadata they contain.

    Args:
        sample_files: Saved ``.npy`` sample files.

    Returns:
        Dictionary with ``pose_files``, ``score_files``, and ``score_only_files``.
    """
    pose_files = []
    score_files = []
    score_only_files = []
    for sample_file in sample_files:
        sample_data = np.load(sample_file, allow_pickle=True).item()
        has_pose = "grasp_pose" in sample_data
        has_scores = "pred_grasp_type_prob" in sample_data
        if has_pose:
            pose_files.append(sample_file)
        if has_scores:
            score_files.append(sample_file)
        if has_scores and not has_pose:
            score_only_files.append(sample_file)
    return {
        "pose_files": pose_files,
        "score_files": score_files,
        "score_only_files": score_only_files,
    }


def is_feasibility_model(config) -> bool:
    """Check whether the configured model uses the new feasibility head.

    Args:
        config: Full Hydra config.

    Returns:
        ``True`` for feasibility-style human models.
    """
    objective = str(cfg_get(config.algo.model, "type_objective", default="")).lower()
    if objective in {"ce", "object_bce", "scene_ranking"}:
        return True
    model_name = str(cfg_get(config.algo.model, "name", default=""))
    return "Feasibility" in model_name or "TypeObjective" in model_name


def load_human_records_for_split(config, split_name: str) -> list:
    """Load human grasp records from a configured split.

    Args:
        config: Full Hydra config.
        split_name: Split name such as ``train`` or ``test``.

    Returns:
        Comparable human pose records.
    """
    records = []
    hand_pos_source = normalize_hand_pos_source(getattr(config.data, "hand_pos_source", "wrist"))
    for data_config in iter_data_configs(config.data):
        if cfg_get(data_config, "dataset_type", default="") != "HumanMultiDexDataset":
            continue
        object_path = cfg_get(data_config, "object_path", "paths.object_path")
        split_path = cfg_get(data_config, "split_path", "paths.split_path")
        grasp_path_root = cfg_get(data_config, "grasp_path", "paths.grasp_path")
        split_file = os.path.join(object_path, split_path, f"{split_name}.json")
        if not os.path.exists(split_file):
            continue
        for obj_id in load_json(split_file):
            grasp_paths = sorted(glob(os.path.join(grasp_path_root, obj_id, "**/*.npy"), recursive=True))
            for grasp_path in grasp_paths:
                records.append(training_pose_record(grasp_path, hand_pos_source))
    return records


def build_human_object_labels(config) -> tuple:
    """Build object-level feasibility labels for the sampled human split.

    Args:
        config: Full Hydra config.

    Returns:
        Tuple ``(label_by_object, split_records)``.
    """
    split_name = str(cfg_get(config.test_data, "test_split", "test.split", default="test"))
    split_records = load_human_records_for_split(config, split_name)
    negative_policy = str(cfg_get(config.algo, "supervision.negative_policy", default="")).lower()
    label_mode = "closed_world_object_complete" if negative_policy == "object_closed_world" else str(
        cfg_get(
            config.data,
            "feasibility_label_mode",
            "feasibility.label_mode",
            default="open_world_positive_only",
        )
    )

    object_counts = count_grasp_types_by_object(split_records)
    labels = {}
    for object_id, counter in object_counts.items():
        feasible = np.asarray(
            [1.0 if int(counter.get(type_id, 0)) > 0 else 0.0 for type_id in TARGET_GRASP_TYPE_IDS],
            dtype=np.float32,
        )
        if label_mode == "closed_world_object_complete":
            tested = np.ones_like(feasible, dtype=np.float32)
        else:
            tested = feasible.copy()
        labels[object_id] = {
            "feasible": feasible,
            "tested": tested,
            "counts": counter,
            "label_mode": label_mode,
        }
    return labels, split_records


def build_human_scene_labels(config) -> tuple:
    """Build scene-level labels with positive, strong-negative, and unknown types.

    Args:
        config: Full Hydra config.

    Returns:
        Tuple ``(label_by_scene_key, split_records)``. Labels use current-scene
        observed types as positives, types never observed for the same canonical
        object as strong negatives, and same-object other observed types as unknown.
    """
    split_name = str(cfg_get(config.test_data, "test_split", "test.split", default="test"))
    split_records = load_human_records_for_split(config, split_name)
    object_positive_types = {}
    scene_positive_types = {}
    scene_object_ids = {}

    for record in split_records:
        object_id = record["canonical_object_id"]
        key = record["scene_key"]
        type_id = int(record["grasp_type_id"])
        object_positive_types.setdefault(object_id, set()).add(type_id)
        scene_positive_types.setdefault(key, set()).add(type_id)
        scene_object_ids[key] = object_id

    labels = {}
    all_types = set(TARGET_GRASP_TYPE_IDS)
    for key, positive_types in scene_positive_types.items():
        object_id = scene_object_ids[key]
        object_types = object_positive_types.get(object_id, set())
        strong_negative_types = all_types - object_types
        unknown_types = object_types - positive_types
        y = np.zeros(len(TARGET_GRASP_TYPE_IDS), dtype=np.float32)
        mask = np.zeros(len(TARGET_GRASP_TYPE_IDS), dtype=np.float32)
        for idx, type_id in enumerate(TARGET_GRASP_TYPE_IDS):
            if type_id in positive_types:
                y[idx] = 1.0
                mask[idx] = 1.0
            elif type_id in strong_negative_types:
                y[idx] = 0.0
                mask[idx] = 1.0
        labels[key] = {
            "canonical_object_id": object_id,
            "scene_key": key,
            "positive_types": sorted(positive_types),
            "strong_negative_types": sorted(strong_negative_types),
            "unknown_types": sorted(unknown_types),
            "y": y,
            "mask": mask,
        }
    return labels, split_records


def score_vector_from_sample(sample_data: dict):
    """Read real-type score vector from one sample dictionary.

    Args:
        sample_data: Saved sample dictionary.

    Returns:
        Five-dimensional score vector for types ``1..5``, or ``None``.
    """
    if "pred_grasp_type_prob" not in sample_data:
        return None
    scores = np.asarray(sample_data["pred_grasp_type_prob"], dtype=np.float64).reshape(-1)
    if scores.size == len(TARGET_GRASP_TYPE_IDS) + 1:
        scores = scores[1:]
    elif scores.size != len(TARGET_GRASP_TYPE_IDS):
        return None
    return scores.astype(np.float64)


def load_score_records(sample_files: list) -> list:
    """Load score records from sample files.

    Args:
        sample_files: Saved sample files containing ``pred_grasp_type_prob``.

    Returns:
        List of object-level score records.
    """
    records = []
    for sample_file in sample_files:
        sample_data = np.load(sample_file, allow_pickle=True).item()
        scores = score_vector_from_sample(sample_data)
        if scores is None:
            continue
        metadata = resolve_sample_scene_metadata(sample_data)
        records.append(
            {
                "path": sample_file,
                "object_name": metadata["object_name"],
                "canonical_object_id": metadata["canonical_object_id"],
                "scene_id": metadata["scene_id"],
                "scene_key": metadata["scene_key"],
                "scores": scores,
            }
        )
    return records


def aggregate_scores_by_object(score_records: list) -> dict:
    """Average repeated score samples by canonical object id.

    Args:
        score_records: Per-file score records.

    Returns:
        Mapping from canonical object id to mean score vector and count.
    """
    grouped = {}
    for record in score_records:
        grouped.setdefault(record["canonical_object_id"], []).append(record["scores"])

    return {
        object_id: {
            "scores": np.mean(np.stack(scores, axis=0), axis=0),
            "sample_count": len(scores),
        }
        for object_id, scores in grouped.items()
    }


def aggregate_scores_by_scene(score_records: list) -> dict:
    """Average repeated score samples by scene key.

    Args:
        score_records: Per-file score records.

    Returns:
        Mapping from scene key to mean score vector and metadata.
    """
    grouped = {}
    for record in score_records:
        grouped.setdefault(record["scene_key"], []).append(record)

    scene_scores = {}
    for key, records in grouped.items():
        scores = [record["scores"] for record in records]
        scene_scores[key] = {
            "scores": np.mean(np.stack(scores, axis=0), axis=0),
            "sample_count": len(scores),
            "canonical_object_id": records[0]["canonical_object_id"],
            "scene_id": records[0]["scene_id"],
            "paths": [record["path"] for record in records],
        }
    return scene_scores


def parse_thresholds(task_cfg) -> list:
    """Read feasibility thresholds from config.

    Args:
        task_cfg: Evaluation task config.

    Returns:
        Sorted unique threshold values.
    """
    threshold_values = getattr(task_cfg, "score_thresholds", None)
    if threshold_values is None:
        threshold_values = [getattr(task_cfg, "score_threshold", 0.5)]
    elif isinstance(threshold_values, (str, bytes)):
        threshold_values = [value for value in str(threshold_values).split(",") if value.strip()]
    elif not isinstance(threshold_values, (list, tuple, ListConfig)):
        threshold_values = [threshold_values]
    elif len(threshold_values) == 0:
        threshold_values = [getattr(task_cfg, "score_threshold", 0.5)]
    thresholds = sorted({float(value) for value in threshold_values})
    return thresholds


def selected_types_for_threshold(scores: np.ndarray, threshold: float) -> list:
    """Convert one score vector into a thresholded feasible type list.

    Args:
        scores: Five-dimensional score vector for real types ``1..5``.
        threshold: Feasibility threshold.

    Returns:
        Selected grasp type ids.
    """
    return [type_id for type_id, score in zip(TARGET_GRASP_TYPE_IDS, scores) if float(score) >= threshold]


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary metrics for thresholded feasibility labels.

    Args:
        y_true: Binary labels.
        y_pred: Binary predictions.

    Returns:
        Dictionary with confusion counts and common rates.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    balanced_accuracy = (
        0.5 * (recall + specificity) if recall is not None and specificity is not None else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def format_optional_float(value, precision: int = 4) -> str:
    """Format optional numeric metrics for reports.

    Args:
        value: Numeric value or ``None``.
        precision: Decimal precision.

    Returns:
        Formatted string or ``NA``.
    """
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def average_precision(y_true: np.ndarray, y_score: np.ndarray):
    """Compute average precision without adding a sklearn dependency.

    Args:
        y_true: Binary labels.
        y_score: Prediction scores.

    Returns:
        Average precision, or ``None`` when positives or negatives are absent.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    positive_count = int(y_true.sum())
    if positive_count == 0 or positive_count == len(y_true):
        return None
    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    precision_at_rank = np.cumsum(sorted_true) / (np.arange(len(sorted_true)) + 1.0)
    return float((precision_at_rank * sorted_true).sum() / positive_count)


def percentile_lines(title: str, values: list, unit: str = "") -> list:
    """Format standard percentile statistics for markdown.

    Args:
        title: Metric title.
        values: Numeric values.
        unit: Optional displayed unit.

    Returns:
        Markdown lines.
    """
    if not values:
        return [f"- {title}: no values"]
    arr = np.asarray(values, dtype=np.float64)
    suffix = f" {unit}" if unit else ""
    return [
        (
            f"- {title}: mean={arr.mean():.4f}{suffix}, p50={np.percentile(arr, 50):.4f}{suffix}, "
            f"p90={np.percentile(arr, 90):.4f}{suffix}, p95={np.percentile(arr, 95):.4f}{suffix}, "
            f"p99={np.percentile(arr, 99):.4f}{suffix}, max={arr.max():.4f}{suffix}"
        )
    ]


def run_score_sanity(score_records: list, sample_dir: str) -> list:
    """Summarize feasibility or categorical score behavior.

    Args:
        score_records: Per-file score records.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score records were available."]
    rows = []
    max_scores = []
    entropy_values = []
    score_sums = []
    for record in score_records:
        scores = np.asarray(record["scores"], dtype=np.float64)
        score_sum = float(scores.sum())
        probs = scores / score_sum if score_sum > 1e-8 else np.ones_like(scores) / len(scores)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
        max_scores.append(float(scores.max()))
        entropy_values.append(entropy)
        score_sums.append(score_sum)
        row = {
            "sample_path": record["path"],
            "canonical_object_id": record["canonical_object_id"],
            "max_score": float(scores.max()),
            "score_sum": score_sum,
            "score_entropy": entropy,
        }
        for type_id, score in zip(TARGET_GRASP_TYPE_IDS, scores):
            row[f"type_{type_id}_score"] = float(score)
        rows.append(row)

    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_score_sanity.csv"))
    lines = [f"- Score records: {len(score_records)}"]
    lines.extend(percentile_lines("max score", max_scores))
    lines.extend(percentile_lines("score sum", score_sums))
    lines.extend(percentile_lines("score entropy", entropy_values))
    return lines


def run_feasibility_evaluation(score_records: list, object_labels: dict, task_cfg, sample_dir: str) -> list:
    """Evaluate object-level human type feasibility scores.

    Args:
        score_records: Score records loaded from saved samples.
        object_labels: Object-level feasibility labels.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No feasibility score samples were available."]

    threshold = float(getattr(task_cfg, "score_threshold", 0.5))
    top_k_values = [int(v) for v in getattr(task_cfg, "coverage_top_k", [1, 2, 3, 5])]
    object_scores = aggregate_scores_by_object(score_records)
    common_ids = sorted(set(object_scores) & set(object_labels))
    rows = []
    per_type_values = {type_id: {"y": [], "score": []} for type_id in TARGET_GRASP_TYPE_IDS}
    coverage_hits = {top_k: [] for top_k in top_k_values}

    for object_id in common_ids:
        scores = object_scores[object_id]["scores"]
        label = object_labels[object_id]
        feasible = label["feasible"]
        tested = label["tested"]
        positives = set(np.nonzero(feasible > 0.5)[0] + 1)
        ranked_types = [type_id for type_id in np.argsort(-scores) + 1]

        for type_idx, type_id in enumerate(TARGET_GRASP_TYPE_IDS):
            if tested[type_idx] <= 0.5:
                continue
            per_type_values[type_id]["y"].append(int(feasible[type_idx] > 0.5))
            per_type_values[type_id]["score"].append(float(scores[type_idx]))

        for top_k in top_k_values:
            selected = set(ranked_types[: min(top_k, len(ranked_types))])
            if positives:
                coverage_hits[top_k].append(len(positives & selected) / len(positives))

        row = {
            "canonical_object_id": object_id,
            "sample_count": int(object_scores[object_id]["sample_count"]),
            "label_mode": label["label_mode"],
            "positive_types": " ".join(str(type_id) for type_id in sorted(positives)),
            "top_types": " ".join(str(type_id) for type_id in ranked_types),
        }
        for type_id, score, is_feasible, is_tested in zip(TARGET_GRASP_TYPE_IDS, scores, feasible, tested):
            row[f"type_{type_id}_score"] = float(score)
            row[f"type_{type_id}_label"] = int(is_feasible > 0.5)
            row[f"type_{type_id}_tested"] = int(is_tested > 0.5)
            row[f"type_{type_id}_pred"] = int(score >= threshold)
        rows.append(row)

    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_feasibility_by_object.csv"))
    lines = [
        f"- Score sample files: {len(score_records)}",
        f"- Objects with scores: {len(object_scores)}",
        f"- Objects with human labels: {len(object_labels)}",
        f"- Common evaluated objects: {len(common_ids)}",
        f"- Score threshold: {threshold:.3f}",
    ]
    if not common_ids:
        lines.append("- No common object ids between score samples and human labels.")
        return lines

    for top_k in top_k_values:
        values = coverage_hits[top_k]
        if values:
            lines.append(f"- Positive type coverage@{top_k}: {np.mean(values):.4f}")

    metric_rows = []
    for type_id in TARGET_GRASP_TYPE_IDS:
        y_true = np.asarray(per_type_values[type_id]["y"], dtype=np.int32)
        y_score = np.asarray(per_type_values[type_id]["score"], dtype=np.float64)
        if y_true.size == 0:
            continue
        y_pred = (y_score >= threshold).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        ap = average_precision(y_true, y_score)
        metric_rows.append(
            {
                "type_id": type_id,
                "tested": int(y_true.size),
                "positives": int(y_true.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": ap,
                "mean_score": float(y_score.mean()),
            }
        )

    write_generic_csv(metric_rows, os.path.join(sample_dir, "evaluation_feasibility_metrics.csv"))
    for row in metric_rows:
        precision = "NA" if row["precision"] is None else f"{row['precision']:.4f}"
        recall = "NA" if row["recall"] is None else f"{row['recall']:.4f}"
        f1 = "NA" if row["f1"] is None else f"{row['f1']:.4f}"
        ap = "NA" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
        lines.append(
            f"- Type {row['type_id']}: tested={row['tested']}, positives={row['positives']}, "
            f"precision={precision}, recall={recall}, f1={f1}, AP={ap}, "
            f"mean_score={row['mean_score']:.4f}"
        )
    lines.append(
        "- Note: AP/precision need tested negatives. In open-world positive-only labels, "
        "negative-dependent metrics may be unavailable or weakly informative."
    )
    return lines


def collect_labeled_scene_pairs(scene_scores: dict, scene_labels: dict) -> tuple:
    """Collect positive and strong-negative scene/type pairs.

    Args:
        scene_scores: Aggregated scores keyed by scene key.
        scene_labels: Human scene labels keyed by scene key.

    Returns:
        Tuple ``(y_true, y_score, rows)`` for labeled positive/strong-negative
        pairs only. Unknown same-object types are intentionally excluded.
    """
    y_true = []
    y_score = []
    rows = []
    for key in sorted(set(scene_scores) & set(scene_labels)):
        scores = scene_scores[key]["scores"]
        label = scene_labels[key]
        for idx, type_id in enumerate(TARGET_GRASP_TYPE_IDS):
            if label["mask"][idx] <= 0.5:
                continue
            pair_label = int(label["y"][idx] > 0.5)
            score = float(scores[idx])
            y_true.append(pair_label)
            y_score.append(score)
            rows.append(
                {
                    "scene_key": key,
                    "canonical_object_id": label["canonical_object_id"],
                    "type_id": type_id,
                    "label": pair_label,
                    "score": score,
                    "label_kind": "positive" if pair_label else "strong_negative",
                }
            )
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_score, dtype=np.float64), rows


def run_human_scene_feasibility(score_records: list, scene_labels: dict, task_cfg, sample_dir: str) -> list:
    """Evaluate score ranking and PR-AUC against human scene labels.

    Args:
        score_records: Score records loaded from saved samples.
        scene_labels: Human scene-level labels.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score samples were available."]
    scene_scores = aggregate_scores_by_scene(score_records)
    common_keys = sorted(set(scene_scores) & set(scene_labels))
    top_k_values = [int(v) for v in getattr(task_cfg, "coverage_top_k", [1, 2, 3, 5])]
    scene_rows = []
    mrr_values = []
    observed_ranks = []
    hit_values = {top_k: [] for top_k in top_k_values}

    for key in common_keys:
        scores = scene_scores[key]["scores"]
        label = scene_labels[key]
        positive_types = list(label["positive_types"])
        ranked_types = [type_id for type_id in np.argsort(-scores) + 1]
        positive_ranks = [ranked_types.index(type_id) + 1 for type_id in positive_types if type_id in ranked_types]
        best_rank = min(positive_ranks) if positive_ranks else None
        if best_rank is not None:
            observed_ranks.append(best_rank)
            mrr_values.append(1.0 / best_rank)
        for top_k in top_k_values:
            selected = set(ranked_types[: min(top_k, len(ranked_types))])
            hit_values[top_k].append(float(bool(set(positive_types) & selected)))
        row = {
            "scene_key": key,
            "canonical_object_id": label["canonical_object_id"],
            "positive_types": " ".join(str(type_id) for type_id in positive_types),
            "strong_negative_types": " ".join(str(type_id) for type_id in label["strong_negative_types"]),
            "unknown_types": " ".join(str(type_id) for type_id in label["unknown_types"]),
            "top_types": " ".join(str(type_id) for type_id in ranked_types),
            "best_positive_rank": best_rank if best_rank is not None else "",
            "sample_count": scene_scores[key]["sample_count"],
        }
        for type_id, score in zip(TARGET_GRASP_TYPE_IDS, scores):
            row[f"type_{type_id}_score"] = float(score)
        scene_rows.append(row)

    write_generic_csv(scene_rows, os.path.join(sample_dir, "evaluation_human_scene_feasibility.csv"))
    y_true, y_score, pair_rows = collect_labeled_scene_pairs(scene_scores, scene_labels)
    write_generic_csv(pair_rows, os.path.join(sample_dir, "evaluation_human_scene_labeled_pairs.csv"))
    pr_auc = average_precision(y_true, y_score) if y_true.size else None
    positive_count = int(y_true.sum()) if y_true.size else 0
    negative_count = int(y_true.size - positive_count) if y_true.size else 0

    lines = [
        f"- Scenes with scores: {len(scene_scores)}",
        f"- Scenes with human labels: {len(scene_labels)}",
        f"- Common evaluated scenes: {len(common_keys)}",
        f"- Labeled scene/type pairs: {int(y_true.size)}",
        f"- Positive pairs: {positive_count}",
        f"- Strong-negative pairs: {negative_count}",
        f"- PR-AUC / Average Precision: {format_optional_float(pr_auc)}",
    ]
    if observed_ranks:
        lines.append(f"- Mean observed positive rank: {np.mean(observed_ranks):.4f}")
        lines.append(f"- MRR: {np.mean(mrr_values):.4f}")
    for top_k in top_k_values:
        values = hit_values[top_k]
        if values:
            lines.append(f"- Observed type hit@{top_k}: {np.mean(values):.4f}")
    lines.append("- Unknown same-object types are excluded from PR-AUC and threshold precision.")
    return lines


def run_feasibility_threshold_sweep(
    score_records: list,
    scene_labels: dict,
    task_cfg,
    sample_dir: str,
) -> list:
    """Sweep thresholds that convert scores into BODex feasibility sets.

    Args:
        score_records: Score records loaded from saved samples.
        scene_labels: Optional human scene labels.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score samples were available."]
    thresholds = parse_thresholds(task_cfg)
    rows = []
    best_f1_row = None
    best_pr_row = None
    for threshold in thresholds:
        budgets = []
        per_type_selected = {type_id: 0 for type_id in TARGET_GRASP_TYPE_IDS}
        y_true = []
        y_pred = []
        y_score = []
        observed_recall_values = []
        for record in score_records:
            scores = np.asarray(record["scores"], dtype=np.float64)
            selected = selected_types_for_threshold(scores, threshold)
            budgets.append(len(selected))
            for type_id in selected:
                per_type_selected[type_id] += 1

            label = scene_labels.get(record["scene_key"]) if scene_labels else None
            if label is None:
                continue
            positive_types = set(label["positive_types"])
            if positive_types:
                observed_recall_values.append(len(positive_types & set(selected)) / len(positive_types))
            for idx, type_id in enumerate(TARGET_GRASP_TYPE_IDS):
                if label["mask"][idx] <= 0.5:
                    continue
                y_true.append(int(label["y"][idx] > 0.5))
                y_pred.append(int(type_id in selected))
                y_score.append(float(scores[idx]))

        row = {
            "threshold": float(threshold),
            "score_records": len(score_records),
            "mean_budget_size": float(np.mean(budgets)),
            "p50_budget_size": float(np.percentile(budgets, 50)),
            "zero_budget_rate": float(np.mean(np.asarray(budgets) == 0)),
            "full_budget_rate": float(np.mean(np.asarray(budgets) == len(TARGET_GRASP_TYPE_IDS))),
        }
        for type_id in TARGET_GRASP_TYPE_IDS:
            row[f"type_{type_id}_selection_rate"] = per_type_selected[type_id] / len(score_records)

        if y_true:
            metrics = binary_metrics(np.asarray(y_true), np.asarray(y_pred))
            pr_auc = average_precision(np.asarray(y_true), np.asarray(y_score))
            row.update(metrics)
            row["pr_auc"] = pr_auc
            row["observed_type_recall"] = float(np.mean(observed_recall_values)) if observed_recall_values else None
            if row.get("f1") is not None and (best_f1_row is None or row["f1"] > best_f1_row["f1"]):
                best_f1_row = row
            if pr_auc is not None:
                best_pr_row = row
        rows.append(row)

    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_feasibility_threshold_sweep.csv"))
    default_threshold = float(getattr(task_cfg, "score_threshold", 0.5))
    default_row = min(rows, key=lambda row: abs(row["threshold"] - default_threshold))
    lines = [
        f"- Thresholds: {', '.join(f'{threshold:.2f}' for threshold in thresholds)}",
        f"- Score records: {len(score_records)}",
        (
            f"- Default threshold {default_row['threshold']:.2f}: "
            f"mean_budget={default_row['mean_budget_size']:.3f}, "
            f"zero_budget={default_row['zero_budget_rate']:.3f}, "
            f"full_budget={default_row['full_budget_rate']:.3f}"
        ),
    ]
    if "precision" in default_row:
        lines.append(
            f"- Default threshold human metrics: precision={format_optional_float(default_row['precision'])}, "
            f"recall={format_optional_float(default_row['recall'])}, "
            f"F1={format_optional_float(default_row['f1'])}, "
            f"FPR={format_optional_float(default_row['fpr'])}, "
            f"PR-AUC={format_optional_float(default_row['pr_auc'])}"
        )
    if best_f1_row is not None:
        lines.append(
            f"- Best F1 threshold: {best_f1_row['threshold']:.2f} "
            f"(F1={format_optional_float(best_f1_row['f1'])}, "
            f"precision={format_optional_float(best_f1_row['precision'])}, "
            f"recall={format_optional_float(best_f1_row['recall'])})"
        )
    if best_pr_row is not None:
        lines.append(f"- PR-AUC is threshold-free over labeled pairs: {format_optional_float(best_pr_row['pr_auc'])}")
    lines.append("- CSV includes budget size, per-type selection rate, and thresholded human metrics.")
    return lines


def run_budget_to_bodex_proxy(
    score_records: list,
    scene_labels: dict,
    task_cfg,
    sample_dir: str,
) -> list:
    """Summarize top-K and threshold budgets as downstream BODex proxies.

    Args:
        score_records: Score records loaded from saved samples.
        scene_labels: Optional human scene labels.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score samples were available."]
    top_k_values = [int(v) for v in getattr(task_cfg, "coverage_top_k", [1, 2, 3, 5])]
    thresholds = parse_thresholds(task_cfg)
    rows = []
    for top_k in top_k_values:
        coverages = []
        for record in score_records:
            label = scene_labels.get(record["scene_key"]) if scene_labels else None
            if label is None or not label["positive_types"]:
                continue
            ranked_types = [type_id for type_id in np.argsort(-record["scores"]) + 1]
            selected = set(ranked_types[: min(top_k, len(ranked_types))])
            positives = set(label["positive_types"])
            coverages.append(len(positives & selected) / len(positives))
        rows.append(
            {
                "mode": "top_k",
                "budget": top_k,
                "mean_budget_size": top_k,
                "human_positive_coverage": float(np.mean(coverages)) if coverages else None,
            }
        )
    for threshold in thresholds:
        budgets = []
        coverages = []
        for record in score_records:
            selected = set(selected_types_for_threshold(record["scores"], threshold))
            budgets.append(len(selected))
            label = scene_labels.get(record["scene_key"]) if scene_labels else None
            if label is None or not label["positive_types"]:
                continue
            positives = set(label["positive_types"])
            coverages.append(len(positives & selected) / len(positives))
        rows.append(
            {
                "mode": "threshold",
                "budget": float(threshold),
                "mean_budget_size": float(np.mean(budgets)),
                "human_positive_coverage": float(np.mean(coverages)) if coverages else None,
            }
        )
    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_budget_to_bodex_proxy.csv"))

    lines = ["- This proxy reports candidate type budget before running BODex."]
    for row in rows:
        if row["mode"] == "top_k":
            lines.append(
                f"- Top-{int(row['budget'])}: mean_budget={row['mean_budget_size']:.3f}, "
                f"human_positive_coverage={format_optional_float(row['human_positive_coverage'])}"
            )
        elif float(row["budget"]) == float(getattr(task_cfg, "score_threshold", 0.5)):
            lines.append(
                f"- Threshold {row['budget']:.2f}: mean_budget={row['mean_budget_size']:.3f}, "
                f"human_positive_coverage={format_optional_float(row['human_positive_coverage'])}"
            )
    return lines


def summarize_nearest_rows_markdown(rows: list, task_cfg) -> list:
    """Summarize nearest-neighbor rows for markdown.

    Args:
        rows: Nearest-neighbor rows.
        task_cfg: Evaluation task config.

    Returns:
        Markdown summary lines.
    """
    if not rows:
        return ["- No comparable pose samples were available."]
    near_copy_trans_m = float(getattr(task_cfg, "near_copy_trans_m", 0.01))
    near_copy_rot_deg = float(getattr(task_cfg, "near_copy_rot_deg", 5.0))
    near_copy_rows = [
        row for row in rows if row["trans_m"] <= near_copy_trans_m and row["rot_deg"] <= near_copy_rot_deg
    ]
    type_matches = [
        row for row in rows if int(row["sample_grasp_type_id"]) == int(row["nearest_train_grasp_type_id"])
    ]
    lines = [
        f"- Comparable pose samples: {len(rows)}",
        (
            f"- Near-copy rows: {len(near_copy_rows)} "
            f"({len(near_copy_rows) / len(rows) * 100.0:.2f}%)"
        ),
        f"- Nearest type match: {len(type_matches) / len(rows) * 100.0:.2f}%",
    ]
    lines.extend(percentile_lines("NN translation distance", [row["trans_m"] for row in rows], "m"))
    lines.extend(percentile_lines("NN rotation distance", [row["rot_deg"] for row in rows], "deg"))
    return lines


def run_pose_diversity_analysis(sample_records: list, task_cfg, sample_dir: str) -> list:
    """Measure within-scene/type generated pose diversity.

    Args:
        sample_records: Comparable sampled pose records.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    max_poses = int(getattr(task_cfg, "diversity_max_poses_per_group", 50))
    groups = {}
    for record in sample_records:
        key = (record["scene_key"], int(record["grasp_type_id"]))
        groups.setdefault(key, []).append(record)

    rows = []
    all_trans = []
    all_rot = []
    for (scene_key_value, type_id), records in sorted(groups.items()):
        if len(records) < 2:
            continue
        records = records[:max_poses]
        trans_values = []
        rot_values = []
        for idx_a in range(len(records)):
            for idx_b in range(idx_a + 1, len(records)):
                distance = pose_distance(records[idx_a], records[idx_b], trans_ref_m=1.0, rot_ref_deg=1.0)
                trans_values.append(distance["trans_m"])
                rot_values.append(distance["rot_deg"])
        if not trans_values:
            continue
        all_trans.extend(trans_values)
        all_rot.extend(rot_values)
        rows.append(
            {
                "scene_key": scene_key_value,
                "canonical_object_id": records[0]["canonical_object_id"],
                "grasp_type_id": type_id,
                "sample_count": len(records),
                "pair_count": len(trans_values),
                "mean_pair_trans_m": float(np.mean(trans_values)),
                "p50_pair_trans_m": float(np.percentile(trans_values, 50)),
                "mean_pair_rot_deg": float(np.mean(rot_values)),
                "p50_pair_rot_deg": float(np.percentile(rot_values, 50)),
            }
        )

    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_pose_diversity.csv"))
    lines = [f"- Scene/type groups with at least two samples: {len(rows)}"]
    lines.extend(percentile_lines("Pairwise translation diversity", all_trans, "m"))
    lines.extend(percentile_lines("Pairwise rotation diversity", all_rot, "deg"))
    return lines


def normalize_object_scale(obj_scale) -> np.ndarray:
    """Normalize scalar or vector object scale to XYZ scale.

    Args:
        obj_scale: Raw object scale from scene config.

    Returns:
        Three-dimensional scale vector.
    """
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Unsupported object scale shape: {scale.shape}")


def object_pose_to_rt(obj_pose) -> tuple:
    """Convert object pose metadata to rotation and translation.

    Args:
        obj_pose: ``4x4`` matrix or ``[x, y, z, qw, qx, qy, qz]`` vector.

    Returns:
        Tuple ``(rotation, translation)``.
    """
    pose = np.asarray(obj_pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]
    pose = pose.reshape(-1)
    if pose.size == 7:
        rot = SciR.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix().astype(np.float32)
        return rot, pose[:3]
    raise ValueError(f"Unsupported object pose shape: {pose.shape}")


def extract_object_meta(scene_cfg: dict) -> tuple:
    """Extract object name, scale, and pose from supported scene config layouts.

    Args:
        scene_cfg: Loaded scene config dictionary.

    Returns:
        Tuple ``(object_name, scale_xyz, rotation, translation)``.
    """
    if "object" in scene_cfg:
        obj_data = scene_cfg["object"]
        obj_name = obj_data.get("name")
        obj_scale = obj_data.get("rel_scale", obj_data.get("scale"))
        obj_pose = obj_data.get("pose")
    else:
        scene = scene_cfg.get("scene", {})
        obj_name = scene_cfg.get("task", {}).get("obj_name")
        if obj_name is None:
            candidates = [
                name
                for name, entry in scene.items()
                if isinstance(entry, dict) and "scale" in entry and "pose" in entry and name != "table"
            ]
            obj_name = candidates[0] if candidates else None
        obj_data = scene.get(obj_name, {}) if obj_name is not None else {}
        obj_scale = obj_data.get("scale")
        obj_pose = obj_data.get("pose")

    if obj_name is None or obj_scale is None or obj_pose is None:
        raise KeyError("Scene config does not contain complete object metadata.")
    obj_rot, obj_trans = object_pose_to_rt(obj_pose)
    return obj_name, normalize_object_scale(obj_scale), obj_rot, obj_trans


def object_bbox_for_sample(sample_data: dict, bbox_cache: dict):
    """Compute or retrieve a world-frame object bounding box for one sample.

    Args:
        sample_data: Saved sample dictionary.
        bbox_cache: Cache keyed by point-cloud and scene paths.

    Returns:
        Tuple ``(bbox_min, bbox_max, used_scene_transform)`` or ``None``.
    """
    pc_path = resolve_existing_path(str(sample_data.get("pc_path", "")))
    if not pc_path or not os.path.exists(pc_path):
        return None
    scene_path = resolve_existing_path(str(sample_data.get("scene_path", "")))
    cache_key = (pc_path, scene_path)
    if cache_key in bbox_cache:
        return bbox_cache[cache_key]

    points = np.asarray(np.load(pc_path, allow_pickle=True), dtype=np.float32).reshape(-1, 3)
    used_scene_transform = False
    if scene_path and os.path.exists(scene_path):
        try:
            scene_cfg = np.load(scene_path, allow_pickle=True).item()
            _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg)
            points = np.matmul(points * obj_scale_xyz[None, :], obj_rot.T) + obj_trans[None, :]
            used_scene_transform = True
        except Exception as exc:
            print(f"[evaluate] Could not transform point cloud with scene config {scene_path}: {exc}")

    bbox = (points.min(axis=0), points.max(axis=0), used_scene_transform)
    bbox_cache[cache_key] = bbox
    return bbox


def point_to_bbox_distance(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Compute Euclidean distance from a point to an axis-aligned bounding box.

    Args:
        point: Three-dimensional point.
        bbox_min: Box minimum corner.
        bbox_max: Box maximum corner.

    Returns:
        Distance in meters, zero when the point is inside the box.
    """
    point = np.asarray(point, dtype=np.float32)
    outside = np.maximum(bbox_min - point, 0.0) + np.maximum(point - bbox_max, 0.0)
    return float(np.linalg.norm(outside))


def object_bbox_dims_for_sample_file(sample_file: str, bbox_cache: dict):
    """Read object bbox dimensions for a saved sample file.

    Args:
        sample_file: Saved sample file path.
        bbox_cache: Shared bbox cache.

    Returns:
        Three-dimensional bbox size vector, or ``None`` when unavailable.
    """
    sample_data = np.load(sample_file, allow_pickle=True).item()
    bbox = object_bbox_for_sample(sample_data, bbox_cache)
    if bbox is None:
        return None
    bbox_min, bbox_max, _ = bbox
    return np.asarray(bbox_max - bbox_min, dtype=np.float64)


def size_bucket_from_dims(dims: np.ndarray, task_cfg) -> str:
    """Assign an object-size sanity bucket.

    Args:
        dims: Bbox dimensions in meters.
        task_cfg: Evaluation task config.

    Returns:
        ``tiny``, ``small``, ``large``, or ``other``.
    """
    large_dim_min = float(getattr(task_cfg, "large_dim_min_m", 0.15))
    small_dim_max = float(getattr(task_cfg, "small_dim_max_m", 0.05))
    tiny_dim_max = float(getattr(task_cfg, "tiny_dim_max_m", 0.03))
    if bool(np.all(dims < tiny_dim_max)):
        return "tiny"
    if bool(np.all(dims < small_dim_max)):
        return "small"
    if bool(np.all(dims > large_dim_min)):
        return "large"
    return "other"


def size_rule_violating_types(selected_types: list, dims: np.ndarray, task_cfg) -> list:
    """Return predicted types that violate coarse object-size rules.

    Args:
        selected_types: Thresholded feasible type ids.
        dims: Object bbox dimensions in meters.
        task_cfg: Evaluation task config.

    Returns:
        Sorted violating type ids.
    """
    selected = set(int(type_id) for type_id in selected_types)
    bucket = size_bucket_from_dims(dims, task_cfg)
    if bucket == "large":
        return sorted(selected & {1, 2, 3})
    if bucket == "small":
        return sorted(selected & {4, 5})
    if bucket == "tiny":
        return sorted(selected & {3, 4, 5})
    return []


def interval_holes_for_selected_types(selected_types: list) -> list:
    """Find ordinal interval holes in a thresholded type set.

    Args:
        selected_types: Thresholded feasible type ids.

    Returns:
        Missing middle type ids such as ``2`` for selected ``{1, 3}``.
    """
    selected = set(int(type_id) for type_id in selected_types)
    holes = []
    for left_type in (1, 2, 3):
        middle_type = left_type + 1
        right_type = left_type + 2
        if left_type in selected and right_type in selected and middle_type not in selected:
            holes.append(middle_type)
    return holes


def continuous_ordinal_violations(scores: np.ndarray, margin: float) -> list:
    """Find continuous score convexity violations along ordered grasp types.

    Args:
        scores: Five-dimensional score vector.
        margin: Minimum violation margin.

    Returns:
        Rows describing each violating triplet.
    """
    violations = []
    for left_idx in range(3):
        middle_idx = left_idx + 1
        right_idx = left_idx + 2
        gap = min(float(scores[left_idx]), float(scores[right_idx])) - float(scores[middle_idx])
        if gap > margin:
            violations.append(
                {
                    "left_type": left_idx + 1,
                    "middle_type": middle_idx + 1,
                    "right_type": right_idx + 1,
                    "gap": gap,
                }
            )
    return violations


def run_size_rule_violations(score_records: list, task_cfg, sample_dir: str) -> list:
    """Evaluate thresholded feasible sets against coarse size rules.

    Args:
        score_records: Score records loaded from saved samples.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score samples were available."]
    thresholds = parse_thresholds(task_cfg)
    bbox_cache = {}
    rows = []
    skipped = 0
    for record in score_records:
        dims = object_bbox_dims_for_sample_file(record["path"], bbox_cache)
        if dims is None:
            skipped += 1
            continue
        bucket = size_bucket_from_dims(dims, task_cfg)
        for threshold in thresholds:
            selected = selected_types_for_threshold(record["scores"], threshold)
            violating_types = size_rule_violating_types(selected, dims, task_cfg)
            rows.append(
                {
                    "threshold": float(threshold),
                    "sample_path": record["path"],
                    "scene_key": record["scene_key"],
                    "canonical_object_id": record["canonical_object_id"],
                    "size_bucket": bucket,
                    "bbox_x_m": float(dims[0]),
                    "bbox_y_m": float(dims[1]),
                    "bbox_z_m": float(dims[2]),
                    "selected_types": " ".join(str(type_id) for type_id in selected),
                    "violating_types": " ".join(str(type_id) for type_id in violating_types),
                    "violation_count": len(violating_types),
                    "has_violation": bool(violating_types),
                }
            )
    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_size_rule_violations.csv"))
    if not rows:
        return [f"- No bbox dimensions were available for score records. Skipped: {skipped}"]
    default_threshold = float(getattr(task_cfg, "score_threshold", 0.5))
    lines = [
        f"- Rows: {len(rows)}",
        f"- Skipped score files without bbox: {skipped}",
        (
            f"- Rules: large all dims > {float(getattr(task_cfg, 'large_dim_min_m', 0.15)):.3f}m "
            "forbid types 1/2/3; "
            f"small all dims < {float(getattr(task_cfg, 'small_dim_max_m', 0.05)):.3f}m "
            "forbid types 4/5; "
            f"tiny all dims < {float(getattr(task_cfg, 'tiny_dim_max_m', 0.03)):.3f}m "
            "forbid types 3/4/5."
        ),
    ]
    for threshold in thresholds:
        threshold_rows = [row for row in rows if abs(row["threshold"] - threshold) < 1e-8]
        if not threshold_rows:
            continue
        violation_count = sum(1 for row in threshold_rows if row["has_violation"])
        if threshold == default_threshold or threshold in (thresholds[0], thresholds[-1]):
            lines.append(
                f"- Threshold {threshold:.2f}: violation scenes={violation_count}/{len(threshold_rows)} "
                f"({violation_count / len(threshold_rows):.4f})"
            )
    return lines


def run_ordinal_consistency(score_records: list, task_cfg, sample_dir: str) -> list:
    """Evaluate type-order consistency in thresholded sets and raw scores.

    Args:
        score_records: Score records loaded from saved samples.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    if not score_records:
        return ["- No score samples were available."]
    thresholds = parse_thresholds(task_cfg)
    margin = float(getattr(task_cfg, "ordinal_margin", 0.0))
    rows = []
    continuous_rows = []
    for record in score_records:
        continuous = continuous_ordinal_violations(record["scores"], margin)
        for violation in continuous:
            continuous_rows.append(
                {
                    "sample_path": record["path"],
                    "scene_key": record["scene_key"],
                    "canonical_object_id": record["canonical_object_id"],
                    **violation,
                }
            )
        for threshold in thresholds:
            selected = selected_types_for_threshold(record["scores"], threshold)
            holes = interval_holes_for_selected_types(selected)
            rows.append(
                {
                    "threshold": float(threshold),
                    "sample_path": record["path"],
                    "scene_key": record["scene_key"],
                    "canonical_object_id": record["canonical_object_id"],
                    "selected_types": " ".join(str(type_id) for type_id in selected),
                    "interval_holes": " ".join(str(type_id) for type_id in holes),
                    "hole_count": len(holes),
                    "has_interval_hole": bool(holes),
                }
            )
    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_ordinal_threshold_holes.csv"))
    write_generic_csv(continuous_rows, os.path.join(sample_dir, "evaluation_ordinal_score_violations.csv"))
    default_threshold = float(getattr(task_cfg, "score_threshold", 0.5))
    default_rows = [row for row in rows if abs(row["threshold"] - default_threshold) < 1e-8]
    hole_count = sum(1 for row in default_rows if row["has_interval_hole"])
    lines = [
        f"- Score records: {len(score_records)}",
        f"- Continuous score violations: {len(continuous_rows)} with margin > {margin:.4f}",
    ]
    if default_rows:
        lines.append(
            f"- Threshold {default_threshold:.2f} interval-hole scenes: {hole_count}/{len(default_rows)} "
            f"({hole_count / len(default_rows):.4f})"
        )
    return lines


def run_bbox_geometry_sanity(pose_files: list, task_cfg, sample_dir: str) -> list:
    """Evaluate sampled wrist targets against object point-cloud bounding boxes.

    Args:
        pose_files: Saved pose sample files.
        task_cfg: Evaluation task config.
        sample_dir: Sample output directory.

    Returns:
        Markdown summary lines.
    """
    max_files = int(getattr(task_cfg, "geometry_max_sample_files", 0))
    if max_files > 0:
        pose_files = pose_files[:max_files]
    bbox_cache = {}
    rows = []
    skipped = 0
    for sample_file in pose_files:
        sample_data = np.load(sample_file, allow_pickle=True).item()
        if "grasp_pose" not in sample_data:
            continue
        bbox = object_bbox_for_sample(sample_data, bbox_cache)
        if bbox is None:
            skipped += 1
            continue
        bbox_min, bbox_max, used_scene_transform = bbox
        pose = np.asarray(sample_data["grasp_pose"], dtype=np.float32).reshape(-1)
        if pose.size < 7:
            skipped += 1
            continue
        grasp_type_source = sample_data.get("pred_grasp_type_id", sample_data.get("grasp_type_id", 0))
        grasp_type_id = int(np.asarray(grasp_type_source).reshape(-1)[0])
        metadata = resolve_sample_scene_metadata(sample_data)
        side_distances = {"right": point_to_bbox_distance(pose[0:3], bbox_min, bbox_max)}
        if grasp_type_id >= 4 and pose.size >= 14:
            side_distances["left"] = point_to_bbox_distance(pose[7:10], bbox_min, bbox_max)
        rows.append(
            {
                "sample_path": sample_file,
                "scene_key": metadata["scene_key"],
                "canonical_object_id": metadata["canonical_object_id"],
                "grasp_type_id": grasp_type_id,
                "active_side_count": len(side_distances),
                "max_bbox_distance_m": max(side_distances.values()),
                "mean_bbox_distance_m": float(np.mean(list(side_distances.values()))),
                "right_bbox_distance_m": side_distances.get("right", 0.0),
                "left_bbox_distance_m": side_distances.get("left", ""),
                "used_scene_transform": bool(used_scene_transform),
            }
        )

    write_generic_csv(rows, os.path.join(sample_dir, "evaluation_bbox_geometry.csv"))
    if not rows:
        return [f"- No bbox geometry rows were available. Skipped files: {skipped}"]
    threshold = float(getattr(task_cfg, "bbox_outlier_threshold_m", 0.2))
    max_distances = [row["max_bbox_distance_m"] for row in rows]
    outliers = [distance for distance in max_distances if distance > threshold]
    lines = [
        f"- Geometry rows: {len(rows)}",
        f"- Skipped files: {skipped}",
        f"- Bbox outlier threshold: {threshold:.4f} m",
        f"- Outlier rows: {len(outliers)} ({len(outliers) / len(rows) * 100.0:.2f}%)",
    ]
    lines.extend(percentile_lines("max wrist-to-bbox distance", max_distances, "m"))
    return lines


def task_evaluate(config) -> None:
    """Hydra task entry point for human model evaluation.

    Args:
        config: Full Hydra config for the current AnyScaleDexLearn run.

    Returns:
        None.
    """
    resolve_type_supervision_config(config)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    task_cfg = config.task
    sample_dir = get_sample_output_dir(config)
    sample_files = collect_sample_files(config, sample_dir)
    inspected = inspect_sample_file_types(sample_files)
    pose_files = inspected["pose_files"]
    score_files = inspected["score_files"]
    score_eval_files = inspected["score_only_files"] or score_files

    require_score_samples = str(getattr(task_cfg, "require_score_samples", "auto")).lower()
    if require_score_samples == "true" or (require_score_samples == "auto" and is_feasibility_model(config)):
        if not score_eval_files:
            raise_missing_samples(
                config,
                sample_dir,
                "No feasibility score sample files were found.",
                "0_any",
            )

    if bool(getattr(task_cfg, "require_pose_samples", True)) and not pose_files:
        raise_missing_samples(
            config,
            sample_dir,
            "No fixed-type pose sample files were found.",
            FIXED_TYPE_SAMPLE_LIST,
        )

    report_path_value = getattr(task_cfg, "report_md", "")
    report_path = str(report_path_value) if report_path_value else os.path.join(sample_dir, "evaluation_report.md")
    report = MarkdownReport(report_path, "Human Prior Score-to-Feasibility Evaluation")
    report.add_section(
        "Sample Inventory",
        [
            f"- Sample directory: `{sample_dir}`",
            f"- Total sample files: {len(sample_files)}",
            f"- Pose sample files: {len(pose_files)}",
            f"- Score sample files: {len(score_files)}",
            f"- Score-only sample files: {len(inspected['score_only_files'])}",
            f"- Report path: `{report_path}`",
        ],
    )

    # Check whether the saved type scores are numerically sane before using
    # them as feasibility or budget signals.
    score_records = load_score_records(score_eval_files)
    report.add_section("Score Sanity", run_score_sanity(score_records, sample_dir))

    human_label_records = []
    scene_labels = {}
    if cfg_get(config.test_data, "dataset_type", default="") == "HumanMultiDexDataset":
        # Human labels are evaluated at scene/sequence level. Same-object types
        # observed elsewhere are treated as unknown rather than negatives.
        scene_labels, human_label_records = build_human_scene_labels(config)
        report.add_section(
            "Human Scene Feasibility",
            run_human_scene_feasibility(score_records, scene_labels, task_cfg, sample_dir),
        )

    report.add_section(
        "Feasibility Threshold Sweep",
        run_feasibility_threshold_sweep(score_records, scene_labels, task_cfg, sample_dir),
    )
    report.add_section("Budget-to-BODex Proxy", run_budget_to_bodex_proxy(score_records, scene_labels, task_cfg, sample_dir))
    report.add_section("Size Rule Violations", run_size_rule_violations(score_records, task_cfg, sample_dir))
    report.add_section("Ordinal Type Consistency", run_ordinal_consistency(score_records, task_cfg, sample_dir))

    train_records = []
    sample_records = []
    if pose_files:
        train_records = load_training_records(config)
        sample_records = load_sample_records(pose_files)
        reference_records = human_label_records if human_label_records else train_records
        reference_name = "human eval split" if human_label_records else "training split"
        print(f"[evaluate] Loaded {len(train_records)} training records.")
        print(f"[evaluate] Loaded {len(human_label_records)} human split label records.")
        print(f"[evaluate] Found {len(sample_files)} sample files in {sample_dir}")
        print(f"[evaluate] Loaded {len(sample_records)} comparable pose sample records.")

        # Inspect selected objects in detail to see whether generated grasps are
        # near copies of reference poses or meaningfully novel alternatives.
        object_rows = run_object_specific_nn_analysis(reference_records, sample_records, task_cfg, sample_dir)
        report.add_section(
            "Object Novelty",
            [f"- Reference records: {reference_name} ({len(reference_records)})"]
            + summarize_nearest_rows_markdown(object_rows, task_cfg),
        )

        # Estimate within-object/type pose diversity so a low NN distance is not
        # mistaken for good multi-modal generation by itself.
        report.add_section("Pose Diversity", run_pose_diversity_analysis(sample_records, task_cfg, sample_dir))

        # Use object point-cloud bounding boxes as a lightweight geometry sanity
        # check for Human/DGN samples without running downstream simulation.
        report.add_section("BBox Geometry", run_bbox_geometry_sanity(pose_files, task_cfg, sample_dir))

    report.add_section(
        "Downstream Note",
        [
            "- This task does not run BODex/Bench. It only evaluates saved samples.",
            "- Use the saved feasibility scores and fixed-type pose samples as inputs to downstream budget tests.",
        ],
    )
    print(f"[evaluate] Wrote markdown report to {report_path}")
