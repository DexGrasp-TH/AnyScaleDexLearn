import csv
import json
import os
import re
from copy import deepcopy
from glob import glob

import numpy as np
from scipy.spatial.transform import Rotation as SciR

try:
    from dexlearn.utils.config import cfg_get, flatten_multidex_data_config
    from omegaconf import ListConfig
except ModuleNotFoundError:
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


MIRROR = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
FIXED_LEFT_HAND_TRANS = np.array([0.0, 0.0, -0.5], dtype=np.float32)
FIXED_LEFT_HAND_ROT = np.eye(3, dtype=np.float32)
VALID_HAND_POS_SOURCES = {"wrist", "index_mcp"}


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
        object_name: Object or sequence name such as ``obj_0_seq_3``.

    Returns:
        Canonical object id, for example ``obj_0`` for ``obj_0_seq_3``.
    """
    base_name = os.path.basename(str(object_name))
    return re.sub(r"([_-]seq[_-]?\d+.*)$", "", base_name)


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
        "canonical_object_id": canonical_object_id(object_name),
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
            scene_cfg = np.load(scene_path, allow_pickle=True).item()
            if "object" in scene_cfg and "name" in scene_cfg["object"]:
                return str(scene_cfg["object"]["name"])
            task_obj_name = scene_cfg.get("task", {}).get("obj_name")
            if task_obj_name is not None:
                return str(task_obj_name)
        except Exception as exc:
            print(f"[overfit_nn] Could not load scene config {scene_path}: {exc}")

    pc_path = str(sample_data.get("pc_path", ""))
    return os.path.basename(os.path.dirname(pc_path)) if pc_path else ""


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
    object_name = resolve_sample_object_name(sample_data)
    return [
        {
            "path": sample_path,
            "object_name": object_name,
            "canonical_object_id": canonical_object_id(object_name),
            "grasp_type_id": grasp_type_id,
            "right_trans": pose[0:3],
            "right_rot": quat_wxyz_to_matrix(pose[3:7]),
            "left_trans": pose[7:10],
            "left_rot": quat_wxyz_to_matrix(pose[10:14]),
        }
    ]


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


def grouped_candidates(train_records: list, key: str) -> dict:
    """Group training records by object name or canonical object id.

    Args:
        train_records: List of training pose records.
        key: Group key, either ``object_name`` or ``canonical_object_id``.

    Returns:
        Dictionary mapping group value to candidate training records.
    """
    groups = {}
    for record in train_records:
        groups.setdefault(record[key], []).append(record)
    return groups


def summarize_rows(rows: list, group_key: str, trans_threshold_m: float, rot_threshold_deg: float) -> None:
    """Print nearest-neighbor summary grouped by sampled object metadata.

    Args:
        rows: Per-sample nearest-neighbor result rows.
        group_key: Row key to group by for printed summaries.
        trans_threshold_m: Translation threshold for near-copy percentage.
        rot_threshold_deg: Rotation threshold for near-copy percentage.

    Returns:
        None.
    """
    print("\nNearest-neighbor summary:")
    print(f"  Rows: {len(rows)}")
    if not rows:
        return

    near_copy = [row for row in rows if row["trans_m"] <= trans_threshold_m and row["rot_deg"] <= rot_threshold_deg]
    print(
        f"  Near-copy threshold: trans <= {trans_threshold_m:.4f} m and rot <= {rot_threshold_deg:.2f} deg"
    )
    print(f"  Near-copy rows: {len(near_copy)} ({len(near_copy) / len(rows) * 100:.1f}%)")

    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    print(f"\nGrouped by {group_key}:")
    print(f"  {'group':<32} {'count':>8} {'mean_t_m':>10} {'min_t_m':>10} {'mean_r_deg':>12} {'min_r_deg':>11}")
    for group_name, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        trans_values = np.asarray([row["trans_m"] for row in group_rows], dtype=np.float64)
        rot_values = np.asarray([row["rot_deg"] for row in group_rows], dtype=np.float64)
        print(
            f"  {str(group_name)[:32]:<32} {len(group_rows):>8} "
            f"{trans_values.mean():>10.4f} {trans_values.min():>10.4f} "
            f"{rot_values.mean():>12.2f} {rot_values.min():>11.2f}"
        )


def write_csv(rows: list, csv_path: str) -> None:
    """Write nearest-neighbor rows to a CSV file.

    Args:
        rows: Per-sample nearest-neighbor result rows.
        csv_path: Destination CSV path.

    Returns:
        None.
    """
    if not rows:
        print("[overfit_nn] No rows to write.")
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[overfit_nn] Wrote nearest-neighbor rows to {csv_path}")


def task_overfit_nn(config) -> None:
    """Hydra task entry point for sampled-to-training nearest-neighbor analysis.

    Args:
        config: Full Hydra config for the current AnyScaleDexLearn run.

    Returns:
        None.
    """
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    task_cfg = config.task
    trans_ref_m = float(getattr(task_cfg, "trans_ref_m", 0.02))
    rot_ref_deg = float(getattr(task_cfg, "rot_ref_deg", 10.0))
    max_sample_files = int(getattr(task_cfg, "max_sample_files", 0))
    group_key = str(getattr(task_cfg, "group_by", "sample_canonical_object_id"))

    train_records = load_training_records(config)
    train_by_object = grouped_candidates(train_records, "object_name")
    train_by_canonical = grouped_candidates(train_records, "canonical_object_id")

    sample_dir = get_sample_output_dir(config)
    sample_files = sorted(glob(os.path.join(sample_dir, "**/*.npy"), recursive=True))
    if max_sample_files > 0:
        sample_files = sample_files[:max_sample_files]
    print(f"[overfit_nn] Loaded {len(train_records)} training records.")
    print(f"[overfit_nn] Found {len(sample_files)} sampled result files in {sample_dir}")

    rows = []
    for sample_file in sample_files:
        for sample_record in sample_pose_records(sample_file):
            candidates = train_by_object.get(sample_record["object_name"])
            if not candidates:
                candidates = train_by_canonical.get(sample_record["canonical_object_id"], train_records)
            rows.append(nearest_train_record(sample_record, candidates, trans_ref_m, rot_ref_deg))

    summarize_rows(
        rows,
        group_key=group_key,
        trans_threshold_m=float(getattr(task_cfg, "near_copy_trans_m", 0.01)),
        rot_threshold_deg=float(getattr(task_cfg, "near_copy_rot_deg", 5.0)),
    )

    output_csv_value = getattr(task_cfg, "output_csv", "")
    output_csv = "" if output_csv_value is None else str(output_csv_value)
    if not output_csv:
        output_csv = os.path.join(sample_dir, "overfit_nearest_train_grasp.csv")
    write_csv(rows, output_csv)
