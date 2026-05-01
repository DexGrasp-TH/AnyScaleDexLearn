import csv
import json
import os
import re
from collections import Counter
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
TARGET_GRASP_TYPE_IDS = (1, 2, 3, 4, 5)


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


def run_nearest_neighbor_analysis(
    train_records: list,
    sample_records: list,
    task_cfg,
    sample_dir: str,
) -> list:
    """Run sampled-to-training nearest-neighbor pose overfit analysis.

    Args:
        train_records: Comparable training pose records.
        sample_records: Comparable sampled pose records.
        task_cfg: Hydra task config containing thresholds and output settings.
        sample_dir: Directory containing saved sample results.

    Returns:
        Per-sample nearest-neighbor result rows.
    """
    trans_ref_m = float(getattr(task_cfg, "trans_ref_m", 0.02))
    rot_ref_deg = float(getattr(task_cfg, "rot_ref_deg", 10.0))
    group_key = str(getattr(task_cfg, "group_by", "sample_canonical_object_id"))
    train_by_object = grouped_candidates(train_records, "object_name")
    train_by_canonical = grouped_candidates(train_records, "canonical_object_id")

    rows = []
    for sample_record in sample_records:
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
    return rows


def count_grasp_types_by_object(records: list) -> dict:
    """Count grasp types for each canonical object.

    Args:
        records: Training or sampled records containing ``canonical_object_id``
            and ``grasp_type_id``.

    Returns:
        Mapping ``canonical_object_id -> Counter({grasp_type_id: count})``.
    """
    grouped_counts = {}
    for record in records:
        object_id = record["canonical_object_id"]
        grasp_type_id = int(record["grasp_type_id"])
        grouped_counts.setdefault(object_id, Counter())[grasp_type_id] += 1
    return grouped_counts


def counter_total(counter: Counter, type_ids=TARGET_GRASP_TYPE_IDS) -> int:
    """Count valid grasp-type entries in a counter.

    Args:
        counter: Grasp-type counter.
        type_ids: Type ids included in the valid distribution.

    Returns:
        Sum of counts for valid type ids.
    """
    return int(sum(int(counter.get(type_id, 0)) for type_id in type_ids))


def counter_probs(counter: Counter, type_ids=TARGET_GRASP_TYPE_IDS) -> np.ndarray:
    """Convert a grasp-type counter into a probability vector.

    Args:
        counter: Grasp-type counter.
        type_ids: Ordered type ids used for the vector.

    Returns:
        Probability vector. Empty counters return zeros.
    """
    counts = np.asarray([counter.get(type_id, 0) for type_id in type_ids], dtype=np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def dominant_type(counter: Counter, type_ids=TARGET_GRASP_TYPE_IDS):
    """Find the most frequent valid grasp type.

    Args:
        counter: Grasp-type counter.
        type_ids: Valid type ids to compare.

    Returns:
        Dominant type id, or ``None`` if there are no valid counts.
    """
    valid_counts = [(type_id, int(counter.get(type_id, 0))) for type_id in type_ids]
    valid_counts = [item for item in valid_counts if item[1] > 0]
    if not valid_counts:
        return None
    return max(valid_counts, key=lambda item: (item[1], -item[0]))[0]


def total_variation_distance(counter_a: Counter, counter_b: Counter, type_ids=TARGET_GRASP_TYPE_IDS) -> float:
    """Compute total variation distance between two grasp-type distributions.

    Args:
        counter_a: First grasp-type counter.
        counter_b: Second grasp-type counter.
        type_ids: Type ids included in the distribution.

    Returns:
        Total variation distance in ``[0, 1]``.
    """
    return float(0.5 * np.abs(counter_probs(counter_a, type_ids) - counter_probs(counter_b, type_ids)).sum())


def format_counter(counter: Counter, type_ids=TARGET_GRASP_TYPE_IDS) -> str:
    """Format grasp-type counts as a compact string.

    Args:
        counter: Grasp-type counter.
        type_ids: Type ids to include.

    Returns:
        String like ``1:3 2:0 3:1 4:0 5:0``.
    """
    return " ".join(f"{type_id}:{int(counter.get(type_id, 0))}" for type_id in type_ids)


def type_percentage(counter: Counter, type_id: int, type_ids=TARGET_GRASP_TYPE_IDS) -> float:
    """Compute one grasp type's percentage within a counter.

    Args:
        counter: Grasp-type counter.
        type_id: Grasp type id whose percentage should be computed.
        type_ids: Type ids included in the denominator.

    Returns:
        Percentage in ``[0, 100]``. Empty counters return ``0.0``.
    """
    total = counter_total(counter, type_ids)
    if total <= 0:
        return 0.0
    return float(counter.get(type_id, 0) / total * 100.0)


def format_percentages(counter: Counter, type_ids=TARGET_GRASP_TYPE_IDS) -> str:
    """Format grasp-type percentages as a compact string.

    Args:
        counter: Grasp-type counter.
        type_ids: Type ids to include.

    Returns:
        String like ``1:75.00% 2:0.00% 3:25.00% 4:0.00% 5:0.00%``.
    """
    return " ".join(f"{type_id}:{type_percentage(counter, type_id, type_ids):.2f}%" for type_id in type_ids)


def global_type_summary(records: list) -> Counter:
    """Count global grasp types across records.

    Args:
        records: Training or sampled records.

    Returns:
        Global grasp-type counter.
    """
    counter = Counter()
    for record in records:
        counter[int(record["grasp_type_id"])] += 1
    return counter


def build_distribution_rows(train_records: list, sample_records: list) -> list:
    """Build per-object grasp-type distribution comparison rows.

    Args:
        train_records: Comparable training records.
        sample_records: Comparable sampled records.

    Returns:
        Rows comparing train and sample distributions for each canonical object.
    """
    train_counts = count_grasp_types_by_object(train_records)
    sample_counts = count_grasp_types_by_object(sample_records)
    all_object_ids = sorted(set(train_counts) | set(sample_counts))
    rows = []
    for object_id in all_object_ids:
        train_counter = train_counts.get(object_id, Counter())
        sample_counter = sample_counts.get(object_id, Counter())
        row = {
            "canonical_object_id": object_id,
            "train_total": counter_total(train_counter),
            "sample_total": counter_total(sample_counter),
            "train_dominant_type": dominant_type(train_counter),
            "sample_dominant_type": dominant_type(sample_counter),
            "dominant_type_match": dominant_type(train_counter) == dominant_type(sample_counter),
            "type_tv_distance": total_variation_distance(train_counter, sample_counter),
            "train_counts": format_counter(train_counter),
            "sample_counts": format_counter(sample_counter),
            "train_percentages": format_percentages(train_counter),
            "sample_percentages": format_percentages(sample_counter),
        }
        for type_id in TARGET_GRASP_TYPE_IDS:
            row[f"train_type_{type_id}"] = int(train_counter.get(type_id, 0))
            row[f"sample_type_{type_id}"] = int(sample_counter.get(type_id, 0))
            row[f"train_type_{type_id}_percent"] = type_percentage(train_counter, type_id)
            row[f"sample_type_{type_id}_percent"] = type_percentage(sample_counter, type_id)
        rows.append(row)
    return rows


def print_grasp_type_counter(title: str, counter: Counter) -> None:
    """Print one global grasp-type distribution.

    Args:
        title: Printed section title.
        counter: Grasp-type counter.

    Returns:
        None.
    """
    total = counter_total(counter)
    print(f"\n{title}:")
    print(f"  Total valid grasps: {total}")
    print(f"  {'type':<8} {'count':>8} {'percent':>9}")
    for type_id in TARGET_GRASP_TYPE_IDS:
        count = int(counter.get(type_id, 0))
        percent = count / total * 100.0 if total else 0.0
        print(f"  {type_id:<8} {count:>8} {percent:>8.2f}%")
    invalid_count = sum(count for type_id, count in counter.items() if type_id not in TARGET_GRASP_TYPE_IDS)
    if invalid_count:
        print(f"  invalid    {invalid_count:>8}")


def summarize_distribution_rows(rows: list, top_k: int) -> None:
    """Print per-object train/sample grasp-type distribution comparison.

    Args:
        rows: Per-object distribution comparison rows.
        top_k: Number of highest-distance objects to print.

    Returns:
        None.
    """
    print("\nGrasp-type distribution by canonical object:")
    if not rows:
        print("  No rows.")
        return

    train_objects = [row for row in rows if row["train_total"] > 0]
    sample_objects = [row for row in rows if row["sample_total"] > 0]
    common_rows = [row for row in rows if row["train_total"] > 0 and row["sample_total"] > 0]
    missing_sample = [row for row in rows if row["train_total"] > 0 and row["sample_total"] == 0]
    sample_only = [row for row in rows if row["train_total"] == 0 and row["sample_total"] > 0]

    print(f"  Train objects: {len(train_objects)}")
    print(f"  Sample objects: {len(sample_objects)}")
    print(f"  Common objects: {len(common_rows)}")
    print(f"  Train objects without samples: {len(missing_sample)}")
    print(f"  Sample objects absent from train: {len(sample_only)}")

    if common_rows:
        tv_values = np.asarray([row["type_tv_distance"] for row in common_rows], dtype=np.float64)
        dominant_matches = [row for row in common_rows if row["dominant_type_match"]]
        print(f"  Mean TV distance: {tv_values.mean():.4f}")
        print(f"  Median TV distance: {np.median(tv_values):.4f}")
        print(f"  Max TV distance: {tv_values.max():.4f}")
        print(
            f"  Dominant type match: {len(dominant_matches)}/{len(common_rows)} "
            f"({len(dominant_matches) / len(common_rows) * 100.0:.1f}%)"
        )

    top_rows = sorted(common_rows, key=lambda row: row["type_tv_distance"], reverse=True)[: max(0, int(top_k))]
    if top_rows:
        print(f"\nTop {len(top_rows)} objects with largest train/sample type-distribution gap:")
        print(
            f"  {'object':<28} {'train_n':>8} {'sample_n':>8} {'tv':>7} "
            f"{'train_dom':>9} {'sample_dom':>10}  train_counts -> sample_counts"
        )
        for row in top_rows:
            print(
                f"  {str(row['canonical_object_id'])[:28]:<28} "
                f"{row['train_total']:>8} {row['sample_total']:>8} {row['type_tv_distance']:>7.3f} "
                f"{str(row['train_dominant_type']):>9} {str(row['sample_dominant_type']):>10}  "
                f"{row['train_counts']} -> {row['sample_counts']}"
            )


def write_distribution_csv(rows: list, csv_path: str) -> None:
    """Write per-object grasp-type distribution rows to a CSV file.

    Args:
        rows: Per-object distribution comparison rows.
        csv_path: Destination CSV path.

    Returns:
        None.
    """
    if not rows:
        print("[overfit_nn] No distribution rows to write.")
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[overfit_nn] Wrote grasp-type distribution rows to {csv_path}")


def run_grasp_type_distribution_analysis(
    train_records: list,
    sample_records: list,
    task_cfg,
    sample_dir: str,
) -> list:
    """Compare train and sample grasp-type distributions per canonical object.

    Args:
        train_records: Comparable training records.
        sample_records: Comparable sampled records.
        task_cfg: Hydra task config containing output settings.
        sample_dir: Directory containing saved sample results.

    Returns:
        Per-object distribution comparison rows.
    """
    print_grasp_type_counter("Global train grasp-type distribution", global_type_summary(train_records))
    print_grasp_type_counter("Global sampled grasp-type distribution", global_type_summary(sample_records))

    rows = build_distribution_rows(train_records, sample_records)
    summarize_distribution_rows(rows, top_k=int(getattr(task_cfg, "distribution_top_k", 20)))

    output_csv_value = getattr(task_cfg, "distribution_output_csv", "")
    output_csv = "" if output_csv_value is None else str(output_csv_value)
    if not output_csv:
        output_csv = os.path.join(sample_dir, "grasp_type_distribution_by_object.csv")
    write_distribution_csv(rows, output_csv)
    return rows


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
        print("[overfit_nn] No object-specific rows to write.")
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[overfit_nn] Wrote object-specific NN rows to {csv_path}")


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
        output_csv = os.path.join(sample_dir, "object_specific_nn_overfit.csv")
    write_object_analysis_csv(rows, output_csv)
    return rows


def task_overfit_nn(config) -> None:
    """Hydra task entry point for sampled-result diagnostics.

    Args:
        config: Full Hydra config for the current AnyScaleDexLearn run.

    Returns:
        None.
    """
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    task_cfg = config.task
    max_sample_files = int(getattr(task_cfg, "max_sample_files", 0))

    train_records = load_training_records(config)

    sample_dir = get_sample_output_dir(config)
    sample_files = sorted(glob(os.path.join(sample_dir, "**/*.npy"), recursive=True))
    if max_sample_files > 0:
        sample_files = sample_files[:max_sample_files]
    sample_records = load_sample_records(sample_files)
    print(f"[overfit_nn] Loaded {len(train_records)} training records.")
    print(f"[overfit_nn] Found {len(sample_files)} sampled result files in {sample_dir}")
    print(f"[overfit_nn] Loaded {len(sample_records)} comparable sampled pose records.")

    run_nearest_neighbor_analysis(train_records, sample_records, task_cfg, sample_dir)
    run_grasp_type_distribution_analysis(train_records, sample_records, task_cfg, sample_dir)
    run_object_specific_nn_analysis(train_records, sample_records, task_cfg, sample_dir)
