import time
import os
import json
import random
import re
from glob import glob

import hydra
import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf
from dexlearn.utils.config import flatten_multidex_data_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import viser

    VISER_AVAILABLE = True
except ImportError:
    viser = None
    VISER_AVAILABLE = False


VISUALIZE_MODE_OPTIONS = ("random_object", "one_object", "one_object_multi_seq", "grasp_type")
CAPTION_ASPECTS = (
    ("scene_id", "Scene ID"),
    ("object_id", "Object ID"),
    ("file", "File"),
    ("given_grasp_type", "Given Grasp Type"),
    ("pred_grasp_type", "Pred Grasp Type"),
    ("pred_grasp_type_prob", "Pred Grasp Type Prob"),
    ("position_source", "Position Source"),
    ("point_cloud", "Point Cloud"),
    ("error", "Error"),
    ("ik_status", "IK Status"),
    ("other", "Other"),
)


def progress_iter(iterable, desc: str, total=None):
    """Wrap an iterable with a progress bar when tqdm is available.

    Args:
        iterable: Iterable to wrap.
        desc: Progress bar description.
        total: Optional total item count for iterables without ``len``.

    Returns:
        An iterable that yields the same items, with a tqdm progress bar when
        the dependency is installed.
    """
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def get_task_value(config: DictConfig, key: str, default):
    value = OmegaConf.select(config, f"task.{key}")
    return default if value is None else value


def normalize_visualize_mode(mode: str):
    aliases = {
        "random": "random_object",
        "random_objects": "random_object",
        "object": "one_object",
        "single_object": "one_object",
        "multi_seq": "one_object_multi_seq",
        "multi_sequence": "one_object_multi_seq",
        "one_object_multi_sequence": "one_object_multi_seq",
        "grasp_type_id": "grasp_type",
        "type": "grasp_type",
    }
    mode = str(mode).lower()
    return aliases.get(mode, mode)


def list_sample_files(output_dir: str):
    return sorted(glob(os.path.join(output_dir, "**/*.npy"), recursive=True))


def infer_object_id_from_sample_path(output_dir: str, sample_file: str):
    """Infer a scene-like object id from the saved sample path.

    Args:
        output_dir: Root directory that contains saved visualization samples.
        sample_file: Path to one saved sample file.

    Returns:
        A slash-separated id inferred from the relative sample path. The first
        relative component is the logger group such as ``0_any`` and is skipped.
    """
    try:
        rel_path = os.path.relpath(sample_file, output_dir)
    except ValueError:
        rel_path = sample_file
    parts = rel_path.split(os.sep)
    if len(parts) >= 3:
        return "/".join(parts[1:-1])
    return os.path.splitext(os.path.basename(sample_file))[0]


def extract_grasp_type_id(data: dict):
    if "pred_grasp_type_id" in data:
        return int(np.asarray(data["pred_grasp_type_id"]).reshape(-1)[0])
    if "grasp_type_id" in data:
        return int(np.asarray(data["grasp_type_id"]).reshape(-1)[0])
    return None


def infer_object_id_from_scene_cfg(scene_cfg: dict, scene_path: str):
    if "scene_id" in scene_cfg:
        return str(scene_cfg["scene_id"])
    if "object" in scene_cfg and isinstance(scene_cfg["object"], dict):
        obj_name = scene_cfg["object"].get("name")
        if obj_name is not None:
            return str(obj_name)
    task_obj_name = scene_cfg.get("task", {}).get("obj_name") if isinstance(scene_cfg.get("task"), dict) else None
    if task_obj_name is not None:
        return str(task_obj_name)
    return os.path.splitext(os.path.basename(scene_path))[0]


def load_visualization_record_payload(record: dict, scene_path_resolver=None, load_scene_cfg: bool = True):
    """Load sample data and optional scene config into a lightweight record.

    Args:
        record: Sample record created by ``build_visualization_sample_index``.
        scene_path_resolver: Optional callable that maps saved scene paths to
            paths that are valid in the current workspace.
        load_scene_cfg: Whether to load the referenced scene config.

    Returns:
        The same record dictionary after adding loaded payload fields.
    """
    if record.get("data") is not None and (not load_scene_cfg or record.get("scene_cfg")):
        return record

    sample_file = record["sample_file"]
    data = np.load(sample_file, allow_pickle=True).item()
    scene_path = str(data.get("scene_path", ""))
    if scene_path_resolver is not None and scene_path:
        scene_path = scene_path_resolver(scene_path)

    scene_cfg = record.get("scene_cfg") or {}
    if load_scene_cfg and scene_path:
        scene_cfg = np.load(scene_path, allow_pickle=True).item()

    record["data"] = data
    record["scene_path"] = scene_path
    record["scene_cfg"] = scene_cfg
    record["grasp_type_id"] = extract_grasp_type_id(data)
    if scene_cfg:
        record["object_id"] = infer_object_id_from_scene_cfg(scene_cfg, scene_path)
    return record


def build_visualization_sample_index(
    output_dir: str,
    scene_path_resolver=None,
    load_payload: bool = True,
    load_scene_cfg: bool = True,
):
    """Build an index for saved visualization samples.

    Args:
        output_dir: Directory containing saved ``.npy`` sample files.
        scene_path_resolver: Optional callable used to resolve saved scene paths.
        load_payload: Whether to load every sample file while indexing.
        load_scene_cfg: Whether to load every scene config while indexing.

    Returns:
        A list of sample records. With ``load_payload=False``, records contain
        only path-derived metadata and are hydrated later for selected samples.
    """
    sample_records = []
    sample_files = list_sample_files(output_dir)
    print(
        f"[visualize] Found {len(sample_files)} saved sample file(s) in {output_dir}. "
        f"load_payload={load_payload}, load_scene_cfg={load_scene_cfg}"
    )
    file_iter = progress_iter(sample_files, desc="Indexing saved visualization samples", total=len(sample_files))
    for idx, sample_file in enumerate(file_iter, 1):
        record = {
            "sample_file": sample_file,
            "data": None,
            "scene_path": "",
            "scene_cfg": {},
            "object_id": infer_object_id_from_sample_path(output_dir, sample_file),
            "grasp_type_id": None,
        }
        if not load_payload:
            sample_records.append(record)
            continue
        try:
            load_visualization_record_payload(
                record,
                scene_path_resolver=scene_path_resolver,
                load_scene_cfg=load_scene_cfg,
            )
        except Exception as exc:
            print(f"[visualize] Skipping unreadable sample {sample_file}: {exc}")
            continue
        if idx % 50000 == 0:
            print(f"[visualize] Indexed {idx}/{len(sample_files)} sample file(s).")
        sample_records.append(record)
    return sample_records


def limit_records(records, max_grasps: int):
    if max_grasps <= 0 or len(records) <= max_grasps:
        return records
    return random.sample(records, k=max_grasps)


def randomize_records(records, max_grasps: int):
    """Return a randomized subset of records.

    Args:
        records: Candidate records or ids.
        max_grasps: Maximum number of records to return. Non-positive values
            return all candidates in randomized order.

    Returns:
        A list sampled from the candidate pool without preferring path-sorted
        prefixes.
    """
    records = list(records)
    if max_grasps <= 0 or len(records) <= max_grasps:
        random.shuffle(records)
        return records
    return random.sample(records, k=max_grasps)


def slice_records_batch(records, max_grasps: int, batch_index: int):
    """Return one circular batch from an ordered record list.

    Args:
        records: Ordered records or ids to batch.
        max_grasps: Maximum number of entries in each batch. Non-positive
            values keep all records in one batch.
        batch_index: Zero-based batch index to show.

    Returns:
        A list containing the selected batch. When ``batch_index`` exceeds the
        number of available batches, selection wraps around.
    """
    records = list(records)
    if max_grasps <= 0 or len(records) <= max_grasps:
        return records
    batch_index = max(0, int(batch_index))
    start = (batch_index * max_grasps) % len(records)
    end = start + max_grasps
    if end <= len(records):
        return records[start:end]
    return records[start:] + records[: end - len(records)]


def object_id_matches(record_object_id: str, target_object_id: str):
    record_object_id = str(record_object_id)
    target_object_id = str(target_object_id)
    return (
        record_object_id == target_object_id
        or record_object_id.startswith(f"{target_object_id}/")
        or os.path.basename(record_object_id) == target_object_id
    )


def natural_sort_key(value):
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", str(value))]


def canonical_object_id(object_id):
    return str(object_id).strip().strip("/")


def base_object_id_from_sequence(object_id):
    object_id = canonical_object_id(object_id)
    if "/" in object_id:
        return object_id.split("/", 1)[0]
    match = re.match(r"^(.+)_seq_[^/]+$", object_id)
    return match.group(1) if match else object_id


def sequence_object_id_matches(record_object_id: str, target_object_id: str):
    return base_object_id_from_sequence(record_object_id) == base_object_id_from_sequence(target_object_id)


def is_our_human_grasp_format_test_data(config: DictConfig):
    try:
        object_path = (
            OmegaConf.select(config, "test_data.object_path")
            or OmegaConf.select(config, "test_data.paths.object_path")
        )
    except Exception:
        object_path = None
    return object_path is not None and "OurHumanGraspFormat" in str(object_path)


def select_evenly_across_sequences(records, max_grasps: int):
    grouped = {}
    for record in sorted(records, key=lambda item: item["sample_file"]):
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)

    selected = []
    sequence_ids = sorted(grouped.keys(), key=natural_sort_key)
    while sequence_ids:
        next_sequence_ids = []
        for sequence_id in sequence_ids:
            if not grouped[sequence_id]:
                continue
            selected.append(grouped[sequence_id].pop(0))
            if max_grasps > 0 and len(selected) >= max_grasps:
                return selected
            if grouped[sequence_id]:
                next_sequence_ids.append(sequence_id)
        sequence_ids = next_sequence_ids
    return selected


def select_random_across_sequences(records, max_grasps: int):
    """Select records randomly while spreading them across sequence ids.

    Args:
        records: Candidate records with ``object_id`` metadata.
        max_grasps: Maximum number of records to return.

    Returns:
        Randomly selected records, round-robin across sequence ids when
        multiple sequence ids are present.
    """
    grouped = {}
    for record in records:
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)
    if not grouped:
        return []

    for group_records in grouped.values():
        random.shuffle(group_records)
    sequence_ids = list(grouped.keys())
    random.shuffle(sequence_ids)

    selected = []
    while sequence_ids:
        next_sequence_ids = []
        for sequence_id in sequence_ids:
            if not grouped[sequence_id]:
                continue
            selected.append(grouped[sequence_id].pop())
            if max_grasps > 0 and len(selected) >= max_grasps:
                return selected
            if grouped[sequence_id]:
                next_sequence_ids.append(sequence_id)
        random.shuffle(next_sequence_ids)
        sequence_ids = next_sequence_ids
    return selected


def object_variant_bucket_id(object_id: str):
    """Return the coarse scale bucket for a scene-like object id.

    Args:
        object_id: Full object id inferred from a saved sample path.

    Returns:
        A bucket id used to interleave scene variants. DGN ids end with tokens
        like ``scale002_pose000_0``; grouping by ``scale002`` prevents the first
        one-object batch from showing only the smallest scale.
    """
    leaf = canonical_object_id(object_id).split("/")[-1]
    match = re.match(r"^(scale\d+)_pose", leaf)
    return match.group(1) if match else leaf


def interleave_object_variant_ids(object_ids):
    """Interleave full object ids across coarse variant buckets.

    Args:
        object_ids: Full object ids, usually one per scene/scale/pose variant.

    Returns:
        A list of object ids ordered round-robin across buckets.
    """
    buckets = {}
    for object_id in sorted(object_ids, key=natural_sort_key):
        buckets.setdefault(object_variant_bucket_id(object_id), []).append(object_id)

    ordered = []
    bucket_ids = sorted(buckets.keys(), key=natural_sort_key)
    while bucket_ids:
        next_bucket_ids = []
        for bucket_id in bucket_ids:
            if not buckets[bucket_id]:
                continue
            ordered.append(buckets[bucket_id].pop(0))
            if buckets[bucket_id]:
                next_bucket_ids.append(bucket_id)
        bucket_ids = next_bucket_ids
    return ordered


def select_one_object_variant_batch(records, max_grasps: int, batch_index: int):
    """Randomly select a one-object batch across full scene variants.

    Args:
        records: Records matched by the selected object id.
        max_grasps: Maximum number of scenes to show.
        batch_index: Zero-based batch index. This is accepted for compatibility
            with the Selection panel state; the selected batch is randomized
            rather than sliced by index.

    Returns:
        Selected records. When a selected base object expands to many DGN
        scale/pose variants, the batch samples scene variants randomly instead
        of taking the first lexicographic scale only.
    """
    del batch_index
    records = list(records)
    if max_grasps <= 0:
        return randomize_records(records, max_grasps)
    grouped = {}
    for record in records:
        grouped.setdefault(canonical_object_id(record["object_id"]), []).append(record)
    if len(grouped) <= 1:
        return randomize_records(records, max_grasps)

    selected_object_ids = randomize_records(grouped.keys(), max_grasps)
    selected = []
    for object_id in selected_object_ids:
        object_records = grouped[object_id]
        selected.append(random.choice(object_records))
    if len(selected) < max_grasps and len(selected) < len(records):
        used_paths = {record["sample_file"] for record in selected}
        remaining = [record for record in records if record["sample_file"] not in used_paths]
        selected.extend(randomize_records(remaining, max_grasps - len(selected)))
    return selected


def annotate_scene_labels(records, mode: str):
    for record in records:
        record.pop("viser_all_label", None)
        record.pop("viser_spatial_group", None)

    for idx, record in enumerate(records):
        record["viser_all_label"] = f"{idx} | {canonical_object_id(record['object_id'])}"
        if mode == "one_object_multi_seq":
            record["viser_spatial_group"] = canonical_object_id(record["object_id"])
    return records


def prefix_caption_with_viser_label(sample_record, caption: str):
    label = sample_record.get("viser_all_label")
    return f"{label} | {caption}" if label else caption


def copy_viser_record_metadata(sample_record, scene_record):
    for key in ("viser_all_label", "viser_spatial_group"):
        if key in sample_record:
            scene_record[key] = sample_record[key]
    return scene_record


def select_visualization_samples(output_dir: str, config: DictConfig, scene_path_resolver=None):
    mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_object"))
    max_grasps = int(get_task_value(config, "max_grasps", 20))
    object_id = get_task_value(config, "object_id", None)
    target_grasp_type_id = get_task_value(config, "target_grasp_type_id", None)
    records = build_visualization_sample_index(output_dir, scene_path_resolver=scene_path_resolver)
    if not records:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    return select_visualization_records(
        records,
        mode,
        max_grasps,
        object_id,
        target_grasp_type_id,
        is_our_human_grasp_format=is_our_human_grasp_format_test_data(config),
    )


def select_visualization_records(
    records,
    mode: str,
    max_grasps: int,
    object_id=None,
    target_grasp_type_id=None,
    is_our_human_grasp_format=False,
    batch_index=None,
):
    mode = normalize_visualize_mode(mode)

    if mode == "random_object":
        grouped = {}
        for record in records:
            group_id = base_object_id_from_sequence(record["object_id"]) if is_our_human_grasp_format else record["object_id"]
            grouped.setdefault(group_id, []).append(record)
        object_ids = randomize_records(grouped.keys(), max_grasps)
        selected = []
        for selected_object_id in object_ids:
            selected.append(random.choice(grouped[selected_object_id]))

    elif mode == "one_object":
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object.")
        selected = [record for record in records if object_id_matches(record["object_id"], object_id)]
        selected = sorted(selected, key=lambda record: record["sample_file"])
        selected = select_one_object_variant_batch(selected, max_grasps, 0 if batch_index is None else batch_index)

    elif mode == "one_object_multi_seq":
        if not is_our_human_grasp_format:
            raise RuntimeError(
                "task.visualize_mode=one_object_multi_seq is only valid when test_data.object_path points to "
                "OurHumanGraspFormat."
            )
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object_multi_seq.")
        selected = [record for record in records if sequence_object_id_matches(record["object_id"], object_id)]
        selected = select_random_across_sequences(selected, max_grasps)

    elif mode == "grasp_type":
        if target_grasp_type_id is None:
            raise ValueError("task.target_grasp_type_id must be set when task.visualize_mode=grasp_type.")
        target_grasp_type_id = int(target_grasp_type_id)
        populate_grasp_type_ids(records)
        selected = [record for record in records if record["grasp_type_id"] == target_grasp_type_id]
        selected = randomize_records(selected, max_grasps)

    else:
        raise ValueError(
            f"Unsupported visualize_mode={mode}. Expected one of {list(VISUALIZE_MODE_OPTIONS)}."
        )

    if not selected:
        example_object_ids = list_available_object_ids(records)[:10]
        example_text = ", ".join(example_object_ids) if example_object_ids else "none"
        raise RuntimeError(
            f"No samples matched visualize_mode={mode}, object_id={object_id}, "
            f"target_grasp_type_id={target_grasp_type_id}. Available object examples: {example_text}"
        )

    selected = annotate_scene_labels(selected, mode)
    print(f"[visualize] Selected {len(selected)} sample(s) with visualize_mode={mode}.")
    return selected


def compact_caption(caption: str):
    return caption.split(" | ", 1)[0]


def split_caption_after_viser_label(record):
    """Split a full caption into the compact viser label and content parts.

    Args:
        record: Scene record containing ``caption`` and optional
            ``viser_all_label`` metadata.

    Returns:
        A tuple ``(label_parts, content_parts)``. Label parts come from
        ``viser_all_label`` when present, so labels like ``0 | obj_1`` are not
        confused with regular caption separators.
    """
    caption = str(record.get("caption", ""))
    label = str(record.get("viser_all_label", ""))
    if label and caption.startswith(f"{label} | "):
        return label.split(" | "), caption[len(label) + 3 :].split(" | ")
    if label and caption == label:
        return label.split(" | "), []
    return [], [part.strip() for part in caption.split(" | ") if part.strip()]


def caption_aspects_for_record(record):
    """Extract named caption aspects from one scene record.

    Args:
        record: Scene record with a full caption.

    Returns:
        A dictionary mapping caption aspect ids to display strings.
    """
    label_parts, content_parts = split_caption_after_viser_label(record)
    aspects = {}
    if label_parts:
        aspects["scene_id"] = label_parts[0]
    if len(label_parts) >= 2:
        aspects["object_id"] = " | ".join(label_parts[1:])

    other_parts = []
    for part in content_parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Given:") or part.startswith("grasp_type_id="):
            aspects["given_grasp_type"] = part
        elif part.startswith("Pred:") or part.startswith("pred_grasp_type_id="):
            aspects["pred_grasp_type"] = part
        elif part.startswith("pred_grasp_type_prob="):
            aspects["pred_grasp_type_prob"] = part
        elif part.startswith("Pos:"):
            aspects["position_source"] = part
        elif part.startswith("PC:"):
            aspects["point_cloud"] = part
        elif part.startswith("err="):
            aspects["error"] = part
        elif "IK:" in part:
            aspects["ik_status"] = part
        elif "file" not in aspects:
            aspects["file"] = part
        else:
            other_parts.append(part)

    if other_parts:
        aspects["other"] = " | ".join(other_parts)
    return aspects


def build_caption_from_aspects(record, selected_aspects):
    """Build a scene caption from selected aspect ids.

    Args:
        record: Scene record with a full caption.
        selected_aspects: Iterable of aspect ids enabled in the web UI.

    Returns:
        A compact caption containing only the selected aspects. Returns an empty
        string when no selected aspect is available for the record.
    """
    aspects = caption_aspects_for_record(record)
    parts = [aspects[aspect_id] for aspect_id, _ in CAPTION_ASPECTS if aspect_id in selected_aspects and aspect_id in aspects]
    return " | ".join(parts)


def set_viser_dropdown_options(dropdown, options, initial_value=None):
    if hasattr(dropdown, "options"):
        try:
            dropdown.options = options
        except Exception as exc:
            print(f"[visualize] Could not update viser dropdown options: {exc}")
    if initial_value is not None:
        try:
            dropdown.value = initial_value
        except Exception as exc:
            print(f"[visualize] Could not update viser dropdown value: {exc}")


def list_available_object_ids(sample_records):
    return tuple(sorted({str(record["object_id"]) for record in sample_records if str(record["object_id"])}, key=natural_sort_key))


def list_available_base_object_ids(sample_records):
    return tuple(
        sorted(
            {base_object_id_from_sequence(record["object_id"]) for record in sample_records if str(record["object_id"])},
            key=natural_sort_key,
        )
    )


def list_available_grasp_type_ids(sample_records):
    return tuple(sorted({int(record["grasp_type_id"]) for record in sample_records if record["grasp_type_id"] is not None}))


def build_grasp_type_options(sample_records, grasp_type_names=None):
    """Build grasp type dropdown options without forcing sample payload reads.

    Args:
        sample_records: Visualization sample records, which may only contain
            path-derived metadata during fast startup.
        grasp_type_names: Optional ordered grasp type names from the dataset.

    Returns:
        A tuple of dropdown option strings. When dataset grasp names are known,
        all concrete grasp types are exposed immediately so startup does not
        need to read every saved sample.
    """
    if grasp_type_names is not None:
        return tuple(format_grasp_type_option(idx, grasp_type_names) for idx in range(1, len(grasp_type_names)))
    grasp_type_ids = list_available_grasp_type_ids(sample_records)
    return tuple(format_grasp_type_option(idx, grasp_type_names) for idx in grasp_type_ids) or ("",)


def populate_grasp_type_ids(sample_records):
    """Populate grasp type metadata for records indexed without full payloads.

    Args:
        sample_records: Visualization sample records created by
            ``build_visualization_sample_index``.

    Returns:
        The number of records whose grasp type id is available after the pass.
    """
    available_count = 0
    record_iter = progress_iter(sample_records, desc="Reading grasp type ids", total=len(sample_records))
    for record in record_iter:
        if record.get("grasp_type_id") is not None:
            available_count += 1
            continue
        data = record.get("data")
        if data is None:
            try:
                data = np.load(record["sample_file"], allow_pickle=True).item()
            except Exception as exc:
                print(f"[visualize] Could not read grasp type from {record['sample_file']}: {exc}")
                continue
        record["grasp_type_id"] = extract_grasp_type_id(data)
        if record["grasp_type_id"] is not None:
            available_count += 1
    return available_count


def build_selection_record_index(sample_records, is_our_human_grasp_format=False):
    """Build reusable record groups for responsive web UI selection.

    Args:
        sample_records: Visualization records indexed from saved sample files.
        is_our_human_grasp_format: Whether sequence-base object grouping is
            valid for ``one_object_multi_seq``.

    Returns:
        A dictionary containing sorted records grouped by exact object id, base
        object id, and random-object group id. Grasp-type groups are populated
        lazily because they require reading saved sample payloads.
    """
    records_by_object = {}
    records_by_base_object = {}
    records_by_random_group = {}
    record_iter = progress_iter(sample_records, desc="Building visualization selection index", total=len(sample_records))
    for record in record_iter:
        object_id = canonical_object_id(record["object_id"])
        base_object_id = base_object_id_from_sequence(object_id)
        random_group_id = base_object_id if is_our_human_grasp_format else object_id
        records_by_object.setdefault(object_id, []).append(record)
        records_by_base_object.setdefault(base_object_id, []).append(record)
        records_by_random_group.setdefault(random_group_id, []).append(record)

    for grouped_records in (
        list(records_by_object.values())
        + list(records_by_base_object.values())
        + list(records_by_random_group.values())
    ):
        grouped_records.sort(key=lambda record: record["sample_file"])

    return {
        "records": sample_records,
        "records_by_object": records_by_object,
        "records_by_base_object": records_by_base_object,
        "records_by_random_group": records_by_random_group,
        "random_group_ids": tuple(sorted(records_by_random_group.keys(), key=natural_sort_key)),
        "object_ids": tuple(sorted(records_by_object.keys(), key=natural_sort_key)),
        "base_object_ids": tuple(sorted(records_by_base_object.keys(), key=natural_sort_key)),
        "grasp_type_records": {},
        "grasp_type_scan_complete": False,
    }


def get_indexed_object_records(selection_index, object_id):
    """Return records for an object id using exact lookup with scan fallback.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        object_id: Object id selected in the web UI or provided by config.

    Returns:
        A sorted list of matching records.
    """
    object_id = canonical_object_id(object_id)
    exact_records = selection_index["records_by_object"].get(object_id)
    if exact_records is not None:
        return exact_records
    base_records = selection_index["records_by_base_object"].get(base_object_id_from_sequence(object_id))
    if base_records is not None:
        return base_records
    return sorted(
        [
            record
            for record in selection_index["records"]
            if object_id_matches(record["object_id"], object_id)
        ],
        key=lambda record: record["sample_file"],
    )


def select_grasp_type_records_lazy(selection_index, target_grasp_type_id: int, max_grasps: int, batch_index: int):
    """Randomly select grasp-type records while reading few payloads.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        target_grasp_type_id: Integer grasp type id to match.
        max_grasps: Maximum records to return. Non-positive values require a
            complete scan and return all matches.
        batch_index: Zero-based batch index. This is accepted for compatibility
            with Selection panel state; selection is randomized.

    Returns:
        A list of records matching the requested grasp type.
    """
    del batch_index
    target_grasp_type_id = int(target_grasp_type_id)
    target_records = selection_index["grasp_type_records"].setdefault(target_grasp_type_id, [])
    if max_grasps <= 0:
        populate_grasp_type_ids(selection_index["records"])
        target_records[:] = [record for record in selection_index["records"] if record["grasp_type_id"] == target_grasp_type_id]
        selection_index["grasp_type_scan_complete"] = True
        return randomize_records(target_records, max_grasps)

    cached_paths = {record["sample_file"] for record in target_records}
    selected = randomize_records(target_records, max_grasps)
    selected_paths = {record["sample_file"] for record in selected}
    if len(selected) < max_grasps and not selection_index["grasp_type_scan_complete"]:
        shuffled_records = list(selection_index["records"])
        random.shuffle(shuffled_records)
        record_iter = progress_iter(
            shuffled_records,
            desc=f"Scanning grasp type {target_grasp_type_id}",
            total=len(shuffled_records),
        )
        for record in record_iter:
            if record.get("grasp_type_id") is None:
                data = record.get("data")
                if data is None:
                    try:
                        data = np.load(record["sample_file"], allow_pickle=True).item()
                    except Exception as exc:
                        print(f"[visualize] Could not read grasp type from {record['sample_file']}: {exc}")
                        continue
                record["grasp_type_id"] = extract_grasp_type_id(data)
            if record["grasp_type_id"] != target_grasp_type_id:
                continue
            if record["sample_file"] not in cached_paths:
                target_records.append(record)
                cached_paths.add(record["sample_file"])
            if record["sample_file"] not in selected_paths:
                selected.append(record)
                selected_paths.add(record["sample_file"])
            if len(selected) >= max_grasps:
                break
        else:
            selection_index["grasp_type_scan_complete"] = True

    if len(selected) < max_grasps and target_records:
        selected_paths = {record["sample_file"] for record in selected}
        remaining = [record for record in target_records if record["sample_file"] not in selected_paths]
        selected.extend(randomize_records(remaining, max_grasps - len(selected)))
    return selected


def select_visualization_records_from_index(
    selection_index,
    mode: str,
    max_grasps: int,
    object_id=None,
    target_grasp_type_id=None,
    is_our_human_grasp_format=False,
    batch_index=0,
):
    """Select visualization records from precomputed web UI indexes.

    Args:
        selection_index: Selection index from ``build_selection_record_index``.
        mode: Visualization mode.
        max_grasps: Maximum number of records in one batch.
        object_id: Optional selected object id.
        target_grasp_type_id: Optional selected grasp type id.
        is_our_human_grasp_format: Whether ``one_object_multi_seq`` is valid.
        batch_index: Zero-based batch index.

    Returns:
        Selected and annotated sample records.
    """
    mode = normalize_visualize_mode(mode)
    batch_index = max(0, int(batch_index))
    if mode == "random_object":
        group_ids = randomize_records(selection_index["random_group_ids"], max_grasps)
        selected = []
        for group_id in group_ids:
            group_records = selection_index["records_by_random_group"][group_id]
            selected.append(random.choice(group_records))
    elif mode == "one_object":
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object.")
        selected = select_one_object_variant_batch(
            get_indexed_object_records(selection_index, object_id),
            max_grasps,
            batch_index,
        )
    elif mode == "one_object_multi_seq":
        if not is_our_human_grasp_format:
            raise RuntimeError(
                "task.visualize_mode=one_object_multi_seq is only valid when test_data.object_path points to "
                "OurHumanGraspFormat."
            )
        if object_id is None:
            raise ValueError("task.object_id must be set when task.visualize_mode=one_object_multi_seq.")
        base_object_id = base_object_id_from_sequence(object_id)
        selected = select_random_across_sequences(selection_index["records_by_base_object"].get(base_object_id, []), max_grasps)
    elif mode == "grasp_type":
        if target_grasp_type_id is None:
            raise ValueError("task.target_grasp_type_id must be set when task.visualize_mode=grasp_type.")
        selected = select_grasp_type_records_lazy(selection_index, int(target_grasp_type_id), max_grasps, batch_index)
    else:
        raise ValueError(f"Unsupported visualize_mode={mode}. Expected one of {list(VISUALIZE_MODE_OPTIONS)}.")

    if not selected:
        example_object_ids = selection_index["object_ids"][:10]
        example_text = ", ".join(example_object_ids) if example_object_ids else "none"
        raise RuntimeError(
            f"No samples matched visualize_mode={mode}, object_id={object_id}, "
            f"target_grasp_type_id={target_grasp_type_id}. Available object examples: {example_text}"
        )

    selected = annotate_scene_labels(selected, mode)
    print(f"[visualize] Selected {len(selected)} sample(s) with visualize_mode={mode}.")
    return selected


def format_grasp_type_option(grasp_type_id: int, grasp_type_names=None):
    if grasp_type_names is not None and 0 <= grasp_type_id < len(grasp_type_names):
        return f"{grasp_type_id}: {grasp_type_names[grasp_type_id]}"
    return str(grasp_type_id)


def parse_grasp_type_option(option: str):
    return int(str(option).split(":", 1)[0])


def pick_initial_option(options, preferred):
    if not options:
        return None
    if preferred is not None:
        preferred = str(preferred)
        for option in options:
            if str(option) == preferred or str(option).startswith(f"{preferred}:"):
                return option
    return options[0]


def pick_initial_object_option(options, preferred):
    if not options:
        return None
    if preferred is not None:
        for option in options:
            if object_id_matches(option, preferred) or sequence_object_id_matches(option, preferred):
                return option
    return options[0]


def format_gui_wrappable_value(value):
    """Format a long GUI value with safe wrap points.

    Args:
        value: Value to display in a markdown GUI panel.

    Returns:
        HTML-escaped text with zero-width break points inserted after common
        object-id separators so long DGN object names can wrap inside the side
        panel while preserving the visible object id.
    """
    text = str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for separator in ("/", "_", "-"):
        text = text.replace(separator, f"{separator}\u200b")
    text = re.sub(r"([A-Za-z0-9]{12})", lambda match: f"{match.group(1)}\u200b", text)
    return text


def build_viser_selection_controls(
    sample_records,
    config,
    build_scene_records,
    grasp_type_names=None,
    is_our_human_grasp_format=False,
):
    selection_index = build_selection_record_index(sample_records, is_our_human_grasp_format=is_our_human_grasp_format)
    mode_options = (
        VISUALIZE_MODE_OPTIONS
        if is_our_human_grasp_format
        else tuple(mode for mode in VISUALIZE_MODE_OPTIONS if mode != "one_object_multi_seq")
    )
    object_options = selection_index["object_ids"] or ("",)
    base_object_options = selection_index["base_object_ids"] or object_options
    grasp_type_options = build_grasp_type_options(sample_records, grasp_type_names)
    initial_mode = normalize_visualize_mode(get_task_value(config, "visualize_mode", "random_object"))
    if initial_mode not in mode_options:
        initial_mode = "random_object"
    initial_object_options = base_object_options if initial_mode in {"one_object", "one_object_multi_seq"} else object_options
    initial_object = pick_initial_object_option(initial_object_options, get_task_value(config, "object_id", None))
    initial_grasp_type = pick_initial_option(grasp_type_options, get_task_value(config, "target_grasp_type_id", None))
    max_grasps = int(get_task_value(config, "max_grasps", 20))
    batch_state = {"key": None, "index": 0}

    def load_scene_records(mode, object_id, grasp_type_option, advance_batch=False):
        mode = normalize_visualize_mode(mode)
        target_grasp_type_id = parse_grasp_type_option(grasp_type_option) if grasp_type_option else None
        selection_key = (mode, str(object_id), target_grasp_type_id)
        if selection_key != batch_state["key"]:
            batch_state["key"] = selection_key
            batch_state["index"] = 0
        elif advance_batch:
            batch_state["index"] += 1
        elif not advance_batch:
            batch_state["index"] = 0
        selected_records = select_visualization_records_from_index(
            selection_index,
            mode,
            max_grasps,
            object_id=object_id,
            target_grasp_type_id=target_grasp_type_id,
            is_our_human_grasp_format=is_our_human_grasp_format,
            batch_index=batch_state["index"],
        )
        return build_scene_records(selected_records)

    return {
        "mode_options": mode_options,
        "initial_mode": initial_mode,
        "object_options": object_options,
        "base_object_options": base_object_options,
        "initial_object": initial_object,
        "grasp_type_options": grasp_type_options,
        "initial_grasp_type": initial_grasp_type,
        "load_scene_records": load_scene_records,
        "batch_state": batch_state,
    }


def build_scene_grid_offsets(num_scenes: int, spacing: float):
    if num_scenes <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = int(np.ceil(np.sqrt(num_scenes)))
    rows = int(np.ceil(num_scenes / cols))
    offsets = []
    for idx in range(num_scenes):
        row = idx // cols
        col = idx % cols
        offsets.append(
            [
                (col - (cols - 1) / 2.0) * spacing,
                ((rows - 1) / 2.0 - row) * spacing,
                0.0,
            ]
        )
    return np.asarray(offsets, dtype=np.float32)


def build_grouped_scene_offsets(scene_records, spacing: float):
    group_to_indices = {}
    for idx, record in enumerate(scene_records):
        group_id = record.get("viser_spatial_group")
        if group_id:
            group_to_indices.setdefault(str(group_id), []).append(idx)

    if len(group_to_indices) <= 1:
        return build_scene_grid_offsets(len(scene_records), spacing)

    offsets = np.zeros((len(scene_records), 3), dtype=np.float32)
    group_ids = sorted(group_to_indices.keys(), key=natural_sort_key)
    cursor_x = 0.0

    for group_idx, group_id in enumerate(group_ids):
        indices = group_to_indices[group_id]
        local_offsets = build_scene_grid_offsets(len(indices), spacing)
        local_min_x = float(local_offsets[:, 0].min())
        local_max_x = float(local_offsets[:, 0].max())
        if group_idx == 0:
            group_origin_x = -local_min_x
        else:
            group_origin_x = cursor_x - local_min_x + 1.5 * spacing
        for local_idx, record_idx in enumerate(indices):
            offsets[record_idx] = local_offsets[local_idx] + np.asarray([group_origin_x, 0.0, 0.0], dtype=np.float32)
        cursor_x = group_origin_x + local_max_x

    return offsets - offsets.mean(axis=0, keepdims=True)


def transform_points(points, offset):
    return np.asarray(points, dtype=np.float32) + np.asarray(offset, dtype=np.float32).reshape(1, 3)


def transform_trimesh_geometry(geometry, offset):
    geometry = geometry.copy()
    geometry.apply_translation(np.asarray(offset, dtype=np.float32))
    return geometry


def add_scene_elements_to_viser(server, scene_elements, scene_name: str, offset):
    handles = []
    for idx, geometry in enumerate(scene_elements):
        name = f"{scene_name}/geometry_{idx}"
        if isinstance(geometry, trimesh.points.PointCloud):
            points = transform_points(geometry.vertices, offset)
            colors = getattr(geometry.visual, "vertex_colors", None)
            if colors is None or len(colors) == 0:
                colors = np.tile(np.asarray([[255, 0, 0]], dtype=np.uint8), (points.shape[0], 1))
            else:
                colors = np.asarray(colors)[:, :3].astype(np.uint8)
            handles.append(server.scene.add_point_cloud(name, points=points, colors=colors, point_size=0.003))
            continue

        if isinstance(geometry, trimesh.Trimesh):
            mesh = transform_trimesh_geometry(geometry, offset)
            handles.append(server.scene.add_mesh_trimesh(name, mesh=mesh))
            continue

        if isinstance(geometry, trimesh.Scene):
            for sub_idx, sub_geometry in enumerate(geometry.geometry.values()):
                sub_name = f"{name}/sub_geometry_{sub_idx}"
                if isinstance(sub_geometry, trimesh.Trimesh):
                    mesh = transform_trimesh_geometry(sub_geometry, offset)
                    handles.append(server.scene.add_mesh_trimesh(sub_name, mesh=mesh))
    return handles


def add_scene_label_to_viser(server, scene_name: str, caption: str, offset, log_prefix: str):
    label_position = np.asarray(offset, dtype=np.float32) + np.asarray([0.0, 0.0, 0.35], dtype=np.float32)
    if hasattr(server.scene, "add_label"):
        try:
            return server.scene.add_label(f"{scene_name}/label", text=caption, position=label_position)
        except TypeError:
            print(f"[{log_prefix}] {scene_name}: {caption}")
    else:
        print(f"[{log_prefix}] {scene_name}: {caption}")
    return None


def remove_viser_handles(handles):
    for handle in handles:
        if handle is not None and hasattr(handle, "remove"):
            handle.remove()


def show_scenes_with_trimesh(scene_records):
    for record in scene_records:
        scene = trimesh.Scene(record["elements"])
        camera_distance = record.get("camera_distance")
        if camera_distance is not None:
            scene.set_camera(
                angles=(np.deg2rad(60.0), 0.0, np.deg2rad(45.0)),
                distance=camera_distance,
                center=scene.centroid,
            )
        scene.show(caption=record["caption"], smooth=bool(record.get("smooth", True)))


def add_caption_gui(
    server,
    scene_records,
    log_prefix: str,
    on_show_full_caption=None,
    on_hide_full_caption=None,
    on_toggle_caption_aspect=None,
):
    if not hasattr(server, "gui") or not scene_records:
        return None

    def build_caption_state(records):
        captions = [record["caption"] for record in records]
        sample_names = [
            record.get("viser_all_label", compact_caption(caption))
            for record, caption in zip(records, captions)
        ]
        options = tuple(f"{idx:04d}: {name}" for idx, name in enumerate(sample_names))
        return captions, options

    captions, options = build_caption_state(scene_records)
    caption_state = {"captions": captions, "options": options, "visible": False}

    with server.gui.add_folder("Captions"):
        caption_dropdown = server.gui.add_dropdown("Sample", options=options, initial_value=options[0])
        show_button = server.gui.add_button("Show Full Caption")
        aspect_buttons = {}
        for aspect_id, aspect_name in CAPTION_ASPECTS:
            aspect_buttons[aspect_id] = server.gui.add_button(aspect_name)
        if hasattr(server.gui, "add_markdown"):
            caption_handle = server.gui.add_markdown("Caption hidden.")
            caption_mode = "markdown"
        elif hasattr(server.gui, "add_text"):
            caption_handle = server.gui.add_text("Full caption", initial_value="Caption hidden.")
            caption_mode = "text"
        else:
            caption_handle = None
            caption_mode = "print"

        def show_caption(index: int):
            text = caption_state["captions"][index]
            if caption_mode == "markdown" and hasattr(caption_handle, "content"):
                caption_handle.content = f"```text\n{text}\n```"
                caption_state["visible"] = True
            elif caption_mode == "text" and hasattr(caption_handle, "value"):
                caption_handle.value = text
                caption_state["visible"] = True
            else:
                print(f"[{log_prefix}] {caption_state['options'][index]}: {text}")
                caption_state["visible"] = True

        def hide_caption():
            if caption_mode == "markdown" and hasattr(caption_handle, "content"):
                caption_handle.content = "Caption hidden."
            elif caption_mode == "text" and hasattr(caption_handle, "value"):
                caption_handle.value = "Caption hidden."
            caption_state["visible"] = False

        @show_button.on_click
        def _on_show_caption(event):
            del event
            selected_idx = caption_state["options"].index(caption_dropdown.value)
            if caption_state["visible"]:
                hide_caption()
                if on_hide_full_caption is not None:
                    on_hide_full_caption()
                return
            show_caption(selected_idx)
            if on_show_full_caption is not None:
                on_show_full_caption(selected_idx)

        for aspect_id, aspect_button in aspect_buttons.items():
            @aspect_button.on_click
            def _on_toggle_aspect(event, aspect_id=aspect_id):
                del event
                if on_toggle_caption_aspect is not None:
                    on_toggle_caption_aspect(aspect_id)

        @caption_dropdown.on_update
        def _on_caption_select(event):
            del event
            selected_idx = caption_state["options"].index(caption_dropdown.value)
            if caption_state["visible"]:
                show_caption(selected_idx)

        def update_records(records, selected_idx: int):
            captions, options = build_caption_state(records)
            caption_state["captions"] = captions
            caption_state["options"] = options
            selected_idx = int(np.clip(selected_idx, 0, len(options) - 1))
            set_viser_dropdown_options(caption_dropdown, options, options[selected_idx])
            hide_caption()

    return {
        "dropdown": caption_dropdown,
        "show_caption": show_caption,
        "hide_caption": hide_caption,
        "update_records": update_records,
        "aspect_buttons": aspect_buttons,
    }


def show_scenes_with_viser(
    scene_records,
    port: int,
    scene_spacing: float,
    display_mode: str = "all",
    scene_id: int = 0,
    log_prefix: str = "visualize",
    selection_controls=None,
    next_batch_loader=None,
):
    if not VISER_AVAILABLE:
        raise ImportError(
            "viser is not installed. Install it in the active environment before using task.visualizer=viser."
        )
    if not scene_records:
        raise RuntimeError("No scene records to visualize.")

    display_mode = display_mode if display_mode in {"all", "single"} else "all"
    scene_id = int(np.clip(scene_id, 0, len(scene_records) - 1))

    server = viser.ViserServer(port=port)
    if hasattr(server.gui, "configure_theme"):
        server.gui.configure_theme(control_layout="floating", control_width="large")
    server.scene.set_up_direction("+z")
    scene_handles = {"value": []}

    records_state = {"records": scene_records}

    def build_sample_options(records):
        return tuple(
            f"{idx:04d}: {record.get('viser_all_label', compact_caption(record['caption']))}"
            for idx, record in enumerate(records)
        )

    sample_options_state = {"options": build_sample_options(records_state["records"])}
    state = {
        "display_mode": display_mode,
        "scene_id": scene_id,
        "show_full_scene_captions": False,
        "caption_aspects": set(),
    }

    with server.gui.add_folder("Display"):
        display_dropdown = server.gui.add_dropdown(
            "View",
            options=("all", "single"),
            initial_value=state["display_mode"],
        )
        scene_dropdown = server.gui.add_dropdown(
            "Scene",
            options=sample_options_state["options"],
            initial_value=sample_options_state["options"][state["scene_id"]],
        )

    def show_full_scene_captions(selected_idx: int):
        state["scene_id"] = int(np.clip(selected_idx, 0, len(records_state["records"]) - 1))
        state["show_full_scene_captions"] = True
        render_current_view()

    def hide_full_scene_captions():
        state["show_full_scene_captions"] = False
        render_current_view()

    def toggle_caption_aspect(aspect_id: str):
        if aspect_id in state["caption_aspects"]:
            state["caption_aspects"].remove(aspect_id)
        else:
            state["caption_aspects"].add(aspect_id)
        state["show_full_scene_captions"] = False
        render_current_view()

    caption_gui = add_caption_gui(
        server,
        scene_records,
        log_prefix,
        on_show_full_caption=show_full_scene_captions,
        on_hide_full_caption=hide_full_scene_captions,
        on_toggle_caption_aspect=toggle_caption_aspect,
    )

    def render_current_view():
        current_records = records_state["records"]
        remove_viser_handles(scene_handles["value"])
        scene_handles["value"] = []
        if state["display_mode"] == "single":
            records_and_offsets = [(state["scene_id"], current_records[state["scene_id"]], np.zeros(3, dtype=np.float32))]
        else:
            offsets = build_grouped_scene_offsets(current_records, scene_spacing)
            records_and_offsets = list(zip(range(len(current_records)), current_records, offsets))

        for idx, record, offset in records_and_offsets:
            scene_name = f"/sample_{idx:04d}"
            scene_handles["value"].extend(add_scene_elements_to_viser(server, record["elements"], scene_name, offset))
            if state["show_full_scene_captions"]:
                label_text = record["caption"]
            else:
                label_text = build_caption_from_aspects(record, state["caption_aspects"])
            label = add_scene_label_to_viser(server, scene_name, label_text, offset, log_prefix) if label_text else None
            if label is not None:
                scene_handles["value"].append(label)

        if hasattr(scene_dropdown, "disabled"):
            scene_dropdown.disabled = state["display_mode"] == "all"
        if caption_gui is not None:
            caption_gui["dropdown"].value = sample_options_state["options"][state["scene_id"]]
            if state["display_mode"] == "single":
                caption_gui["show_caption"](state["scene_id"])
            elif not state["show_full_scene_captions"]:
                caption_gui["hide_caption"]()

        if state["display_mode"] == "single":
            print(f"[{log_prefix}] Showing scene {state['scene_id']}: {compact_caption(current_records[state['scene_id']]['caption'])}")
        else:
            print(f"[{log_prefix}] Showing all {len(current_records)} scenes.")

    def update_scene_records(new_scene_records):
        if not new_scene_records:
            print(f"[{log_prefix}] Selection produced no scenes; keeping the current view.")
            return
        records_state["records"] = new_scene_records
        state["scene_id"] = int(np.clip(state["scene_id"], 0, len(new_scene_records) - 1))
        state["show_full_scene_captions"] = False
        state["caption_aspects"].clear()
        sample_options_state["options"] = build_sample_options(new_scene_records)
        set_viser_dropdown_options(
            scene_dropdown,
            sample_options_state["options"],
            sample_options_state["options"][state["scene_id"]],
        )
        if caption_gui is not None:
            caption_gui["update_records"](new_scene_records, state["scene_id"])
        render_current_view()

    if selection_controls is not None and hasattr(server, "gui"):
        def object_options_for_mode(mode):
            mode = normalize_visualize_mode(mode)
            if mode in {"one_object", "one_object_multi_seq"}:
                return selection_controls["base_object_options"]
            return selection_controls["object_options"]

        with server.gui.add_folder("Selection"):
            mode_dropdown = server.gui.add_dropdown(
                "Mode",
                options=selection_controls["mode_options"],
                initial_value=selection_controls["initial_mode"],
            )
            initial_object_options = object_options_for_mode(selection_controls["initial_mode"])
            object_dropdown = server.gui.add_dropdown(
                "Object",
                options=initial_object_options,
                initial_value=selection_controls["initial_object"],
            )
            object_info_handle = (
                server.gui.add_markdown("")
                if hasattr(server.gui, "add_markdown")
                else None
            )
            grasp_type_dropdown = server.gui.add_dropdown(
                "Grasp Type",
                options=selection_controls["grasp_type_options"],
                initial_value=selection_controls["initial_grasp_type"],
            )
            apply_button = server.gui.add_button("Apply Selection")
            next_batch_button = server.gui.add_button("Next Batch")
            status_handle = server.gui.add_markdown("Selection loaded.") if hasattr(server.gui, "add_markdown") else None

            def update_object_info():
                if object_info_handle is None or not hasattr(object_info_handle, "content"):
                    return
                object_info_handle.content = (
                    "Selected Object: "
                    f"{format_gui_wrappable_value(object_dropdown.value)}"
                )

            update_object_info()

            @apply_button.on_click
            def _on_apply_selection(event):
                del event
                try:
                    new_scene_records = selection_controls["load_scene_records"](
                        str(mode_dropdown.value),
                        str(object_dropdown.value),
                        str(grasp_type_dropdown.value),
                    )
                except Exception as exc:
                    if status_handle is not None and hasattr(status_handle, "content"):
                        status_handle.content = f"Selection failed: `{exc}`"
                    print(f"[{log_prefix}] Selection failed: {exc}")
                    return
                if status_handle is not None and hasattr(status_handle, "content"):
                    status_handle.content = f"Loaded {len(new_scene_records)} scene(s)."
                update_scene_records(new_scene_records)

            @next_batch_button.on_click
            def _on_next_selection_batch(event):
                del event
                try:
                    new_scene_records = selection_controls["load_scene_records"](
                        str(mode_dropdown.value),
                        str(object_dropdown.value),
                        str(grasp_type_dropdown.value),
                        advance_batch=True,
                    )
                except Exception as exc:
                    if status_handle is not None and hasattr(status_handle, "content"):
                        status_handle.content = f"Next batch failed: `{exc}`"
                    print(f"[{log_prefix}] Next batch failed: {exc}")
                    return
                batch_index = selection_controls.get("batch_state", {}).get("index", 0)
                if status_handle is not None and hasattr(status_handle, "content"):
                    status_handle.content = f"Loaded batch {batch_index + 1} with {len(new_scene_records)} scene(s)."
                update_scene_records(new_scene_records)

            @mode_dropdown.on_update
            def _on_selection_mode_update(event):
                del event
                options = object_options_for_mode(str(mode_dropdown.value))
                preferred = object_dropdown.value if hasattr(object_dropdown, "value") else None
                initial_object = pick_initial_object_option(options, preferred)
                set_viser_dropdown_options(object_dropdown, options, initial_object)
                update_object_info()

            @object_dropdown.on_update
            def _on_selection_object_update(event):
                del event
                update_object_info()

    if next_batch_loader is not None and hasattr(server, "gui"):
        with server.gui.add_folder("Dataloader"):
            next_batch_button = server.gui.add_button("Next Batch")
            batch_status = (
                server.gui.add_markdown("Current batch loaded.")
                if hasattr(server.gui, "add_markdown")
                else None
            )

            @next_batch_button.on_click
            def _on_next_batch(event):
                del event
                try:
                    new_scene_records = next_batch_loader()
                except Exception as exc:
                    if batch_status is not None and hasattr(batch_status, "content"):
                        batch_status.content = f"Next batch failed: `{exc}`"
                    print(f"[{log_prefix}] Next batch failed: {exc}")
                    return
                update_scene_records(new_scene_records)
                if batch_status is not None and hasattr(batch_status, "content"):
                    batch_status.content = f"Loaded next batch with {len(new_scene_records)} scene(s)."

    @display_dropdown.on_update
    def _on_display_update(event):
        del event
        state["display_mode"] = str(display_dropdown.value)
        render_current_view()

    @scene_dropdown.on_update
    def _on_scene_update(event):
        del event
        state["scene_id"] = sample_options_state["options"].index(scene_dropdown.value)
        if state["display_mode"] == "single":
            render_current_view()

    render_current_view()

    print(f"[{log_prefix}] Viser server running at http://localhost:{port}")
    print(f"[{log_prefix}] Rendering {len(records_state['records'])} scene(s). Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\n[{log_prefix}] Shutting down.")
    finally:
        remove_viser_handles(scene_handles["value"])


def infer_visualize_mode(config: DictConfig):
    configured = OmegaConf.select(config, "task.mode")
    if configured not in (None, "auto"):
        return str(configured)
    if bool(OmegaConf.select(config, "data.human")):
        return "human"
    if bool(OmegaConf.select(config, "test_data.human")):
        return "human"
    if OmegaConf.select(config, "test_data.robot_urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "test_data.robot.urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "data.robot_urdf_path") is not None:
        return "robot"
    if OmegaConf.select(config, "data.robot.urdf_path") is not None:
        return "robot"
    return "human"


def get_output_dir(config: DictConfig) -> str:
    return os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )


def resolve_human_dataset_path(path):
    """Replace legacy /AnyScaleGrasp/ prefixes with AnyScaleGraspDataset."""
    if "/AnyScaleGrasp/" in path:
        path = path.split("/AnyScaleGrasp/", 1)[1]
        dataset_root = os.environ.get("AnyScaleGraspDataset")
        if not dataset_root:
            raise ValueError("AnyScaleGraspDataset environment variable not set")
        return os.path.join(dataset_root, path)
    return path


def resolve_robot_dataset_path(path: str, object_root: str) -> str:
    if os.path.exists(path):
        return path

    if "/AnyScaleGrasp/" in path:
        relative_path = path.split("/AnyScaleGrasp/", 1)[1]
        dataset_root = os.environ.get("AnyScaleGraspDataset")
        if dataset_root:
            resolved = os.path.join(dataset_root, relative_path)
            if os.path.exists(resolved):
                return resolved

    if "/object/" in path and "/object/" in object_root:
        suffix = path.split("/object/", 1)[1]
        dataset_root = object_root.split("/object/", 1)[0]
        resolved = os.path.join(dataset_root, "object", suffix)
        if os.path.exists(resolved):
            return resolved

    return path


def visualize_with_trimesh(verts, faces, joints=None, color=[200, 200, 250, 255]):
    hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    hand_mesh.visual.face_colors = color
    scene_contents = [hand_mesh]
    if joints is not None:
        for joint in joints:
            sphere = trimesh.creation.uv_sphere(radius=0.005)
            sphere.visual.face_colors = [255, 0, 0, 255]
            sphere.apply_translation(joint)
            scene_contents.append(sphere)
    return scene_contents


def infer_pc_source_from_sample_file(sample_file):
    base_name = os.path.basename(sample_file)
    if base_name.startswith("complete_point_cloud"):
        return "complete"
    if base_name.startswith("partial_pc"):
        return "partial"

    try:
        sample_data = np.load(sample_file, allow_pickle=True).item()
    except Exception:
        return "unknown"
    pc_path = str(sample_data.get("pc_path", ""))
    pc_base = os.path.basename(pc_path)
    if "processed_data" in pc_path or pc_base == "complete_point_cloud.npy":
        return "complete"
    if "vision_data" in pc_path or pc_base.startswith("partial_pc"):
        return "partial"
    return "unknown"


def normalize_object_scale(obj_scale, scene_path):
    scale = np.asarray(obj_scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        return np.repeat(scale, 3)
    if scale.size == 3:
        return scale
    raise ValueError(f"Unsupported object scale shape {scale.shape} in scene config: {scene_path}")


def object_pose_to_rt(obj_pose, scene_path):
    from dexlearn.utils.rot import numpy_quaternion_to_matrix

    pose = np.asarray(obj_pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]

    pose = pose.reshape(-1)
    if pose.size == 7:
        rot = numpy_quaternion_to_matrix(pose[3:7].reshape(1, 4))[0].astype(np.float32)
        return rot, pose[:3]

    raise ValueError(f"Unsupported object pose shape {pose.shape} in scene config: {scene_path}")


def extract_object_meta(scene_cfg, scene_path):
    """Extract object name, scale, and pose transform from scene config."""
    if "object" in scene_cfg:
        obj_data = scene_cfg["object"]
        obj_name = obj_data.get("name")
        obj_scale = obj_data.get("rel_scale", obj_data.get("scale"))
        obj_pose = obj_data.get("pose")
    else:
        scene = scene_cfg.get("scene")
        if scene is None:
            raise KeyError(f"Could not find 'scene' in scene config: {scene_path}")

        obj_name = scene_cfg.get("task", {}).get("obj_name")
        if obj_name is None:
            candidates = [
                name
                for name, entry in scene.items()
                if isinstance(entry, dict) and "scale" in entry and "pose" in entry and name != "table"
            ]
            if not candidates:
                raise KeyError(f"Could not infer object name from scene config: {scene_path}")
            obj_name = candidates[0]

        if obj_name not in scene:
            raise KeyError(f"Object '{obj_name}' not found in scene config: {scene_path}")
        obj_data = scene[obj_name]
        obj_scale = obj_data.get("scale")
        obj_pose = obj_data.get("pose")

    if obj_name is None:
        raise KeyError(f"Could not find object name in scene config: {scene_path}")
    if obj_scale is None:
        raise KeyError(f"Could not find object scale for '{obj_name}' in scene config: {scene_path}")
    if obj_pose is None:
        raise KeyError(f"Could not find object pose for '{obj_name}' in scene config: {scene_path}")

    obj_scale_xyz = normalize_object_scale(obj_scale, scene_path)
    obj_rot, obj_trans = object_pose_to_rt(obj_pose, scene_path)
    return obj_name, obj_scale_xyz, obj_rot, obj_trans


def transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans):
    # complete_point_cloud.npy is normalized and must be transformed to world coordinates.
    scaled_pc = pc * obj_scale_xyz[None, :]
    return np.matmul(scaled_pc, obj_rot.T) + obj_trans[None, :]


def task_visualize_human(config: DictConfig) -> None:
    import torch
    from pytorch3d.transforms import matrix_to_axis_angle
    from manopth.manolayer import ManoLayer
    from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

    from dexlearn.dataset import GRASP_TYPES
    from dexlearn.utils.human_hand import get_wrist_translation_from_target, normalize_hand_pos_source
    from dexlearn.utils.logger import Logger
    from dexlearn.utils.util import set_seed

    set_seed(config.seed)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    Logger(config)
    visualizer = str(get_task_value(config, "visualizer", "trimesh")).lower()
    if visualizer not in {"trimesh", "viser"}:
        raise ValueError(f"Unsupported visualizer={visualizer}. Expected one of ['trimesh', 'viser'].")
    viser_port = int(get_task_value(config, "viser_port", 8080))
    viser_scene_spacing = float(get_task_value(config, "viser_scene_spacing", 0.8))
    viser_display_mode = str(get_task_value(config, "viser_display_mode", "all"))
    viser_scene_id = int(get_task_value(config, "viser_scene_id", 0))
    is_our_human_grasp_format = is_our_human_grasp_format_test_data(config)

    mano_layers = {}
    for side in ["left", "right"]:
        mano_layers[side] = ManoLayer(
            center_idx=0,
            mano_root="third_party/manopth/mano/models",
            side=side,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
        ).to(config.device)

    output_dir = get_output_dir(config)
    pc_source = str(getattr(config.test_data, "pc_source", "partial")).lower()
    if pc_source not in {"partial", "complete"}:
        raise ValueError(f"Unsupported pc_source={pc_source}. Expected one of ['partial', 'complete'].")

    all_sample_records = build_visualization_sample_index(
        output_dir,
        scene_path_resolver=resolve_human_dataset_path,
        load_payload=False,
    )
    if not all_sample_records:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    def build_human_scene_records(sample_records):
        scene_records = []
        record_iter = progress_iter(sample_records, desc="Building human visualization scenes", total=len(sample_records))
        for sample_record in record_iter:
            load_visualization_record_payload(sample_record, scene_path_resolver=resolve_human_dataset_path)
            sample_file = sample_record["sample_file"]

            data = sample_record["data"]
            grasp_pose = data["grasp_pose"]
            scene_path = sample_record["scene_path"]
            scene_cfg = sample_record["scene_cfg"]
            if not scene_cfg:
                raise RuntimeError(f"Scene config is unavailable for sample: {sample_file}")

            grasp_type_str = ""
            if "grasp_type_id" in data:
                gt_type_id = int(data["grasp_type_id"])
                grasp_type_str += f" | Given: {GRASP_TYPES[gt_type_id]}"
            if "pred_grasp_type_id" in data:
                pred_type_id = int(data["pred_grasp_type_id"])
                grasp_type_str += f" | Pred: {GRASP_TYPES[pred_type_id]}"
            if "pred_grasp_type_prob" in data:
                pred_grasp_type_prob = np.asarray(data["pred_grasp_type_prob"]).reshape(-1)
                prob_str = ", ".join([f"{p:.3f}" for p in pred_grasp_type_prob.tolist()])
                grasp_type_str += f" | pred_grasp_type_prob=[{prob_str}]"
            grasp_pos_source = normalize_hand_pos_source(data.get("grasp_pos_source", "wrist"))
            grasp_type_str += f" | Pos: {grasp_pos_source}"
            grasp_type_str += f" | PC: {infer_pc_source_from_sample_file(sample_file)}"

            if bool(OmegaConf.select(config, "test_data.human")):
                _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
                pc_path = resolve_human_dataset_path(data["pc_path"])
                raw_pc = np.load(pc_path, allow_pickle=True)
                idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
                pc = transform_complete_pc(raw_pc[idx], obj_scale_xyz, obj_rot, obj_trans)
            else:
                pc_path = resolve_human_dataset_path(data["pc_path"])
                raw_pc = np.load(pc_path, allow_pickle=True)
                idx = np.random.choice(raw_pc.shape[0], config.data.num_points, replace=True)
                pc = raw_pc[idx]
                if pc_source == "complete":
                    _, obj_scale_xyz, obj_rot, obj_trans = extract_object_meta(scene_cfg, scene_path)
                    pc = transform_complete_pc(pc, obj_scale_xyz, obj_rot, obj_trans)

            scene_elements = []
            poses = np.split(grasp_pose, len(grasp_pose) // 7)
            side_names = ["right", "left"]
            colors = [
                [180, 200, 255, 220],
                [210, 190, 250, 220],
            ]

            for i, p in enumerate(poses):
                if np.allclose(p, 0, atol=1e-3):
                    continue

                hand_pose_np = posQuat2Isometry3d(p[:3], quatWXYZ2XYZW(p[3:7]))
                side = side_names[i]
                m_layer = mano_layers[side]
                hand_target_pos = torch.from_numpy(p[:3]).to(config.device).float()
                hand_rot_mat = torch.from_numpy(hand_pose_np[:3, :3]).to(config.device).unsqueeze(0).float()

                mano_params = torch.cat(
                    [matrix_to_axis_angle(hand_rot_mat), torch.zeros((1, 45), device=config.device)], dim=-1
                )
                verts, joints = m_layer(mano_params, th_betas=torch.zeros((1, 10), device=config.device))
                wrist_trans = get_wrist_translation_from_target(hand_target_pos, joints[0], grasp_pos_source)

                hand_pose_np[:3, 3] = wrist_trans.cpu().numpy()
                scene_elements.append(trimesh.creation.axis(transform=hand_pose_np, origin_size=0.01))

                v_np = ((verts[0] / 1000.0) + wrist_trans).cpu().numpy()
                f_np = m_layer.th_faces.cpu().numpy()
                scene_elements.extend(visualize_with_trimesh(v_np, f_np, None, color=colors[i]))
                if grasp_pos_source == "index_mcp":
                    mcp_pose = np.eye(4)
                    mcp_pose[:3, :3] = hand_rot_mat[0].cpu().numpy()
                    mcp_pose[:3, 3] = hand_target_pos.cpu().numpy()
                    scene_elements.append(
                        trimesh.creation.axis(
                            transform=mcp_pose,
                            origin_size=0.008,
                            axis_radius=0.003,
                            axis_length=0.08,
                        )
                    )

            scene_elements.append(trimesh.points.PointCloud(pc, colors=[255, 0, 0, 255]))
            scene_elements.append(trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3))
            caption = prefix_caption_with_viser_label(sample_record, f"{os.path.basename(sample_file)}{grasp_type_str}")
            scene_record = {
                "elements": scene_elements,
                "caption": caption,
            }
            copy_viser_record_metadata(sample_record, scene_record)
            if visualizer == "trimesh":
                show_scenes_with_trimesh([scene_record])
            else:
                scene_records.append(scene_record)
        return scene_records

    if visualizer == "viser":
        selection_controls = build_viser_selection_controls(
            all_sample_records,
            config,
            build_human_scene_records,
            grasp_type_names=GRASP_TYPES,
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        scene_records = selection_controls["load_scene_records"](
            selection_controls["initial_mode"],
            selection_controls["initial_object"],
            selection_controls["initial_grasp_type"],
        )
        show_scenes_with_viser(
            scene_records,
            port=viser_port,
            scene_spacing=viser_scene_spacing,
            display_mode=viser_display_mode,
            scene_id=viser_scene_id,
            log_prefix="visualize_human",
            selection_controls=selection_controls,
        )
    else:
        initial_sample_records = select_visualization_records(
            all_sample_records,
            get_task_value(config, "visualize_mode", "random_object"),
            int(get_task_value(config, "max_grasps", 20)),
            get_task_value(config, "object_id", None),
            get_task_value(config, "target_grasp_type_id", None),
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        build_human_scene_records(initial_sample_records)


def build_robot_components(config: DictConfig):
    import pytorch_kinematics as pk

    from mr_utils.robot.pk_helper import PytorchKinematicsHelper
    from mr_utils.robot.pk_visualizer import Visualizer

    urdf_path = (
        OmegaConf.select(config, "test_data.robot_urdf_path")
        or OmegaConf.select(config, "data.robot_urdf_path")
        or get_task_value(config, "robot_urdf_path", None)
    )
    mesh_dir_path = (
        OmegaConf.select(config, "test_data.robot_mesh_dir_path")
        or OmegaConf.select(config, "data.robot_mesh_dir_path")
        or get_task_value(config, "robot_mesh_dir_path", None)
    )
    metadata_group = (
        OmegaConf.select(config, "test_data.metadata_group")
        or OmegaConf.select(config, "data.metadata_group")
        or get_task_value(config, "metadata_group", None)
    )
    if urdf_path is None:
        raise ValueError("Missing robot URDF path. Set test_data.robot_urdf_path or task.robot_urdf_path.")
    if mesh_dir_path is None:
        raise ValueError("Missing robot mesh dir path. Set test_data.robot_mesh_dir_path or task.robot_mesh_dir_path.")
    if metadata_group is None:
        raise ValueError("Missing robot metadata group. Set test_data.metadata_group or task.metadata_group.")

    chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=config.device)
    robot_helper = PytorchKinematicsHelper(
        chain,
        base_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=config.device,
    )
    robot_visualizer = Visualizer(urdf_path, mesh_dir_path, device=config.device)

    metadata_path = os.path.join(config.test_data.grasp_path, metadata_group, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Joint metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        joint_metadata = json.load(f)

    joint_names = joint_metadata["joint_names"]
    wrist_link_names = joint_metadata["wrist_body_names"]
    right_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("ra_")]
    left_arm_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("la_")]
    right_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("rh_")]
    left_hand_indices = [idx for idx, name in enumerate(joint_names) if name.startswith("lh_")]

    ik_solver_kwargs = dict(
        pos_tolerance=1e-3,
        rot_tolerance=1e-2,
        max_iterations=50,
        retry_configs=None,
        num_retries=20,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="any",
        debug=False,
        lr=0.2,
        regularlization=1e-3,
    )
    for wrist_link_name in wrist_link_names:
        robot_helper.create_serial_chain(wrist_link_name)
        robot_helper.create_ik_solver(wrist_link_name, **ik_solver_kwargs)

    return {
        "joint_names": joint_names,
        "wrist_link_names": wrist_link_names,
        "right_arm_indices": right_arm_indices,
        "left_arm_indices": left_arm_indices,
        "right_hand_indices": right_hand_indices,
        "left_hand_indices": left_hand_indices,
        "robot_helper": robot_helper,
        "robot_visualizer": robot_visualizer,
    }


def recover_pc_path(sample_file: str, scene_cfg: dict, config: DictConfig) -> str:
    original_name = re.sub(r"_\d+\.npy$", ".npy", os.path.basename(sample_file))
    return os.path.join(config.test_data.object_path, config.test_data.pc_path, scene_cfg["scene_id"], original_name)


def load_robot_visualization_pc(sample_file: str, scene_cfg: dict, config: DictConfig):
    pc_path = recover_pc_path(sample_file, scene_cfg, config)
    raw_pc = np.load(pc_path, allow_pickle=True)
    idx = np.random.choice(raw_pc.shape[0], config.test_data.num_points, replace=True)
    return raw_pc[idx]


def solve_stage_pose(stage_qpos: np.ndarray, robot_assets: dict, device: str):
    import torch
    from pytorch3d import transforms as pttf

    half_dim = stage_qpos.shape[0] // 2
    right_qpos = torch.from_numpy(stage_qpos[:half_dim]).to(device=device, dtype=torch.float32).unsqueeze(0)
    left_qpos = torch.from_numpy(stage_qpos[half_dim:]).to(device=device, dtype=torch.float32).unsqueeze(0)

    right_trans, right_quat, right_joint = right_qpos[:, :3], right_qpos[:, 3:7], right_qpos[:, 7:]
    left_trans, left_quat, left_joint = left_qpos[:, :3], left_qpos[:, 3:7], left_qpos[:, 7:]

    right_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    right_matrix[:, :3, :3] = pttf.quaternion_to_matrix(right_quat)
    right_matrix[:, :3, 3] = right_trans

    left_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    left_matrix[:, :3, :3] = pttf.quaternion_to_matrix(left_quat)
    left_matrix[:, :3, 3] = left_trans

    right_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][0], right_matrix)
    left_sol = robot_assets["robot_helper"].solve_ik_batch(robot_assets["wrist_link_names"][1], left_matrix)

    combined_q = torch.zeros((1, len(robot_assets["joint_names"])), device=device, dtype=torch.float32)
    right_q = right_sol["q"].to(device=device, dtype=combined_q.dtype)
    left_q = left_sol["q"].to(device=device, dtype=combined_q.dtype)
    combined_q[:, robot_assets["right_arm_indices"]] = right_q[:, robot_assets["right_arm_indices"]]
    combined_q[:, robot_assets["left_arm_indices"]] = left_q[:, robot_assets["left_arm_indices"]]
    combined_q[:, robot_assets["right_hand_indices"]] = right_joint
    combined_q[:, robot_assets["left_hand_indices"]] = left_joint

    robot_pose = torch.zeros((1, 7 + combined_q.shape[-1]), device=device, dtype=torch.float32)
    robot_pose[:, 3] = 1.0
    robot_pose[:, 7:] = combined_q
    return robot_pose, right_sol["success"][0].item(), left_sol["success"][0].item()


def task_visualize_robot(config: DictConfig) -> None:
    from dexlearn.dataset import GRASP_TYPES
    from dexlearn.utils.logger import Logger
    from dexlearn.utils.util import set_seed

    set_seed(config.seed)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    Logger(config)
    visualizer = str(get_task_value(config, "visualizer", "trimesh")).lower()
    if visualizer not in {"trimesh", "viser"}:
        raise ValueError(f"Unsupported visualizer={visualizer}. Expected one of ['trimesh', 'viser'].")
    viser_port = int(get_task_value(config, "viser_port", 8080))
    viser_scene_spacing = float(get_task_value(config, "viser_scene_spacing", 1.0))
    viser_display_mode = str(get_task_value(config, "viser_display_mode", "all"))
    viser_scene_id = int(get_task_value(config, "viser_scene_id", 0))
    is_our_human_grasp_format = is_our_human_grasp_format_test_data(config)

    output_dir = get_output_dir(config)
    robot_scene_path_resolver = lambda path: resolve_robot_dataset_path(path, config.test_data.object_path)
    all_sample_records = build_visualization_sample_index(
        output_dir,
        scene_path_resolver=robot_scene_path_resolver,
        load_payload=False,
    )
    if not all_sample_records:
        raise RuntimeError(f"No saved grasp files found in {output_dir}")

    robot_assets = build_robot_components(config)
    stage_names = ["grasp_qpos"]
    stage_colors = [(255, 120, 120, 180)]

    def build_robot_scene_records(sample_records):
        scene_records = []
        record_iter = progress_iter(sample_records, desc="Building robot visualization scenes", total=len(sample_records))
        for sample_record in record_iter:
            load_visualization_record_payload(sample_record, scene_path_resolver=robot_scene_path_resolver)
            sample_file = sample_record["sample_file"]

            data = sample_record["data"]
            scene_path = sample_record["scene_path"]
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene config not found: {scene_path}")
            scene_cfg = sample_record["scene_cfg"]
            if not scene_cfg:
                raise RuntimeError(f"Scene config is unavailable for sample: {sample_file}")

            pc = load_robot_visualization_pc(sample_file, scene_cfg, config)
            scene_elements = [
                trimesh.points.PointCloud(pc, colors=[255, 165, 0, 255]),
                trimesh.creation.axis(origin_size=0.01, axis_radius=0.001, axis_length=0.3),
            ]

            ik_status = []
            for stage_name, color in zip(stage_names, stage_colors):
                robot_pose, right_ok, left_ok = solve_stage_pose(data[stage_name], robot_assets, config.device)
                robot_assets["robot_visualizer"].set_robot_parameters(robot_pose, joint_names=robot_assets["joint_names"])
                scene_elements.append(robot_assets["robot_visualizer"].get_robot_trimesh_data(i=0, color=color))
                ik_status.append(f"{stage_name} IK: R={int(right_ok)} L={int(left_ok)}")

            caption = f"{os.path.basename(sample_file)} | err={float(data['grasp_error']):.4f}"
            if "grasp_type_id" in data:
                grasp_type_id = int(data["grasp_type_id"])
                caption = f"{caption} | grasp_type_id={GRASP_TYPES[grasp_type_id]}"
            if "pred_grasp_type_id" in data:
                pred_grasp_type_id = int(data["pred_grasp_type_id"])
                caption = f"{caption} | pred_grasp_type_id={GRASP_TYPES[pred_grasp_type_id]}"
            if "pred_grasp_type_prob" in data:
                pred_grasp_type_prob = np.asarray(data["pred_grasp_type_prob"]).reshape(-1)
                prob_str = ", ".join([f"{p:.3f}" for p in pred_grasp_type_prob.tolist()])
                caption = f"{caption} | pred_grasp_type_prob=[{prob_str}]"
            if bool(get_task_value(config, "show_ik_status", True)):
                caption = f"{caption} | {' ; '.join(ik_status)}"
            caption = prefix_caption_with_viser_label(sample_record, caption)

            scene_record = {
                "elements": scene_elements,
                "caption": caption,
                "camera_distance": float(get_task_value(config, "camera_distance", 1.0)),
                "smooth": False,
            }
            copy_viser_record_metadata(sample_record, scene_record)
            if visualizer == "trimesh":
                show_scenes_with_trimesh([scene_record])
            else:
                scene_records.append(scene_record)
        return scene_records

    if visualizer == "viser":
        selection_controls = build_viser_selection_controls(
            all_sample_records,
            config,
            build_robot_scene_records,
            grasp_type_names=GRASP_TYPES,
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        scene_records = selection_controls["load_scene_records"](
            selection_controls["initial_mode"],
            selection_controls["initial_object"],
            selection_controls["initial_grasp_type"],
        )
        show_scenes_with_viser(
            scene_records,
            port=viser_port,
            scene_spacing=viser_scene_spacing,
            display_mode=viser_display_mode,
            scene_id=viser_scene_id,
            log_prefix="visualize_robot",
            selection_controls=selection_controls,
        )
    else:
        initial_sample_records = select_visualization_records(
            all_sample_records,
            get_task_value(config, "visualize_mode", "random_object"),
            int(get_task_value(config, "max_grasps", 20)),
            get_task_value(config, "object_id", None),
            get_task_value(config, "target_grasp_type_id", None),
            is_our_human_grasp_format=is_our_human_grasp_format,
        )
        build_robot_scene_records(initial_sample_records)


def task_visualize(config: DictConfig) -> None:
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    mode = infer_visualize_mode(config)
    if mode == "human":
        task_visualize_human(config)
        return
    if mode == "robot":
        task_visualize_robot(config)
        return
    raise ValueError(f"Unsupported visualization mode={mode}. Expected 'auto', 'human', or 'robot'.")


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    task_visualize(config)


if __name__ == "__main__":
    main()
