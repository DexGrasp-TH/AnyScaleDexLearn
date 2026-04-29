import os
from glob import glob
import numpy as np
from collections import Counter
from copy import deepcopy
import importlib.util
import random

try:
    from omegaconf import DictConfig, ListConfig
    from dexlearn.utils.config import cfg_get, flatten_multidex_data_config
except ModuleNotFoundError:
    DictConfig = object
    ListConfig = list

    def cfg_get(config, *keys, default=None):
        for key in keys:
            if hasattr(config, key):
                return getattr(config, key)
        return default

    def flatten_multidex_data_config(config):
        return config


DEXLEARN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_grasp_types():
    grasp_types_path = os.path.join(DEXLEARN_ROOT, "dataset", "grasp_types.py")
    spec = importlib.util.spec_from_file_location("dexlearn_grasp_types", grasp_types_path)
    grasp_types_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grasp_types_module)
    return grasp_types_module.GRASP_TYPES


GRASP_TYPES = _load_grasp_types()


TARGET_GRASP_TYPE_IDS = list(range(1, len(GRASP_TYPES)))


def _load_json(path: str):
    import json

    with open(path, "r") as f:
        return json.load(f)


def _iter_data_configs(data_config: DictConfig):
    flatten_multidex_data_config(data_config)
    object_path = cfg_get(data_config, "object_path", "paths.object_path")
    if isinstance(object_path, ListConfig):
        for object_path in object_path:
            single_config = deepcopy(data_config)
            single_config.object_path = object_path
            flatten_multidex_data_config(single_config)
            yield single_config
    else:
        yield data_config


def _human_grasp_type_id(grasp_data: dict) -> int:
    l_contacts = grasp_data["hand"]["left"]["contacts"] if grasp_data["hand"]["left"] else [False] * 5
    r_contacts = grasp_data["hand"]["right"]["contacts"] if grasp_data["hand"]["right"] else [False] * 5
    has_l, has_r = any(l_contacts), any(r_contacts)
    l_count, r_count = sum(l_contacts), sum(r_contacts)

    if not (has_l or has_r):
        raise ValueError("Grasp data has no contact")

    if has_l and has_r:
        return 5 if (l_count > 3 or r_count > 3) else 4

    # Left-only human grasps are mirrored into the same right-hand type classes used by training.
    count = l_count if has_l else r_count
    if count <= 2:
        return 1
    if count == 3:
        return 2
    return 3


def _robot_grasp_type_id(grasp_data: dict) -> int:
    raw_grasp_type = np.asarray(grasp_data["grasp_type"]).reshape(-1)[0]
    if isinstance(raw_grasp_type, bytes):
        raw_grasp_type = raw_grasp_type.decode("utf-8")
    else:
        raw_grasp_type = str(raw_grasp_type)

    grasp_type = next((gt for gt in GRASP_TYPES if gt.endswith(raw_grasp_type)), GRASP_TYPES[0])
    return int(grasp_type.split("_", 1)[0])


def _collect_human_train_types(data_config: DictConfig):
    split_path = os.path.join(data_config.object_path, data_config.split_path, "train.json")
    obj_ids = _load_json(split_path)

    grasp_type_ids = []
    for obj_id in obj_ids:
        grasp_paths = sorted(glob(os.path.join(data_config.grasp_path, obj_id, "**/*.npy"), recursive=True))
        for grasp_path in grasp_paths:
            grasp_data = np.load(grasp_path, allow_pickle=True).item()
            grasp_type_ids.append(_human_grasp_type_id(grasp_data))
    return grasp_type_ids


def _collect_robot_train_types(data_config: DictConfig):
    split_path = os.path.join(data_config.object_path, data_config.split_path, "train.json")
    obj_ids = _load_json(split_path)

    grasp_type_ids = []
    for obj_id in obj_ids:
        grasp_paths = sorted(glob(os.path.join(data_config.grasp_path, "*", obj_id, "**/*.npy"), recursive=True))
        for grasp_path in grasp_paths:
            grasp_data = np.load(grasp_path, allow_pickle=True).item()
            grasp_type_ids.append(_robot_grasp_type_id(grasp_data))
    return grasp_type_ids


def _collect_train_grasp_types(config: DictConfig):
    grasp_type_ids = []
    for data_config in _iter_data_configs(config.data):
        if data_config.dataset_type == "HumanMultiDexDataset":
            grasp_type_ids.extend(_collect_human_train_types(data_config))
        elif data_config.dataset_type == "RobotMultiDexDataset":
            grasp_type_ids.extend(_collect_robot_train_types(data_config))
        else:
            raise NotImplementedError(
                f"task=stat only supports exact train distribution for HumanMultiDexDataset "
                f"and RobotMultiDexDataset, got {data_config.dataset_type}"
            )
    return grasp_type_ids


def _resolve_ckpt_path(config: DictConfig) -> str:
    if config.ckpt is not None and os.path.exists(str(config.ckpt)):
        return str(config.ckpt)

    ckpt_dir = os.path.join(config.output_folder, config.wandb.id, "ckpts")
    if config.ckpt is None:
        all_ckpts = sorted(glob(os.path.join(ckpt_dir, "step_**.pth")))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        return all_ckpts[-1]

    return os.path.join(ckpt_dir, f"step_{config.ckpt}.pth")


def _extract_type_ids(value):
    return [int(v) for v in np.asarray(value).reshape(-1)]


def _valid_counter(grasp_type_ids):
    counter = Counter()
    ignored_counter = Counter()
    for grasp_type_id in grasp_type_ids:
        if grasp_type_id in TARGET_GRASP_TYPE_IDS:
            counter[grasp_type_id] += 1
        else:
            ignored_counter[grasp_type_id] += 1
    return counter, ignored_counter


def _print_distribution(counter: Counter, title: str):
    total = sum(counter.values())
    print(f"\n{title}:")
    print(f"  {'grasp_type':<16} {'count':>10} {'percent':>10}")
    for type_id in TARGET_GRASP_TYPE_IDS:
        count = counter[type_id]
        percentage = count / total * 100 if total else 0.0
        print(f"  {GRASP_TYPES[type_id]:<16} {count:>10} {percentage:>9.1f}%")
    print(f"  {'total':<16} {total:>10} {100.0 if total else 0.0:>9.1f}%")


def _print_comparison(train_counter: Counter, sample_counter: Counter, sample_title: str):
    train_total = sum(train_counter.values())
    sample_total = sum(sample_counter.values())

    print("\nOverall Grasp Type Distribution Comparison:")
    print(
        f"  {'grasp_type':<16} "
        f"{'train_count':>12} {'train_%':>9} "
        f"{'sample_count':>13} {'sample_%':>9}"
    )
    for type_id in TARGET_GRASP_TYPE_IDS:
        train_count = train_counter[type_id]
        sample_count = sample_counter[type_id]
        train_pct = train_count / train_total * 100 if train_total else 0.0
        sample_pct = sample_count / sample_total * 100 if sample_total else 0.0
        print(
            f"  {GRASP_TYPES[type_id]:<16} "
            f"{train_count:>12} {train_pct:>8.1f}% "
            f"{sample_count:>13} {sample_pct:>8.1f}%"
        )
    print(
        f"  {'total':<16} "
        f"{train_total:>12} {100.0 if train_total else 0.0:>8.1f}% "
        f"{sample_total:>13} {100.0 if sample_total else 0.0:>8.1f}%"
    )
    print(f"\nSampled distribution source: {sample_title}")


def task_stat(config: DictConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    flatten_multidex_data_config(config.data)
    flatten_multidex_data_config(config.test_data)
    ckpt_path = _resolve_ckpt_path(config)

    output_dir = os.path.join(
        ckpt_path.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )

    train_types = _collect_train_grasp_types(config)
    train_counter, ignored_train_counter = _valid_counter(train_types)
    _print_distribution(train_counter, "Training Data Grasp Type Distribution")
    if ignored_train_counter:
        print(f"  Ignored non-target train type ids: {dict(sorted(ignored_train_counter.items()))}")

    all_files = glob(os.path.join(output_dir, "**/*.npy"), recursive=True)
    print(f"\nFound {len(all_files)} sampled result files in {output_dir}")

    pred_types = []
    gt_types = []

    for file_path in all_files:
        data = np.load(file_path, allow_pickle=True).item()
        if "pred_grasp_type_id" in data:
            pred_types.extend(_extract_type_ids(data["pred_grasp_type_id"]))
        if "grasp_type_id" in data:
            gt_types.extend(_extract_type_ids(data["grasp_type_id"]))

    sample_title = "pred_grasp_type_id" if pred_types else "grasp_type_id"
    sample_types = pred_types if pred_types else gt_types
    sample_counter, ignored_sample_counter = _valid_counter(sample_types)
    _print_distribution(sample_counter, "Sampled Data Grasp Type Distribution")
    if ignored_sample_counter:
        print(f"  Ignored non-target sampled type ids: {dict(sorted(ignored_sample_counter.items()))}")

    _print_comparison(train_counter, sample_counter, sample_title)

    # Keep the legacy sections because they are useful when checking saved sample metadata.
    if pred_types:
        print("\nPredicted Grasp Type Distribution:")
        pred_counter = Counter(pred_types)
        for type_id in sorted(pred_counter.keys()):
            count = pred_counter[type_id]
            percentage = count / len(pred_types) * 100
            print(f"  {GRASP_TYPES[type_id]}: {count} ({percentage:.1f}%)")

    if gt_types:
        print("\nGround Truth Grasp Type Distribution:")
        gt_counter = Counter(gt_types)
        for type_id in sorted(gt_counter.keys()):
            count = gt_counter[type_id]
            percentage = count / len(gt_types) * 100
            print(f"  {GRASP_TYPES[type_id]}: {count} ({percentage:.1f}%)")
