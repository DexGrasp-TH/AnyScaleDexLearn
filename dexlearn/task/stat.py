import sys
import os
from glob import glob
import numpy as np
from collections import Counter
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.dataset import GRASP_TYPES


def task_stat(config: DictConfig) -> None:
    set_seed(config.seed)
    logger = Logger(config)

    output_dir = os.path.join(
        config.ckpt.replace("ckpts", "tests").replace(".pth", ""),
        config.test_data.name,
    )

    all_files = glob(os.path.join(output_dir, "**/*.npy"), recursive=True)
    print(f"Found {len(all_files)} result files")

    pred_types = []
    gt_types = []

    for file_path in all_files:
        data = np.load(file_path, allow_pickle=True).item()
        if "pred_grasp_type_id" in data:
            pred_types.append(int(data["pred_grasp_type_id"]))
        if "grasp_type_id" in data:
            gt_types.append(int(data["grasp_type_id"]))

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
