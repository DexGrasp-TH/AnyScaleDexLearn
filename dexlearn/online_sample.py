import sys
import os
import argparse
from os.path import join as pjoin
from glob import glob
import random
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.utils.config import flatten_multidex_data_config, resolve_type_supervision_config
from dexlearn.dataset import create_test_dataloader
from dexlearn.dataset import minkowski_collate_fn
from dexlearn.network.models import *

def main_func(config: DictConfig) -> None:
    resolve_type_supervision_config(config)
    set_seed(config.seed)
    flatten_multidex_data_config(config.data)
    config.wandb.mode = "disabled"
    logger = Logger(config)

    sp_voxel_size = (
        config.algo.model.backbone.voxel_size
        if "MinkUNet" in config.algo.model.backbone.name
        else None
    )

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


    # assemble input data
    ret_dict = {}

    # read point cloud
    pc_path = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexLearn/assets/object/DGN_2k/vision_data/azure_kinect_dk/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose000_0/partial_pc_00.npy"
    raw_pc = np.load(pc_path, allow_pickle=True)
    idx = np.random.choice(
        raw_pc.shape[0], config.data.num_points, replace=True
    )
    pc = raw_pc[idx]

    # centering the point cloud
    pc_centroid = np.mean(pc, axis=-2, keepdims=True)
    pc = pc - pc_centroid # normalization

    ret_dict["point_clouds"] = pc  # (N, 3)
    ret_dict["grasp_type_id"] = 0 # not used
    if sp_voxel_size is not None:
        ret_dict["coors"] = pc / sp_voxel_size  # (N, 3)
        ret_dict["feats"] = pc  # (N, 3)

    data = minkowski_collate_fn([ret_dict])
    for k, v in data.items():
        if (
            "Int" not in v.type()
            and "Long" not in v.type()
            and "Short" not in v.type()
        ):
            v = v.float()
        data[k] = v.to(config.device)


    # model inference
    with torch.no_grad():
        result = model.sample(data, config.algo.test_grasp_num)
        if isinstance(result, dict):
            pred_grasp_type_prob = result["pred_grasp_type_prob"]
            pred_grasp_type_id = result.get("pred_grasp_type_id")
            save_dict = {
                "pred_grasp_type_prob": pred_grasp_type_prob.detach().cpu().numpy(),
            }
            if pred_grasp_type_id is not None:
                save_dict["pred_grasp_type_id"] = pred_grasp_type_id.detach().cpu().numpy()

            save_dir = "output/test"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "test.npy"), save_dict)
            return

        robot_pose, log_prob = result
        # select top k predictions with higher log_prob
        topk_indices = torch.topk(log_prob, config.algo.test_topk, dim=1).indices
        batch_indices = (
            torch.arange(robot_pose.size(0))
            .unsqueeze(1)
            .expand(-1, config.algo.test_topk)
        )
        robot_pose = robot_pose[batch_indices, topk_indices]
        log_prob = log_prob[batch_indices, topk_indices]

        save_dict = {
            "pregrasp_qpos": robot_pose[..., 0, :].detach().cpu().numpy(),
            "grasp_qpos": robot_pose[..., 1, :].detach().cpu().numpy(),
            "squeeze_qpos": robot_pose[..., 2, :].detach().cpu().numpy(),
            "grasp_error": -log_prob.detach().cpu().numpy(),
        }

        save_dir = "output/test"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "test.npy"), save_dict)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        required=True,
        help="experiment folder, e.g. shadow_tabletop_debug",
    )
    args, unknown = parser.parse_known_args()

    sys.argv = (
        sys.argv[:1]
        + unknown
        + list(OmegaConf.load(f"output/{args.exp_name}/.hydra/overrides.yaml"))
    )

    # remove duplicated args. Note: cmd has the priority!
    check_dict = {}
    for argv in sys.argv[1:]:
        arg_key = argv.split("=")[0]
        if arg_key not in check_dict:
            check_dict[arg_key] = True
        else:
            sys.argv.remove(argv)

    hydra.main(config_path="config", config_name="base", version_base=None)(main_func)()
