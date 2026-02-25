import sys
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_train_dataloader
from dexlearn.network.models import *


def task_train(config: DictConfig):
    set_seed(config.seed)
    logger = Logger(config)
    train_loader, val_loader = create_train_dataloader(config)

    model = eval(config.algo.model.name)(config.algo.model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.algo.lr)
    scheduler = CosineAnnealingLR(optimizer, config.algo.max_iter, eta_min=config.algo.lr_min)

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(config.device)
        optimizer.load_state_dict(ckpt["optimizer"])
        cur_iter = ckpt["iter"]
        for _ in range(cur_iter):
            scheduler.step()
        print(f"loaded ckpt from {config.ckpt}")
    else:
        cur_iter = 0

    # training
    model.to(config.device)
    model.train()

    for it in trange(cur_iter, config.algo.max_iter):
        optimizer.zero_grad()
        data = train_loader.get()
        result_dict = model(data)
        loss = 0
        for k, v in result_dict.items():
            if k in config.algo.loss_weight:
                loss += config.algo.loss_weight[k] * v
            elif "loss" in k:
                print(f"{k} is not used in loss!")
        loss.backward()
        debug_flag = False
        for p in model.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                try:
                    if torch.isnan(p.grad).any():
                        p.grad.zero_()
                        debug_flag = True
                except Exception as e:
                    print("Wrong p", p)
                    print("grad", p.grad)
                    print("shape", p.shape)
                    raise e

        if debug_flag:
            print("grad is nan!")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.algo.grad_clip)
        if it == 0:
            continue
        optimizer.step()
        scheduler.step()

        result_dict["lr"] = torch.tensor(scheduler.get_last_lr())
        if (it + 1) % config.algo.log_every == 0:
            logger.log({k: v.mean().item() for k, v in result_dict.items()}, "train", it)

        if (it + 1) % config.algo.save_every == 0:
            logger.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iter=it + 1,
                ),
                it + 1,
            )

        if (it + 1) % config.algo.val_every == 0:
            with torch.no_grad():
                model.eval()
                result_dicts = []
                for _ in range(config.algo.val_num):
                    data = val_loader.get()
                    result_dict = model(data)
                    result_dicts.append(result_dict)
                logger.log(
                    {
                        k: torch.cat([(dic[k] if len(dic[k].shape) else dic[k][None]) for dic in result_dicts]).mean()
                        for k in result_dicts[0].keys()
                    },
                    "eval",
                    it,
                )
                model.train()

    return
