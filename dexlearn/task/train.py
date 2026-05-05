import sys
import os
import time
import textwrap
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.config import resolve_type_supervision_config
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_train_dataloader
from dexlearn.network.models import *


def _maybe_cuda_sync(device: str, enabled: bool):
    if enabled and isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _markdown_text(value) -> str:
    """Escape a value for safe use inside the markdown registry.

    Args:
        value: Object to stringify and place into the markdown registry.

    Returns:
        A markdown-safe string.
    """
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _summarize_model_features(config: DictConfig) -> str:
    """Build a compact feature summary for the model registry.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        Human-readable summary of key options that distinguish this run.
    """
    user_note = OmegaConf.select(config, "model_registry.key_features")
    if user_note:
        return str(user_note)

    parts = [
        f"data={config.data_name}",
        f"algo={config.algo_name}",
        f"max_iter={config.algo.max_iter}",
    ]
    pc_random = OmegaConf.select(config, "data.point_cloud.random_pc_across_sequences")
    if pc_random is not None:
        parts.append(f"random_pc_across_sequences={bool(pc_random)}")
    scale_enabled = OmegaConf.select(config, "data.augmentation.scale.enabled")
    if scale_enabled is not None:
        scale_min = OmegaConf.select(config, "data.augmentation.scale.min")
        scale_max = OmegaConf.select(config, "data.augmentation.scale.max")
        parts.append(f"scale_aug={bool(scale_enabled)}[{scale_min},{scale_max}]")
    train_sampling_unit = OmegaConf.select(config, "data.sampling.train_unit")
    if train_sampling_unit is not None:
        parts.append(f"train_sampling_unit={train_sampling_unit}")
    type_objective = OmegaConf.select(config, "algo.model.type_objective")
    if type_objective is not None:
        scope = OmegaConf.select(config, "algo.supervision.scope")
        negative_policy = OmegaConf.select(config, "algo.supervision.negative_policy")
        parts.append(f"type_objective={type_objective}[scope={scope},negative={negative_policy}]")
    type_balance_enabled = OmegaConf.select(config, "algo.supervision.balancing.enabled")
    if type_balance_enabled is not None:
        sampler_alpha = OmegaConf.select(config, "algo.supervision.balancing.sampler.alpha")
        loss_beta = OmegaConf.select(config, "algo.supervision.balancing.loss_weight.beta")
        parts.append(
            f"type_balancing={bool(type_balance_enabled)}"
            f"[sampler_alpha={sampler_alpha},loss_beta={loss_beta}]"
        )
    return "; ".join(parts)


def _format_feature_lines(summary: str, width: int = 76) -> list[str]:
    """Format a feature summary as short markdown bullet lines."""
    items = [part.strip() for part in str(summary).split(";") if part.strip()]
    if not items:
        return ["- Key Features:", "  -"]

    lines = ["- Key Features:"]
    for item in items:
        wrapped = textwrap.wrap(
            _markdown_text(item),
            width=width,
            initial_indent="  - ",
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or ["  -"])
    return lines


def _build_model_registry_entry(
    exp_name: str,
    timestamp: str,
    data_name: str,
    algo_name: str,
    max_iter,
    feature_summary: str,
    notes: str = "",
) -> str:
    """Build one markdown registry entry with wrapped lines."""
    lines = [
        f"## {_markdown_text(exp_name)}",
        f"- Timestamp: `{_markdown_text(timestamp)}`",
        f"- Data: `{_markdown_text(data_name)}`",
        f"- Algo: `{_markdown_text(algo_name)}`",
        f"- Max Iter: `{_markdown_text(max_iter)}`",
        *_format_feature_lines(feature_summary),
        f"- Notes: {_markdown_text(notes)}",
        "",
    ]
    return "\n".join(lines)


def _resolve_model_registry_path(config: DictConfig) -> str:
    """Resolve the markdown model registry path from Hydra config.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        Absolute path to the markdown registry file.
    """
    registry_path = OmegaConf.select(config, "model_registry.path") or "docs/trained_models.md"
    registry_path = str(registry_path)
    if os.path.isabs(registry_path):
        return registry_path
    try:
        root_dir = hydra.utils.get_original_cwd()
    except ValueError:
        root_dir = os.getcwd()
    return os.path.join(root_dir, registry_path)


def _append_model_registry_entry(config: DictConfig) -> None:
    """Append the current training run to the markdown model registry.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        None.
    """
    if not bool(OmegaConf.select(config, "model_registry.enabled")):
        return

    path = _resolve_model_registry_path(config)
    registry_dir = os.path.dirname(path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    header = (
        "# Trained Models\n\n"
        "This registry is appended automatically by `task=train` when\n"
        "`model_registry.enabled=true`. Use `model_registry.key_features`\n"
        "to describe intentional differences from previous runs.\n\n"
    )
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)

    exp_name = str(config.wandb.id)
    with open(path, "r", encoding="utf-8") as f:
        existing = f.read()
    if f"| {exp_name} |" in existing or f"## {exp_name}\n" in existing:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = _build_model_registry_entry(
        exp_name=exp_name,
        timestamp=timestamp,
        data_name=config.data_name,
        algo_name=config.algo_name,
        max_iter=config.algo.max_iter,
        feature_summary=_summarize_model_features(config),
        notes="",
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def task_train(config: DictConfig):
    resolve_type_supervision_config(config)
    set_seed(config.seed)
    logger = Logger(config)
    _append_model_registry_entry(config)
    train_loader, val_loader = create_train_dataloader(config)
    timing_cfg = config.timing
    timing_enabled = timing_cfg.enabled
    timing_sync_cuda = timing_enabled and timing_cfg.cuda_sync
    timing_buffer = {
        "data_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
        "optim_time": 0.0,
        "iter_time": 0.0,
        "count": 0,
    }

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
        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            iter_start = time.perf_counter()
        optimizer.zero_grad()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            data_start = time.perf_counter()
        data = train_loader.get()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["data_time"] += time.perf_counter() - data_start
            forward_start = time.perf_counter()
        result_dict = model(data)

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["forward_time"] += time.perf_counter() - forward_start
        loss = 0
        for k, v in result_dict.items():
            if k in config.algo.loss_weight:
                loss += config.algo.loss_weight[k] * v
            elif "loss" in k:
                print(f"{k} is not used in loss!")

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            backward_start = time.perf_counter()
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

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["backward_time"] += time.perf_counter() - backward_start
        if it == 0:
            if timing_enabled:
                for key in timing_buffer:
                    timing_buffer[key] = 0.0 if key != "count" else 0
            continue

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            optim_start = time.perf_counter()
        optimizer.step()
        scheduler.step()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["optim_time"] += time.perf_counter() - optim_start
            timing_buffer["iter_time"] += time.perf_counter() - iter_start
            timing_buffer["count"] += 1

        result_dict["lr"] = torch.tensor(scheduler.get_last_lr())
        if timing_enabled and (it + 1) % config.algo.log_every == 0 and timing_buffer["count"] > 0:
            result_dict.update(
                {
                    "time_data": torch.tensor(timing_buffer["data_time"] / timing_buffer["count"]),
                    "time_forward": torch.tensor(timing_buffer["forward_time"] / timing_buffer["count"]),
                    "time_backward": torch.tensor(timing_buffer["backward_time"] / timing_buffer["count"]),
                    "time_optim": torch.tensor(timing_buffer["optim_time"] / timing_buffer["count"]),
                    "time_iter": torch.tensor(timing_buffer["iter_time"] / timing_buffer["count"]),
                }
            )
        if (it + 1) % config.algo.log_every == 0:
            logger.log({k: v.mean().item() for k, v in result_dict.items()}, "train", it)
            if timing_enabled:
                for key in timing_buffer:
                    timing_buffer[key] = 0.0 if key != "count" else 0

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
