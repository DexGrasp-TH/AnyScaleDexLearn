from omegaconf import DictConfig, OmegaConf, open_dict


TYPE_OBJECTIVE_ALIASES = {
    "type_ce": "ce",
    "scene_ce": "ce",
    "softmax": "ce",
    "softmax_ce": "ce",
    "bce": "object_bce",
    "feasibility": "object_bce",
    "object_prior_bce": "object_bce",
    "ranking": "scene_ranking",
    "pairwise": "scene_ranking",
    "sampled_negative": "scene_ranking",
}

TYPE_OBJECTIVE_DEFAULTS = {
    "ce": {"scope": "record", "negative_policy": "softmax"},
    "object_bce": {"scope": "object", "negative_policy": "object_closed_world"},
    "scene_ranking": {"scope": "record", "negative_policy": "sampled_ranking"},
}

VALID_TYPE_OBJECTIVES = set(TYPE_OBJECTIVE_DEFAULTS)
VALID_SUPERVISION_SCOPES = {"record", "sequence", "object"}
VALID_NEGATIVE_POLICIES = {
    "softmax",
    "open_world_positive_only",
    "object_closed_world",
    "sampled_ranking",
}


def cfg_get(config, *keys, default=None):
    """Read the first existing config key from dotted key candidates."""
    if config is None:
        return default
    for key in keys:
        value = OmegaConf.select(config, key)
        if value is not None:
            return value
    return default


def _set_top_level(config, key, value):
    if not isinstance(config, DictConfig):
        setattr(config, key, value)
        return
    with open_dict(config):
        config[key] = value


def _ensure_config_section(config, key):
    """Ensure a nested config section exists.

    Args:
        config: OmegaConf config object that should own the section.
        key: Section name to create when absent.

    Returns:
        The existing or newly created config section.
    """
    if OmegaConf.select(config, key) is None:
        with open_dict(config):
            config[key] = {}
    return OmegaConf.select(config, key)


def normalize_type_objective(value):
    """Normalize type-objective aliases used by CLI overrides.

    Args:
        value: Objective name supplied by config or CLI.

    Returns:
        Canonical objective name.
    """
    objective = str(value).strip().lower()
    objective = TYPE_OBJECTIVE_ALIASES.get(objective, objective)
    if objective not in VALID_TYPE_OBJECTIVES:
        raise ValueError(
            f"Unsupported algo.model.type_objective={value}. "
            f"Expected one of {sorted(VALID_TYPE_OBJECTIVES)}."
        )
    return objective


def _infer_type_objective(config):
    """Infer a type objective for old configs that do not set one explicitly.

    Args:
        config: Full Hydra config.

    Returns:
        Canonical objective name, or ``None`` when the config is not a human
        multi-type hierarchical setup.
    """
    explicit = OmegaConf.select(config, "algo.model.type_objective")
    if explicit is not None:
        return normalize_type_objective(explicit)

    model_name = str(OmegaConf.select(config, "algo.model.name") or "")
    data_type = str(OmegaConf.select(config, "data.dataset_type") or "")
    if data_type != "HumanMultiDexDataset":
        return None
    if model_name == "HierarchicalFeasibilityModel":
        return "object_bce"
    if model_name in {"HierarchicalModel", "HierarchicalTypeCEModel"}:
        return "ce"
    if model_name == "HierarchicalTypeObjectiveModel":
        return "ce"
    return None


def resolve_type_supervision_config(config):
    """Resolve and validate the human multi-type supervision config.

    Args:
        config: Full Hydra config. The function mutates ``config`` in place so
            downstream model and dataset builders see canonical fields.

    Returns:
        The same config object.
    """
    objective = _infer_type_objective(config)
    if objective is None:
        return config

    defaults = TYPE_OBJECTIVE_DEFAULTS[objective]
    algo_cfg = _ensure_config_section(config, "algo")
    model_cfg = _ensure_config_section(algo_cfg, "model")
    supervision_cfg = _ensure_config_section(algo_cfg, "supervision")

    scope = cfg_get(supervision_cfg, "scope", default="auto")
    if scope is None or str(scope).lower() == "auto":
        scope = defaults["scope"]
    scope = str(scope).strip().lower()

    negative_policy = cfg_get(supervision_cfg, "negative_policy", default="auto")
    if negative_policy is None or str(negative_policy).lower() == "auto":
        negative_policy = defaults["negative_policy"]
    negative_policy = str(negative_policy).strip().lower()

    if scope not in VALID_SUPERVISION_SCOPES:
        raise ValueError(
            f"Unsupported algo.supervision.scope={scope}. "
            f"Expected one of {sorted(VALID_SUPERVISION_SCOPES)}."
        )
    if negative_policy not in VALID_NEGATIVE_POLICIES:
        raise ValueError(
            f"Unsupported algo.supervision.negative_policy={negative_policy}. "
            f"Expected one of {sorted(VALID_NEGATIVE_POLICIES)}."
        )

    if objective == "ce" and (scope not in {"record", "sequence"} or negative_policy != "softmax"):
        raise ValueError("type_objective=ce requires scope=record|sequence and negative_policy=softmax")
    if objective == "object_bce" and (
        scope != "object" or negative_policy not in {"open_world_positive_only", "object_closed_world"}
    ):
        raise ValueError(
            "type_objective=object_bce requires scope=object and "
            "negative_policy=open_world_positive_only|object_closed_world"
        )
    if objective == "scene_ranking" and (scope not in {"record", "sequence"} or negative_policy != "sampled_ranking"):
        raise ValueError(
            "type_objective=scene_ranking requires scope=record|sequence and negative_policy=sampled_ranking"
        )

    with open_dict(supervision_cfg):
        supervision_cfg.scope = scope
        supervision_cfg.negative_policy = negative_policy
    with open_dict(model_cfg):
        model_cfg.type_objective = objective
        model_cfg.supervision_scope = scope
        model_cfg.negative_policy = negative_policy
        model_cfg.ranking_loss = cfg_get(supervision_cfg, "ranking.loss", default="logistic")
        model_cfg.ranking_margin = float(cfg_get(supervision_cfg, "ranking.margin", default=1.0))
        focal_gamma = cfg_get(supervision_cfg, "balancing.focal_gamma", default=None)
        if focal_gamma is not None:
            model_cfg.focal_gamma = float(focal_gamma)
    return config


def apply_type_supervision_to_data_config(config, data_config, mode):
    """Inject objective-derived runtime view fields into a data config copy.

    Args:
        config: Full Hydra config.
        data_config: Data config copy passed to a dataset constructor.
        mode: Dataset mode, usually ``train``, ``eval``, or ``test``.

    Returns:
        The mutated ``data_config``.
    """
    resolve_type_supervision_config(config)
    objective = OmegaConf.select(config, "algo.model.type_objective")
    if objective is None or mode == "test":
        return data_config

    supervision_cfg = OmegaConf.select(config, "algo.supervision")
    balancing_cfg = OmegaConf.select(config, "algo.supervision.balancing")
    ranking_cfg = OmegaConf.select(config, "algo.supervision.ranking")
    scope = cfg_get(supervision_cfg, "scope", default=TYPE_OBJECTIVE_DEFAULTS[objective]["scope"])
    negative_policy = cfg_get(
        supervision_cfg,
        "negative_policy",
        default=TYPE_OBJECTIVE_DEFAULTS[objective]["negative_policy"],
    )

    _set_top_level(data_config, "type_objective", objective)
    _set_top_level(data_config, "supervision_scope", scope)
    _set_top_level(data_config, "negative_policy", negative_policy)
    _set_top_level(data_config, "feasibility_enabled", objective == "object_bce")
    _set_top_level(
        data_config,
        "feasibility_label_mode",
        "closed_world_object_complete" if negative_policy == "object_closed_world" else "open_world_positive_only",
    )
    _set_top_level(data_config, "ranking_enabled", objective == "scene_ranking")
    _set_top_level(data_config, "ranking_negatives_per_positive", int(cfg_get(ranking_cfg, "negatives_per_positive", default=4)))
    _set_top_level(data_config, "ranking_negative_sampling", cfg_get(ranking_cfg, "negative_sampling", default="uniform"))

    if balancing_cfg is not None:
        balancing_enabled = bool(cfg_get(balancing_cfg, "enabled", default=False))
        _set_top_level(data_config, "type_balancing_enabled", balancing_enabled)
        _set_top_level(data_config, "type_sampler_enabled", bool(cfg_get(balancing_cfg, "sampler.enabled", default=True)))
        _set_top_level(data_config, "type_sampler_alpha", float(cfg_get(balancing_cfg, "sampler.alpha", default=1.0)))
        _set_top_level(
            data_config,
            "type_sampler_object_uniform",
            bool(cfg_get(balancing_cfg, "sampler.object_uniform", default=True)),
        )
    return data_config


def flatten_multidex_data_config(config):
    """Expose flat compatibility aliases for the nested multi-dex data configs.

    The data YAMLs are organized by prefix, while existing dataset and utility
    code still expects flat names such as ``object_path`` and ``num_points``.
    This helper keeps those call sites stable during the config cleanup.
    """
    aliases = {
        "runtime.num_workers": "num_workers",
        "model.traj_length": "traj_length",
        "model.joint_num": "joint_num",
        "paths.grasp_path": "grasp_path",
        "paths.object_path": "object_path",
        "paths.split_path": "split_path",
        "paths.pc_path": "pc_path",
        "point_cloud.num_points": "num_points",
        "point_cloud.preload": "preload_point_clouds",
        "point_cloud.source": "pc_source",
        "point_cloud.random_pc_across_sequences": "random_pc_across_sequences",
        "sampling.train_unit": "train_sampling_unit",
        "augmentation.pc_centering": "pc_centering",
        "augmentation.rotation": "rotation_aug",
        "augmentation.z_rotation": "z_rotation_aug",
        "augmentation.xy_rotation.enabled": "xy_rotation_aug",
        "augmentation.xy_rotation.max_angle_deg": "xy_rotation_max_angle_deg",
        "augmentation.translation.enabled": "translation_aug",
        "augmentation.translation.range": "translation_range",
        "augmentation.scale.enabled": "scale_aug",
        "augmentation.scale.min": "scale_min",
        "augmentation.scale.max": "scale_max",
        "augmentation.pc_noise.enabled": "pc_noise_aug",
        "augmentation.pc_noise.scale": "pc_noise_scale",
        "feasibility.enabled": "feasibility_enabled",
        "feasibility.label_mode": "feasibility_label_mode",
        "type_balancing.enabled": "type_balancing_enabled",
        "type_balancing.sampler.enabled": "type_sampler_enabled",
        "type_balancing.sampler.alpha": "type_sampler_alpha",
        "type_balancing.sampler.object_uniform": "type_sampler_object_uniform",
        "type_balancing.loss_weight.enabled": "type_loss_weight_enabled",
        "type_balancing.loss_weight.beta": "type_loss_weight_beta",
        "type_balancing.loss_weight.reference": "type_loss_weight_reference",
        "human_hand.position_source": "hand_pos_source",
        "human_hand.load_mano_params": "load_mano_params",
        "test.split": "test_split",
        "test.scene_cfg": "test_scene_cfg",
        "test.mini": "mini_test",
        "robot.urdf_path": "robot_urdf_path",
        "robot.mesh_dir_path": "robot_mesh_dir_path",
        "robot.metadata_group": "metadata_group",
    }
    for nested_key, flat_key in aliases.items():
        value = cfg_get(config, flat_key, nested_key)
        if value is not None:
            _set_top_level(config, flat_key, value)
    return config
