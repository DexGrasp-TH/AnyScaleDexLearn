from omegaconf import DictConfig, OmegaConf, open_dict


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
        "augmentation.pc_centering": "pc_centering",
        "augmentation.rotation": "rotation_aug",
        "augmentation.z_rotation": "z_rotation_aug",
        "augmentation.xy_rotation.enabled": "xy_rotation_aug",
        "augmentation.xy_rotation.max_angle_deg": "xy_rotation_max_angle_deg",
        "augmentation.translation.enabled": "translation_aug",
        "augmentation.translation.range": "translation_range",
        "augmentation.pc_noise.enabled": "pc_noise_aug",
        "augmentation.pc_noise.scale": "pc_noise_scale",
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
