# Trained Models

This registry is appended automatically by `task=train` when
`model_registry.enabled=true`. Use `model_registry.key_features`
to describe intentional differences from previous runs.

## humanMulti_humanMultiHierar_debug8
- Timestamp: `2026-04-30 13:48:27`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=False`
  - `scale_aug=False[0.8,1.2]`
- Notes: categorical 基线；易优化。

## humanMulti_humanMultiHierar_debug9
- Timestamp: `2026-04-30 13:48:28`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=True`
  - `scale_aug=True[0.8,1.2]`
- Notes: aug/random-PC 基线；OOD 长尾更好。

## humanMulti_humanMultiHierar_debug10
- Timestamp: `2026-04-30 16:13:08`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=False`
  - `scale_aug=True[0.8,1.2]`
  - `type_balancing=True[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: 无 random-PC 的 type balancing 消融。

## humanMulti_humanMultiHierar_debug11
- Timestamp: `2026-04-30 16:13:08`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=True`
  - `scale_aug=True[0.8,1.2]`
  - `type_balancing=True[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: type balancing + random-PC；长尾更稳。

## humanMulti_humanMultiHierar_debug12
- Timestamp: `2026-05-01 10:38:53`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=False`
  - `scale_aug=True[0.8,1.2]`
  - `feasibility=True[open_world_positive_only]`
  - `type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: 原 open-world 默认；无 random-PC。

## humanMulti_humanMultiHierar_debug13
- Timestamp: `2026-05-01 10:38:53`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=False`
  - `scale_aug=True[0.8,1.2]`
  - `feasibility=True[closed_world_object_complete]`
  - `type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: 旧 humanMulti 默认；closed-world；无 random-PC。

## humanMulti_humanMultiHierar_debug14
- Timestamp: `2026-05-01 10:38:53`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=True`
  - `scale_aug=True[0.8,1.2]`
  - `feasibility=True[open_world_positive_only]`
  - `type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: open-world + random-PC。

## humanMulti_humanMultiHierar_debug15
- Timestamp: `2026-05-01 10:38:53`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - `data=humanMulti`
  - `algo=humanMultiHierar`
  - `max_iter=20000`
  - `random_pc_across_sequences=True`
  - `scale_aug=True[0.8,1.2]`
  - `feasibility=True[closed_world_object_complete]`
  - `type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]`
- Notes: closed-world + random-PC。
## humanMulti_humanMultiHierar_debug18
- Timestamp: `2026-05-01 12:25:48`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=True
  - scale_aug=True[0.8,1.2]
  - feasibility=True[open_world_positive_only]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: open-world + random-PC。
## humanMulti_humanMultiHierar_debug19
- Timestamp: `2026-05-01 12:25:48`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=True
  - scale_aug=True[0.8,1.2]
  - feasibility=True[closed_world_object_complete]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 当前 humanMulti 默认；closed-world + random-PC。
## humanMulti_humanMultiHierar_debug16
- Timestamp: `2026-05-01 12:25:48`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.8,1.2]
  - feasibility=True[open_world_positive_only]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes:
## humanMulti_humanMultiHierar_debug17
- Timestamp: `2026-05-01 12:25:48`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.8,1.2]
  - feasibility=True[closed_world_object_complete]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes:
## humanMulti_humanMultiHierar_debug21
- Timestamp: `2026-05-01 17:58:16`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - type_objective_object_bce_object_closed_world
- Notes: 
## humanMulti_humanMultiHierar_debug22
- Timestamp: `2026-05-01 17:58:17`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - type_objective_scene_ranking_sampled_negative
- Notes: 
## humanMulti_humanMultiHierar_debug20
- Timestamp: `2026-05-01 17:58:17`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - type_objective_ce_record_softmax
- Notes: 
## humanMulti_humanMultiHierar_debug24
- Timestamp: `2026-05-01 18:08:11`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - type_objective_scene_ranking_type_uniform_sampler_no_loss_weight
- Notes: 
## humanMulti_humanMultiHierar_debug23
- Timestamp: `2026-05-01 18:08:11`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - type_objective_ce_type_uniform_sampler_no_loss_weight
- Notes: 
## humanMulti_humanMultiHierar_debug26
- Timestamp: `2026-05-03 10:45:21`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.8,1.2]
  - train_sampling_unit=object_uniform
  - type_objective=ce[scope=record,negative=softmax]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 
## humanMulti_humanMultiHierar_debug25
- Timestamp: `2026-05-03 10:45:21`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.8,1.2]
  - train_sampling_unit=record_uniform
  - type_objective=ce[scope=record,negative=softmax]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 
## humanMulti_humanMultiHierar_debug27
- Timestamp: `2026-05-11 20:37:57`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.9,1.1]
  - train_sampling_unit=posed_object_uniform
  - type_objective=ce[scope=record,negative=softmax]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 
## humanMulti_humanMultiHierar_debug29
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A2_A1_plus_point_dropout_0.1
- Notes: 
## humanMulti_humanMultiHierar_debug30
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A3_A1_plus_scale_aug_0.85_1.15
- Notes: 
## humanMulti_humanMultiHierar_debug33
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A5_posed_object_sampler_hard_CE_control
- Notes: 
## humanMulti_humanMultiHierar_debug31
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A4_A1_plus_clipped_jitter_sigma_0.002
- Notes: 
## humanMulti_humanMultiHierar_debug28
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A1_posed_object_soft_CE_default_aug_no_point_dropout
- Notes: 
## humanMulti_humanMultiHierar_debug32
- Timestamp: `2026-05-11 22:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A0_record_uniform_hard_CE_baseline
- Notes: 
## humanMulti_humanMultiHierar_debug39
- Timestamp: `2026-05-12 00:48:45`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A5_posed_object_sampler_hard_CE_control_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug36
- Timestamp: `2026-05-12 00:48:45`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A3_A1_plus_scale_aug_0.85_1.15_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug35
- Timestamp: `2026-05-12 00:48:45`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A2_A1_plus_point_dropout_0.1_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug37
- Timestamp: `2026-05-12 00:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A4_A1_plus_clipped_jitter_sigma_0.002_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug38
- Timestamp: `2026-05-12 00:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A0_record_uniform_hard_CE_baseline_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug34
- Timestamp: `2026-05-12 00:48:46`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - A1_posed_object_soft_CE_default_aug_no_point_dropout_new_object_split
- Notes: 
## humanMulti_humanMultiHierar_debug40
- Timestamp: `2026-05-12 09:07:00`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `20000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=20000
  - random_pc_across_sequences=False
  - scale_aug=True[0.9,1.1]
  - train_sampling_unit=record_uniform
  - type_objective=ce[scope=record,negative=softmax]
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 
## humanMulti_humanMultiHierar_debug45
- Timestamp: `2026-05-12 09:38:36`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `3000`
- Key Features:
  - T5_debug35_type_only_posed_object_hard_CE_3000iter_save500
- Notes: 
## humanMulti_humanMultiHierar_debug42
- Timestamp: `2026-05-12 09:38:36`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `5000`
- Key Features:
  - T2_debug35_type_only_5000iter_save500
- Notes: 
## humanMulti_humanMultiHierar_debug44
- Timestamp: `2026-05-12 09:38:36`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `3000`
- Key Features:
  - T4_debug35_type_only_record_uniform_soft_labels_3000iter_save500
- Notes: 
## humanMulti_humanMultiHierar_debug41
- Timestamp: `2026-05-12 09:38:36`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `3000`
- Key Features:
  - T1_debug35_type_only_3000iter_save500
- Notes: 
## humanMulti_humanMultiHierar_debug43
- Timestamp: `2026-05-12 09:38:36`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `5000`
- Key Features:
  - T3_debug35_type_only_5000iter_lr5e-4_save500
- Notes: 
## humanMulti_humanMultiHierar_debug47
- Timestamp: `2026-05-12 10:10:37`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T1_debug41_type_only_500iter_save100_hidden128_lr1e-3_posed_object
- Notes: 
## humanMulti_humanMultiHierar_debug50
- Timestamp: `2026-05-12 10:10:38`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T4_debug41_type_only_500iter_save100_hidden256_lr2e-4_posed_object
- Notes: 
## humanMulti_humanMultiHierar_debug46
- Timestamp: `2026-05-12 10:10:38`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T0_debug41_type_only_500iter_save100_hidden256_lr1e-3_posed_object
- Notes: 
## humanMulti_humanMultiHierar_debug51
- Timestamp: `2026-05-12 10:10:39`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T5_debug41_type_only_500iter_save100_hidden256_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug49
- Timestamp: `2026-05-12 10:10:39`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T3_debug41_type_only_500iter_save100_hidden256_lr5e-4_posed_object
- Notes: 
## humanMulti_humanMultiHierar_debug48
- Timestamp: `2026-05-12 10:10:39`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `500`
- Key Features:
  - T2_debug41_type_only_500iter_save100_hidden64_lr1e-3_posed_object
- Notes: 
## humanMulti_humanMultiHierar_debug56
- Timestamp: `2026-05-12 11:00:49`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T4_debug51_type_only_300iter_save50_hidden256_lr5e-4_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug55
- Timestamp: `2026-05-12 11:00:49`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T3_debug51_type_only_300iter_save50_hidden512_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug57
- Timestamp: `2026-05-12 11:00:49`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T5_debug51_type_only_300iter_save50_hidden256_lr2e-4_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug52
- Timestamp: `2026-05-12 11:00:49`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T0_debug51_type_only_300iter_save50_hidden256_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug54
- Timestamp: `2026-05-12 11:00:50`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T2_debug51_type_only_300iter_save50_hidden64_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug53
- Timestamp: `2026-05-12 11:00:50`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - T1_debug51_type_only_300iter_save50_hidden128_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug59
- Timestamp: `2026-05-12 11:28:35`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - S1_debug54_type_only_300iter_save50_hidden16_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug58
- Timestamp: `2026-05-12 11:28:35`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - S0_debug54_type_only_300iter_save50_hidden32_lr1e-3_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug60
- Timestamp: `2026-05-12 11:55:16`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `200`
- Key Features:
  - debug40_step010000_frozen_encoder_new_hidden64_type_head_200iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug61_stage1
- Timestamp: `2026-05-12 12:48:56`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - integrated_stage1_diffusion_encoder_10000iter_save2500_loss_type0_freeze_type_head_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug61_stage2
- Timestamp: `2026-05-12 13:33:40`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - integrated_stage2_frozen_stage1_encoder_train_yaml_configured_type_head_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug62
- Timestamp: `2026-05-12 13:56:48`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `1000`
- Key Features:
  - data=humanMulti
  - algo=humanMultiHierar
  - max_iter=1000
  - random_pc_across_sequences=False
  - scale_aug=True[0.9,1.1]
  - train_sampling_unit=record_uniform
  - type_objective=ce[scope=record,negative=softmax]
  - train_type_only=True
  - type_balancing=False[sampler_alpha=0.5,loss_beta=0.25]
- Notes: 
## humanMulti_humanMultiHierar_debug63_stage1
- Timestamp: `2026-05-12 14:32:58`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - integrated_stage1_diffusion_encoder_10000iter_save2500_loss_diffusion1_loss_type0.005_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug63_stage2
- Timestamp: `2026-05-12 15:22:51`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - integrated_stage2_frozen_stage1_encoder_train_yaml_configured_continued_type_head_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug64
- Timestamp: `2026-05-12 15:40:59`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - debug64_reset_type_head_from_debug63_stage1_frozen_encoder_300iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug65_type
- Timestamp: `2026-05-12 16:24:38`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - independent_from_scratch_type_only_shared_arch_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug65_stage1
- Timestamp: `2026-05-12 16:42:08`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - integrated_stage1_diffusion_encoder_10000iter_save2500_loss_diffusion1_loss_type0.005_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug65_diffusion
- Timestamp: `2026-05-12 16:48:18`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - independent_from_scratch_diffusion_only_shared_arch_10000iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_human_prior_0_type
- Timestamp: `2026-05-12 20:55:00`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - independent_from_scratch_type_only_shared_arch_300iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_human_prior_0_diffusion
- Timestamp: `2026-05-12 20:56:33`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - independent_from_scratch_diffusion_only_shared_arch_10000iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_human_prior_2_type
- Timestamp: `2026-05-12 23:27:25`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - independent_from_scratch_type_only_shared_arch_300iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug73_type
- Timestamp: `2026-05-12 23:28:31`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `300`
- Key Features:
  - independent_from_scratch_type_only_shared_arch_300iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_human_prior_2_diffusion
- Timestamp: `2026-05-12 23:29:00`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - independent_from_scratch_diffusion_only_shared_arch_10000iter_record_uniform_soft_labels
- Notes: 
## humanMulti_humanMultiHierar_debug73_diffusion
- Timestamp: `2026-05-12 23:30:23`
- Data: `humanMulti`
- Algo: `humanMultiHierar`
- Max Iter: `10000`
- Key Features:
  - independent_from_scratch_diffusion_only_shared_arch_10000iter_record_uniform_soft_labels
- Notes: 
## leapspMulti_robotMultiHierar_debug0
- Timestamp: `2026-05-20 20:05:16`
- Data: `leapspMulti`
- Algo: `robotMultiHierar`
- Max Iter: `50000`
- Key Features:
  - data=leapspMulti
  - algo=robotMultiHierar
  - max_iter=50000
  - type_objective=availability[scope=None,negative=None]
- Notes: 
