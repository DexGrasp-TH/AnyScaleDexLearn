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
