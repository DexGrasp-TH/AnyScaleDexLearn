# Trained Models

This table is appended automatically by `task=train` when `model_registry.enabled=true`. Use `model_registry.key_features` to describe intentional differences from previous runs.

| Exp Name | Timestamp | Data | Algo | Max Iter | Key Features / Differences | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| humanMulti_humanMultiHierar_debug8 | 2026-04-30 13:48:27 | humanMulti | humanMultiHierar | 20000 | data=humanMulti; algo=humanMultiHierar; max_iter=20000; random_pc_across_sequences=False; scale_aug=False[0.8,1.2] |  |
| humanMulti_humanMultiHierar_debug9 | 2026-04-30 13:48:28 | humanMulti | humanMultiHierar | 20000 | data=humanMulti; algo=humanMultiHierar; max_iter=20000; random_pc_across_sequences=True; scale_aug=True[0.8,1.2] |  |
| humanMulti_humanMultiHierar_debug11 | 2026-04-30 16:13:08 | humanMulti | humanMultiHierar | 20000 | data=humanMulti; algo=humanMultiHierar; max_iter=20000; random_pc_across_sequences=True; scale_aug=True[0.8,1.2]; type_balancing=True[sampler_alpha=0.5,loss_beta=0.25] |  |
| humanMulti_humanMultiHierar_debug10 | 2026-04-30 16:13:08 | humanMulti | humanMultiHierar | 20000 | data=humanMulti; algo=humanMultiHierar; max_iter=20000; random_pc_across_sequences=False; scale_aug=True[0.8,1.2]; type_balancing=True[sampler_alpha=0.5,loss_beta=0.25] |  |
