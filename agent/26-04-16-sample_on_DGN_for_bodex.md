# Sample on DGB for BODex Optimization

## Overall Task

I have an generative model trained on collected human grasp dataset. 

I want to use this generative model to sample human-inspired grasp wrist poses on the DGN object assets (or other assets). An example sample process is `python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=DGNMulti exp_name=debug1`.

I want to use the sampled wrist poses as the initial values for optimization by BimanBODex.

## General Requests (Important)

* Write concise and necessary comments to your generated code for better user understanding.

## Plan for supporting complete object pointcloud (Finished)

1. Add a point-cloud source switch in `dexlearn/dataset/robot_multidex.py`:
   - Use a config flag `pc_source` with options:
     - `partial` (default): keep current `partial_pc**.npy` pipeline.
     - `complete`: load normalized full object cloud from `processed_data/<obj_name>/complete_point_cloud.npy`.

2. In test/sampling data loading (`_load_test_data`), parse object metadata from `scene_cfg`:
   - `obj_name` from `scene_cfg["task"]["obj_name"]`.
   - `obj_scale` and `obj_pose` from `scene_cfg["scene"][obj_name]`.

3. For `pc_source == "complete"`, build NN input with full geometric transform:
   - Sample points from `complete_point_cloud.npy`.
   - Apply object scale to normalized cloud.
   - Apply object pose to place points into scene/world frame:
     - rotation + translation from pose.
   - This ensures complete cloud input is consistent with scene placement.

4. Keep backward compatibility:
   - No behavior change when `pc_source` is absent or set to `partial`.
   - Preserve existing `save_path/scene_path/pc_path` outputs.

5. Add robust checks:
   - Raise clear errors for missing complete cloud files or malformed scene config entries.
   - Keep preloading logic compatible with both partial and complete paths.

## Plan for index-MCP-based human grasp training (Finished)

1. Add a dataset pre-process task `dexlearn/task/human_preprocess.py`.
   - Iterate over the source human grasp files under `config.data.grasp_path`.
   - For each active hand, compute the MANO joints from the stored hand rotation, MANO pose, MANO betas, and translation.
   - Extract MANO joint 5 as the index MCP position in world coordinates.
   - Save it back into the grasp file as `grasp_data["hand"][side]["index_mcp_pos"]`.
   - Keep all existing fields unchanged.
   - Add config for this task, including source scope, overwrite behavior, and MANO model path.

2. Add a position-source switch to `dexlearn/config/data/*.yaml`.
   - Introduce a config field such as `hand_pos_source: wrist` or `hand_pos_source: index_mcp`.
   - Default it to `wrist` for backward compatibility.
   - Use the same option in human training/eval configs first; test configs do not need hand labels, but the chosen source must still be propagated to inference outputs.

3. Update `dexlearn/dataset/human_multidex.py` to load either wrist translation or index MCP position.
   - In `_extract_hand_poses()`, continue loading wrist rotation as before.
   - When `hand_pos_source == "wrist"`, keep the current behavior.
   - When `hand_pos_source == "index_mcp"`, set `{side}_hand_trans` from `grasp_data["hand"][side]["index_mcp_pos"]`.
   - Preserve the current mirroring logic for left-only grasps:
     - Mirror the left-hand `index_mcp_pos` into the right-hand frame when mirroring.
   - Preserve the fixed placeholder left-hand behavior for right-only grasps.
   - Keep point-cloud centering and rotation augmentation operating on whichever position source is loaded.
   - Raise a clear error if `hand_pos_source == "index_mcp"` but the preprocessed field is missing.

4. Update `dexlearn/scripts/check_human_dataloader.py` so visualization respects the chosen position source.
   - Continue using `{side}_hand_rot` and MANO parameters to construct the hand shape.
   - If the dataloader position source is wrist position, keep the current visualization path.
   - If the dataloader position source is index MCP position:
     - First compute the hand mesh/joints in a neutral global position using the stored wrist rotation.
     - Read the generated MANO index MCP joint.
     - Translate the whole hand so that this generated MCP point matches the loaded `{side}_hand_trans`.
   - Skip the fixed placeholder left hand for right-only grasps as before.

5. Update sample saving in `dexlearn/task/sample.py`.
   - Keep the current `grasp_pose` tensor shape and wrist-orientation format.
   - Add a saved metadata flag indicating the meaning of the position component, for example `grasp_pos_source` or `hand_pos_source`.
   - Set it from the model/data config used at inference time.
   - Keep decentering logic consistent with the selected source:
     - Add centroid back to the right-hand position always.
     - Add centroid back to the left-hand position only for bimanual grasp types.

6. Update `dexlearn/task/visualize_human.py` to visualize either wrist-position outputs or index-MCP-position outputs.
   - Read the saved position-source flag from each sample file.
   - If the sample uses wrist position, keep the current rendering path.
   - If the sample uses index MCP position:
     - Treat `grasp_pose[:3]` (and the left-hand counterpart when present) as target index MCP positions.
     - Use the saved wrist orientation to build the MANO hand at a neutral translation.
     - Translate the hand mesh so its MANO joint 5 aligns with the target MCP position.
   - Keep support for right-only and bimanual grasp types.

7. Keep backward compatibility and output clarity.
   - Existing checkpoints, datasets, and saved sample files should continue to work when `hand_pos_source` is absent or set to `wrist`.
   - New preprocessing should be additive, not destructive.
   - Sample results should explicitly state which position source they contain, so downstream BODex initialization and later conversion tasks are not ambiguous.

8. Validate in small steps after implementation.
   - Run the preprocessing task on a small subset and inspect that `index_mcp_pos` is written correctly.
   - Run `dexlearn/scripts/check_human_dataloader.py` once with `hand_pos_source=wrist` and once with `hand_pos_source=index_mcp`.
   - Run a small sampling job and confirm the saved metadata flag is present.
   - Run `dexlearn/task/visualize_human.py` on both output types and verify the hand mesh aligns with the intended contact point.

## Run Log

Record what you have done here after every job.

- 2026-04-16: Added implementation plan in the "Plan for supporting complete object pointcloud" section, updated to explicitly include object pose transformation (scale + rotation + translation) for complete pointcloud sampling.
- 2026-04-16: Implemented the plan in `dexlearn/dataset/robot_multidex.py` by adding `pc_source` switch (`partial`/`complete`), complete-cloud loading from `processed_data/<obj_name>/complete_point_cloud.npy`, and scale+pose transformation to world frame. Added robust scene/object parsing and error checks. Updated `dexlearn/config/test_data/DGNMulti.yaml` to use `pc_source: complete` by default. Ran smoke test for both `pc_source=partial` and `pc_source=complete`, and both paths loaded successfully.
- 2026-04-16: Updated `dexlearn/task/visualize_test.py` to align complete-pointcloud transformation with `dexlearn/dataset/robot_multidex.py`: added `pc_source` handling, robust scene/object metadata parsing, and scale + pose (rotation/translation) transform for `pc_source=complete`.
- 2026-04-16: Updated `dexlearn/task/visualize_test.py` to distinguish mixed saved results by point-cloud source (`complete` vs `partial`) when sampling visualization files, and added source tag in visualization caption.
- 2026-04-16: Renamed human visualization task from `visualize_test` to `visualize_human` across task module/config/imports and README command examples (`task=visualize_human`) to align naming with `visualize_robot`.
- 2026-04-16: Updated `dexlearn/task/visualize_human.py` caption to include `pred_grasp_type_prob` (3-decimal list format), consistent with `dexlearn/task/visualize_robot.py`.
- 2026-04-16: Added `dexlearn/scripts/check_test_dataloader.py` to visualize only object point clouds from test dataloader, with caption fields (`grasp_type`, `scene_path`, `pc_path`) and a Z-up-friendly initial camera view.
- 2026-04-16: Added `reorder_samples_by_pred_grasp_type()` to `dexlearn/task/visualize_human.py` and applied it to sampled files (target predicted grasp type order: 1,2,3,4,5), matching the behavior in `dexlearn/task/visualize_robot.py`.
- 2026-04-16: Updated `dexlearn/dataset/human_multidex.py` to track fixed placeholder left-hand poses explicitly, and skip point-cloud centering / rotation augmentation on that non-sense left hand for right-hand-only grasps. Updated `dexlearn/task/sample.py` so `_decenter_human_pose()` adds centroid back to the left-hand translation only for bimanual grasp types (ids 4, 5), matching robot decentering behavior.
- 2026-04-16: Simplified `dexlearn/task/visualize_human.py` sample selection to match `dexlearn/task/visualize_robot.py`: use group-directory discovery / optional `task.include_groups`, balanced per-group sampling, then round-robin reorder by predicted grasp type `1,2,3,4,5,1,2,...`. Added `include_groups` to `dexlearn/config/task/visualize_human.yaml` for parity with robot visualization.
- 2026-04-16: Added new task `dexlearn/task/human_prior_format.py` and registered it in `dexlearn/task/__init__.py`. The task reads saved human sample results, preserves all original fields, and adds `wrist_pos`, `wrist_quat`, and `index_mcp_pos` derived from `grasp_pose` using the same neutral MANO setup as `dexlearn/task/visualize_human.py` (index MCP = MANO joint 5). Added `dexlearn/config/task/human_prior_format.yaml` with `include_groups`, `output_suffix`, and `mano_root`. Verified by `py_compile`, pose-splitting helper import test, and a MANO smoke test that produced a finite MCP point.
- 2026-04-16: Did not change code. Read the current implementations of `dexlearn/dataset/human_multidex.py`, `dexlearn/scripts/check_human_dataloader.py`, `dexlearn/task/sample.py`, and `dexlearn/task/visualize_human.py`, then added the below implementation plan for switching human NN position targets from wrist position to index MCP position.
- 2026-04-16: Implemented the index-MCP human position path. Added shared helpers in `dexlearn/utils/human_hand.py` and a new preprocessing task `dexlearn/task/human_preprocess.py` with config `dexlearn/config/task/human_preprocess.yaml` to compute and save `grasp_data["hand"][side]["index_mcp_pos"]`. Added `hand_pos_source: wrist` to `dexlearn/config/data/humanMulti.yaml` and updated `dexlearn/dataset/human_multidex.py` to load either wrist translation or index MCP position, with clear erroring if preprocessing has not been run. Updated `dexlearn/scripts/check_human_dataloader.py` and `dexlearn/task/visualize_human.py` to align MANO hands by wrist or index MCP depending on the configured/saved position source. Updated `dexlearn/task/sample.py` to save `grasp_pos_source`, `dexlearn/utils/logger.py` to persist scalar metadata in saved samples, and `dexlearn/task/human_prior_format.py` to correctly recover both `wrist_pos` and `index_mcp_pos` for either saved position-source mode. Verified by `py_compile`, helper smoke tests for MCP-to-wrist alignment, a MANO smoke test in `human_prior_format.py`, and a preprocessing-side MANO smoke test in `human_preprocess.py`.
- 2026-04-16: Updated `dexlearn/scripts/check_human_dataloader.py` for `hand_pos_source == "index_mcp"` to also visualize the loaded index-MCP frame directly, using the loaded MCP position and the wrist orientation, instead of visualizing a recomputed MCP point from MANO joints. Verified by `py_compile`.
- 2026-04-16: Updated `dexlearn/task/visualize_human.py` for `grasp_pos_source == "index_mcp"` to also visualize the NN-predicted index-MCP frame directly, using the inferred MCP position from `grasp_pose[:3]` and the wrist orientation, instead of visualizing a recomputed MCP point from MANO joints. Verified by `py_compile`.




## What I want you to do next


## TODO (in the future)
