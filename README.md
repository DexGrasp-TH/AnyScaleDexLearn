# DexLearn

DexLearn trains and evaluates grasp generation models for human-hand and robot-hand settings. The main workflows are:

- `train`: train a model and save checkpoints
- `sample`: generate grasps from a trained checkpoint
- `visualize_*`: inspect generated grasps
- `evaluate`: evaluate saved human-model samples
- `scene_budget`: build human-only geometry scene budget labels and train the budget head

## Prerequisites

- Linux environment with NVIDIA GPU support
- Python `3.10`
- Conda
- CUDA-compatible PyTorch environment
- Dataset path available through `AnyScaleGraspDataset`

## Installation

1. Clone the repository with submodules.
   ```bash
   git clone --recursive <repository-url>
   cd AnyScaleDexLearn
   # Or if already cloned:
   git submodule update --init --recursive --progress
   ```

2. Create the conda environment.
   ```bash
   conda create -n anyscalelearn python=3.10
   conda activate anyscalelearn
   ```

3. Install PyTorch.
   ```bash
   conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install mkl==2024.0.0
   ```

4. Install PyTorch3D.
   ```bash
   wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2
   conda install -y --use-local ./pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2
   ```

5. Install third-party dependencies.
   ```bash
   pip install -e ./third_party/diffusers
   pip install -e ./third_party/manopth
   pip install -e ./third_party/nflows
   pip install -e ./third_party/pytorch_kinematics
   pip install -e ./third_party/utils_python
   ```
   Place mano models in `./third_party/manopath/mano/models`.

6. Install `MinkowskiEngine`.
   Follow the setup notes in [detailed steps](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2566883469) and [NVTX_DISABLE](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2886016764).
   ```bash
   cd third_party/MinkowskiEngine
   sudo apt install libopenblas-dev
   export CUDA_HOME=/usr/local/cuda-12.4  # adjust if needed
   python setup.py install --blas=openblas
   cd ../..
   ```

7. Install DexLearn and runtime Python packages.
   ```bash
   pip install -e .
   pip install hydra-core
   pip install trimesh
   pip install viser
   pip install 'pyglet<2'
   pip install chumpy --no-build-isolation
   pip install opencv-python
   pip install numpy==1.26.4
   ```

## Preparation

1. Export the dataset path in each terminal.
   ```bash
   # local
   export AnyScaleGraspDataset=/data/dataset/AnyScaleGrasp
   # server
   export AnyScaleGraspDataset=/data/mingrui/dataset/AnyScaleGrasp
   ```

2. Create the object symlink in `assets`.
   ```bash
   ln -s ${AnyScaleGraspDataset}/object ./assets/object
   ```

3. Confirm the dataset path exists before running training or evaluation commands.

## Arguments

- `exp_name`: experiment name used in output paths
- `DATA_NAME`: training dataset config name
- `TEST_DATA_NAME`: test dataset config name
- `ckpt`: checkpoint step or checkpoint path to load

Availabel configs for robot workflow:

- `DATA_NAME`: `shadowMulti`, `leapMulti`
- `TEST_DATA_NAME`: `shadowMulti`, `leapMulti`

## Outputs

- Training checkpoints are saved under `output/<data>_<algo>_<exp_name>/ckpts/`
- Sampled results are saved under `output/<data>_<algo>_<exp_name>/tests/step_<ckpt>/`
- `visualize` reads sampled grasps from the corresponding `tests` directory

## Robot Workflow

### Check Dataloader

Inspect robot dataloader samples before training or debugging. The script buffers samples so visualization follows a fixed grasp-type order such as `1 2 3 4 5`, then `1 2 3 4 5` again when available.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/check_robot_dataloader.py data=<DATA_NAME> exp_name=<EXP_NAME>
```

### Train

Train a robot grasp model.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=robotMultiHierar data=<DATA_NAME> num_workers=24 prefetch_factor=2 exp_name=<EXP_NAME>
```

### Sample

Generate robot grasps from a trained checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
```

### Visualize

Visualize sampled robot grasps. The current visualization sampler is controlled by `task.visualize_mode`; the old group-balanced grasp-type cycling behavior is deprecated.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>

# e.g., python dexlearn/main.py task=visualize algo=robotMultiHierar data=leapMulti test_data=leapMulti exp_name=dataset_full_1

# Web visualizer. The browser UI can switch views and apply object or grasp-type selections.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualizer=viser algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>

# New sample selection modes.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=random_object task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=one_object task.object_id=<OBJECT_ID> task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=grasp_type task.target_grasp_type_id=<1-5> task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
```

## Human Workflow

### Preprocess

Compute and save `index_mcp_pos` into the source human grasp files before training with index-MCP positions.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=human_preprocess data=humanMulti exp_name=<EXP_NAME>

# e.g., CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=human_preprocess data=humanMulti exp_name=debug1
```

### Check Dataloader

Inspect human dataloader samples after preprocessing and before training. This visualization follows the configured `hand_pos_source` in `dexlearn/config/data/humanMulti.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/check_human_dataloader.py data=humanMulti data.hand_pos_source=<wrist/index_mcp> exp_name=<EXP_NAME> 

# e.g.: CUDA_VISIBLE_DEVICES=0 python tests/check_human_dataloader.py data=humanMulti data.hand_pos_source=index_mcp exp_name=debug1
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME>

# e.g.: CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME>
```

### Sample

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<EXP_NAME>

# e.g,: CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=DGNMulti exp_name=<EXP_NAME>
# e.g,: CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<EXP_NAME>
```

### Object Human Prior Export

Export object-scene human prior scores and hand-position seeds for downstream
BODex synthesis. The default task writes one 5-type budget score vector per
scene and writes 20 unsorted pose samples per scene and grasp type. When
`data.hand_pos_source=index_mcp`, the export stores `index_mcp_pos` and
`wrist_quat` directly without running MANO to infer `wrist_pos`.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=obj_human_prior_export \
  data=humanMulti \
  algo=humanMultiHierar \
  test_data=DGNMulti \
  algo.batch_size=1024 # number of parallel scenes \
  exp_name=<EXP_NAME> \
  ckpt=<CKPT> \
```

Outputs are written to
`output/humanMulti_humanMultiHierar_<EXP_NAME>/obj_human_prior/step_<CKPT>/`
unless `task.output_dir` is set. Per-scene files are stored under a subdirectory
named after `test_data.object_path`'s final component, preserving the original
scene id hierarchy, for example `.../step_<CKPT>/DGN_2k/<object>/<env>/<scene>.npy`.


### Visualize

```bash
python dexlearn/main.py task=visualize data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>

# e.g, python dexlearn/main.py task=visualize data=humanMulti algo=humanMultiHierar test_data=humanMulti ckpt=010000 exp_name=<EXP_NAME>

# Web visualizer with multi-scene layout and runtime object or grasp-type selection.
python dexlearn/main.py task=visualize task.visualizer=viser task.viser_port=8080 task.viser_display_mode=single task.viser_scene_id=0 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>

# New sample selection modes.
python dexlearn/main.py task=visualize task.visualize_mode=random_object task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualize_mode=one_object task.object_id=<OBJECT_ID> task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualizer=viser task.visualize_mode=one_object_multi_seq task.object_id=obj_0_seq_0 task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualize_mode=grasp_type task.target_grasp_type_id=<1-5> task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
```

### Evaluate

Evaluate an already sampled human model run. Run `task=sample` first; this task
does not generate samples.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=evaluate algo=humanMultiHierar data=humanMulti test_data=humanMulti exp_name=<EXP_NAME> ckpt=<CKPT>
```

### Scene Budget

Build human-only scene-budget labels and train the independent geometry budget
head. This task does not train the CE or diffusion model. The default label
source is `task.label_source=hierarchy_count`. The task writes
`scene_budget_label_hierarchy.csv`, a compact per-grasp table grouped as
canonical object, scene id, pose class, grasp type, and grasp record. It then
aggregates those rows in memory into one direct-count row per pose-class scene:
`(component_idx, split, canonical_object_id, pose_class_id)`. The raw label is
`grasp_record_count`; the training target is
`log_count_multiplier = log(clip(grasp_record_count / mean_train_count))`.

By default `task.splits=[train,test]`, so the label build reads the explicit
`train.json` and `test.json` files under the configured object split directory.
The budget head trains on `split=train` rows and validates on `split=test` rows;
`scene_budget` does not create its own random validation split.

The budget head is intentionally small because the direct-count dataset is tiny:
default hidden dimensions are `[16, 16]`, dropout is `0.1`, weight decay is
`0.001`, and validation-MSE early stopping is enabled. These defaults are meant
for a conservative geometry-only budget baseline, not a high-capacity predictor.

The budget head input uses three yaw-invariant bounding-box dimensions:
`bbox_xy_major`, `bbox_xy_minor`, and `bbox_z`. The canonical point cloud is
scaled and transformed by the stored object pose before measuring the bbox. The
XY box uses the minimum-area rectangle over the tabletop plane rather than fixed
world X/Y axes.

Set `task.train.input_type=pointcloud` to train the budget head from object
point clouds instead of bbox features. This path uses the same `WrappedMinkUNet`
backbone family as `task=train`, can initialize from a main training checkpoint
with `task.train.pointcloud.encoder_checkpoint=<CKPT>`, and supports Z-yaw
augmentation through `task.train.pointcloud.z_yaw_aug=true`. The default
`task.train.input_type=bbox` remains the lightweight baseline.

```bash
python dexlearn/main.py task=scene_budget data=humanMulti algo=humanMultiHierar exp_name=<EXP_NAME>
```

Default outputs are written to
`output/humanMulti_humanMultiHierar_<EXP_NAME>/scene_budget/`:

- `scene_budget_label_hierarchy.csv`: compact per-grasp canonical-object / scene / pose-class / type table
- `scene_budget_summary.json`: feature normalization, direct-count statistics, and checks
- `geometry_budget_head.pth`: trained independent budget head checkpoint
- `budget_head_predictions.csv`: train/test target and predicted budget multipliers
- `budget_head_train_summary.json`: train/validation metrics
- `budget_head_train_multiplier_scatter.png`: train-set target-vs-predicted multiplier plot
- `budget_head_test_multiplier_scatter.png`: test-set target-vs-predicted multiplier plot
- `scene_budget_run_summary.json`: resolved task config and output paths

Common overrides:

```bash
# Only build scene-budget labels, without training the budget head.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.mode=build_labels

# Train with the point-cloud encoder input instead of bbox features.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.train.input_type=pointcloud \
  task.train.pointcloud.encoder_checkpoint=<PATH_TO_TRAIN_CKPT>

# Run budget-head inference on the same test_data interface used by task=sample.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  test_data=DGNMulti \
  exp_name=<EXP_NAME> \
  task.mode=predict \
  task.inference.checkpoint=<PATH_TO>/geometry_budget_head.pth

# Use the legacy nearest-scene diverse-grasp-class label source for ablation.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.label_source=legacy_nearest_n \
  task.legacy_nearest_n.nearest_scene_num=16 \
  task.legacy_nearest_n.orientation_threshold_deg=30.0 \
  task.legacy_nearest_n.direction_threshold_deg=30.0 \
  task.legacy_nearest_n.posed_object_translation_threshold_m=0.1 \
  task.legacy_nearest_n.posed_object_rotation_threshold_deg=45.0 \
  task.label_structure.pose_class_rotation_threshold_deg=45.0 \
  task.label_structure.pose_class_bbox_proportion_threshold=0.2 \
  task.legacy_nearest_n.clip_min=0.5 \
  task.legacy_nearest_n.clip_max=3.0
```
