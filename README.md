# DexLearn

DexLearn trains and evaluates grasp generation models for human-hand and robot-hand settings. The main workflows are:

- `train`: train a model and save checkpoints
- `sample`: generate grasps from a trained checkpoint
- `visualize_*`: inspect generated grasps
- `stat`: compute summary statistics for generated grasps

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
- `visualize_robot` reads sampled robot grasps from the corresponding `tests` directory

## Robot Workflow

### Check Dataloader

Inspect robot dataloader samples before training or debugging. The script buffers samples so visualization follows a fixed grasp-type order such as `1 2 3 4 5`, then `1 2 3 4 5` again when available.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/scripts/check_robot_dataloader.py data=<DATA_NAME> exp_name=<EXP_NAME>
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

Visualize sampled robot grasps. Saved results are reordered by predicted grasp type so the viewer shows one grasp from each type in sequence, for example `1 2 3 4 5`, then `1 2 3 4 5` again when available.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize_robot algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
```

### Stat

Compute summary statistics for sampled robot grasps.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=stat algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
```

## Human Workflow

### Train

```bash
# single wrist pose, normalizing flow
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=h_nflow data=human exp_name=<EXP_NAME>

# single wrist pose, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=h_diffusion data=human exp_name=<EXP_NAME>

# bimanual wrist pose, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=hbi_diffusion data=humanbi exp_name=<EXP_NAME>

# multi-hand hierarchical model
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME>
```

### Sample

```bash
# single wrist pose, test on human_obj
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=human algo=h_diffusion test_data=human_obj exp_name=<EXP_NAME>

# single wrist pose, test on DGN
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=human algo=h_diffusion test_data=DGN exp_name=<EXP_NAME>

# bimanual wrist pose, test on DGN_grasp_type
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanbi algo=hbi_diffusion test_data=DGN_grasp_type ckpt=050000 exp_name=<EXP_NAME>

# multi-hand hierarchical model
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<EXP_NAME>
```

### Visualize

```bash
# test on DGN, normalizing flow
python dexlearn/main.py task=visualize_test data=human algo=h_nflow test_data=DGN exp_name=<EXP_NAME>

# single wrist pose, test on human_obj
python dexlearn/main.py task=visualize_test data=human algo=h_diffusion test_data=human_obj exp_name=<EXP_NAME>

# single wrist pose, test on DGN
python dexlearn/main.py task=visualize_test data=human algo=h_diffusion test_data=DGN exp_name=<EXP_NAME>

# bimanual wrist pose, test on DGN_grasp_type
python dexlearn/main.py task=visualize_test data=humanbi algo=hbi_diffusion test_data=DGN_grasp_type ckpt=050000 exp_name=<EXP_NAME>
```