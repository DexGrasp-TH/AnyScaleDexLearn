# DexLearn

Learning-based grasp synthesis baselines (e.g., diffusion model and normalizing flow) for dexterous hands, used in [BODex (ICRA 2025)](https://pku-epic.github.io/BODex/) and [Dexonomy (RSS 2025)](https://pku-epic.github.io/Dexonomy/)


## TODO list

- [x] Support BODex and Dexonomy datasets
- [x] Release grasp type classifier for Dexonomy

## Installation

### 1. Clone the repository with submodules
```bash
git clone --recursive <repository-url>
cd AnyScaleDexLearn
# Or if already cloned:
git submodule update --init --recursive --progress
```

### 2. Create conda environment
```bash
conda create -n dexlearn python=3.10 
conda activate dexlearn
```

### 3. Install PyTorch
```bash
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4. Install PyTorch3D
```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.8-py310_cu121_pyt222.tar.bz2
```

### 5. Install third-party dependencies

**Diffusers:**
```bash
cd third_party/diffusers
pip install -e .
cd ../..
```

**manopth:**
```bash
cd third_party/manopth
pip install -e .
cd ../..
```

**MinkowskiEngine:**

Before installation, complete the steps in [detailed steps](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2566883469) and [NVTX_DISABLE](https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-2886016764).

```bash
cd third_party/MinkowskiEngine
sudo apt install libopenblas-dev
export CUDA_HOME=/usr/local/cuda-12.4  # Adjust to your CUDA version
python setup.py install --blas=openblas
cd ../..
```

**nflows:**
```bash
cd third_party/nflows
pip install -e .
cd ../..
```

**pytorch_kinematics:**
```bash
cd third_party/pytorch_kinematics
pip install -e .
cd ../..
```

### 6. Install DexLearn
```bash
pip install -e .
pip install numpy==1.26.4
pip install hydra-core
```

### 7. Fix potential MKL errors (if needed)
If you encounter errors like `undefined symbol: iJIT_NotifyEvent`:
```bash
conda install -c conda-forge mkl=2020.2 -y
```

## Mingrui

### Installation
```bash
pip install trimesh
pip install 'pyglet<2'
pip install chumpy
pip install opencv-python
pip install 'numpy<1.24'

cd third_party/manopth
pip install -e .
```

### Previous
```bash
# from mingrui

# train
CUDA_VISIBLE_DEVICES=x python -m dexlearn.train exp_name=debugx algo=nflow data=bodex_tabletop_xxx
# test
CUDA_VISIBLE_DEVICES=x python -m dexlearn.sample -e bodex_tabletop_xxx_nflow_debugx
```

### Human Grasp

Export the dataset path:
```bash
# on local
export AnyScaleGraspDataset=/data/dataset/AnyScaleGrasp
# on server
export AnyScaleGraspDataset=/data/mingrui/dataset/AnyScaleGrasp
```

Train:
```bash
# single wrist pose, normalizing flow
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=h_nflow data=human exp_name=<XXX> 
# single wrist pose, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=h_diffusion data=human exp_name=<XXX>

# bimanual wrist pose, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=hbi_diffusion data=humanbi exp_name=<XXX>

CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<XXX>
```

Sample (Inference): 
```bash
# single wrist pose, test on human_obj, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=human algo=h_diffusion test_data=human_obj exp_name=<XXX>
# single wrist pose, test on DGN, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=human algo=h_diffusion test_data=DGN exp_name=<XXX>

# bimanual wrist pose, test on DGN_grasp_type, diffusion
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample data=humanbi algo=hbi_diffusion test_data=DGN_grasp_type ckpt=050000 exp_name=<XXX>

CUDA_VISIBLE_DEVICES=0  python dexlearn/main.py task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<XXX>
```

Visualize test results:
```bash
# test on DGN, normalizing flow
python dexlearn/main.py task=visualize_test data=human algo=h_nflow test_data=DGN exp_name=<XXX>

# single wrist pose, test on human_obj, diffusion
python dexlearn/main.py task=visualize_test data=human algo=h_diffusion test_data=human_obj exp_name=<XXX>
# single wrist pose, test on DGN, diffusion
python dexlearn/main.py task=visualize_test data=human algo=h_diffusion test_data=DGN exp_name=<XXX>

# bimanual wrist pose, test on DGN_grasp_type, diffusion
python dexlearn/main.py task=visualize_test data=humanbi algo=hbi_diffusion test_data=DGN_grasp_type ckpt=050000 exp_name=<XXX>
```