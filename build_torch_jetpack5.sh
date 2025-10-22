#!/bin/bash
set -e

# ==========================================
#  Build PyTorch from source on JetPack 5.x
#  (Python 3.10 + CUDA 11.8)
# ==========================================

# --- Configurable versions ---
PYTORCH_VERSION="v2.1.0"     # can also try v2.2.0
TORCHVISION_VERSION="v0.16.0"
ARCH="7.2"                   # Xavier (for Orin use 8.7)

# --- Environment setup ---
echo "[1/6] Setting up environment..."
export CUDA_HOME=/usr/local/cuda
export CUDNN_INCLUDE_DIR=/usr/include
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Jetson build flags
export USE_CUDA=1
export USE_NCCL=0
export USE_SYSTEM_NCCL=0
export USE_DISTRIBUTED=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_MKLDNN=0
export USE_FBGEMM=0
export USE_OPENMP=1
export CMAKE_CUDA_ARCHITECTURES=72
export TORCH_CUDA_ARCH_LIST="7.2"
export CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc

# --- Swap setup (recommended for memory) ---
if [ ! -f /swapfile ]; then
  echo "[2/6] Creating 8GB swap (recommended for build)..."
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi

# --- Dependencies ---
echo "[3/6] Installing dependencies..."
sudo apt update
sudo apt install -y python3.10-dev cmake ninja-build git libopenblas-dev libblas-dev libeigen3-dev libatlas-base-dev

pip install --upgrade pip setuptools wheel

# --- Clone PyTorch ---
echo "[4/6] Cloning PyTorch $PYTORCH_VERSION ..."
cd ~
if [ ! -d pytorch ]; then
  git clone --branch $PYTORCH_VERSION https://github.com/pytorch/pytorch.git
fi
cd pytorch
git submodule sync
git submodule update --init --recursive

# --- Build wheel ---
echo "[5/6] Building PyTorch wheel (this takes hours)..."
python3 setup.py bdist_wheel

# --- Install wheel ---
echo "[6/6] Installing wheel..."
pip install dist/torch-*-cp310-*-linux_aarch64.whl

# --- Optional: build torchvision ---
echo "[Optional] Building torchvision $TORCHVISION_VERSION ..."
cd ~
if [ ! -d vision ]; then
  git clone --branch $TORCHVISION_VERSION https://github.com/pytorch/vision.git
fi
cd vision
export BUILD_VERSION=$TORCHVISION_VERSION
python3 setup.py bdist_wheel
pip install dist/torchvision-*-cp310-*-linux_aarch64.whl

# --- Verify ---
python3 -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "✅ Build complete! Wheel is in ~/pytorch/dist/"
