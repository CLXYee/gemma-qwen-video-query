#!/bin/bash
# =======================================================
# Jetson Environment Builder (Gemma3 Video Query)
# Cross-compatible with JetPack 5.x–6.x and Python 3.8–3.11
# Auto-detects CUDA/Python version, handles wheel sources,
# and installs verified stable dependencies.
# =======================================================

set -e
ENV_NAME="gemma3"
JETSON_PYPI_BASE="https://pypi.jetson-ai-lab.io"
DRY_RUN=false

# =======================================================
# Parse arguments
# =======================================================
for arg in "$@"; do
    case $arg in
        --check) DRY_RUN=true ;;
    esac
done

echo "----------------------------------------------------"
echo "🔧 Detecting JetPack and system configuration..."
echo "----------------------------------------------------"

# =======================================================
# Detect JetPack / L4T version
# =======================================================
JETPACK_VERSION=""
if [[ -f /etc/nv_tegra_release ]]; then
    L4T_VERSION=$(grep -oP 'R[0-9]+' /etc/nv_tegra_release | tr -d 'R')
    echo "[INFO] Detected L4T R${L4T_VERSION}"

    if (( L4T_VERSION >= 36 )); then
        JETPACK_VERSION=6
    elif (( L4T_VERSION >= 34 )); then
        JETPACK_VERSION=5
    else
        JETPACK_VERSION=4
    fi
    echo "[INFO] Mapped to JetPack $JETPACK_VERSION.x"
else
    echo "[WARN] JetPack not detected (non-Jetson system?)"
fi

# =======================================================
# Detect CUDA version
# =======================================================
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION="cpu"
fi
echo "[INFO] CUDA version: $CUDA_VERSION"

# =======================================================
# Detect Python version to use
# =======================================================
if (( JETPACK_VERSION < 5 )); then
    PYTHON_VERSION=3.8
elif (( JETPACK_VERSION == 5 )); then
    PYTHON_VERSION=3.8
else
    PYTHON_VERSION=3.10
fi
echo "[INFO] Target Python version: $PYTHON_VERSION"

# =======================================================
# Determine Jetson PyPI source
# =======================================================
case $JETPACK_VERSION in
    6) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126" ;;
    5) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp5/cu114" ;;
    4) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp4/cu102" ;;
    *) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126" ;;
esac
echo "[INFO] Using Jetson PyPI index: $JETSON_INDEX_URL"

# =======================================================
# DRY-RUN MODE
# =======================================================
if $DRY_RUN; then
    echo "----------------------------------------------------"
    echo "🧪 DRY-RUN MODE ENABLED — no changes will be made"
    echo "----------------------------------------------------"
    echo "Would perform the following actions:"
    echo " - Create environment '$ENV_NAME'"
    echo " - Python version: $PYTHON_VERSION"
    echo " - CUDA version: $CUDA_VERSION"
    echo " - Install verified wheels for torch/transformers/numpy"
    echo "----------------------------------------------------"
    exit 0
fi

# =======================================================
# Create or reuse environment
# =======================================================
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "$ENV_NAME"; then
        echo "[INFO] Reusing existing conda environment '$ENV_NAME'..."
        conda activate $ENV_NAME
    else
        echo "[INFO] Creating new conda environment '$ENV_NAME'..."
        conda create -y -n $ENV_NAME python=$PYTHON_VERSION
        conda activate $ENV_NAME
    fi
else
    if [[ -d "$ENV_NAME" ]]; then
        echo "[INFO] Reusing Python venv '$ENV_NAME'..."
        source $ENV_NAME/bin/activate
    else
        echo "[INFO] Creating new venv '$ENV_NAME'..."
        python$PYTHON_VERSION -m venv $ENV_NAME
        source $ENV_NAME/bin/activate
    fi
fi

pip install --upgrade pip wheel setuptools

# =======================================================
# Install PyTorch
# =======================================================
echo "----------------------------------------------------"
echo "Installing PyTorch for CUDA=$CUDA_VERSION / Python=$PYTHON_VERSION"
echo "----------------------------------------------------"

if python -c "import torch" &> /dev/null; then
    echo "[OK] PyTorch already installed."
else
    if [[ "$PYTHON_VERSION" == "3.8" ]]; then
        # --- NOTE: Verified Ultralytics wheels (torch 2.2.0 + torchvision 0.17.2)
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    else
        # --- NOTE: Use official PyTorch CUDA wheels for JetPack 6+
        case $CUDA_VERSION in
            12.*) pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126 ;;
            11.*) pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 ;;
            cpu)  pip install torch torchvision torchaudio ;;
            *)    pip install torch torchvision torchaudio ;;
        esac
    fi
fi

# =======================================================
# Install jetson-utils 
# =======================================================
echo "----------------------------------------------------"
echo "Installing jetson-utils from source..."
echo "----------------------------------------------------"
sudo apt update
sudo apt install -y cmake build-essential git python3-dev

WORKDIR=$(pwd)
cd /tmp

if [ -d "jetson-utils" ]; then sudo rm -rf jetson-utils; fi
git clone --recursive https://github.com/dusty-nv/jetson-utils
cd jetson-utils
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ../
make -j$(nproc)
sudo make install
sudo ldconfig
cd $WORKDIR

# =======================================================
# Install Python dependencies
# =======================================================
echo "----------------------------------------------------"
echo "Installing Python dependencies..."
echo "----------------------------------------------------"

# --- NOTE: Cross-version verified versions
case $PYTHON_VERSION in
    3.8)
        PYTHON_PACKAGES=(
            "transformers==4.37.2"
            "numpy==1.24.4"
            "Pillow==10.2.0"
            "pygame==2.5.2"
        )
        ;;
    3.9|3.10|3.11)
        PYTHON_PACKAGES=(
            "transformers==4.57.0"
            "numpy==1.26.4"
            "Pillow==11.3.0"
            "pygame==2.6.1"
        )
        ;;
    *)
        PYTHON_PACKAGES=(
            "transformers"
            "numpy"
            "Pillow"
            "pygame"
        )
        ;;
esac

for pkg in "${PYTHON_PACKAGES[@]}"; do
    pip install "$pkg" || echo "[WARN] Failed to install $pkg — continuing..."
done

pip cache purge || true

# =======================================================
# Summary
# =======================================================
echo "----------------------------------------------------"
echo "✅ Environment setup complete!"
echo "----------------------------------------------------"
echo "JetPack:   ${JETPACK_VERSION:-Unknown}"
echo "CUDA:      ${CUDA_VERSION}"
python -c "import sys; print(f'Python:   {sys.version.split()[0]}')"
python -c "import torch; print(f'Torch:    {torch.__version__}')"
python -c "import jetson_utils; print(f'jetson-utils: {jetson_utils.__version__}')"
echo "----------------------------------------------------"
if command -v conda &> /dev/null; then
    echo "To activate: conda activate $ENV_NAME"
else
    echo "To activate: source $ENV_NAME/bin/activate"
fi
echo
echo "Then run: python video_query.py --on_video"
echo "----------------------------------------------------"
