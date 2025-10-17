#!/bin/bash
# =======================================================
# Jetson Environment Builder (Gemma3 Video Query)
# Auto-detects JetPack / CUDA / Python version
# Supports reusing existing envs + dry-run mode
# =======================================================

set -e
ENV_NAME="gemma3"
REQUIREMENTS_FILE="requirements.txt"
JETSON_PYPI_BASE="https://pypi.jetson-ai-lab.io"

DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --check)
            DRY_RUN=true
            ;;
    esac
done

echo "----------------------------------------------------"
echo "🔧 Detecting JetPack and system configuration..."
echo "----------------------------------------------------"

# Detect JetPack version
JETPACK_VERSION=""
if [[ -f /etc/nv_tegra_release ]]; then
    JETPACK_VERSION=$(grep -oP 'R[0-9]+.[0-9]+' /etc/nv_tegra_release | sed 's/R//')
    echo "[INFO] Detected JetPack $JETPACK_VERSION"
else
    echo "[WARN] JetPack not detected (non-Jetson system?)"
fi

# Detect CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION="cpu"
fi
echo "[INFO] CUDA version: $CUDA_VERSION"

# Determine Python version
if [[ "$JETPACK_VERSION" == "" || "$(echo "$JETPACK_VERSION < 5.1" | bc)" -eq 1 ]]; then
    PYTHON_VERSION=3.8
    echo "[INFO] JetPack ≤ 5 → using Python 3.8 (legacy Jetson)"
else
    PYTHON_VERSION=3.10
    echo "[INFO] JetPack ≥ 6 → using Python 3.10"
fi

# Determine PyPI source
case $JETPACK_VERSION in
    6.*) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126" ;;
    5.*) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp5/cu118" ;;
    4.*) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp4/cu102" ;;
    *)   JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126" ;;
esac

echo "[INFO] Using Jetson PyPI index: $JETSON_INDEX_URL"

# ------------------------------------------------------
# DRY-RUN MODE
# ------------------------------------------------------
if $DRY_RUN; then
    echo "----------------------------------------------------"
    echo "🧪 DRY-RUN MODE ENABLED — no changes will be made"
    echo "----------------------------------------------------"
    echo "Would perform the following actions:"
    echo " - Create or reuse environment '$ENV_NAME'"
    echo " - Python version: $PYTHON_VERSION"
    echo " - Install PyTorch for CUDA=$CUDA_VERSION"
    if [[ "$PYTHON_VERSION" == "3.8" ]]; then
        echo "   (Ultralytics prebuilt wheel)"
    else
        echo "   (PyTorch official CUDA wheel via $JETSON_INDEX_URL)"
    fi
    echo " - Clone and build jetson-utils from dusty-nv/jetson-utils"
    echo " - Install Python dependencies from requirements.txt"
    echo "----------------------------------------------------"
    echo "✅ Safe to run without --check to perform actual installation"
    exit 0
fi

# ------------------------------------------------------
# Create or reuse environment
# ------------------------------------------------------
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "$ENV_NAME"; then
        echo "[INFO] Environment '$ENV_NAME' already exists. Reusing..."
        conda activate $ENV_NAME
    else
        echo "[INFO] Creating new Conda environment '$ENV_NAME'..."
        conda create -y -n $ENV_NAME python=$PYTHON_VERSION
        conda activate $ENV_NAME
    fi
else
    if [[ -d "$ENV_NAME" ]]; then
        echo "[INFO] Python venv '$ENV_NAME' already exists. Reusing..."
        source $ENV_NAME/bin/activate
    else
        echo "[INFO] Creating new venv '$ENV_NAME'..."
        python$PYTHON_VERSION -m venv $ENV_NAME
        source $ENV_NAME/bin/activate
    fi
fi

pip install --upgrade pip wheel setuptools

# ------------------------------------------------------
# Install PyTorch dynamically
# ------------------------------------------------------
echo "----------------------------------------------------"
echo "Installing PyTorch for CUDA=$CUDA_VERSION / Python=$PYTHON_VERSION"
echo "----------------------------------------------------"

if python -c "import torch" &> /dev/null; then
    echo "[OK] PyTorch already installed. Linking existing build."
else
    if [[ "$PYTHON_VERSION" == "3.8" ]]; then
        echo "[INFO] Using Ultralytics prebuilt PyTorch wheels for Jetson (Python 3.8)"
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    else
        case $CUDA_VERSION in
            12.*) pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126 ;;
            11.*) pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 ;;
            cpu)  pip install torch torchvision torchaudio ;;
            *)    pip install torch torchvision torchaudio ;;
        esac
    fi
fi

# ------------------------------------------------------
# Install jetson-utils from GitHub if missing
# ------------------------------------------------------
echo "----------------------------------------------------"
echo "Installing jetson-utils from dusty-nv/jetson-utils..."
echo "----------------------------------------------------"

if python -c "import jetson_utils; from jetson_utils import videoSource, videoOutput" &> /dev/null; then
    echo "[OK] jetson-utils already installed and functional."
else
    echo "[BUILD] Building jetson-utils from source..."
    git clone --depth=1 https://github.com/dusty-nv/jetson-utils.git /tmp/jetson-utils
    cd /tmp/jetson-utils
    mkdir -p build && cd build
    cmake ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd -
    echo "[DONE] jetson-utils successfully installed."
fi

# ------------------------------------------------------
# Install remaining dependencies
# ------------------------------------------------------
if [ ! -f "$REQUIREMENTS_FILE" ]; then
cat > $REQUIREMENTS_FILE <<EOL
transformers
numpy
Pillow
pygame
argparse
EOL
fi

pip install -r $REQUIREMENTS_FILE --extra-index-url $JETSON_INDEX_URL
pip cache purge || true

# ------------------------------------------------------
# Summary
# ------------------------------------------------------
echo "----------------------------------------------------"
echo "✅ Environment setup complete!"
echo "----------------------------------------------------"
echo "JetPack:   ${JETPACK_VERSION:-Unknown}"
echo "CUDA:      ${CUDA_VERSION}"
python -c "import sys; print(f'Python:   {sys.version.split()[0]}')"
python -c "import torch; print(f'Torch:    {torch.__version__}')"
python -c "import jetson_utils; print(f'jetson-utils: {jetson_utils.__version__}')"
echo "----------------------------------------------------"
echo "To activate:"
if command -v conda &> /dev/null; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source $ENV_NAME/bin/activate"
fi
echo
echo "Then run:"
echo "  python video_query.py"
echo "  python video_query.py --on_video"
echo "----------------------------------------------------"
