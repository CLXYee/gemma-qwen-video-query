#!/bin/bash
# =======================================================
# Jetson Environment Builder for Gemma3 Video Agent
# Smart version (auto-detects JetPack/CUDA/Python, links deps)
# =======================================================

set -e  # Exit on error

ENV_NAME="gemma3"
PYTHON_VERSION=${1:-3.10}
REQUIREMENTS_FILE="requirements.txt"
JETSON_PYPI_BASE="https://pypi.jetson-ai-lab.io"

echo "----------------------------------------------------"
echo "Setting up environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "----------------------------------------------------"

# --------------------------
#  Create environment
# --------------------------
if command -v conda &> /dev/null; then
    echo "[INFO] Conda detected → creating environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
else
    echo "[INFO] Using Python venv..."
    python$PYTHON_VERSION -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

pip install --upgrade pip wheel setuptools

# --------------------------
#  Detect JetPack / CUDA
# --------------------------
echo "----------------------------------------------------"
echo "Detecting JetPack and CUDA versions..."
echo "----------------------------------------------------"

JETPACK_VERSION=""
CUDA_VERSION="cpu"

if [[ -f /etc/nv_tegra_release ]]; then
    JETPACK_VERSION=$(grep -oP 'R[0-9]+.[0-9]+' /etc/nv_tegra_release | sed 's/R//')
    echo "[INFO] Detected JetPack $JETPACK_VERSION"
else
    echo "[WARN] Could not detect JetPack version (non-Jetson system?)"
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    echo "[INFO] CUDA detected: $CUDA_VERSION"
else
    echo "[INFO] CUDA not found, defaulting to CPU mode."
fi

# --------------------------
#  Choose dependency index
# --------------------------
case $JETPACK_VERSION in
    6.*)
        JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126"
        ;;
    5.*)
        JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp5/cu118"
        ;;
    *)
        JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126"
        ;;
esac

echo "[INFO] Using Jetson PyPI index: $JETSON_INDEX_URL"

# --------------------------
#  Check existing deps
# --------------------------
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
SYSTEM_SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || echo "/usr/lib/python$PYTHON_VERSION/site-packages")

echo "----------------------------------------------------"
echo "Checking for system-installed Jetson dependencies..."
echo "----------------------------------------------------"

PACKAGES=("torch" "torchvision" "jetson_utils")
for PKG in "${PACKAGES[@]}"; do
    if python3 -c "import $PKG" &> /dev/null; then
        PKG_PATH=$(python3 -c "import $PKG, os; print(os.path.dirname($PKG.__file__))")
        if [ ! -d "$SITE_PACKAGES/$PKG" ]; then
            echo "[LINK] Linking $PKG from $PKG_PATH → $SITE_PACKAGES"
            ln -s "$PKG_PATH" "$SITE_PACKAGES/$PKG"
        else
            echo "[SKIP] $PKG already linked in environment"
        fi
    else
        echo "[INSTALL] $PKG not found system-wide → will install."
    fi
done

# --------------------------
#  Install PyTorch
# --------------------------
echo "----------------------------------------------------"
echo "Installing PyTorch (based on CUDA=$CUDA_VERSION)..."
echo "----------------------------------------------------"

if ! python -c "import torch" &> /dev/null; then
    case $CUDA_VERSION in
        12.6)
            pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
            ;;
        11.*)
            pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
            ;;
        cpu)
            pip install torch torchvision torchaudio
            ;;
        *)
            pip install torch torchvision torchaudio
            ;;
    esac
else
    echo "[INFO] torch already available in environment."
fi

# --------------------------
#  Ensure jetson-utils video features
# --------------------------
echo "----------------------------------------------------"
echo "Ensuring jetson-utils with videoSource/videoOutput..."
echo "----------------------------------------------------"

if ! python -c "import jetson_utils; from jetson_utils import videoSource, videoOutput" &> /dev/null; then
    echo "[INSTALL] Installing compatible jetson-utils from $JETSON_INDEX_URL"
    pip install --extra-index-url $JETSON_INDEX_URL jetson-utils
else
    echo "[OK] jetson-utils already includes videoSource/videoOutput"
fi

# --------------------------
#  Install remaining dependencies
# --------------------------
echo "----------------------------------------------------"
echo "Installing remaining dependencies..."
echo "----------------------------------------------------"

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

# --------------------------
#  Cleanup
# --------------------------
echo "----------------------------------------------------"
echo "Cleaning up pip cache..."
echo "----------------------------------------------------"
pip cache purge || true

# --------------------------
#  Finish
# --------------------------
echo "----------------------------------------------------"
echo "Environment setup complete!"
echo "To activate, run:"
if command -v conda &> /dev/null; then
    echo "    conda activate $ENV_NAME"
else
    echo "    source $ENV_NAME/bin/activate"
fi
echo
echo "Then launch:"
echo "    python video_query.py"
echo "    python video_query.py --on_video"
echo "    python video_query.py --model_id google/gemma-3-4b-it"
echo "----------------------------------------------------"
