#!/bin/bash
# =======================================================
# Non-Jetson Environment Builder (Gemma3 Image/Video Query)
# Python 3.8–3.11, CPU or CUDA supported
# =======================================================

set -e
ENV_NAME="video_query"
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
echo "🔧 Detecting system configuration..."
echo "----------------------------------------------------"

# =======================================================
# Detect CUDA version (if available)
# =======================================================
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION="cpu"
fi
echo "[INFO] CUDA version: $CUDA_VERSION"

# =======================================================
# Choose Python version
# =======================================================
PYTHON_VERSION=3.10
echo "[INFO] Target Python version: $PYTHON_VERSION"

# =======================================================
# DRY-RUN MODE
# =======================================================
if $DRY_RUN; then
    echo "----------------------------------------------------"
    echo "🧪 DRY-RUN MODE ENABLED — no changes will be made"
    echo "Would perform the following actions:"
    echo " - Create environment '$ENV_NAME'"
    echo " - Python version: $PYTHON_VERSION"
    echo " - Install PyTorch ($CUDA_VERSION) and verified dependencies"
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
    if [[ "$CUDA_VERSION" == "cpu" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | tr -d .)
    fi
fi

# =======================================================
# Install Python dependencies
# =======================================================
echo "----------------------------------------------------"
echo "Installing Python dependencies..."
echo "----------------------------------------------------"

PYTHON_PACKAGES=(
    "transformers==4.57.0"
    "numpy==1.26.4"
    "Pillow==11.3.0"
    "pygame==2.6.1"
    "termcolor"
    "tabulate"
    "docker"
    "accelerate"
    "ffmpeg"
)

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
echo "CUDA:      ${CUDA_VERSION}"
python -c "import sys; print(f'Python:   {sys.version.split()[0]}')"
python -c "import torch; print(f'Torch:    {torch.__version__}')"
echo "----------------------------------------------------"
if command -v conda &> /dev/null; then
    echo "To activate: conda activate $ENV_NAME"
else
    echo "To activate: source $ENV_NAME/bin/activate"
fi
echo
echo "Then run: python video_query.py --on_video"
echo "----------------------------------------------------"
