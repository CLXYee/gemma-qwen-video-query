#!/bin/bash
# ==============================================
# Jetson Environment Builder for Gemma3 Video Agent
# Flexible for multiple JetPack and Python versions
# ==============================================

set -e  # Exit on any error

ENV_NAME="gemma3"
PYTHON_VERSION=${1:-3.10}   # Allow optional argument to override Python version
REQUIREMENTS_FILE="requirements.txt"

echo "----------------------------------------------------"
echo "Setting up environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "----------------------------------------------------"

# Detect Conda or fallback to venv
if command -v conda &> /dev/null
then
    echo "Conda detected. Creating environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
else
    echo "Conda not found. Using Python venv..."
    python$PYTHON_VERSION -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

echo "----------------------------------------------------"
echo "Detecting JetPack / CUDA version..."
echo "----------------------------------------------------"

# Detect CUDA version if installed
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    echo "CUDA detected: $CUDA_VERSION"
else
    echo "CUDA not detected. Installing CPU PyTorch version."
    CUDA_VERSION="cpu"
fi

# Install PyTorch dynamically based on CUDA availability
pip install --upgrade pip wheel setuptools

if [ "$CUDA_VERSION" == "cpu" ]; then
    echo "Installing CPU PyTorch..."
    pip install torch torchvision torchaudio
else
    echo "Installing PyTorch for CUDA $CUDA_VERSION..."
    # Map CUDA version to official NVIDIA wheel URL if needed
    case $CUDA_VERSION in
        12.6)
            TORCH_WHL="torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0"
            INDEX_URL="https://download.pytorch.org/whl/cu126"
            ;;
        11.*)
            TORCH_WHL="torch torchvision torchaudio"
            INDEX_URL="https://download.pytorch.org/whl/cu113"  # adjust as needed
            ;;
        *)
            TORCH_WHL="torch torchvision torchaudio"
            INDEX_URL=""
            ;;
    esac

    if [ -z "$INDEX_URL" ]; then
        pip install $TORCH_WHL
    else
        pip install $TORCH_WHL --extra-index-url $INDEX_URL
    fi
fi

echo "----------------------------------------------------"
echo "Installing remaining dependencies..."
echo "----------------------------------------------------"

# Create requirements.txt if missing
if [ ! -f "$REQUIREMENTS_FILE" ]; then
cat > $REQUIREMENTS_FILE <<EOL
transformers
jetson-utils
numpy
Pillow
pygame
argparse
EOL
fi

pip install -r $REQUIREMENTS_FILE

echo "----------------------------------------------------"
echo "Cleaning up pip cache..."
echo "----------------------------------------------------"
pip cache purge || true

echo "----------------------------------------------------"
echo "Environment setup complete!"
echo "To activate, run:"
echo
if command -v conda &> /dev/null
then
    echo "    conda activate $ENV_NAME"
else
    echo "    source $ENV_NAME/bin/activate"
fi
echo
echo "Then launch:"
echo "    python video_query.py"
echo "    or"
echo "    python video_query.py --on_video"
echo "    or"
echo "    python video_query.py --model_id google/gemma-3-4b-it"
echo "----------------------------------------------------"
