#!/bin/bash
# ==============================================
# Jetson Environment Builder for Gemma3 Video Agent
# Compatible with JetPack 6.x (CUDA 12.2 / cuDNN 9)
# ==============================================

set -e  # Exit on any error
ENV_NAME="gemma3"
PYTHON_VERSION="3.10"
REQUIREMENTS_FILE="requirements.txt"

echo "----------------------------------------------------"
echo "Setting up environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "----------------------------------------------------"

# Check if conda exists, otherwise fallback to venv
if command -v conda &> /dev/null
then
    echo "Conda detected. Creating environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
else
    echo "Conda not found. Falling back to Python venv..."
    python$PYTHON_VERSION -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

echo "----------------------------------------------------"
echo "Installing PyTorch for JetPack 6 (CUDA 12.2)..."
echo "----------------------------------------------------"

# Install PyTorch (official NVIDIA wheel)
pip install --upgrade pip wheel setuptools
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu122

echo "----------------------------------------------------"
echo "Installing remaining dependencies..."
echo "----------------------------------------------------"

# Create requirements.txt if missing
if [ ! -f "$REQUIREMENTS_FILE" ]; then
cat > $REQUIREMENTS_FILE <<EOL
torch==2.3.0
transformers==4.44.2
jetson-utils==0.5.1
numpy==1.26.4
Pillow==10.4.0
pygame==2.5.2
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
