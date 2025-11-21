#!/bin/bash
# =======================================================
# Jetson Environment Builder (Gemma3 Video Query)
# Cross-compatible with JetPack 5.x–6.x and Python 3.8–3.11
# Auto-detects CUDA/Python version, handles wheel sources,
# and installs verified stable dependencies.
# =======================================================

set -e
ENV_NAME="video_query"
JETSON_PYPI_BASE="https://pypi.jetson-ai-lab.io"
DRY_RUN=false

export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

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
    echo " [WARNING] Using Python 3.8. Newer transformers version is incompatible with Python 3.8 but is required for Gemma and Qwen. Please upgrade to a newer version."
elif (( JETPACK_VERSION == 5 )); then
    PYTHON_VERSION=3.10 
    echo " [WARNING] Using Python 3.10 for Jetpack 5.x. If using CUDA, please ensure to use a compatible torch and torchvision version."
else
    PYTHON_VERSION=3.10
fi
echo "[INFO] Target Python version: $PYTHON_VERSION"

# =======================================================
# Determine Jetson PyPI source
# =======================================================
case $JETPACK_VERSION in
    6) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp6/cu126" ;;
    5) JETSON_INDEX_URL="$JETSON_PYPI_BASE/jp5/cu118" ;;
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
    echo " Check if the information are correct before running the script"
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

pip install numpy==1.26.4

if python -c "import torch" &> /dev/null; then
    echo "[OK] PyTorch already installed."
else
    echo "Installing PyTorch"
    if [[ "$PYTHON_VERSION" == "3.8" ]]; then
        # --- NOTE: Verified Ultralytics wheels (torch 2.2.0 + torchvision 0.17.2)
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
        pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    else
        # --- NOTE: Use official PyTorch CUDA wheels for JetPack 6+
        case $CUDA_VERSION in
            12.*) pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1 https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1 https://pypi.jetson-ai-lab.io/jp6/cu126/+f/014/eff8ba676c7a3/bitsandbytes-0.47.0.dev0-cp310-cp310-linux_aarch64.whl#sha256=014eff8ba676c7a3830b9430744115af50790d2f7ff1b57f155a8839bcc39104;;            
            11.*)
                echo "Installing custom PyTorch wheels from local repo..."
                cat ../utils/jp5_wheels/torch_wheel_part_* > /tmp/torch_custom.tar.gz
                tar -xzvf /tmp/torch_custom.tar.gz -C /tmp
                pip install /tmp/torch-*.whl
                pip install ../utils/jp5_wheels/torchvision*.whl
                ;;
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

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=$(which python3)

# 1) Detect target site-packages for the active Python environment
TARGET_SITE_PACKAGES=$($PYTHON_BIN - <<'PY'
import site, sys
paths = [site.getusersitepackages()] + site.getsitepackages()
for p in paths:
    if p.startswith(sys.prefix):
        print(p)
        break
PY
)

echo "Detected target site-packages: $TARGET_SITE_PACKAGES"

# 2) Gather system dist-packages candidates (force look in system dist-packages)
#    we pick site.getsitepackages() entries that end with 'dist-packages' and NOT in sys.prefix
SYSTEM_DISTS=$($PYTHON_BIN - <<'PY'
import site, sys
dists = []
for p in site.getsitepackages():
    if p.endswith('dist-packages') and not p.startswith(sys.prefix):
        dists.append(p)
# Also include /usr/lib/pythonX.Y/dist-packages fallback if not already present
# (construct using sys.version_info)
fallback = f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
if fallback not in dists:
    dists.append(fallback)
# print one per line
print("\n".join(dists))
PY
)

echo "System dist-packages candidates:"
echo "$SYSTEM_DISTS"

# 3) Look for jetson_utils in each system dist-packages candidate
SOURCE_PARENT=""
while IFS= read -r distp; do
    [ -z "$distp" ] && continue
    if [ -d "$distp/jetson_utils" ] || [ -f "$distp/jetson_utils_python.so" ]; then
        SOURCE_PARENT="$distp"
        break
    fi
done <<< "$SYSTEM_DISTS"

if [ -n "$SOURCE_PARENT" ]; then
    echo "Found jetson-utils in system dist-packages: $SOURCE_PARENT"

    # Prepare source paths (either directory or file may exist)
    SRC_PKG_DIR="$SOURCE_PARENT/jetson_utils"
    SRC_SO="$SOURCE_PARENT/jetson_utils_python.so"

    # ensure destination exists
    sudo mkdir -p "$TARGET_SITE_PACKAGES"

    # Compare realpaths to avoid copying into itself (same env case)
    if [ -e "$SRC_PKG_DIR" ]; then
        SRC_REAL=$(realpath "$SRC_PKG_DIR")
        DST_REAL=$(realpath "$TARGET_SITE_PACKAGES/jetson_utils" 2>/dev/null || true)

        if [ -n "$DST_REAL" ] && [ "$SRC_REAL" = "$DST_REAL" ]; then
            echo "Package directory is already the same as target. No copy needed:"
            echo "  $SRC_REAL"
        else
            echo "Copying package directory from system dist to target site-packages..."
            sudo cp -r "$SRC_PKG_DIR" "$TARGET_SITE_PACKAGES/"
            echo "Copied: $SRC_PKG_DIR -> $TARGET_SITE_PACKAGES/"
        fi
    fi

    if [ -f "$SRC_SO" ]; then
        # If .so already exists at destination, compare real paths
        DST_SO="$TARGET_SITE_PACKAGES/jetson_utils_python.so"
        if [ -e "$DST_SO" ] && [ "$(realpath "$SRC_SO")" = "$(realpath "$DST_SO")" ]; then
            echo "Binary .so already present and identical. No copy needed."
        else
            echo "Copying binary .so from system dist to target site-packages..."
            sudo cp "$SRC_SO" "$TARGET_SITE_PACKAGES/"
            echo "Copied: $SRC_SO -> $TARGET_SITE_PACKAGES/"
        fi
    fi

    echo "Done. jetson-utils copied from system dist-packages to target site-packages."

else
    echo "No jetson-utils found in system dist-packages candidates."
    echo "Now checking whether jetson-utils already exists in the target site-packages..."

    # check in target site-packages
    if [ -d "$TARGET_SITE_PACKAGES/jetson_utils" ] || [ -f "$TARGET_SITE_PACKAGES/jetson_utils_python.so" ]; then
        echo "jetson-utils already present in the target site-packages: $TARGET_SITE_PACKAGES"
        echo "No action needed."
    else
        echo "ERROR: jetson-utils not found in system dist-packages nor in target site-packages."
        echo "You can either:"
        echo "  - build the wheel and pip install it into this environment"
        echo "  - or place the built files into: $TARGET_SITE_PACKAGES/"
        exit 1
    fi
fi

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
            "numpy==1.26.4"
            "Pillow"
            "pygame"
        )
        ;;
esac

for pkg in "${PYTHON_PACKAGES[@]}"; do
    pip install "$pkg" || echo "[WARN] Failed to install $pkg — continuing..."
done
pip install termcolor tabulate docker ffmpeg matplotlib
pip install accelerate
sudo apt install -y ffmpeg mesa-utils

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
