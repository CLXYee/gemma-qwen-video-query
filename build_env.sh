#!/bin/bash

# Function to detect if the device is Jetson
is_jetson() {
    if [ -f /proc/device-tree/model ] && grep -q "NVIDIA Jetson" /proc/device-tree/model 2>/dev/null; then
        return 0  # Jetson
    elif uname -a | grep -qi "tegra"; then
        return 0  # Jetson
    elif [ -f /etc/nv_tegra_release ]; then
        return 0  # Jetson
    else
        return 1  # Not Jetson
    fi
}

# Check if --check flag is passed
CHECK_MODE=false
if [[ "$1" == "--check" ]]; then
    CHECK_MODE=true
fi

if is_jetson; then
    DEVICE="Jetson"
    SCRIPT_PATH="script/build_env_jetson.sh"
else
    DEVICE="Non-Jetson"
    SCRIPT_PATH="script/build_env_notjetson.sh"
fi

# Echo device type
echo "Detected device: $DEVICE"

# Make the respective script executable
chmod +x "$SCRIPT_PATH"

# Execute the script
if $CHECK_MODE; then
    echo "Running $SCRIPT_PATH in --check mode..."
    "$SCRIPT_PATH" --check
else
    "$SCRIPT_PATH"
fi
