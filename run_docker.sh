#!/bin/bash
set -euo pipefail

# -------------------------------
# Detect JetPack version (host)
# -------------------------------
JETPACK_VERSION=""
if [[ -f /etc/nv_tegra_release ]]; then
    L4T_VERSION=$(grep -oP 'R[0-9]+' /etc/nv_tegra_release | tr -d 'R')
    echo "[INFO] Detected L4T R${L4T_VERSION}"

    if (( L4T_VERSION >= 36 )); then
        JETPACK_VERSION=6
    elif (( L4T_VERSION >= 34 )); then
        JETPACK_VERSION=5
    elif (( L4T_VERSION >= 32 )); then
        JETPACK_VERSION=4
    else
        echo "❌ Unsupported L4T version: $L4T_VERSION"
        exit 1
    fi
    echo "[INFO] Mapped to JetPack ${JETPACK_VERSION}.x"
else
    echo "⚠️ /etc/nv_tegra_release not found. Are you on a Jetson? Exiting."
    exit 1
fi

# -------------------------------
# Map JetPack version to base image
# -------------------------------
case "$JETPACK_VERSION" in
    4) BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.7.1-py3" ;;
    5) BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.3.1-py3" ;;
    6) BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3" ;;
    *) echo "❌ Unsupported JetPack version: $JETPACK_VERSION"; exit 1 ;;
esac

# Trim whitespace just in case
BASE_IMAGE=$(echo "$BASE_IMAGE" | tr -d '[:space:]')
echo "[INFO] Using Docker base image: '$BASE_IMAGE'"

# -------------------------------
# Build Docker image if not exists
# -------------------------------
IMAGE_NAME="gemma_qwen"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[INFO] Building Docker image '$IMAGE_NAME'..."
    cat > Dockerfile <<EOF
FROM $BASE_IMAGE
WORKDIR /workspace
COPY . /workspace

# -------------------------------
# Install build dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3-venv python3-pip git cmake build-essential \
    meson ninja-build pkg-config \
    libglib2.0-dev libssl-dev libsctp-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libnice-dev\
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav gstreamer1.0-tools \
    libglew-dev libgles2-mesa-dev libsoup2.4-dev \
    libprotobuf-dev protobuf-compiler libjson-glib-dev \
    libgstrtspserver-1.0 libgstrtspserver-1.0-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Build gst-plugins-bad (WebRTC) if not present
# -------------------------------
RUN if [ ! -f /usr/include/gst/webrtc/webrtc.h ]; then \
        echo "[INFO] gst-webrtc not found, building from source..."; \
        cd /tmp && \
        git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-bad.git && \
        cd gst-plugins-bad && \
        git checkout 1.19.2 && \
        meson setup build \
            -Dexamples=disabled \
            -Dtests=disabled \
            -Dopenh264=disabled \
            -Donnx=disabled \
            -Dwebrtc=enabled \
            -Donvif=disabled \
            -Dopencv=disabled && \
        ninja -C build && \
        ninja -C build install && \
        ldconfig; \
    else \
        echo "[INFO] gst-webrtc already available"; \
    fi

RUN mkdir -p /usr/include/gst && \
    ln -sf /usr/include/gstreamer-1.0/gst/webrtc /usr/include/gst/webrtc

CMD ["bash"]
EOF

    docker build -t "$IMAGE_NAME" .
else
    echo "[INFO] Docker image '$IMAGE_NAME' already exists. Skipping build."
fi

# -------------------------------
# Run container and launch build_env.sh
# -------------------------------
echo "[INFO] Running container '$IMAGE_NAME'..."
docker run -it --rm \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$(pwd)":/workspace \
    -v /usr/include/gstreamer-1.0/gst/webrtc:/usr/include/gstreamer-1.0/gst/webrtc \
    -v /usr/include/gstreamer-1.0/gst:/usr/include/gst \
    gemma_qwen \
    bash -c "
        echo '[INFO] Binding WebRTC headers into container...'

        # Ensure link location exists
        mkdir -p /usr/include/gst

        # Link gst → gstreamer-1.0/gst
        ln -sf /usr/include/gstreamer-1.0/gst /usr/include/gst/
        echo '[INFO] Linked /usr/include/gst → /usr/include/gstreamer-1.0/gst'

        chmod +x /workspace/docker_build_env.sh
        /workspace/docker_build_env.sh
        exec bash
    "

