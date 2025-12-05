FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
WORKDIR /workspace
COPY . /workspace

# -------------------------------
# Install build dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing     python3-venv python3-pip git cmake build-essential     meson ninja-build pkg-config     libglib2.0-dev libssl-dev libsctp-dev     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libnice-dev    gstreamer1.0-plugins-base gstreamer1.0-plugins-good     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly     gstreamer1.0-libav gstreamer1.0-tools     libglew-dev libgles2-mesa-dev libsoup2.4-dev     libprotobuf-dev protobuf-compiler libjson-glib-dev     libgstrtspserver-1.0 libgstrtspserver-1.0-dev     && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Build gst-plugins-bad (WebRTC) if not present
# -------------------------------
RUN if [ ! -f /usr/include/gst/webrtc/webrtc.h ]; then         echo "[INFO] gst-webrtc not found, building from source...";         cd /tmp &&         git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-bad.git &&         cd gst-plugins-bad &&         git checkout 1.19.2 &&         meson setup build             -Dexamples=disabled             -Dtests=disabled             -Dopenh264=disabled             -Donnx=disabled             -Dwebrtc=enabled             -Donvif=disabled             -Dopencv=disabled &&         ninja -C build &&         ninja -C build install &&         ldconfig;     else         echo "[INFO] gst-webrtc already available";     fi

RUN mkdir -p /usr/include/gst &&     ln -sf /usr/include/gstreamer-1.0/gst/webrtc /usr/include/gst/webrtc

CMD ["bash"]
