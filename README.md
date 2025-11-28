# 🎥 Gemma3 Real-Time Video Agent (Jetson Optimized)

This project enables real-time video captioning and visual understanding using **Google’s Gemma3 vision-language model**, optimized for **NVIDIA Jetson devices**.  
It uses CUDA-accelerated inference and a lightweight **Pygame display backend** for stable video rendering.

---

## 🚀 Features

- 🔍 Real-time image understanding using **Gemma** and **Qwen** models
- ⚡ Optimized for Jetson (CUDA / Tensor Cores)  
- 🧠 Threaded inference + safe video display loop  
- 🧾 Automatic prompt logging and CSV history  
- 🪶 Minimal dependencies, fast setup  

---

## 🧩 Environment Setup

### 1. Clone this repository
```bash
git clone https://github.com/CLXYee/gemma-qwen-video-query.git
cd gemma-qwen-video-query
````

### 2. Build the environment
A ready-to-run setup script is included.

```bash
chmod +x build_env.sh
./build_env.sh
```
You can check the environment configuration before building 
```bash
./build_env.sh --check
```

This script will:

* Create a Conda (or venv) environment named `video_query`
* Check if the device is a Jetson or non-Jetson device
* Check Jetpack version and install respective dependencies
* Install all other dependencies 

> ⚠️ If you don't have Conda installed, the script will automatically fall back to Python `venv`.


---

### 3. Activate the environment

#### If using Conda:

```bash
conda activate video_query
```

#### If using venv:

```bash
source video_query/bin/activate
```

---

### 4. Verify installation

```bash
python -m torch.utils.collect_env
```

You should see:

```
CUDA available: True
GPU type: NVIDIA Orin / Xavier / Nano
```

---

## 🧠 Running the Video Agent

The main entry script is `video_query.py`.
It accepts command-line arguments for flexibility in model selection and prompts.

### Example usage

```bash
python video_query.py \
  --model_id google/gemma-3-4b-it \
  --prompt "Describe the scene in one sentence." \
  --max_new_tokens 16 \
  --on_video
```

### Arguments

| Argument           | Description                    | Default                                           |
| ------------------ | ------------------------------ | ------------------------------------------------- |
| `--model_id`       | Gemma3 model to load           | `google/gemma-3-4b-it`                            |
| `--prompt`         | Custom prompt for captioning   | `"Describe the image precisely within 10 words."` |
| `--max_new_tokens` | Maximum tokens for generation  | `16`                                              |
| `--on_video`       | Enable real-time video display | (flag only)                                       |
| `--mode`           | Toggle display mode            | `Choose between "video" and "image" mode`         |

---

## 🧰 Project Structure

```
├── build_env.sh            # Jetson environment setup script
├── usage.md                # Usage examples
├── video_query.py          # Main entry point
├── video_agent.py          # Inference + display loop for video-based query
├── image_agent.py          # Inference + display loop for image-based query
├── camera.py               # Video source (Jetson camera input)
├── display.py              # Pygame-based safe video output
├── image_source.py         # Image fetcher for video-like display
├── model.py                # Gemma3 model class wrapper
└── utils/                  # Helper modules (CUDA utils, image tools, etc.)
└── script/                 # Supporting environment setup script
└── output/                 # Sample outputs during video query (.csv, .mp4)
└── selected/               # Image samples for demo
```

---

## 🧪 Performance Tips

* If you notice **lagging inference**, reduce model size or increase display sleep:

  ```python
  time.sleep(0.02)
  ```
* If you experience **GL context errors**, ensure no other process (like `nvv4l2`) is using the camera.
* For faster warmup, use `torch.compile()` on supported Jetson builds.

---

## 🧾 License

This project is released under the **MIT License**.
© 2025 NVIDIA / Google / Contributors.

---

## 💬 Support

For Jetson-related issues:

* NVIDIA Jetson Forum: [https://forums.developer.nvidia.com/c/agx-xavier/74](https://forums.developer.nvidia.com/c/agx-xavier/74)
* PyTorch Jetson wheels: [https://forums.developer.nvidia.com/t/pytorch-for-jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
