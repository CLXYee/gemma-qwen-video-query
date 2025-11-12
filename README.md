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
##### 2.1 On Jetson Devices
A ready-to-run setup script is included for Jetson devices.

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
* Check Jetpack version and install respective dependencies
* Install all other dependencies 

> ⚠️ If you don't have Conda installed, the script will automatically fall back to Python `venv`.
> ⚠️ For Jetpack 5.1.2, manually install [torch](https://entuedu-my.sharepoint.com/my?id=%2Fpersonal%2Fc230032%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FIntern%2FDemo%2Ftorch%2D2%2E6%2E0a0%2Bgit1eba9b3%2Dcp310%2Dcp310%2Dlinux%5Faarch64%2Ewhl&parent=%2Fpersonal%2Fc230032%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FIntern%2FDemo&ga=1) and [torchvision](https://entuedu-my.sharepoint.com/my?id=%2Fpersonal%2Fc230032%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FIntern%2FDemo%2Ftorchvision%2D0%2E21%2E0%2B7af6987%2Dcp310%2Dcp310%2Dlinux%5Faarch64%2Ewhl&parent=%2Fpersonal%2Fc230032%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FIntern%2FDemo&ga=1) wheels. Then, run `pip install <filename>` in the <b>activated environment</b>


##### 2.2 On Non-Jetson Devices

An environment building script is included for Non-Jetson devices.
```bash
chmod +x build_env_notjetson.sh
./build_env_notjetson.sh 
``` 
This script will:

* Create a Conda (or venv) environment named `video_query`
* Install all dependencies needed for Python 3.10

---

### 3. Activate the environment

#### If using Conda:

```bash
conda activate gemma3
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
├── requirements.txt        # Python dependencies
├── video_query.py          # Main entry point
├── video_agent.py          # Inference + display loop
├── camera.py               # Video source (Jetson camera input)
├── display.py              # Pygame-based safe video output
├── model.py                # Gemma3 model class wrapper
└── utils/                  # Helper modules (CUDA utils, image tools, etc.)
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
