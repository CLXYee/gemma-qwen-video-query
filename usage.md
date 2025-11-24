# USAGE — Gemma-Qwen Video Query

This file shows common usage examples for running the repository tools (live video, video files, and image-folder processing).

Prerequisites
- Prepare environment:
  - Jetson: ./build_env.sh (auto-selects Jetson script) or script/build_env_jetson.sh
  - Non-Jetson: script/build_env_notjetson.sh
- Ensure Python 3.10+ and required packages installed per project scripts.

Quick examples

1) Live camera (display + save prompts)
- Use a USB camera (/dev/video0) and Gemma model:
  ```
  python video_query.py --mode video --source /dev/video0 --on_video \
    --model_id google/gemma-3-4b-it \
    --save_output output/prompt_history_gemma.csv \
    --save_video --video_path output/gemma_output.mp4
  ```

2) Process a video file (headless)
- Run inference without GUI and save prompt history:
  ```
  python video_query.py --mode video --source ./input/video.mp4 --headless \
    --model_id google/gemma-3-4b-it \
    --save_output output/prompt_history_gemma.csv \
    --return_tensors np
  ```

3) Process an image folder (interactive / plotting)
- Iterate images in folder and show results with matplotlib:
  ```
  python video_query.py --mode image --image_source ./selected/Global \
    --plot plt --model_id google/gemma-3-4b-it \
    --save_output output/prompt_history_gemma.csv
  ```

4) Jetson-optimized run (use CUDA frames)
- Jetson agents and CUDA tensor return:
  ```
  python video_query.py --mode video --source /dev/video0 --on_video \
    --model_id google/gemma-3-4b-it --return_tensors cuda
  ```

Common CLI options
- --mode: image | video
- --source: camera device, path, or RTSP
- --image_source: folder for image mode
- --model_id: model name (e.g., google/gemma-3-4b-it)
- --on_video / --headless: enable GUI overlay vs headless
- --save_output <path>: CSV prompt history output
- --save_video --video_path <path>: record output video
- --plot: plt | cv2 (image-mode display backend)
- --return_tensors: cuda | np | pt (frame tensor format)

Outputs produced
- CSV prompt history (default under output/)
- Optional recorded MP4 (output/)
- On-screen overlays when not headless

Troubleshooting
- If camera doesn't open, confirm device path and permissions.
- On Jetson, confirm jetson-utils installed via provided scripts.
- If inference is slow, try smaller models or set headless mode.

If a specific example or a short script for your environment is needed, provide the target (camera, file, or Jetson) and desired outputs.
