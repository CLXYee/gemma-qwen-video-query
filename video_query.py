# video_query.py
import argparse
import time
from camera import VideoSource
from video_agent import LiveVideoAgent
from display import VideoOutput


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma3 live video query with optional display"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/dev/video0",
        help="Video source (e.g. /dev/video0, rtsp://, file path)"
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=60,
        help="Video frame rate"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video display width (remember to set --on_video as well)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video display height (remember to set --on_video as well)"
    )
    parser.add_argument(
        "--on_video",
        action="store_true",
        help="Enable live video display and caption rendering"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable display output (useful for headless mode)"
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Save VLM output as a csv"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="prompt_history.csv",
        help="Save VLM output as a csv"
    )
    parser.add_argument(
        "--save_video", 
        action="store_true",
        help="Save video with VLM output"
    )
    parser.add_argument(
        "--video_path", 
        type=str,
        default="output.mp4",
        help="Path to save video with VLM output"
    )
    parser.add_argument(
        "--on_server", #TBC
        type=str,
        default="rtc",
        help="Stream live video display and caption rendering on server"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-3-4b-it",
        help="Define the model id of the VLM to be used"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image precisely.",
        help="Define prompt to pass to the VLM"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Define the maximum output token of the VLM"
    )
    parser.add_argument(
        "--return_tensors",
        type=str,
        default='cuda',
        choices=['cuda', 'np', 'pt'],
        help="Tensors output format: 'cuda' (GPU), 'np' (NumPy), 'pt' (PyTorch). Defaults to 'cuda'.",
    )


    args = parser.parse_args()
    parser.print_help()

    # -----------------------------
    # Initialize components
    # -----------------------------
    print(f"[INFO] Loading model and initializing video source: {args.source}")
    if "gemma" in args.model_id:
        from model import Gemma3ImageDescriber
        describer = Gemma3ImageDescriber(model_id = args.model_id)
    elif "Qwen" in args.model_id:
        from model import QwenImageDescriber
        describer = QwenImageDescriber(model_id=args.model_id)
    else:
        print("[Warning] Model not available yet. Stay tuned! For now, please use vision-language-models from the Gemma family")
        return
    video_source = VideoSource(args.source, video_input_framerate=args.frame_rate, return_tensors=args.return_tensors)

    if args.save_video and not args.on_video:
        print("[INFO] --save_video enabled but --on_video not detected. Automatically enabling --on_video for frame fetching and rendering")
        args.on_video = True
    if args.headless or args.on_video==False:
        video_output = None
    else:
        video_output = VideoOutput(width=args.width, height=args.height)

    agent = LiveVideoAgent(describer, 
                           video_source, video_output, 
                           prompt=args.prompt, max_tokens=args.max_tokens,
                           save_output = args.save_output, output_file=args.output_file,
                           save_video = args.save_video, video_path = args.video_path,
                           on_server = args.on_server
                           )
    agent.start()

    # -----------------------------
    # Run display or background mode
    # -----------------------------
    try:
        if args.on_video and not args.headless:
            print("[INFO] Starting video display loop...")
            while True:
                agent.display_loop()
        else:
            print("[INFO] Running without display (inference only mode)...")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, stopping agent...")
        agent.stop()


if __name__ == "__main__":
    main()

# Stream on server 
# save output to server args