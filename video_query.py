# video_query.py
import argparse
import time
from model import Gemma3ImageDescriber
from camera import VideoSource
from video_agent import LiveVideoAgent
from display import PyVideoOutput


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
        "--no_display",
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
        "--save_video", #TBC
        type=bool,
        default=False,
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
        default="Describe the image precisely within 10 words. Only output the description. Do not provide additional explanations.",
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
        help="Return tensor of the VideoSource (cuda, np, pt)"
    )

    args = parser.parse_args()
    parser.print_help()

    # -----------------------------
    # Initialize components
    # -----------------------------
    print(f"[INFO] Loading model and initializing video source: {args.source}")
    describer = Gemma3ImageDescriber(model_id = args.model_id)
    video_source = VideoSource(args.source, return_tensors=args.return_tensors)

    if args.no_display or args.on_video==False:
        video_output = None
    else:
        video_output = PyVideoOutput(width=args.width, height=args.height)

    agent = LiveVideoAgent(describer, 
                           video_source, video_output, 
                           prompt=args.prompt, max_tokens=args.max_tokens,
                           save_output = args.save_output, output_file=args.output_file,
                           save_video = args.save_video, video_path = args.video_path
                           )
    agent.start()

    # -----------------------------
    # Run display or background mode
    # -----------------------------
    try:
        if args.on_video and not args.no_display:
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

# To be implemented
# Stream on server 
# save output to server args
# Video saving
