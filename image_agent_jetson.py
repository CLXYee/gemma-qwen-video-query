# live_image_agent.py
import threading
import time
import csv
import os
from jetson_utils import cudaMemcpy
from utils.utils import cudaToNumpy
import numpy as np

class LiveImageAgent:
    """
    Live image agent for processing frames from an ImageSource.
    Similar to LiveVideoAgent, but assumes ImageSource and VideoOutput
    are created externally and passed in.
    """
    def __init__(self, describer, image_source, video_output, 
                 prompt_history=None, prompt=None, max_tokens=16,
                 save_output=True, output_file="prompt_history.csv",
                 save_video=False, video_path="output.mp4"):
        
        self.describer = describer
        self.image_source = image_source
        self.video_output = video_output
        self.prompt_history = prompt_history or []
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.save_output = save_output
        self.output_file = output_file
        self.save_video = save_video
        self.video_path = video_path

        self.latest_cuda_frame = None
        self.last_caption = "Loading..."
        self.frame_lock = threading.Lock()
        self.inference_thread = None
        self.running = False
        self.catch_time = []
        self.i = 1

        # FFmpeg process placeholder if saving video
        self.ffmpeg_process = None
    
    def on_frame(self, frame):
        """Receive frame from ImageSource and trigger inference."""
        if frame is None:
            return

        try:
            with self.frame_lock:
                self.latest_cuda_frame = frame

            # Only allow one inference thread at a time
            if self.inference_thread is None or not self.inference_thread.is_alive():
                self.inference_thread = threading.Thread(
                    target=self._run_inference, args=(frame,), daemon=True
                )
                self.inference_thread.start()
        except Exception as e:
            print(f"[LiveImageAgent] on_frame error: {e}")

    def _run_inference(self, cuda_frame):
        """Run description on frame asynchronously."""
        try:
            np_frame = cudaToNumpy(cuda_frame)
            start_time = time.time()
            description = self.describer.describe_frame(np_frame, self.prompt, self.max_tokens)
            elapsed = time.time() - start_time
            print(f"[{self.i}/100] Inference time: {elapsed:.2f}s")
            self.catch_time.append(elapsed)
            self.i += 1
            if len(self.catch_time) == 100:
                print("[PROCESS STOPPING] Average inference time: {:.2f}s".format(np.mean(self.catch_time)))
                self.stop()

            self.last_caption = description
            self.prompt_history.append({"timeframe": time.time(), "description": description})

            # Save to CSV every 5 entries
            if len(self.prompt_history) % 5 == 0 and self.save_output:
                file_exists = os.path.isfile(self.output_file)
                with open(self.output_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["timeframe", "description"])
                    if not file_exists:
                        writer.writeheader()
                    for entry in self.prompt_history[-5:]:
                        writer.writerow(entry)
                self.prompt_history = []

        except Exception as e:
            print(f"[LiveImageAgent] inference error: {e}")
    
    def display_loop(self):
        """Render the latest frame with caption."""
        while self.running:
            frame_to_render = None
            with self.frame_lock:
                if self.latest_cuda_frame is not None:
                    try:
                        frame_to_render = cudaMemcpy(self.latest_cuda_frame)
                        caption = self.last_caption or "Loading..."
                    except Exception as e:
                        print(f"[Display] Failed to copy frame: {e}")
                        frame_to_render = None

            if frame_to_render is not None:
                try:
                    annotated = self.video_output.overlay_text(frame_to_render, caption, position=(10, 30))
                    self.video_output.render(annotated)
                except Exception as e:
                    print(f"[Display] Render error: {e}")
            else:
                time.sleep(0.1)

            time.sleep(0.05)

    def start(self):
        """Start receiving frames from ImageSource."""
        print("[LiveImageAgent] Starting...")
        self.running = True
        self.image_source.start(self.on_frame)

    def stop(self):
        """Stop all processes."""
        print("[LiveImageAgent] Stopping...")
        self.running = False
        self.image_source.stop()
