#video_agent.py
import threading
import time
from jetson_utils import cudaMemcpy
from utils.utils import cudaToNumpy
import numpy as np
from PIL import Image
import subprocess
import queue
import csv
import os

class LiveVideoAgent:
    def __init__(self, describer, video_source, video_output, 
                 prompt_history=None, skip_during_inference=True, 
                 prompt=None, max_tokens=16,
                 save_output = True, output_file = "prompt_history.csv",
                 save_video = False, video_path = "output.mp4",
                 on_server = None):
        
        self.describer = describer
        self.video_source = video_source
        self.video_output = video_output
        self.prompt_history = prompt_history or []
        self.skip_during_inference = skip_during_inference
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.save_output = save_output
        self.save_video = save_video
        self.on_server = on_server

        if self.save_output:
            self.prompt_history_file = output_file
        if self.save_video:
            self.video_path = video_path
            self.fps = 15 # Adjust if needed
            self.ffmpeg_process = None

        self.running = False
        self.inference_thread = None

        self.latest_cuda_frame = None
        self.last_caption = "Loading..."
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_lock = threading.Lock()

        self.display_thread = None
    
    def on_frame(self, frame):
        if frame is None:
            return

        try:
            with self.frame_lock:
                self.latest_cuda_frame = frame

            if self.inference_thread is None or not self.inference_thread.is_alive():
                self.inference_thread = threading.Thread(
                    target=self._run_inference,
                    args=(frame,),
                    daemon=True
                )
                self.inference_thread.start()

        except Exception as e:
            print(f"[LiveVideoAgent] ERROR: {e}")

    def _run_inference(self, cuda_frame):
        try:
            np_frame = cudaToNumpy(cuda_frame)
            #np_frame = Image.fromarray(np_frame,'RGB')
            cur_time = time.time()
            description = self.describer.describe_frame(np_frame,self.prompt,self.max_tokens)
            print("Inference time: {:.2f}s".format(time.time() - cur_time))
            self.last_caption = description
            self.prompt_history.append({"timeframe": time.time(), "description": description})

            # Save to CSV every 5 new entries
            if len(self.prompt_history) % 5 == 0:
                if (self.save_output):
                    file_exists = os.path.isfile(self.prompt_history_file)
                    with open(self.prompt_history_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=["timeframe", "description"])
                        if not file_exists:
                            writer.writeheader()
                        for entry in self.prompt_history[-5:]:  
                            writer.writerow(entry)
                self.prompt_history = [] # Clear written history

        except Exception as e:
            print(f"[Error in inference]: {e}")
            import traceback
            traceback.print_exc()

    def display_loop(self):
        print("[Display] started")

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

                    if self.save_video and self.ffmpeg_process:
                        try:
                            np_frame = cudaToNumpy(annotated)
                            self.ffmpeg_process.stdin.write(np_frame.astype(np.uint8).tobytes())
                        except Exception as e:
                            print(f"[FFmpeg] Error writing frame {e}")
                        
                except Exception as e:
                    print(f"[Display] Render error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("[Display] Cannot find valid frame to render")
                time.sleep(1)

            time.sleep(0.1)

    def start(self):
        """Start live video processing."""
        print("[LiveVideoAgent] Starting...")
        self.running = True
        if self.save_video:
            self.ffmpeg_process = subprocess.Popen([
                'ffmpeg', '-y', '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{self.video_output.width}x{self.video_output.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-an',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                self.video_path
            ], stdin=subprocess.PIPE)
        self.video_source.start(self.on_frame)
        
    def stop(self):
        """Stop all processes."""
        print("[LiveVideoAgent] Stopping...")
        self.running = False
        self.video_source.stop()
        if self.save_video and self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)
