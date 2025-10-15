#video_agent.py
import threading
import time
from jetson_utils import cudaMemcpy
from utils.utils import cudaToNumpy
import queue
import csv
import os

class LiveVideoAgent:
    def __init__(self, describer, video_source, video_output, 
                 prompt_history=None, skip_during_inference=True, 
                 prompt=None, max_tokens=16,
                 save_output = True, output_file = "prompt_history.csv",
                 save_video = False, video_path = "output.mp4"):
        self.describer = describer
        self.video_source = video_source
        self.video_output = video_output
        self.prompt_history = prompt_history or []
        self.skip_during_inference = skip_during_inference
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.save_output = save_output
        self.save_video = save_video

        if self.save_output:
            self.prompt_history_file = output_file
        if self.save_video:
            self.video_file = video_path

        self.running = False
        self.inference_thread = None

        self.latest_cuda_frame = None
        self.last_caption = "Loading..."
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_lock = threading.Lock()

        # Display thread
        self.display_thread = None
    
    def on_frame(self, frame):
        if frame is None:
            print("[LiveVideoAgent] WARNING: frame is None")
            return

        try:
            # Store frame for display
            with self.frame_lock:
                self.latest_cuda_frame = frame

            # Run inference
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
            cur_time = time.time()
            description = self.describer.describe_frame(np_frame,self.prompt,self.max_tokens)
            print("Inference time: {:.2f}s".format(time.time() - cur_time))
            self.last_caption = description
            self.prompt_history.append({"timeframe": time.time(), "description": description})

            # Save to CSV every 5 new entries
            if (self.save_output):
                if len(self.prompt_history) % 5 == 0:
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

                except Exception as e:
                    print(f"[Display] Render error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("[Display] Cannot find valid frame to render")
                time.sleep(1)

            time.sleep(0.15)

    def start(self):
        """Start live video processing."""
        print("[LiveVideoAgent] Starting...")
        self.running = True
        self.video_source.start(self.on_frame)


    def stop(self):
        """Stop all processes."""
        print("[LiveVideoAgent] Stopping...")
        self.running = False
        self.video_source.stop()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)
