# image_agent.py
import threading
import time
import os
import csv
import queue
import numpy as np
import cv2
from PIL import Image

# Optional Jetson display support
try:
    import jetson.utils
    USE_JETSON = True
except ImportError:
    USE_JETSON = False


class ImageAgent:
    def __init__(self, describer, image_folder="./selected",
                 prompt=None, max_tokens=16,
                 save_output=True, output_file="prompt_history.csv",
                 save_video=False, video_path="output.mp4"):

        self.describer = describer
        self.image_folder = image_folder
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.save_output = save_output
        self.output_file = output_file
        self.save_video = save_video
        self.video_path = video_path

        self.prompt_history = []
        self.catch_time = []
        self.i = 1
        self.running = False

        self.inference_thread = None
        self.display_thread = None
        self.frame_lock = threading.Lock()

        self.latest_frame = None
        self.last_caption = "Loading..."
        self.current_filename = None
        self.frame_queue = queue.Queue(maxsize=2)

        # Prepare image list
        if not os.path.exists(self.image_folder):
            raise FileNotFoundError(f"[ImageAgent] Image folder not found: {self.image_folder}")

        self.images = sorted([
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.index = 0

        if self.save_video:
            self.fps = 0.2  # 1 frame every 5 seconds
            self.video_writer = None

        # Set up Jetson or OpenCV display
        if USE_JETSON:
            self.display = jetson.utils.videoOutput()
        else:
            self.display = None

    # -------------------------------
    # Core image/frame processing
    # -------------------------------
    def on_frame(self, frame, filename):
        """Handle new frame input and start inference thread."""
        if frame is None:
            return

        with self.frame_lock:
            self.latest_frame = frame
            self.current_filename = filename

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = threading.Thread(
                target=self._run_inference,
                args=(frame, filename),
                daemon=True
            )
            self.inference_thread.start()

    def _run_inference(self, frame, filename):
        """Run inference on one frame (numpy array)"""
        try:
            cur_time = time.time()
            description = self.describer.describe_frame(frame, self.prompt, self.max_tokens)
            elapsed = time.time() - cur_time
            print(f"[{self.i}] {filename} - Inference time: {elapsed:.2f}s")

            self.catch_time.append(elapsed)
            self.i += 1
            self.last_caption = description

            self.prompt_history.append({
                "timeframe": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "description": description
            })

            if len(self.prompt_history) % 5 == 0:
                self._save_history()

        except Exception as e:
            print(f"[Error in inference for {filename}]: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------
    # Display and overlay
    # -------------------------------
    def _overlay_text(self, frame, text, max_width=1200, max_height=900):
        """
        Overlay caption text on image with word wrapping and consistent font size.
        """
        # Resize first
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            h, w = frame.shape[:2]

        annotated = frame.copy()

        # Convert RGB -> BGR for OpenCV
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3  # consistent size
        thickness = 1
        line_spacing = 5

        # Maximum width for text area
        max_text_width = int(w * 0.9)  # leave 10% padding on right

        # Split text into words and wrap lines
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width > max_text_width:
                    if current_line:  # push current line
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        # Overlay lines
        y0 = 40
        dy = int(cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + line_spacing)
        for i, line in enumerate(lines):
            y = y0 + i*dy
            cv2.putText(annotated, line, (20, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    def display_loop(self):
        """Continuously display frames with captions (like video)."""
        print("[Display] Starting display loop... (Press ESC to quit)")

        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                caption = self.last_caption

            if frame is None:
                time.sleep(0.1)
                continue

            annotated = self._overlay_text(frame, caption)

            # Render frame
            if USE_JETSON:
                cuda_img = jetson.utils.cudaFromNumpy(annotated)
                self.display.RenderOnce(cuda_img)
                self.display.SetStatus("Gemma3 Image Describer")
            else:
                cv2.imshow("Gemma3 Image Describer", annotated)
                key = cv2.waitKey(100)
                if key == 27:  # ESC
                    self.stop()
                    cv2.destroyAllWindows()
                    break

            # Optionally save to video
            if self.save_video and self.video_writer:
                self.video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # -------------------------------
    # File I/O and logs
    # -------------------------------
    def _save_history(self):
        """Save prompt history to CSV"""
        if not self.save_output or not self.prompt_history:
            return
        file_exists = os.path.isfile(self.output_file)
        with open(self.output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["timeframe", "filename", "description"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.prompt_history)
        self.prompt_history = []

    # -------------------------------
    # Start/stop lifecycle
    # -------------------------------
    def start(self):
        """Start looping through images as frames"""
        print(f"[ImageAgent] Starting... Found {len(self.images)} images.")
        self.running = True

        # Set up video writer if saving video
        if self.save_video and self.images:
            sample_img = Image.open(self.images[0])
            width, height = sample_img.size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc,
                                                self.fps, (width, height))

        # Start display thread
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

        # Loop through images
        for filename in self.images:
            if not self.running:
                break

            try:
                img = Image.open(filename).convert('RGB')
                self.last_caption = "Loading..."
                np_frame = np.array(img)
            except Exception as e:
                print(f"[ImageAgent] Failed to load {filename}: {e}")
                continue

            self.on_frame(np_frame, os.path.basename(filename))
            time.sleep(10)  # wait 5 seconds before next image

        if self.catch_time:
            print(f"[PROCESS COMPLETE] Average inference time: {np.mean(self.catch_time):.2f}s")

        self._save_history()

        # Wait for user to close display manually
        print("[ImageAgent] All images processed. Press ESC in display window to exit.")
        self.display_thread.join()


    def stop(self):
        """Stop all processes"""
        if not self.running:
            return
        print("[ImageAgent] Stopping...")
        self.running = False

        if self.save_video and self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)

        if not USE_JETSON:
            cv2.destroyAllWindows()
