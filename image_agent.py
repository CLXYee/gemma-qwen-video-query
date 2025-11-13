# image_agent.py
import threading
import time
import os
import csv
import queue
import numpy as np
import cv2
from PIL import Image
import unicodedata

try:
    import jetson.utils
    USE_JETSON = True
except ImportError:
    USE_JETSON = False


class LiveImageAgent:
    def __init__(self, describer, image_folder="./selected",
                 prompt=None, max_tokens=16,
                 save_output=False, output_file="prompt_history.csv",
                 save_video=False, video_path="output.mp4"):

        self.describer = describer
        self.image_folder = image_folder
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.save_output = save_output
        self.output_file = output_file
        self.save_video = save_video
        self.video_path = video_path
        self.stop_event = threading.Event()

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
            self.fps = 60  # default fps
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

            if self.save_output:
                if len(self.prompt_history) % 5 == 0:
                    self._save_history()

        except Exception as e:
            print(f"[Error in inference for {filename}]: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------
    # Display and overlay
    # -------------------------------
    def _overlay_text(self, frame, text, max_width=1536, max_height=864):
        """
        Overlay caption text on image with word wrapping, consistent font size,
        and a single semi-transparent background rectangle for readability.
        """
        # Resize first
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            h, w = frame.shape[:2]

        annotated = frame.copy()

        # Convert RGB -> BGR for OpenCV display
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # consistent size
        thickness = 1
        line_spacing = 5

        # Maximum width for text area 
        max_text_width = int(w * 0.9)  # leave 10% padding on right

        # Normalize fancy quotes/dashes to ASCII equivalents
        text = (text.replace("’", "'")
                    .replace("‘", "'")
                    .replace("“", '"')
                    .replace("”", '"')
                    .replace("–", "-")
                    .replace("—", "-"))

        # Also remove any non-printable Unicode characters
        text = ''.join(ch for ch in text if ord(ch) < 128)

        # Split text into words and wrap lines
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width > max_text_width:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        # Calculate the bounding rectangle for all text
        y0 = 40
        dy = int(cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + line_spacing)
        text_height = len(lines) * dy
        max_line_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)

        # Rectangle coordinates: top-left and bottom-right
        x1, y1 = 15, y0 - 10
        x2, y2 = x1 + max_line_width + 20, y1 + text_height + 10

        # Draw a single semi-transparent rectangle
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Draw all lines of text on top of the rectangle
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(annotated, line, (20, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated


    def display_loop(self):
        """Continuously display frames with captions (like video)."""
        print("[Display] Starting display loop... (Press ESC to quit)")

        while self.running and not self.stop_event.is_set():
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
                key = cv2.waitKey(50)
                if key == 27:  # ESC
                    self.stop()
                    cv2.destroyAllWindows()
                    break

            if self.stop_event.is_set():
                break
            
            if self.save_video and self.video_writer:
                try:
                    # Ensure frame matches the initialized video size
                    frame_to_write = cv2.resize(annotated, (self.video_width, self.video_height))
                    # Ensure correct color format (BGR)
                    self.video_writer.write(frame_to_write)
                except Exception as e:
                    print(f"[VideoWriter Error] Failed to write frame: {e}")

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
        """Start looping through images as frames continuously"""
        if not self.images:
            print("[ImageAgent] No images found to display.")
            return

        print(f"[ImageAgent] Starting... Found {len(self.images)} images.")
        self.stop_event.clear()
        self.running = True

        # Set up video writer if saving video
        if self.save_video and self.images:
            sample_img = Image.open(self.images[0])
            self.video_width, self.video_height = sample_img.size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # try XVID if mp4v causes issues
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                max(60, self.fps),
                (self.video_width, self.video_height)
            )

        # Start display thread
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

        try:
            # Continuous looping through images
            while self.running and not self.stop_event.is_set():
                for filename in self.images:
                    if self.stop_event.is_set() or not self.running:
                        break


                    try:
                        img = Image.open(filename).convert('RGB')
                        self.last_caption = "Loading..."
                        np_frame = np.array(img)
                    except Exception as e:
                        print(f"[ImageAgent] Failed to load {filename}: {e}")
                        continue

                    # Send frame to inference + display
                    self.on_frame(np_frame, os.path.basename(filename))
                    time.sleep(10)  # wait 10 seconds before next image

        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Gracefully stopping...")
            self.stop_event.set()
            self.stop()

        finally:
            # Always release video writer and cleanup
            if self.catch_time:
                print(f"[PROCESS COMPLETE] Average inference time: {np.mean(self.catch_time):.2f}s")

            self._save_history()
            if self.save_video and self.video_writer:
                print(f"[Video] Saved to: {self.video_path}")
                self.video_writer.release()
                self.video_writer = None

            self.running = False
            print("[ImageAgent] Stopped.")
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=1)
            if not USE_JETSON:
                cv2.destroyAllWindows()

    def stop(self):
        """Stop all processes gracefully."""
        if not self.running:
            return

        print("[ImageAgent] Stopping...")
        self.running = False
        self.stop_event.set()

        # Wait for display thread to exit
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)

        # Release video writer last
        if self.save_video and self.video_writer:
            print("[VideoWriter] Releasing video writer...")
            self.video_writer.release()
            self.video_writer = None
            print(f"[VideoWriter] Saved video to {self.video_path}")

        if not USE_JETSON:
            cv2.destroyAllWindows()

        print("[ImageAgent] Stopped.")

