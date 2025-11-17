# image_agent.py
import threading
import time
import os
import csv
import queue
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont
import unicodedata

def detect_jetson():
    try:
        import jetson.utils
        return True
    except ImportError:
        pass

    # Check device-tree model 
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r") as f:
                model = f.read()
                if "NVIDIA" in model or "Jetson" in model:
                    return True
        except:
            pass
    return False

USE_JETSON = detect_jetson()
if USE_JETSON:
    import jetson.utils



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
        Jetson-safe text overlay using PIL everywhere.
        Works with or without cv2 installed.
        """

        # Resize using PIL instead of cv2 (more stable on Jetson)
        img = Image.fromarray(frame)

        w, h = img.size
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        # Normalize text (same as your version)
        text = (text.replace("’", "'").replace("‘", "'")
                    .replace("“", '"').replace("”", '"')
                    .replace("–", "-").replace("—", "-"))

        text = ''.join(ch for ch in text if ord(ch) < 128)

        # Font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Wrap text manually
        max_text_px = int(img.size[0] * 0.9)
        words = text.split(" ")
        lines, line = [], ""

        for word in words:
            test = (line + " " + word).strip()
            wtest, _ = draw.textsize(test, font=font)
            if wtest > max_text_px:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        # Background rectangle
        y0 = 40
        line_height = font.getbbox("A")[3] + 6  
        box_height = len(lines) * line_height + 10
        box_width = max(draw.textsize(l, font=font)[0] for l in lines) + 20

        draw.rectangle(
            [(10, y0 - 10), (10 + box_width, y0 + box_height)],
            fill=(0, 0, 0, 180)
        )

        # Draw lines
        y = y0
        for line in lines:
            draw.text((20, y), line, fill=(255, 255, 255), font=font)
            y += line_height

        return np.array(img)


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

