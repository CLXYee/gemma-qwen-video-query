# image_source.py
import threading
import time
import glob
import os
from PIL import Image
import numpy as np
from jetson_utils import cudaMemcpy
from utils.image import cuda_image
from utils.utils import cudaToNumpy  # Assuming you have this utility

class ImageSource:
    """
    Treat a folder of images as a video source.
    Continuously passes frames (images) to a callback in the same way as VideoSource.
    """
    def __init__(self, folder_path, return_tensors='cuda', loop=True):
        """
        Args:
            folder_path: Path to folder containing images.
            return_tensors: 'np' | 'pt' | 'cuda' — format for returned frames.
            loop: Whether to loop back to the first image after finishing.
        """
        self.folder_path = folder_path
        self.return_tensors = return_tensors
        self.loop = loop

        # Collect all image paths
        self.image_paths = sorted(
            glob.glob(os.path.join(self.folder_path, "*.*"))
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.folder_path}")

        self.running = False
        self.thread = None
        self._busy = False  # skip new frames while inference running

    def capture(self, image_path):
        """Load image and return in desired format."""
        img = Image.open(image_path).convert("RGB")
        np_img = np.array(img)

        if self.return_tensors == 'np':
            return np_img
        elif self.return_tensors == 'pt':
            import torch
            return torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        elif self.return_tensors == 'cuda':
            return cuda_image(np_img)  # Will be copied via cudaMemcpy in _inference_thread
        else:
            raise ValueError(f"Unsupported return_tensors: {self.return_tensors}")

    def start(self, callback, threaded=True):
        """
        Start passing images to callback(frame) like a video source.
        The same image is passed until inference finishes and 10-second wait completes.
        """
        self.running = True

        def loop():
            idx = 0
            while self.running:
                image_path = self.image_paths[idx]
                if self._busy:
                    # Skip to next iteration but keep passing the last frame
                    time.sleep(0.1)
                    continue

                frame = self.capture(image_path)
                self._busy = True
                threading.Thread(
                    target=self._inference_thread, args=(callback, frame), daemon=True
                ).start()

                # Wait until inference completes + 10 seconds display
                start_time = time.time()
                while self._busy and time.time() - start_time < 10:
                    if not self.running:
                        break
                    # Keep displaying the same frame
                    time.sleep(0.1)

                # Move to next image
                idx += 1
                if idx >= len(self.image_paths):
                    if self.loop:
                        idx = 0
                    else:
                        break

        if threaded:
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        else:
            loop()

    def _inference_thread(self, callback, frame):
        """
        Pass frame to the agent callback, then release busy flag after completion.
        """
        try:
            safe_frame = cudaMemcpy(frame)
            callback(safe_frame)
        except Exception as e:
            print(f"[ImageSource] Inference error: {e}")
        finally:
            # Keep the _busy True for 10 seconds after inference
            # The start() loop handles this timing
            self._busy = False

    def stop(self):
        """Stop sending images."""
        self.running = False
        if self.thread:
            self.thread.join()
        print("[ImageSource] Stopped.")
