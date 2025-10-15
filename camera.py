#video.py
import threading
import time
import torch

from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize, cudaFont, cudaMemcpy
from utils.utils import cudaToNumpy

def wrap_text(font, image, text='', x=5, y=5, **kwargs):
    """"
    Utility for cudaFont that draws text on a image with word wrapping.
    Returns the new y-coordinate after the text wrapping was applied.
    """
    text_color=kwargs.get("color", font.White) 
    background_color=kwargs.get("background", font.Gray40)
    line_spacing = kwargs.get("line_spacing", 38)
    line_length = kwargs.get("line_length", image.width // 16)

    text = text.split()
    current_line = ""

    for n, word in enumerate(text):
        if len(current_line) + len(word) <= line_length:
            current_line = current_line + word + " "
            
            if n == len(text) - 1:
                font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
                return y + line_spacing
        else:
            current_line = current_line.strip()
            font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
            current_line = word + " "
            y=y+line_spacing
    return y
    

class VideoSource:
    """
    Capture frames from a camera, video file, or stream using jetson-utils.
    Automatically skips frames while inference is still running.
    """
    def __init__(self, source="/dev/video0", return_tensors='cuda',
                 video_input_width=None, video_input_height=None, 
                 video_input_codec=None, video_input_framerate=None, 
                 video_input_save=None, **kwargs):
        """
        Args:
            source: Camera device, file path, or stream URL.
            return_tensors: 'np' | 'pt' | 'cuda' — format for returned frames.
        """

        super().__init__(**kwargs)
        options = {}
        
        if video_input_width:
            options['width'] = video_input_width
            
        if video_input_height:
            options['height'] = video_input_height
            
        if video_input_codec:
            options['codec'] = video_input_codec
 
        if video_input_framerate:
            options['framerate'] = video_input_framerate
            
        if video_input_save:
            options['save'] = video_input_save

        self.source = source
        self.return_tensors = return_tensors
        self.cap = videoSource(source, options=options)  # automatically detects camera/stream type
        self.running = False
        self.thread = None
        self._busy = False  # skip frames while inference running
        print("###Camera Status: ", self.cap.IsStreaming())

    def capture(self):
        """
        Capture a single frame and return it in the specified format.
        """
        #frame = self.cap.Capture(format='rgb8',timeout=1000)
        frame = self.cap.Capture()

        if frame is None:
            raise RuntimeError("Failed to capture frame from source.")
        
        if self.return_tensors == 'np':
            frame_np = cudaToNumpy(frame)
            return frame_np
        elif self.return_tensors == 'pt':
            frame_np = cudaToNumpy(frame)
            return torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        elif self.return_tensors == 'cuda':
            return frame
        else:
            raise ValueError(f"Unsupported return_tensors: {self.return_tensors}")

    def start(self, callback, threaded=True, interval=0.03):
        """
        Continuously capture frames and send to callback(frame),
        skipping new frames if previous inference is still running.
        """
        self.running = True

        def loop():
            while self.running:
                try:
                    if self._busy:
                        # Skip this frame — inference still running
                        _ = self.cap.Capture(timeout=300)
                        time.sleep(interval)
                        continue

                    frame = self.capture()
                    self._busy = True
                    threading.Thread(
                        target=self._inference_thread, args=(callback, frame), daemon=True
                    ).start()

                except Exception as e:
                    print(f"[VideoSource] Error: {e}")
                    time.sleep(1)

        if threaded:
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        else:
            loop()

    def _inference_thread(self, callback, frame):
        """
        Run inference in a separate thread and release busy flag after completion.
        """
        try:
            safe_frame = cudaMemcpy(frame)
            callback(safe_frame)
        except Exception as e:
            print(f"[VideoSource] Inference error: {e}")
        finally:
            self._busy = False  # ready for next frame

    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.thread:
            self.thread.join()
        print("[VideoSource] Stopped.")


class VideoOutput:
    """
    Display frames and overlay text using jetson-utils.
    """
    def __init__(self, output_source="display://0"):
        """
        Args:
            output_source: Output display or stream (e.g., 'display://0', 'file://output.mp4')
        """
        self.output = videoOutput(output_source)
        self.running = False
        self.font = cudaFont()

    def overlay_text(self, frame, text, position=(10, 30), color=(255, 255, 255, 255), background=None):
        """
        Draw text on CUDA frame using jetson-utils' cudaFont (GPU overlay).
        Uses wrap_text internally for consistent rendering with background support.
        
        Args:
            frame: CUDA-mapped image
            text: Text string to render (single line recommended)
            position: (x, y) tuple for top-left corner
            color: RGBA tuple for text color
            background: Optional background color (e.g., self.font.Gray40)
        """
        try:
            x, y = position
            # Use wrap_text for consistent styling and error resilience
            # It handles single-line fine and supports background
            background = self.font.Gray40
            wrap_text(self.font, frame, text=text, x=x, y=y, color=color, background=background)
            
            return frame
        except Exception as e:
            print(f"[VideoOutput] Text overlay error: {e}")
            return frame

    def display(self, frame):
        if not self.output.IsStreaming():
            return False

        try:
            # frame should already be a safe copy
            self.output.Render(frame)
            cudaDeviceSynchronize()  # ← Only needed here
            print("[VideoOutput] Frame rendered successfully.")
            return True
        except Exception as e:
            print(f"[VideoOutput] Render error: {e}")
            return False

    def start(self):
        self.running = True
        print("[VideoOutput] Display started. Press Ctrl+C to quit.")

    def stop(self):
        self.running = False
        self.output.Close()
        print("[VideoOutput] Display stopped.")
