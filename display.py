import time
import logging
import torch
from jetson_utils import cudaFont
from utils.utils import cudaToNumpy
from utils.vision import PyDisplay
from utils.image import wrap_text

class VideoSource:
    """
    Capture images from a camera or video stream, with automatic reconnect.
    """
    def __init__(self, video_input='/dev/video0', video_input_width=1280, video_input_height=720,
                 video_input_framerate=60, return_tensors='cuda'):
        self.video_input = video_input
        self.width = video_input_width
        self.height = video_input_height
        self.framerate = video_input_framerate
        self.return_tensors = return_tensors

        self.stream = None
        self.open_stream()

    def open_stream(self):
        from jetson_utils import videoSource
        options = {}
        if self.width: options['width'] = self.width
        if self.height: options['height'] = self.height
        if self.framerate: options['framerate'] = self.framerate

        try:
            self.stream = videoSource(self.video_input, options=options)
            logging.info(f"[VideoSource] Camera {self.video_input} opened successfully")
        except Exception as e:
            logging.error(f"[VideoSource] Failed to open camera {self.video_input}: {e}")
            self.stream = None

    def reconnect(self):
        """Attempt to reopen stream if disconnected."""
        while self.stream is None:
            logging.warning(f"[VideoSource] Reconnecting to {self.video_input}...")
            try:
                self.open_stream()
                if self.stream is not None:
                    logging.info(f"[VideoSource] Reconnected to {self.video_input}")
            except Exception as e:
                logging.error(f"[VideoSource] Reconnect failed: {e}")
                time.sleep(2.0)

    def capture(self):
        """Capture a single frame and return as cudaImage / torch / numpy."""
        if self.stream is None:
            self.reconnect()

        retries = 0
        while retries < 5:
            img = self.stream.Capture(format='rgb8', timeout=2500)
            if img is not None:
                # Convert to desired format
                if self.return_tensors == 'pt':
                    img = torch.as_tensor(img, device='cuda')
                elif self.return_tensors == 'np':
                    img = cudaToNumpy(img)
                elif self.return_tensors != 'cuda':
                    raise ValueError(f"return_tensors should be 'cuda', 'np', or 'pt', got {self.return_tensors}")
                return img
            else:
                logging.warning(f"[VideoSource] Capture timeout, retrying... ({retries+1}/5)")
                retries += 1
        return None

class VideoOutput:
    """
    Display images safely using PySafeDisplay (Pygame backend)
    """
    def __init__(self, width=1280, height=720):
        self.display = PyDisplay(width=width, height=height)
        self.width = width
        self.height = height
        self.font = cudaFont()

    def render(self, cuda_img):
        """Render a single frame safely."""
        self.display.render(cuda_img)

    def overlay_text(self, frame, text, position=(10, 30)):
        """
        Draw text on CUDA frame using jetson-utils' cudaFont (GPU overlay).
        Uses wrap_text internally for consistent rendering with background support.
        """
        try:
            x, y = position
            color = self.font.White
            background = self.font.Gray40
            wrap_text(self.font, frame, text=text, x=x, y=y, color=color, background=background)
            
            return frame
        except Exception as e:
            print(f"[VideoOutput] Text overlay error: {e}")
            return frame

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # We are using camera.py->VideoSource in the main program instead
    source = VideoSource(video_input='/dev/video0', video_input_width=1280, video_input_height=720)
    # Video output using pygame
    output = VideoOutput(width=1280, height=720)

    logging.info("[MAIN] Starting capture and display loop...")
    
    while True:
        frame = source.capture()
        if frame is not None:
            output.render(frame)
        else:
            logging.warning("[MAIN] No frame captured, trying to reconnect...")
            source.reconnect()
