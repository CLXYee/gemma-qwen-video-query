import numpy as np
import pygame
from jetson_utils import cudaDeviceSynchronize
from utils.utils import cudaToNumpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class PyDisplay:
    def __init__(self, width=1280, height=720):
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11"   # software mode

        pygame.init()

        self.width = width
        self.height = height

        # IMPORTANT: no hardware surfaces, no double buffer, no fullscreen
        flags = pygame.RESIZABLE    # safe for Jetson

        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            flags
        )

        self.clock = pygame.time.Clock()

    def render(self, cuda_img):
        img = cudaToNumpy(cuda_img)
        cudaDeviceSynchronize()

        # Convert to 8-bit RGB for Pygame
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Create surface from image (keep orientation as before)
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(img)))

        # Fit the image into the display while preserving aspect ratio (no stretching)
        surf_w, surf_h = surf.get_width(), surf.get_height()
        if surf_w == 0 or surf_h == 0:
            return

        # Scale to fit (contain) -- do not upscale beyond original size
        scale = min(self.width / surf_w, self.height / surf_h, 1.0)
        new_w, new_h = int(surf_w * scale), int(surf_h * scale)

        if (new_w, new_h) != (surf_w, surf_h):
            try:
                # higher quality scaling when available
                surf = pygame.transform.smoothscale(surf, (new_w, new_h))
            except Exception:
                surf = pygame.transform.scale(surf, (new_w, new_h))

        # Center the image on the screen (black background)
        self.screen.fill((0, 0, 0))
        x = (self.width - surf.get_width()) // 2
        y = (self.height - surf.get_height()) // 2
        self.screen.blit(surf, (x, y))

        pygame.display.flip()
        self.clock.tick(60)

    def soft_render(self, cuda_img, max_display_dim=128):
        """
        Render a CUDA image to pygame screen using software surfaces.
        Optimized for Jetson with limited GPU memory.
        
        Args:
            cuda_img: CUDA image tensor
            max_display_dim: maximum width/height for display to reduce memory usage
        """
        # Convert CUDA image to numpy
        img = cudaToNumpy(cuda_img)
        cudaDeviceSynchronize()

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Ensure 3 channels (RGB)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        # Downscale large images to save memory
        h, w = img.shape[:2]
        scale = min(max_display_dim / h, max_display_dim / w, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple nearest-neighbor downscale using slicing (no cv2)
            img = img[::h//new_h, ::w//new_w, :]

        # Create pygame surface
        try:
            surf = pygame.surfarray.make_surface(np.flipud(np.rot90(img)))
        except Exception as e:
            print("Error creating surface:", e)
            return

        # Scale to fit screen without upscaling
        surf_w, surf_h = surf.get_width(), surf.get_height()
        scale_fit = min(self.width / surf_w, self.height / surf_h, 1.0)
        if scale_fit < 1.0:
            new_w, new_h = int(surf_w * scale_fit), int(surf_h * scale_fit)
            try:
                surf = pygame.transform.smoothscale(surf, (new_w, new_h))
            except Exception:
                surf = pygame.transform.scale(surf, (new_w, new_h))

        # Center on screen
        self.screen.fill((0, 0, 0))
        x = (self.width - surf.get_width()) // 2
        y = (self.height - surf.get_height()) // 2
        self.screen.blit(surf, (x, y))

        pygame.display.flip()
        self.clock.tick(60)

class MatplotlibDisplay:
    """
    Display images safely using matplotlib instead of pygame.
    Designed for SSH sessions or low GPU memory environments.
    """
    def __init__(self, width=1280, height=720, max_display_dim=720):
        self.width = width
        self.height = height
        self.max_display_dim = max_display_dim

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        plt.ion()  # interactive mode
        self.im_obj = None
        self.ax.axis('off')
        self.fig.show()
        self.fig.canvas.draw()

    def render(self, cuda_img):
        """
        Render a CUDA image using matplotlib (like pygame.render).
        """
        img = cudaToNumpy(cuda_img)
        cudaDeviceSynchronize()

        # Convert to 8-bit RGB
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        # Downscale large images
        h, w = img.shape[:2]
        scale = min(self.max_display_dim / h, self.max_display_dim / w, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = img[::h//new_h, ::w//new_w, :]

        # Display image
        if self.im_obj is None:
            self.im_obj = self.ax.imshow(img, interpolation='nearest')
        else:
            self.im_obj.set_data(img)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def soft_render(self, cuda_img, max_display_dim=None):
        """
        Alias for render, to match PyDisplay.soft_render interface.
        """
        if max_display_dim:
            self.max_display_dim = max_display_dim
        self.render(cuda_img)

    from PIL import ImageFont, ImageDraw

    def _overlay_text_numpy(self, img, text, position=(10,30),
                            text_color=(255,255,255), background_color=(64,64,64),
                            line_spacing_ratio=0.05, base_font_size=16, line_length_ratio=0.05):
        """
        Draw word-wrapped text on a numpy image using PIL, with dynamic font size.
        
        Args:
            img: HWC, uint8 NumPy array
            position: (x, y) starting point
            line_spacing_ratio: fraction of image height to use as line spacing
            base_font_size: font size for reference height
            line_length_ratio: fraction of image width to use as max line length
        """
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        h, w = img.shape[:2]

        # Dynamic font size based on image height
        font_size = max(8, int(base_font_size * h / 720))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Dynamic line spacing
        line_spacing = int(h * line_spacing_ratio)

        # Max characters per line
        line_length = max(10, int(w * line_length_ratio))

        # Split text into words
        words = text.split()
        current_line = ""
        y = position[1]

        for n, word in enumerate(words):
            if len(current_line + word) <= line_length:
                current_line += word + " "
                if n == len(words) - 1:
                    self._draw_text_line(draw, current_line.strip(), position[0], y, font, text_color, background_color)
            else:
                self._draw_text_line(draw, current_line.strip(), position[0], y, font, text_color, background_color)
                current_line = word + " "
                y += line_spacing

        return np.array(pil_img)

    def _draw_text_line(self, draw, text, x, y, font, text_color, background_color):
        """
        Draw one line of text with background rectangle.
        Compatible with Pillow >= 10.
        """
        # Get bounding box of text
        bbox = draw.textbbox((x, y), text, font=font)
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top

        # Draw background rectangle
        draw.rectangle([x, y, x + width, y + height], fill=background_color)

        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)