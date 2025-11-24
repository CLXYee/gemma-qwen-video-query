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
        screen_w, screen_h = 1920,1080
        dpi = 100
        self.fig, self.ax = plt.subplots(figsize=(screen_w/dpi, screen_h/dpi), dpi=dpi)

        self.ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.ax.set_aspect('equal', adjustable='box')

        manager = plt.get_current_fig_manager()
        manager.set_window_title('')
        manager.window.overrideredirect(True)

        plt.ion()  # interactive mode
        self.im_obj = None
        self.fig.show()
        self.fig.canvas.draw()

    def render(self, cuda_img):
        """
        Render a CUDA image using matplotlib.
        """
        #img = cudaToNumpy(cuda_img)
        img = cuda_img
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
            self.ax.set_aspect('equal', adjustable='box')

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

    def _overlay_text_numpy(self, img, text, position=(10,30),
                        text_color=(255,255,255), background_color=(64,64,64),
                        base_font_size=13, margin_ratio=0.02):
        """
        Draw word-wrapped text on a numpy image using PIL, with dynamic font size and
        a single rectangle background for all lines.
        """
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        h, w = img.shape[:2]

        # Dynamic font size
        font_size = max(8, int(base_font_size * h / 720))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Line spacing proportional to font size
        line_spacing = int(font_size * 0.2)

        # Maximum line width in pixels
        margin = int(w * margin_ratio)
        max_line_width = w - 2 * margin

        x, y = position
        words = text.split()
        lines = []
        current_line = ""

        # Wrap text into lines
        for word in words:
            test_line = current_line + (word + " ")
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_line_width:
                current_line = test_line
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())

        # Compute total rectangle height and width
        max_line_width_px = 0
        total_height = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            max_line_width_px = max(max_line_width_px, line_width)
            total_height += line_height + line_spacing
        total_height -= line_spacing  # no extra spacing after last line

        # Draw single background rectangle
        draw.rectangle([x - 1, y - 1, x + max_line_width_px + 3, y + total_height + 3], fill=background_color)

        # Draw text line by line
        y_offset = y
        for line in lines:
            draw.text((x, y_offset), line, font=font, fill=text_color)
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            y_offset += line_height + line_spacing

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