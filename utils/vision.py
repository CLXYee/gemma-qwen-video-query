import numpy as np
import pygame
from jetson_utils import cudaDeviceSynchronize
from utils.utils import cudaToNumpy


class PyDisplay:
    def __init__(self, width=None, height=None):
        pygame.init()
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h

        # If width/height not provided, use full desktop resolution
        if width is None or height is None:
            self.width = screen_w
            self.height = screen_h
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
        else:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((self.width, self.height))

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