import numpy as np
import pygame
from jetson_utils import cudaDeviceSynchronize
from utils.utils import cudaToNumpy

class PyDisplay:
    def __init__(self, width=1280, height=720):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

    def render(self, cuda_img):
        img = cudaToNumpy(cuda_img)
        cudaDeviceSynchronize()

        # Convert to 8-bit RGB for Pygame
        if img.dtype != np.uint8:
            img = (img*255).astype(np.uint8)

        # UYVY or other formats may need conversion
        # Using RGB for now
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(img)))
        self.screen.blit(surf, (0,0))
        pygame.display.flip()
        self.clock.tick(60)
