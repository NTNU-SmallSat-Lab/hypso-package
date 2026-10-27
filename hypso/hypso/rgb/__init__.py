"""RGB images: colour-accurate renderings of hyperspectral cubes (CIE colour
matching functions, Magnusson et al. 2020), and HYPSO-2's RGB camera images."""
from .camera import BAYER_PATTERN, camera_image_time, demosaic_bilinear, find_camera_image, read_camera_image
from .hsi2rgb import HSI2RGB
