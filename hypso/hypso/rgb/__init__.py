"""RGB images: colour-accurate renderings of hyperspectral cubes (CIE colour
matching functions, Magnusson et al. 2020), and HYPSO-2's RGB camera images."""
from .camera import BAYER_PATTERN, camera_image_time, demosaic_bilinear, find_camera_image, read_camera_image
from .hsi2rgb import HSI2RGB
from .registration import (CameraRegistration, camera_lonlat, camera_to_hsi_pixels, georeference_camera,
                           hsi_to_camera_pixels, register_camera)
from .maps import camera_gsd, plot_rgb_map, regrid_camera
from .product import camera_filename, camera_product, open_camera, write_camera
