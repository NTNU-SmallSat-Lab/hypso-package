import os
import sys

sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1
from satpy.composites import GenericCompositor
from satpy.writers import to_image
import matplotlib.pyplot as plt


dir_path = '/home/cameron/Dokumenter/Data/frohavet'
nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')
cloud_mask_quantile = 0.6


satobj = Hypso1(hypso_path=nc_file, points_path=points_file, verbose=True)
satobj.generate_l1b_cube()
satobj.generate_geometry()
satobj.write_l1b_nc_file()
satobj.generate_l2a_cube(product_name='acolite')
satobj.write_l2a_nc_file()


print(satobj.l2a_cube)

s = satobj.get_l2a_satpy_scene()

# RGB Hypso image
compositor = GenericCompositor("rgb")

#R:630nm, G:550nm, B:480nm
# Red, Green, Blue, Alpha
red_wl = 630
green_wl = 550
blue_wl = 480

red_idx = satobj.get_closest_wavelength_index(red_wl)
green_idx = satobj.get_closest_wavelength_index(green_wl)
blue_idx = satobj.get_closest_wavelength_index(blue_wl)

red_band = 'band_' + str(red_idx)
green_band = 'band_' + str(green_idx)
blue_band = 'band_' + str(blue_idx)

#composite = compositor([s[red_band][:,::3], s[green_band][:,::3], s[blue_band][:,::3], s['band_5'][:,::3]]) 
composite = compositor([s[red_band][:,::3], s[green_band][:,::3], s[blue_band][:,::3]]) 

rgb_img = to_image(composite) 
rgb_img.stretch()
rgb_img.gamma(2)

rgb_xr_image = rgb_img

rgb_img = rgb_img.pil_image()

rgb_img.save('./' + satobj.capture_name + '_acolite_rgb.png')