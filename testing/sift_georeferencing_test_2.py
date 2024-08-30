import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package')

from hypso import Hypso1
import os
import numpy as np
from pyproj import Transformer
from pyproj import Proj
from pyresample import geometry
import satpy
from satpy.composites import GenericCompositor
from satpy.writers import to_image

from PIL import Image
from pycoast import ContourWriterAGG

dir_path = '/home/cameron/Nedlastinger'
l1a_nc_file = os.path.join(dir_path, 'griegloppa_2024-08-20T09-23-59Z-l1a.nc')
points_file = os.path.join(dir_path, 'sift-bin.points')

dir_path = '/home/cameron/Dokumenter/Data/frohavet'
l1a_nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')


satobj = Hypso1(path=l1a_nc_file, points_path=points_file, verbose=True)
satobj.load_points_file(path=points_file, image_mode='standard', origin_mode='cube')

satobj.generate_georeferencing()
#satobj.generate_geometry()
#satobj.generate_l1b_cube()

l1b_scene = satobj.get_l1a_satpy_scene()

bbox = satobj.get_bbox()


area_id = satobj.capture_target
proj_id = 'roi'
description = 'roi'

lon_0 = (bbox[2] - bbox[0])/2 + bbox[0]
lat_ts = (bbox[3] - bbox[1])/2 + bbox[1]

p = Proj(proj="stere", ellps="bessel", lat_0=90.0, lon_0=lon_0, lat_ts=lat_ts)

projection = {"proj": "stere", 
              "ellps": "bessel", 
              "lat_0": 90.0, 
              "lon_0": lon_0, 
              "lat_ts": lat_ts, 
              "units": "m"}

lower_left_x, lower_left_y = p(bbox[0], bbox[1])
upper_right_x, upper_right_y = p(bbox[2], bbox[3])
area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

res = 300 #meters

width = (area_extent[2] - area_extent[0]) / res
height = (area_extent[3] - area_extent[1]) / res

area_def = geometry.AreaDefinition(area_id, proj_id, description, projection,  width, height, area_extent)

resampled_l1a_scene = l1b_scene.resample(area_def, resampler='bilinear', fill_value=np.NaN)

# RGB Hypso image
compositor = GenericCompositor("rgb")



red_idx = 60
green_idx = 40
blue_idx = 20

red_band = 'band_' + str(red_idx)
green_band = 'band_' + str(green_idx)
blue_band = 'band_' + str(blue_idx)
alpha_band = 'band_' + str(blue_idx)

composite = compositor([resampled_l1a_scene[red_band], resampled_l1a_scene[green_band], resampled_l1a_scene[blue_band]]) 

rgb_img = to_image(composite) 
rgb_img.stretch_linear()
rgb_img.gamma(1.2)

rgb_xr_image = rgb_img

rgb_img = rgb_img.pil_image()