import os
import sys
import matplotlib.pyplot as plt
from pyproj import CRS
from pyresample import geometry
import numpy as np


sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1

dir_path = '/home/cameron/Dokumenter/Data/frohavet'
nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')

satobj = Hypso1(hypso_path=nc_file, points_path=points_file, verbose=True)

#satobj.generate_l1b_cube()

l1a_scene = satobj.get_l1a_satpy_scene()

z = l1a_scene.coarsest_area(['band_3', 'band_4'])

print(z)