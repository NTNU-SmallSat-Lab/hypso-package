import os
import sys

sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1


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