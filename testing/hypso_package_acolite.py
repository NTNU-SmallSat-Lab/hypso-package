import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package')

from hypso import Hypso1
import os

dir_path = '/home/cameron/Nedlastinger'

l1a_nc_file = os.path.join(dir_path, 'grieghammerfest_2024-08-17T10-38-36Z-l1a.nc')
points_file = os.path.join(dir_path, 'grieghammerfest_2024-08-17T10-38-36Z-l1a.points')

satobj = Hypso1(path=l1a_nc_file, verbose=True)

# OPTIONAL: Georeferencing
#satobj.load_points_file(path=points_file, image_mode='standard', origin_mode='cube')
#satobj.generate_georeferencing()

satobj.generate_geometry()
satobj.generate_l1b_cube()
satobj.write_l1b_nc_file()

# REQUIRED: Path to your ACOLITE directory
#satobj.set_acolite_path(path='/home/cameron/Projects/acolite/') 
#satobj.generate_l2a_cube(product_name="acolite")
#satobj.get_l2a_cube()
#satobj.write_l2a_nc_file()