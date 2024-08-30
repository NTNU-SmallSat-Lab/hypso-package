import sys
sys.path.insert(0, '/home/cameron/Projects/hypso-package')


print(sys.path)

from hypso import Hypso1
import os

if True:
    dir_path = '/home/cameron/Dokumenter/Data/frohavet'
    l1a_nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
    l1b_nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1b.nc')
    l2a_nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l2a-acolite.nc')
    points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')

if False:
    dir_path = '/home/cameron/Dokumenter/Data/erie'
    l1a_nc_file = os.path.join(dir_path, 'erie_2022-07-20_1539Z-l1a.nc')
    points_file = os.path.join(dir_path, 'erie_2022-07-20_1539Z-bin3.points')


satobj = Hypso1(path=l1a_nc_file, points_path=points_file, verbose=True)
satobj.load_points_file(path=points_file)

satobj.generate_georeferencing()
satobj.generate_geometry()
satobj.generate_l1b_cube()
satobj.write_l1b_nc_file()

satobj.set_acolite_path(path='/home/cameron/Projects/acolite/')

satobj.generate_l2a_cube(product_name="6sv1")

l2a_cube = satobj.get_l2a_cube()