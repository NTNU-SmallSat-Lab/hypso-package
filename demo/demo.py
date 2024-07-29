
#import sys
#sys.path.append('../')
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1




dir_path = '/home/cameron/Dokumenter/Data/erie'
nc_file = os.path.join(dir_path, 'erie_2022-08-27_1605Z-l1a.nc')
points_file = os.path.join(dir_path, 'erie_2022-08-27_1605Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-10-02_1559Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-10-02_1559Z-bin3.points')


satobj = Hypso1(hypso_path=nc_file, points_path=points_file, verbose=True)


satobj.run_atmospheric_correction(product='test')

#print(satobj.l1b_cube)
#print(satobj.l1b_cube.shape)

print(satobj.l2a_cube)
print(satobj.l2a_cube['6SV1'].shape)
