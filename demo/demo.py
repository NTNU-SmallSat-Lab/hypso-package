
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


satobj = Hypso1(nc_file, points_path=points_file)




#plt.imshow(satobj.l1b_cube[:,:,40])
#plt.savefig('test.png')

#satobj.create_l1b_nc_file()




