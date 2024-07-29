
#import sys
#sys.path.append('../')
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1




#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2022-08-27_1605Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2022-08-27_1605Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-10-02_1559Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-10-02_1559Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-05-17_1553Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-05-17_1553Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-06-03_1612Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-06-03_1612Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-06-04_1557Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-06-04_1557Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-06-17_1541Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-06-17_1541Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2023-08-20_1538Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2023-08-20_1538Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/erie'
#nc_file = os.path.join(dir_path, 'erie_2022-07-19_1550Z-l1a.nc')
#points_file = os.path.join(dir_path, 'erie_2022-07-19_1550Z-bin3.points')

# flipped
dir_path = '/home/cameron/Dokumenter/Data/erie'
nc_file = os.path.join(dir_path, 'erie_2022-07-20_1539Z-l1a.nc')
points_file = os.path.join(dir_path, 'erie_2022-07-20_1539Z-bin3.points')


#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-03_0941Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-03_0941Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-16_0945Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-16_0945Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-18_1035Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-18_1035Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-15_1006Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-15_1006Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-19_1014Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-19_1014Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-04-26_1049Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-04-26_1049Z-bin3.points')

#dir_path = '/home/cameron/Dokumenter/Data/frohavet'
#nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
#points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')


satobj = Hypso1(hypso_path=nc_file, points_path=points_file, verbose=True)


#satobj._run_calibration()

#print(satobj.latitudes.shape)
#satobj.display_geometry_information()

#satobj.run_geometry_computation()

#satobj.run_atmospheric_correction(product='6SV1')

#print(satobj.l1b_cube)
#print(satobj.l1b_cube.shape)

#print(satobj.l2a_cube)
#print(satobj.l2a_cube['6SV1'].shape)

#l1b_cube = satobj.get_l1b_cube()
#l2a_cube = satobj.get_l2a_cube(product="MACHI")

#print(l1b_cube)
#print(l2a_cube)


satobj._run_land_mask(product="global")
#plt.imshow(arr=satobj.land_mask)
plt.imsave(fname='global.png', arr=satobj.land_mask)

satobj._run_land_mask(product="threshold")
#plt.imshow(satobj.land_mask)
plt.imsave(fname='threshold.png', arr=satobj.land_mask)

satobj._run_land_mask(product="NDWI")
#plt.imshow(satobj.land_mask)
plt.imsave(fname='NDWI.png', arr=satobj.land_mask)


plt.imsave(fname='l1b_40.png', arr=satobj.get_l1b_cube()[:,:,40])

plt.imsave(fname='l1a_40.png', arr=satobj.get_l1a_cube()[:,:,40])

plt.imsave(fname='latitudes.png', arr=satobj.latitudes)
plt.imsave(fname='longitudes.png', arr=satobj.longitudes)

print(satobj.datacube_flipped)


print(satobj.latitudes[0,0])
print(satobj.latitudes[-1,-1])

print(satobj.longitudes[0,0])
print(satobj.longitudes[-1,-1])