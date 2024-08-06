
#import sys
#sys.path.append('../')
import os
import sys
import matplotlib.pyplot as plt
from pyproj import CRS
from pyresample import geometry
import numpy as np
import numpy as np


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

#satobj.generate_l1b_cube()

#satobj.write_l1b_nc_file()



if True:
    satobj.get_l1a_cube()

    satobj.generate_l1b_cube()
    satobj.get_l1b_cube()

    satobj.write_l1b_nc_file()

    satobj.generate_land_mask(land_mask="global")
    satobj.get_active_land_mask()

    satobj.generate_land_mask(land_mask="ndwi")
    satobj.get_active_land_mask()

    satobj.generate_land_mask(land_mask="threshold")
    satobj.get_active_land_mask()

    satobj.get_land_mask(land_mask="global")
    satobj.get_land_mask(land_mask="ndwi")
    satobj.get_land_mask(land_mask="threshold")

    satobj.set_active_land_mask(land_mask="ndwi")
    satobj.set_active_land_mask(land_mask="threshold")
    satobj.set_active_land_mask(land_mask="global")

    satobj.get_active_mask()

    satobj.generate_cloud_mask(cloud_mask="global")
    satobj.get_cloud_mask(cloud_mask="global")
    satobj.get_active_cloud_mask()

    satobj.set_active_land_mask(land_mask="ndwi")
    satobj.get_active_mask()


    #satobj.generate_l2a_cube(product="6sv1")
    #satobj.get_l2a_cube(product="6sv1")

    lat = satobj.latitudes[200,500]
    lon = satobj.longitudes[200,500]

    satobj.get_l1a_spectrum(latitude=lat, longitude=lon)
    satobj.get_l1b_spectrum(latitude=lat, longitude=lon)
    satobj.get_l2a_spectrum(latitude=lat, longitude=lon, product="6sv1")

    #satobj.plot_l1a_spectrum(latitude=lat, longitude=lon)
    #satobj.plot_l1b_spectrum(latitude=lat, longitude=lon)
    #satobj.plot_l2a_spectrum(latitude=lat, longitude=lon, product="6sv1")

    #satobj.generate_toa_reflectance()
    #satobj.get_toa_reflectance()


    satobj.generate_chlorophyll_estimates('band_ratio')
    satobj.get_chlorophyll_estimates(product='band_ratio')

    model = "/home/cameron/Dokumenter/Chlorophyll_NN_Models/model_6sv1_aqua_tuned.joblib"
    satobj.generate_chlorophyll_estimates(product='6sv1_aqua', model=model)
    satobj.get_chlorophyll_estimates(product="6sv1_aqua")


exit()