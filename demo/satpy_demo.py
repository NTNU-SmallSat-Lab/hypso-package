import os
import sys
import matplotlib.pyplot as plt
from pyproj import CRS
from pyresample import geometry
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import re
import glob

import satpy
from satpy import Scene
from satpy.dataset.dataid import WavelengthRange
from pyresample import image
from pyresample import geometry

from pyresample.geometry import SwathDefinition
from pyresample import load_area

from satpy.composites import GenericCompositor
from satpy.writers import to_image

sys.path.insert(0, '/home/cameron/Projects/hypso-package')

# Import Satellite Object
from hypso import Hypso1

sys.path.insert(0,'/home/cameron/Projects/')




dir_path = '/home/cameron/Dokumenter/Data/frohavet'
nc_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-l1a.nc')
points_file = os.path.join(dir_path, 'frohavet_2024-05-06_1017Z-bin3.points')

satobj = Hypso1(hypso_path=nc_file, points_path=points_file, verbose=True)

dt = satobj.capture_datetime

l1b_cube = satobj.get_l1a_cube()

band = 40

#wl = satobj.wavelengths
wl = range(0,120)

#scene['chlor_a'].attrs['area'] = scene[band_549nm_name].attrs['area']

ds_info = {
    'file_type': 'netcdf',
    'resolution': None,
    'name': 'band_' + str(band),
    'standard_name': 'sensor_band_identifier',
    'coordinates': ['latitude', 'longitude'],
    'units': "%",
    'wavelength': WavelengthRange(min=wl[band], central=wl[band], max=wl[band], unit="nm"),
    'start_time': dt,
    'end_time': dt,
    'modifiers': (),
    'ancillary_variables': []
}


lat_info = {
    'file_type': 'netcdf',
    'resolution': None,
    'standard_name': 'latitude',
    'units': 'degrees_north',
    'resolution': -999, # TODO
    'start_time': dt,
    'end_time': dt,
    'modifiers': (),
    'ancillary_variables': []
}


lon_info = {
    'file_type': 'netcdf',
    'resolution': None,
    'standard_name': 'longitude',
    'units': 'degrees_east',
    'resolution': -999, # TODO
    'start_time': dt,
    'end_time': dt,
    'modifiers': (),
    'ancillary_variables': []

}


scn = Scene()

latitudes = xr.DataArray(satobj.latitudes, dims=["y", "x"])
longitudes = xr.DataArray(satobj.longitudes, dims=["y", "x"])

swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

scn['latitude'] = latitudes
scn['latitude'].attrs.update(lat_info)
scn['latitude'].attrs['area'] = swath_def

scn['longitude'] = longitudes
scn['longitude'].attrs.update(lon_info)
scn['longitude'].attrs['area'] = swath_def


for band in [89, 70, 59]:

    data = l1b_cube[:,:,band].to_numpy()
    key = 'band_' + str(band)
    scn[key] = xr.DataArray(data, dims=["y", "x"])
    scn[key].attrs.update(ds_info)
    scn[key].attrs['area'] = swath_def


#scn.show(key)

lon_min = longitudes.data.min()
lon_max = longitudes.data.max()
lat_min = latitudes.data.min()
lat_max = latitudes.data.max()

bbox = (lon_min,lat_min,lon_max,lat_max)
#bbox = (-3, 60.25,13.75,67.4)
print(bbox)

area_id = 'frohavet'
proj_id = 'roi'
description = 'roi'
projection = CRS.from_proj4("+proj=latlon")
width = 1000
height = 1000
area_extent = bbox

area_def = geometry.AreaDefinition(area_id, proj_id, description, projection,  width, height, area_extent)

resampled_scn = scn.resample(area_def, resampler='bilinear', fill_value=np.NaN)

alpha = np.where(np.isnan(resampled_scn['band_70']), 0, 1)

s = scn
compositor = GenericCompositor("overview")
composite = compositor([s['band_89'][:,::3], s['band_70'][:,::3], s['band_59'][:,::3], s['band_89'][:,::3]]) # Red, Green, Blue, Alpha
gamma = 2
img = to_image(composite) 
#img.invert([False, False, False])
img.stretch("linear")
img.gamma([gamma, gamma, gamma, gamma])
img.save('./scn.png')

s = resampled_scn
compositor = GenericCompositor("overview")
composite = compositor([s['band_89'], s['band_70'], s['band_59'], s['band_89']]) # Red, Green, Blue, Alpha
gamma = 2
img = to_image(composite) 
#img.invert([False, False, False])
img.stretch("linear")
img.gamma([gamma, gamma, gamma, gamma])
img.save('./resampled_scn.png')