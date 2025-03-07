# SatPy functions

from typing import Union

from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

import numpy as np
import xarray as xr

from datetime import datetime



def get_l1a_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        use_indirect_georef: bool = False
                        ) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = range(0,120)

    standard_name = datacube.attrs['description']
    units = datacube.attrs['units']

    dims = satobj.dim_names_2d
    start_time = satobj.capture_datetime
    end_time = satobj.capture_datetime

    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution,
                        standard_name=standard_name,
                        units=units,
                        dims=dims,
                        start_time=start_time,
                        end_time=end_time)
    
    return scene




def get_l1b_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        use_indirect_georef: bool = False
                        ) -> Scene:

    datacube = satobj.l1b_cube
    wavelengths = satobj.wavelengths

    standard_name = datacube.attrs['description']
    units = datacube.attrs['units']

    dims = satobj.dim_names_2d
    start_time = satobj.capture_datetime
    end_time = satobj.capture_datetime

    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution,
                        standard_name=standard_name,
                        units=units,
                        dims=dims,
                        start_time=start_time,
                        end_time=end_time)
    
    return scene



def get_l1c_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        use_indirect_georef: bool = False
                        ) -> Scene:

    datacube = satobj.l1c_cube
    wavelengths = satobj.wavelengths

    standard_name = datacube.attrs['description']
    units = datacube.attrs['units']

    dims = satobj.dim_names_2d
    start_time = satobj.capture_datetime
    end_time = satobj.capture_datetime

    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution,
                        standard_name=standard_name,
                        units=units,
                        dims=dims,
                        start_time=start_time,
                        end_time=end_time)
    
    return scene


def get_l1d_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        use_indirect_georef: bool = False
                        ) -> Scene:

    datacube = satobj.l1d_cube
    wavelengths = satobj.wavelengths

    standard_name = datacube.attrs['description']
    units = datacube.attrs['units']

    dims = satobj.dim_names_2d
    start_time = satobj.capture_datetime
    end_time = satobj.capture_datetime

    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution,
                        standard_name=standard_name,
                        units=units,
                        dims=dims,
                        start_time=start_time,
                        end_time=end_time)
    
    return scene


def get_datacube_satpy_scene(datacube: xr.DataArray,
                        wavelengths: np.ndarray,
                        latitudes: np.ndarray,
                        longitudes: np.ndarray,
                        resolution: float,
                        standard_name: str = "datacube",
                        units: str = "units",
                        dims: list = ['y', 'x'],
                        start_time: datetime = datetime.now(),
                        end_time: datetime = datetime.now(),
                        ) -> Scene:

    latitudes_xr = xr.DataArray(latitudes, dims=dims)
    longitudes_xr = xr.DataArray(longitudes, dims=dims)

    scene = Scene()

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)


    attrs = {
            'file_type': None,
            'resolution': resolution,
            'name': None,
            'standard_name': standard_name,
            'coordinates': ['latitude', 'longitude'],
            'units': units,
            'start_time': start_time,
            'end_time': end_time,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for i, wl in enumerate(wavelengths):

        data = datacube[:,:,i]

        data = data.reset_coords(drop=True)
            
        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=dims)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene


