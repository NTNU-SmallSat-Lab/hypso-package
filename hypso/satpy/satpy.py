# SatPy functions

from typing import Union

from ..hypso import Hypso
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

import numpy as np
import xarray as xr



def get_l1a_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        indirect: bool = False
                        ) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = satobj.wavelengths

    if indirect:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(satobj=satobj,
                        datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution)
    
    return scene





def get_l1b_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        indirect: bool = False
                        ) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = satobj.wavelengths

    if indirect:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(satobj=satobj,
                        datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution)
    
    return scene




def get_l1c_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        indirect: bool = False
                        ) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = satobj.wavelengths

    if indirect:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(satobj=satobj,
                        datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution)
    
    return scene




def get_l1d_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        indirect: bool = False
                        ) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = satobj.wavelengths

    if indirect:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  

    scene = get_datacube_satpy_scene(satobj=satobj,
                        datacube=datacube,
                        wavelengths=wavelengths,
                        latitudes=latitudes,
                        longitudes=longitudes,
                        resolution=resolution)
    
    return scene






def get_datacube_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        datacube: xr.DataArray,
                        wavelengths: np.ndarray,
                        latitudes: np.ndarray,
                        longitudes: np.ndarray,
                        resolution
                        ) -> Scene:

    latitudes_xr = xr.DataArray(latitudes, dims=satobj.dim_names_2d)
    longitudes_xr = xr.DataArray(longitudes, dims=satobj.dim_names_2d)

    scene = Scene()

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)


    attrs = {
            'file_type': None,
            'resolution': resolution,
            'name': None,
            'standard_name': datacube.attrs['description'],
            'coordinates': ['latitude', 'longitude'],
            'units': datacube.attrs['units'],
            'start_time': satobj.capture_datetime,
            'end_time': satobj.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }   

    for i, wl in enumerate(wavelengths):

        data = datacube[:,:,i]

        data = data.reset_coords(drop=True)
            
        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene


