# SatPy functions

from typing import Union

from ..hypso import Hypso
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

import xarray as xr



def get_l1a_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

    datacube = satobj.l1a_cube
    wavelengths = range(0,120)

    scene = get_datacube_satpy_scene(satobj=satobj, datacube=datacube, wavelengths=wavelengths)

    return scene


def get_l1b_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

    datacube = satobj.l1b_cube
    wavelengths = satobj.wavelengths

    scene = get_datacube_satpy_scene(satobj=satobj, datacube=datacube, wavelengths=wavelengths)

    return scene


def get_l1c_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

    datacube = satobj.l1c_cube
    wavelengths = satobj.wavelengths

    scene = get_datacube_satpy_scene(satobj=satobj, datacube=datacube, wavelengths=wavelengths)

    return scene


def get_l2a_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

    datacube = satobj.l2a_cube
    wavelengths = satobj.wavelengths

    scene = get_datacube_satpy_scene(satobj=satobj, datacube=datacube, wavelengths=wavelengths)

    return scene



def get_datacube_satpy_scene(satobj: Union[Hypso1, Hypso2],
                        datacube,
                        wavelengths
                        ) -> Scene:

    scene = _generate_satpy_scene(satobj=satobj)
    swath_def= _generate_swath_definition(satobj=satobj)

    attrs = {
            'file_type': None,
            'resolution': satobj.resolution,
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









# def get_l1a_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

#     scene = _generate_satpy_scene(satobj=satobj)
#     swath_def= _generate_swath_definition(satobj=satobj)

#     try:
#         cube = satobj.l1a_cube
#     except:
#         return None

#     attrs = {
#             'file_type': None,
#             'resolution': satobj.resolution,
#             'name': None,
#             'standard_name': cube.attrs['description'],
#             'coordinates': ['latitude', 'longitude'],
#             'units': cube.attrs['units'],
#             'start_time': satobj.capture_datetime,
#             'end_time': satobj.capture_datetime,
#             'modifiers': (),
#             'ancillary_variables': []
#             }   

#     # TODO: use dimensions from l1a cube
#     wavelengths = range(0,120)

#     for i, wl in enumerate(wavelengths):

#         data = cube[:,:,i]

#         data = data.reset_coords(drop=True)
            
#         name = 'band_' + str(i+1)
#         scene[name] = data
#         #scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
#         scene[name].attrs.update(attrs)
#         scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
#         scene[name].attrs['band'] = i
#         scene[name].attrs['area'] = swath_def

#     return scene


def generate_satpy_scene(satobj: Union[Hypso1, Hypso2], datasets: dict) -> Scene:

    scene = satobj._generate_satpy_scene()
    swath_def= satobj._generate_swath_definition()

    attrs = {
            'file_type': None,
            'resolution': satobj.resolution,
            'name': None,
            'standard_name': None,
            'coordinates': ['latitude', 'longitude'],
            'units': None,
            'start_time': satobj.capture_datetime,
            'end_time': satobj.capture_datetime,
            'modifiers': (),
            'ancillary_variables': []
            }

    for key, dataset in dataset.items():

            scene[key] = dataset
            scene[key].attrs.update(attrs)
            scene[key].attrs['name'] = key
            scene[key].attrs['standard_name'] = key
            scene[key].attrs['area'] = swath_def

            try:
                scene[key].attrs.update(dataset.attrs)
            except AttributeError:
                pass


    return scene


def _generate_satpy_scene(satobj: Union[Hypso1, Hypso2]) -> Scene:

    scene = Scene()

    latitudes, longitudes = _generate_latlons(satobj=satobj)

    swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

    latitude_attrs = {
                        'file_type': None,
                        'resolution': satobj.resolution,
                        'standard_name': 'latitude',
                        'units': 'degrees_north',
                        'start_time': satobj.capture_datetime,
                        'end_time': satobj.capture_datetime,
                        'modifiers': (),
                        'ancillary_variables': []
                        }

    longitude_attrs = {
                        'file_type': None,
                        'resolution': satobj.resolution,
                        'standard_name': 'longitude',
                        'units': 'degrees_east',
                        'start_time': satobj.capture_datetime,
                        'end_time': satobj.capture_datetime,
                        'modifiers': (),
                        'ancillary_variables': []
                        }

    #scene['latitude'] = latitudes
    #scene['latitude'].attrs.update(latitude_attrs)
    #scene['latitude'].attrs['area'] = swath_def
    #scene['longitude'] = longitudes
    #scene['longitude'].attrs.update(longitude_attrs)
    #scene['longitude'].attrs['area'] = swath_def

    return scene






def _generate_latlons(satobj: Union[Hypso1, Hypso2]) -> tuple[xr.DataArray, xr.DataArray]:

    latitudes = xr.DataArray(satobj.latitudes, dims=satobj.dim_names_2d)
    longitudes = xr.DataArray(satobj.longitudes, dims=satobj.dim_names_2d)

    return latitudes, longitudes

def _generate_swath_definition(satobj: Union[Hypso1, Hypso2]) -> SwathDefinition:

    latitudes, longitudes = _generate_latlons(satobj=satobj)
    swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

    return swath_def



