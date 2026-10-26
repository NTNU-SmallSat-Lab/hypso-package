# SatPy functions

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

import numpy as np
import xarray as xr

from datetime import datetime



def product_scene(product, use_direct_georef: bool = False) -> Scene:
    """A satpy Scene of a hypso.Product (any georeferenced level): one
    dataset per band, on the product's swath."""
    from datetime import datetime as _datetime
    from hypso.processing.capture import acquisition_time

    suffix = "_direct" if use_direct_georef else ""
    geometry = product.geometry
    if f"latitude{suffix}" not in geometry:
        raise ValueError(f"{product.level} product has no geolocation")
    cube = product.cube
    time = _datetime.fromisoformat(acquisition_time(product.tree.attrs))
    return get_datacube_satpy_scene(datacube=cube, wavelengths=cube["radiation_wavelength"].values,
                                    latitudes=geometry[f"latitude{suffix}"].values,
                                    longitudes=geometry[f"longitude{suffix}"].values,
                                    resolution=product.resolution, standard_name=cube.name,
                                    units=product.tree.dataset[cube.name].attrs.get("units", ""),
                                    dims=["y", "x"], start_time=time, end_time=time)


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
        if "lines" in data.dims:
            data = data.rename({"lines": dims[0], "samples": dims[1]})

        data = data.reset_coords(drop=True)
            
        name = 'band_' + str(i+1)
        scene[name] = data
        #scene[name] = xr.DataArray(data, dims=dims)
        scene[name].attrs.update(attrs)
        scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
        scene[name].attrs['band'] = i
        scene[name].attrs['area'] = swath_def

    return scene


