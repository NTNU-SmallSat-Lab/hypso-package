# SatPy functions

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition

import numpy as np
import xarray as xr

from datetime import datetime



def product_scene(product, use_direct_georef: bool = False) -> Scene:
    """A satpy Scene of a hypso.Product (any georeferenced level): one
    dataset per band, on the product's swath, or on its map grid if it is
    regridded."""
    from datetime import datetime as _datetime
    from hypso.processing.capture import acquisition_time

    suffix = "_direct" if use_direct_georef else ""
    geometry = product.geometry
    if f"latitude{suffix}" not in geometry:
        raise ValueError(f"{product.level} product has no geolocation")
    time = _datetime.fromisoformat(acquisition_time(product.tree.attrs))
    if product.level == "L2B":
        return _variables_scene(product, geometry, suffix, time)
    cube = product.cube
    return get_datacube_satpy_scene(datacube=cube, wavelengths=cube["radiation_wavelength"].values,
                                    latitudes=geometry[f"latitude{suffix}"].values,
                                    longitudes=geometry[f"longitude{suffix}"].values,
                                    resolution=product.resolution, standard_name=cube.name,
                                    units=product.tree.dataset[cube.name].attrs.get("units", ""),
                                    dims=["y", "x"], start_time=time, end_time=time,
                                    area=product.area_definition() if product.grid else None)


def _variables_scene(product, geometry, suffix: str, time) -> Scene:
    """A product without a band cube (L2B): one dataset per float variable."""
    lat = xr.DataArray(geometry[f"latitude{suffix}"].values, dims=["y", "x"])
    lon = xr.DataArray(geometry[f"longitude{suffix}"].values, dims=["y", "x"])
    area = product.area_definition() if product.grid else SwathDefinition(lons=lon, lats=lat)
    sensor = "hypso1" if product.sat_id == "HYPSO-1" else "hypso2"
    scene = Scene()
    for name, var in product.variables.data_vars.items():
        if var.dtype.kind != "f":
            continue
        scene[name] = xr.DataArray(var.values, dims=["y", "x"], attrs={
            **{k: v for k, v in var.attrs.items() if k not in ("coordinates", "grid_mapping")},
            "name": name, "area": area, "start_time": time, "end_time": time,
            "platform_name": product.sat_id, "sensor": sensor})
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
                        area=None,
                        ) -> Scene:

    latitudes_xr = xr.DataArray(latitudes, dims=dims)
    longitudes_xr = xr.DataArray(longitudes, dims=dims)

    scene = Scene()

    # A map grid (AreaDefinition) when given, else the swath of latitude/longitude.
    swath_def = area if area is not None else SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)


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


