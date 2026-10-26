"""Resample a hypso.Product's cube, derived variables and flags onto a
pyresample AreaDefinition/SwathDefinition (nearest neighbour).

Building block for the regridding TODO (REFACTOR_PROGRESS.md: common grids,
another sensor's grid); the output is a plain xr.Dataset on the target grid."""
import numpy as np
import xarray as xr

from .resamplers import resample_dataarray_kd_tree_nearest

_RENAME = {"lines": "y", "samples": "x", "bands": "band"}


def resample_product(product, area_def, radius_of_influence: float = None) -> xr.Dataset:
    """xr.Dataset on area_def's grid: the product cube (y, x, band, with
    the source band wavelengths), each derived variable and each flag
    (y, x), and the grid's latitude/longitude. radius_of_influence (m)
    defaults to the product's ground resolution."""
    geometry = product.geometry
    if "latitude" not in geometry:
        raise ValueError(f"{product.level} product has no geolocation to resample from")
    latitudes, longitudes = geometry["latitude"].values, geometry["longitude"].values
    radius = radius_of_influence if radius_of_influence is not None else product.resolution

    def resample(data: xr.DataArray) -> xr.DataArray:
        data = data.rename({k: v for k, v in _RENAME.items() if k in data.dims}).reset_coords(drop=True)
        return resample_dataarray_kd_tree_nearest(area_def=area_def, data=data, latitudes=latitudes,
                                                  longitudes=longitudes, radius_of_influence=radius)

    cube = product.cube
    out = {cube.name: resample(cube.astype(float))}
    for name in product.variables.data_vars:
        out[name] = resample(product.variables[name].astype(float))
    for name, flag in product.flags.items():
        # Nearest neighbour keeps flags boolean; pixels outside the swath are False.
        resampled = resample(flag.astype(float))
        out[f"flag_{name}"] = resampled.fillna(0).astype(bool)

    grid_lon, grid_lat = area_def.get_lonlats()
    ds = xr.Dataset(out, coords={"latitude": (("y", "x"), grid_lat), "longitude": (("y", "x"), grid_lon),
                                 "wavelength": ("band", cube["radiation_wavelength"].values)})
    ds.attrs.update(sat_id=product.sat_id, target=product.target, processing_level=product.level)
    return ds
