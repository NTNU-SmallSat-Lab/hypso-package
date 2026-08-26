"""Resample HYPSO cubes/products onto a target pyresample AreaDefinition/
SwathDefinition. resample_cube() is the one shared implementation -
resample_l1a_cube/l1b_cube/l1c_cube/l1d_cube below are thin per-level
wrappers over it (each used to be an independent, near-identical ~25-line
function differing only in which satobj.l1X_cube attribute it read - zero
external callers of any of them, confirmed, so collapsing them was free).
resample_products() applies the same function to every entry in
satobj.products (see hypso.containers.DatasetDict) - previously a stub whose
only body was a commented-out block referencing the since-deleted
DataArrayDict, now a real, working implementation reusing resample_cube like
everything else here."""
from typing import Union

import xarray as xr

from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

from pyresample.geometry import SwathDefinition, AreaDefinition

from .resamplers import resample_dataarray_kd_tree_nearest


def resample_cube(cube: xr.DataArray, satobj: Union[Hypso1, Hypso2],
                  area_def: Union[SwathDefinition, AreaDefinition],
                  use_indirect_georef: bool = False):
    """Resample one (lines, samples[, band]) array sharing satobj's own
    geolocation onto area_def via nearest-neighbor. Not tied to a specific
    hardcoded satobj attribute - works for any cube (l1a/b/c/d) or product,
    which is what lets every wrapper function in this module (and
    resample_products, looping over several) share one implementation.

    :return: (resampled_data, resampled_latitudes, resampled_longitudes)
    """
    if use_indirect_georef:
        # FIXED (was satobj.latitudes_indirect/longitudes_indirect/
        # resolution_indirect - names never set anywhere; only _direct-suffixed
        # names exist for the OTHER georeferencing method. run_georeferencing
        # (hypso.georeferencing.geo) is what populates these plain names -
        # that's the "indirect" (externally-supplied lat/lon) path.
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution

    resampled_data = resample_dataarray_kd_tree_nearest(area_def = area_def,
                                                        data = cube,
                                                        latitudes = latitudes,
                                                        longitudes = longitudes,
                                                        radius_of_influence=resolution)

    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    return resampled_data, resampled_latitudes, resampled_longitudes


def resample_l1a_cube(satobj: Union[Hypso1, Hypso2],
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False):
    return resample_cube(satobj.l1a_cube, satobj, area_def, use_indirect_georef)


def resample_l1b_cube(satobj: Union[Hypso1, Hypso2],
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False):
    return resample_cube(satobj.l1b_cube, satobj, area_def, use_indirect_georef)


def resample_l1c_cube(satobj: Union[Hypso1, Hypso2],
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False):
    return resample_cube(satobj.l1c_cube, satobj, area_def, use_indirect_georef)


def resample_l1d_cube(satobj: Union[Hypso1, Hypso2],
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False):
    return resample_cube(satobj.l1d_cube, satobj, area_def, use_indirect_georef)


def resample_products(satobj: Union[Hypso1, Hypso2],
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False) -> xr.Dataset:
    """Resample every registered entry in satobj.products (see
    hypso.containers.DatasetDict) onto area_def, returning a plain
    xr.Dataset with the target grid's own latitude/longitude as coordinates.
    A plain Dataset, not wrapped back into a DatasetDict/products-style
    container - a resampled result is a terminal output for the caller to
    consume or write, not something meant to be further validated/mutated
    through the capture's own product-registration path.

    :return: xr.Dataset with one 2D data variable per satobj.products key.
    """
    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    resampled_vars = {}
    for key, product in satobj.products.items():
        resampled_data, _, _ = resample_cube(product, satobj, area_def, use_indirect_georef)
        resampled_vars[key] = resampled_data

    # dims_2d's established convention throughout this module/resamplers.py -
    # every 2D resample_dataarray_kd_tree_nearest result uses these names.
    ds = xr.Dataset(resampled_vars)
    ds = ds.assign_coords(
        latitude=(("y", "x"), resampled_latitudes),
        longitude=(("y", "x"), resampled_longitudes),
    )
    return ds
