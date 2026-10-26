"""Regridding: a product (L1C, L1D or L2A) from sensor geometry onto a map
grid, as a product of the same level (REFACTOR_PROGRESS.md TODO 2).

The gridded product has dims (y, x[, bands]) with CF projection coordinates
x/y and a `crs` grid-mapping variable, which ESA SNAP and GIS tools read as
a map-projected raster; latitude/longitude are kept as 2D auxiliary
coordinates. Per variable:

- product cube and float variables: `method` - "nearest" by default, so
  every grid pixel holds one measured spectrum, unmixed; "bilinear" on
  request (it blends neighbouring pixels' spectra);
- azimuth angles: like the rest, but bilinear on their sine and cosine (no
  0/360 wrap artefacts);
- flags and other integer variables: nearest neighbour;
- pixel_flags gains an `outside_swath` flag for grid pixels the capture
  does not cover;
- *_direct geolocation and angles are dropped (the grid's latitude/longitude
  replace both georeferencings).
"""
import logging

import numpy as np
import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition

from hypso.io.product import GRID_DIMS, GRID_MAPPING, product_name
from hypso.io.provenance import add_step
from hypso.masks.flags import FLAGS_NAME, decode_flags, encode_flags

from .areas import (METRES_PER_DEGREE, coordinate_attrs, geolocation, grid_area, grid_name,
                    resolution as product_resolution, source_definition, utm_crs)
from .resample import METHODS, Resampler, default_radius, target_resolution_metres

logger = logging.getLogger(__name__)

OUTSIDE_FLAG = "outside_swath"
_LEVELS = ("L1C", "L1D", "L2A")


def _target_area(tree, target, resolution) -> AreaDefinition:
    if isinstance(target, AreaDefinition):
        if resolution is not None:
            raise ValueError("resolution is set by the AreaDefinition; do not pass it as well")
        return target
    latitudes, longitudes = geolocation(tree)
    crs = utm_crs(latitudes, longitudes) if target == "utm" else pyproj.CRS.from_user_input(target)
    if resolution is None:
        # The product's own (coarser) GSD: no oversampling by default.
        metres = product_resolution(tree)
        resolution = round(metres / METRES_PER_DEGREE, 4) if crs.is_geographic else float(round(metres))
    return grid_area(crs, resolution, latitudes, longitudes)


def _flag_descriptions(flags: xr.DataArray) -> dict:
    out = {}
    for part in flags.attrs.get("comment", "").split("; "):
        name, _, text = part.partition(": ")
        if text:
            out[name] = text
    return out


def regrid(tree: xr.DataTree, target="utm", resolution: float = None, method: str = "nearest",
           radius_of_influence: float = None, max_gb: float = 4.0) -> xr.DataTree:
    """The product tree on a map grid (same level).

    target: "utm" (WGS84 UTM zone of the scene centre), a CRS pyproj
        accepts (e.g. "EPSG:3035", 4326), or a pyresample AreaDefinition
        (e.g. another product's grid, Product.area_definition()).
    resolution: pixel size in the CRS's units (m, or degrees for a
        geographic CRS); default the product's resolution. Not with an
        AreaDefinition.
    method: "nearest" (default; spectra are not mixed) or "bilinear", for
        the cube and float variables.
    radius_of_influence: metres; default 1.5 times the coarser of source and
        target pixel size.
    max_gb: refuse grids whose cube would exceed this many GB (a fine grid
        over a long capture gets large quickly).
    """
    level = tree.attrs.get("processing_level")
    if level not in _LEVELS:
        raise ValueError(f"regrid needs an {', '.join(_LEVELS)} product, got {level} (no geolocation: "
                         f"calibrate and georeference it first)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    ds = tree.to_dataset(inherit=False)
    cube_name = product_name(tree)
    cube = ds[cube_name]
    area = _target_area(tree, target, resolution)

    size_gb = area.shape[0] * area.shape[1] * cube.sizes["bands"] * cube.dtype.itemsize / 1e9
    if size_gb > max_gb:
        raise ValueError(f"the {area.shape[0]} x {area.shape[1]} grid would make a {size_gb:.1f} GB cube "
                         f"(max_gb={max_gb}); use a coarser resolution or raise max_gb")
    source_metres = product_resolution(tree)
    target_metres = target_resolution_metres(area)
    radius = radius_of_influence or default_radius(source_metres, target_metres)
    resampler = Resampler(source_definition(tree), area, radius, bilinear=method == "bilinear")
    logger.info("Regridding %s %s onto %s (%d x %d, %s)", level, cube_name, area.crs.name, area.shape[0],
                area.shape[1], method)

    out = xr.Dataset(attrs=dict(ds.attrs))
    out[cube_name] = xr.DataArray(resampler(cube.values, method), dims=GRID_DIMS + ("bands",),
                                  coords={name: ("bands", cube[name].values)
                                          for name in ("wavelength", "radiation_wavelength", "fwhm")},
                                  attrs=dict(cube.attrs))
    grid_lon, grid_lat = area.get_lonlats()
    out["latitude"] = (GRID_DIMS, grid_lat)
    out["longitude"] = (GRID_DIMS, grid_lon)

    skip = {cube_name, "latitude", "longitude", "crs_wgs84", GRID_MAPPING, FLAGS_NAME}
    dropped = []
    for name, var in ds.data_vars.items():
        if name in skip:
            continue
        if name.endswith("_direct") or var.ndim != 2:
            dropped.append(name)
            continue
        values = var.values
        if values.dtype.kind in "iub":
            data = resampler.nearest(values)
        elif "azimuth" in name and method == "bilinear":
            radians = np.deg2rad(values)
            data = np.rad2deg(np.arctan2(resampler(np.sin(radians), method), resampler(np.cos(radians), method)))
            data = np.mod(data, 360.0).astype(values.dtype)
        else:
            data = resampler(values, method)
        out[name] = (GRID_DIMS, data, dict(var.attrs))

    covered = resampler.coverage(method)
    masks, descriptions = {}, {}
    if FLAGS_NAME in ds:
        flags = xr.DataArray(resampler.nearest(ds[FLAGS_NAME].values), dims=GRID_DIMS,
                             attrs=dict(ds[FLAGS_NAME].attrs))
        masks = {name: flag.values for name, flag in decode_flags(flags).items()}
        descriptions = _flag_descriptions(ds[FLAGS_NAME])
    masks[OUTSIDE_FLAG] = ~covered
    descriptions[OUTSIDE_FLAG] = "grid pixel not covered by the capture (no data)"
    out[FLAGS_NAME] = encode_flags(masks, descriptions, dims=GRID_DIMS)

    crs = area.crs
    x, y = area.get_proj_vectors()
    x_attrs, y_attrs = coordinate_attrs(crs)
    out.coords["x"] = ("x", x, x_attrs)
    out.coords["y"] = ("y", y, y_attrs)
    out[GRID_MAPPING] = ((), np.int32(0), crs.to_cf())
    size_x, size_y = abs(float(area.pixel_size_x)), abs(float(area.pixel_size_y))
    name = grid_name(crs, size_x, size_y)
    out.attrs.update(grid=name, grid_crs=crs.to_string(), grid_resolution=max(size_x, size_y),
                     grid_resolution_x=size_x, grid_resolution_y=size_y)

    result = tree.copy()
    result.dataset = out
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    parameters = {"grid": name, "crs": crs.to_string(), "shape": f"{area.shape[0]} x {area.shape[1]}",
                  "resolution_x": size_x, "resolution_y": size_y, "method": method,
                  "flags_method": "nearest", "azimuth_method": "sin/cos " + method if method == "bilinear"
                  else "nearest", "radius_of_influence_m": float(radius),
                  "source_grid": tree.attrs.get("grid") or "swath",
                  "covered_fraction": float(covered.mean())}
    if dropped:
        parameters["dropped_variables"] = " ".join(dropped)
    add_step(result, "regrid", input_file=source or f"{level} (in memory)", parameters=parameters)
    return result
