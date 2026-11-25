"""Pixel windows: a product's pixels in a size x size box around a site and
their per-band statistics (Product.window).

This is the product side of a matchup only. Ground truth, time matching,
other sensors, QC thresholds and the matchup protocol itself belong to the
processing pipeline (hypso-processing-pipeline), which calls this.

The statistics are those of the pipeline's extraction.py for HYPSO: find
the nearest pixel, take the box clipped at the image edges, NaN-aware
mean/median/std per band, CV = std / median, cv_median over a band range
(default 405-570 nm, the pipeline's convention), a pixel counted valid
when any band in that range is finite. No outlier rejection: per the usual
matchup protocol (Bailey & Werdell 2006) heterogeneity shows in the CV
instead. Differences: flagged pixels can be excluded (`exclude_flags`,
pixel_flags names), and the mean angles are the window's, not the whole
scene's. The nearest pixel is found geodesically (Product.pixel), not by
Euclidean distance in degrees as the pipeline does.
"""
import warnings

import numpy as np
import xarray as xr

from hypso.io.product import _extra_variables, band_coords, has_cube, product_name, spatial_dims
from hypso.masks.flags import FLAGS_NAME, decode_flags

CV_BAND_RANGE = (405.0, 570.0)
_ANGLES = ("solar_zenith", "sensor_zenith", "relative_azimuth")


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6_371_000.0 * np.arcsin(np.sqrt(a)))


def window_statistics(tree: xr.DataTree, center: tuple, latitude: float, longitude: float, size: int = 3,
                      exclude_flags=None, cv_band_range=CV_BAND_RANGE) -> xr.Dataset:
    """Window statistics around a given pixel.

    Parameters
    ----------
    tree : xarray.DataTree
        Product tree.
    center : tuple of int
        ``(line, sample)`` of the centre pixel (``(y, x)`` when gridded).
    latitude, longitude : float
        The site, for the reported centre distance.
    size, exclude_flags, cv_band_range
        As in :meth:`hypso.Product.window`.

    Returns
    -------
    xarray.Dataset
        See :meth:`hypso.Product.window`.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be odd and positive, got {size}")
    ds = tree.dataset
    dims = spatial_dims(ds)
    if not has_cube(tree):
        return _variable_window(tree, center, latitude, longitude, size, exclude_flags)
    cube = ds[product_name(tree)]
    line, sample = center
    radius = size // 2
    lines = slice(max(line - radius, 0), min(line + radius + 1, cube.shape[0]))
    samples = slice(max(sample - radius, 0), min(sample + radius + 1, cube.shape[1]))
    box = {dims[0]: lines, dims[1]: samples}
    pixels = cube.isel(box).values.astype(np.float64)

    excluded = np.zeros(pixels.shape[:2], bool)
    if exclude_flags:
        if FLAGS_NAME not in ds:
            raise ValueError(f"exclude_flags given but the product has no {FLAGS_NAME}")
        flags = decode_flags(ds[FLAGS_NAME].isel(box))
        unknown = set(exclude_flags) - set(flags)
        if unknown:
            raise ValueError(f"unknown flags {sorted(unknown)}; the product has {sorted(flags)}")
        for name in exclude_flags:
            excluded |= flags[name].values
    pixels[excluded] = np.nan

    wavelengths = cube["radiation_wavelength"].values
    in_range = (wavelengths >= cv_band_range[0]) & (wavelengths <= cv_band_range[1])
    with warnings.catch_warnings():
        # all-NaN bands (e.g. bands an AC tool does not retrieve) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(pixels, axis=(0, 1))
        median = np.nanmedian(pixels, axis=(0, 1))
        std = np.nanstd(pixels, axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = std / median
        cv_median = float(np.nanmedian(cv[in_range]))
        angles = {f"mean_{name}": float(np.nanmean(ds[name].isel(box).values)) for name in _ANGLES if name in ds}
    valid = np.any(np.isfinite(pixels[:, :, in_range]), axis=2)

    band = ("bands",)
    coords = band_coords(cube)
    center_lat = float(ds["latitude"].values[line, sample])
    center_lon = float(ds["longitude"].values[line, sample])
    result = xr.Dataset(
        {"mean": (band, mean), "median": (band, median), "std": (band, std), "cv": (band, cv),
         "valid_count": (band, np.isfinite(pixels).sum(axis=(0, 1))),
         "pixels": (("window_line", "window_sample", "bands"), pixels),
         "excluded": (("window_line", "window_sample"), excluded)},
        coords=coords,
        attrs={"variable": cube.name, "units": cube.attrs.get("units", ""),
               "site_latitude": float(latitude), "site_longitude": float(longitude),
               "center_line": int(line), "center_sample": int(sample),
               "center_latitude": center_lat, "center_longitude": center_lon,
               "center_distance_m": _haversine_m(latitude, longitude, center_lat, center_lon),
               "box_size": f"{pixels.shape[0]}x{pixels.shape[1]}", "box_pixel_count": int(valid.size),
               "pixel_valid": int(valid.sum()), "percent_valid": float(100 * valid.mean()),
               "cv_median": cv_median, "cv_band_range": list(cv_band_range),
               "excluded_flags": " ".join(exclude_flags or []), **angles})
    return result


def _variable_window(tree: xr.DataTree, center, latitude, longitude, size, exclude_flags) -> xr.Dataset:
    """:func:`window` of a product without a band cube (L2B): the statistics
    of each float product variable, over a ``products`` dimension instead of
    ``bands``."""
    ds = tree.dataset
    dims = spatial_dims(ds)
    names = [n for n in _extra_variables(ds, []) if ds[n].dtype.kind == "f"]
    line, sample = center
    radius = size // 2
    shape = tuple(ds.sizes[d] for d in dims)
    box = {dims[0]: slice(max(line - radius, 0), min(line + radius + 1, shape[0])),
           dims[1]: slice(max(sample - radius, 0), min(sample + radius + 1, shape[1]))}
    pixels = np.stack([ds[n].isel(box).values.astype(np.float64) for n in names], axis=-1)
    excluded = np.zeros(pixels.shape[:2], bool)
    if exclude_flags:
        if FLAGS_NAME not in ds:
            raise ValueError(f"exclude_flags given but the product has no {FLAGS_NAME}")
        flags = decode_flags(ds[FLAGS_NAME].isel(box))
        unknown = set(exclude_flags) - set(flags)
        if unknown:
            raise ValueError(f"unknown flags {sorted(unknown)}; the product has {sorted(flags)}")
        for name in exclude_flags:
            excluded |= flags[name].values
    pixels[excluded] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(pixels, axis=(0, 1))
        median = np.nanmedian(pixels, axis=(0, 1))
        std = np.nanstd(pixels, axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = std / median
        angles = {f"mean_{name}": float(np.nanmean(ds[name].isel(box).values)) for name in _ANGLES if name in ds}
    center_lat = float(ds["latitude"].values[line, sample])
    center_lon = float(ds["longitude"].values[line, sample])
    dim = ("products",)
    return xr.Dataset(
        {"mean": (dim, mean), "median": (dim, median), "std": (dim, std), "cv": (dim, cv),
         "valid_count": (dim, np.isfinite(pixels).sum(axis=(0, 1))),
         "pixels": (("window_line", "window_sample", "products"), pixels),
         "excluded": (("window_line", "window_sample"), excluded)},
        coords={"products": names, "units": ("products", [ds[n].attrs.get("units", "") for n in names])},
        attrs={"level": tree.attrs.get("processing_level"), "l2b_group": tree.attrs.get("l2b_group", ""),
               "site_latitude": float(latitude), "site_longitude": float(longitude),
               "center_line": int(line), "center_sample": int(sample),
               "center_latitude": center_lat, "center_longitude": center_lon,
               "center_distance_m": _haversine_m(latitude, longitude, center_lat, center_lon),
               "box_size": f"{pixels.shape[0]}x{pixels.shape[1]}",
               "excluded_flags": " ".join(exclude_flags or []), **angles})
