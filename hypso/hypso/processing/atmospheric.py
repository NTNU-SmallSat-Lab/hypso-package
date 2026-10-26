"""L1C/L1D -> L2A: atmospheric correction.

correct_atmosphere(tree, method, ...) returns an L2A tree: the method's
product cube on the input's bands, the input's geometry, pixel_flags and
metadata, an `atmospheric_correction` global attribute (used in file
names), and an `l2a_<method>` provenance step.

Methods:
- "dps": dark pixel subtraction, in-process on the L1D cube.
- external tools (Polymer, ACOLITE, OC-SMART) - added one at a time.
"""
import numpy as np
import xarray as xr

from hypso.io.product import PRODUCT_DIMS, product_name
from hypso.io.provenance import ac_attrs, add_step
from hypso.masks.flags import FLAGS_NAME, apply_flags

_INPUT_LEVEL = {"dps": "L1D"}


def _input_cube(tree: xr.DataTree, masked: bool, mask_flags) -> xr.DataArray:
    cube = tree.dataset[product_name(tree)]
    if not masked:
        return cube
    if FLAGS_NAME not in tree.dataset:
        raise ValueError("masked=True but the product has no pixel_flags")
    return apply_flags(cube, tree.dataset[FLAGS_NAME], mask_flags)


def _l2a_tree(tree: xr.DataTree, method: str, variable: str, cube: np.ndarray, parameters: dict,
              ac: dict) -> xr.DataTree:
    root = tree.to_dataset(inherit=False)
    source = root[product_name(tree)]
    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out[variable] = xr.DataArray(cube, dims=PRODUCT_DIMS, coords={
        name: ("bands", source[name].values) for name in ("wavelength", "radiation_wavelength", "fwhm")})
    out.attrs["processing_level"] = "L2A"
    out.attrs["atmospheric_correction"] = method

    result = tree.copy()
    result.dataset = out
    input_source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, f"l2a_{method}", input_file=input_source or f"{tree.attrs['processing_level']} (in memory)",
             parameters=parameters, ac=ac)
    return result


def _dps(tree, masked, mask_flags, statistic="min", percentile=None):
    # hypso.ac.ac_dark_pixel_subtraction's math, with NaN-aware statistics so
    # masked (NaN) pixels are ignored instead of making the whole dark
    # spectrum NaN; identical on unmasked input.
    cube = _input_cube(tree, masked, mask_flags).values
    if statistic == "min":
        dark_spectrum = np.nanmin(cube, axis=(0, 1))
    elif statistic == "percentile":
        percentile = 1 if percentile is None else percentile
        dark_spectrum = np.nanpercentile(cube, percentile, axis=(0, 1))
    else:
        raise ValueError("statistic must be 'min' or 'percentile'")
    corrected = np.clip(cube - dark_spectrum, 0, None)
    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "statistic": statistic, "dark_spectrum": [float(v) for v in dark_spectrum]}
    if percentile is not None:
        parameters["percentile"] = percentile
    return _l2a_tree(tree, "dps", "Rrs", corrected, parameters,
                     ac_attrs("dps", version="hypso-package", settings_effective={"statistic": statistic,
                                                                                  "percentile": percentile}))


_METHODS = {"dps": _dps}


def correct_atmosphere(tree: xr.DataTree, method: str, masked: bool = False, mask_flags=None,
                       **options) -> xr.DataTree:
    """Atmospheric correction of an L1C/L1D tree with `method`.

    masked: give the method NaN where pixel_flags are set (all flags, or
        only `mask_flags`, e.g. ["land", "cloud"]). The input product itself
        stays unmasked.
    options: method-specific - dps: statistic ("min" | "percentile"),
        percentile.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; available: {sorted(_METHODS)}")
    level = tree.attrs.get("processing_level")
    if level != _INPUT_LEVEL[method]:
        raise ValueError(f"{method} needs an {_INPUT_LEVEL[method]} product, got {level}")
    return _METHODS[method](tree, masked, mask_flags, **options)
