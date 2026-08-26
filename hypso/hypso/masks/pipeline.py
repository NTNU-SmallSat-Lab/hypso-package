"""Per-capture mask orchestration, extracted from HypsoCapture (part of the
HypsoCapture breakup called for in the approved refactor plan, see
REFACTOR_PROGRESS.md). Bodies are moved verbatim from HypsoCapture.py, not
rewritten - each function takes `satobj` explicitly (matching
hypso.calibration.pipeline/hypso.georeferencing.geo's pattern: these read
several HypsoCapture attributes/methods, so taking satobj as a parameter
avoids either duplicating that state or coupling this module tightly to
HypsoCapture's internals via a stored reference).

Not to be confused with the sibling jon_cnn_classifier.py/
jonas_svm_classifier.py in this same package - those are the actual mask
*algorithms* (CNN/SVM label decoding); this module is the *container* that
holds land_mask/cloud_mask/custom masks and applies them to cubes, regardless
of which algorithm (or none) produced them.

masked_l1a_cube/masked_l1b_cube/masked_l1c_cube/masked_l1d_cube were four
near-identical bodies differing only in which cube attribute they read -
collapsed into one get_masked_cube(satobj, level), matching the
_format_cube_dataarray/resample_cube precedent already established
elsewhere in this refactor for the same shape of duplication.
"""
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
import netCDF4 as nc

from hypso.containers import as_dataarray, DatasetDict

_LEVEL_TO_CUBE_ATTR = {
    "l1a": "_l1a_cube",
    "l1b": "_l1b_cube",
    "l1d": "_l1d_cube",
}


def format_mask_dataarray(satobj, data: Union[np.ndarray, xr.DataArray], description: str) -> xr.DataArray:
    """Validate/wrap a 2D (lines, samples) boolean-ish mask array. Shared by
    land_mask/cloud_mask and set_custom_mask - a mask is a mask regardless of
    what it represents, so there's one validation path, not one per name."""
    attributes = {
                  'description': description,
                  'method': None
                 }

    data = as_dataarray(data, tuple(satobj.dim_names_2d), num_dims=2,
                        dim_shape=tuple(satobj.spatial_dimensions))
    data = satobj._update_dataarray_attrs(data, attributes)

    return data


def format_land_mask_dataarray(satobj, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:
    return format_mask_dataarray(satobj, data, "Land mask")


def format_cloud_mask_dataarray(satobj, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:
    return format_mask_dataarray(satobj, data, "Cloud mask")


def set_custom_mask(satobj, name: str, value: Union[np.ndarray, xr.DataArray, None],
                    description: str = None) -> None:
    """Register (or clear, if value is None) a named custom mask - e.g. an
    externally-produced sea/land/cloud classification, not just the built-in
    land_mask/cloud_mask slots. Any number of custom masks may be registered
    at once; all of them (plus land_mask/cloud_mask, if set) are OR'd
    together by unified_mask and applied by get_masked_cube - no
    further wiring needed once registered.

    :param name: key this mask is stored/removed under (also used in
        load_mask_from_file's `name=` argument).
    :param value: 2D (lines, samples) boolean-ish array/DataArray, True where
        a pixel should be masked out. None removes this mask.
    :param description: optional human-readable note, stored on the
        DataArray's `description` attribute (defaults to `name`).
    """
    if value is None:
        satobj._custom_masks.pop(name, None)
        return None

    satobj._custom_masks[name] = format_mask_dataarray(satobj, value, description or name)
    return None


def clear_custom_masks(satobj) -> None:
    """Remove every registered custom mask (land_mask/cloud_mask are unaffected)."""
    satobj._custom_masks = DatasetDict(dim_names=('y', 'x'), num_dims=2)
    return None


def load_mask_from_file(satobj, path: Union[str, Path], name: str = None, variable: str = None,
                        dtype=np.bool_, invert: bool = False) -> xr.DataArray:
    """Load a 2D (lines, samples) mask from disk and, if `name` is given,
    register it via set_custom_mask in the same call.

    Supported formats, dispatched by file extension:
      - .nc: reads `variable` (required) from the file's root group via
        netCDF4 - for a mask produced by another tool (e.g. a sea/land/cloud
        classification saved as its own NetCDF product).
      - .npy: numpy .npy array.
      - .dat/.bin: raw binary, reshaped to satobj.spatial_dimensions using
        `dtype` (matches the convention HYPSO's own indirect-georeferencing
        lat/lon files use).

    :param path: path to the mask file.
    :param name: if given, also calls set_custom_mask(name, data) - the mask
        is registered and immediately reflected in get_masked_cube.
    :param variable: required for .nc input - the variable name to read.
    :param dtype: numpy dtype to interpret raw binary (.dat/.bin) data as.
    :param invert: if True, flip the mask (use when the source file marks
        *valid* pixels with True rather than masked-out pixels).
    :return: the loaded mask as a validated xr.DataArray (same object stored
        under `name`, if `name` was given).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.nc':
        if variable is None:
            raise ValueError("load_mask_from_file requires variable=... for a .nc source file")
        with nc.Dataset(path, format="NETCDF4") as f:
            data = np.array(f.variables[variable][:])
    elif suffix == '.npy':
        data = np.load(path)
    elif suffix in ('.dat', '.bin'):
        data = np.fromfile(path, dtype=dtype).reshape(satobj.spatial_dimensions)
    else:
        raise ValueError(f"load_mask_from_file: unsupported file extension {suffix!r} "
                         f"(expected .nc, .npy, .dat, or .bin)")

    data = data.astype(bool)
    if invert:
        data = ~data

    if name is not None:
        set_custom_mask(satobj, name, data)
        return satobj._custom_masks[name]

    return format_mask_dataarray(satobj, data, path.name)


def unified_mask(satobj) -> xr.DataArray:
    """OR land_mask, cloud_mask, and every registered custom mask (see
    set_custom_mask/load_mask_from_file) together - get_masked_cube applies
    whatever this returns, so a custom mask needs no changes there."""
    masks = [m for m in (satobj._land_mask, satobj._cloud_mask) if m is not None]
    masks.extend(satobj._custom_masks.values())

    if not masks:
        return None

    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask

    return combined


def get_masked_cube(satobj, level: str) -> xr.DataArray:
    """masked_l1a_cube/masked_l1b_cube/masked_l1c_cube/masked_l1d_cube's
    shared body - see _LEVEL_TO_CUBE_ATTR for which underlying cube attribute
    each level reads.

    l1c is special-cased to satobj.l1c_cube (the public property, a deepcopy
    of _l1b_cube relabeled - see its own getter) rather than a private
    _l1c_cube attribute: FIXED bug (was self._l1c_cube.where(...) before this
    extraction) - _l1c_cube is never actually populated anywhere
    (_generate_l1c_cube_impl never sets it), so masked_l1c_cube always
    silently returned None. Now mirrors l1c_cube's own getter instead.
    """
    if level == "l1c":
        cube = satobj.l1c_cube
    else:
        cube = getattr(satobj, _LEVEL_TO_CUBE_ATTR[level])
    mask = unified_mask(satobj)

    if mask is not None:
        return cube.where(~mask, other=np.nan)

    return cube
