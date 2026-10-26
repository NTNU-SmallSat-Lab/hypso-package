"""Masks as one CF flag variable: products are
written unmasked, with land/cloud/custom masks as bits of `pixel_flags`
(dims (lines, samples)), the CF flag_masks / flag_meanings encoding that
ESA SNAP turns into masks and NASA uses for l2_flags.

    pixel_flags = land(1) | cloud(2) | <custom masks, registration order>

Use apply_flags() where masked data is needed (e.g. AC input staging)
instead of writing a separate NaN-masked copy of the product.
"""
import re

import numpy as np
import xarray as xr

FLAGS_NAME = "pixel_flags"
SPATIAL_DIMS = ("lines", "samples")
_MEANING = re.compile(r"[A-Za-z0-9_]+")


def _flag_dtype(count: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if count <= np.iinfo(dtype).bits:
            return dtype
    raise ValueError(f"{count} masks do not fit in one flag variable (max 64)")


def encode_flags(masks: dict, descriptions: dict = None, dims=SPATIAL_DIMS) -> xr.DataArray:
    """Encode boolean masks as one CF flag variable, ``pixel_flags``.

    Parameters
    ----------
    masks : dict
        ``{name: 2D boolean array}``, True where flagged, in bit order (land
        and cloud first by convention). Names become ``flag_meanings``
        words, so they may only contain letters, digits and ``_``.
    descriptions : dict, optional
        ``{name: text}``, written to the variable's ``comment``.
    dims : tuple of str, default ("lines", "samples")
        ``("y", "x")`` for a gridded product.

    Returns
    -------
    xarray.DataArray
        Unsigned integer flags with ``flag_masks`` and ``flag_meanings``.
    """
    if not masks:
        raise ValueError("no masks to encode")
    names = list(masks)
    for name in names:
        if not _MEANING.fullmatch(name):
            raise ValueError(f"mask name {name!r} may only contain letters, digits and '_'")
    dtype = _flag_dtype(len(names))
    bits = np.array([1 << i for i in range(len(names))], dtype=dtype)
    first = np.asarray(masks[names[0]])
    flags = np.zeros(first.shape, dtype=dtype)
    for bit, name in zip(bits, names):
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != first.shape:
            raise ValueError(f"mask {name!r} has shape {mask.shape}, expected {first.shape}")
        flags |= np.where(mask, bit, 0).astype(dtype)
    attrs = {"long_name": "Pixel classification flags",
             "flag_masks": bits,
             "flag_meanings": " ".join(names)}
    if descriptions:
        attrs["comment"] = "; ".join(f"{n}: {descriptions[n]}" for n in names if descriptions.get(n))
    return xr.DataArray(flags, dims=tuple(dims), attrs=attrs, name=FLAGS_NAME)


def decode_flags(flags: xr.DataArray) -> dict:
    """Decode a CF flag variable into ``{name: 2D boolean DataArray}``."""
    names = flags.attrs["flag_meanings"].split()
    # atleast_1d: netCDF returns a one-flag variable's flag_masks as a scalar.
    bits = np.atleast_1d(flags.attrs["flag_masks"])
    return {name: (flags & bit) != 0 for name, bit in zip(names, bits)}


def apply_flags(cube: xr.DataArray, flags: xr.DataArray, names=None) -> xr.DataArray:
    """``cube`` with NaN wherever any of the named flags is set.

    Parameters
    ----------
    cube : xarray.DataArray
        Data with the flags' spatial dimensions.
    flags : xarray.DataArray
        A flag variable (:func:`encode_flags`).
    names : list of str, optional
        Flags to apply; default all.

    Raises
    ------
    KeyError
        For a name that is not a flag.
    """
    decoded = decode_flags(flags)
    names = list(decoded) if names is None else list(names)
    unknown = sorted(set(names) - set(decoded))
    if unknown:
        raise KeyError(f"unknown flags {unknown}; available: {sorted(decoded)}")
    mask = np.zeros(flags.shape, dtype=bool)
    for name in names:
        mask |= decoded[name].values
    return cube.where(~xr.DataArray(mask, dims=flags.dims))
