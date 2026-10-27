"""HYPSO-2's RGB camera: reading its images (groundwork for camera support).

A capture directory holds the camera image as ``<capture id>-rgb/<target>_
<YYYY_MM_DD>T<HH_MM_SS>.bmp``: one 8-bit raw Bayer mosaic (3840 x 2748),
taken about 2 s before the HSI capture starts. The mosaic is RGGB; the ground
processing's demosaiced copy (``processing-temp/rgb-camera-image-debayered.png``)
is this module's bilinear RGGB demosaic rotated 180 degrees (checked on the
2025-04-04 Ariake capture: correlation 0.98-1.00 per channel). The file has
no georeferencing; co-registration with the HSI image is a REFACTOR_PROGRESS.md
TODO.

Examples
--------
>>> image = read_camera_image(find_camera_image(capture_dir))
>>> image.sel(channel="G")
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import convolve

#: Colour filter layout of the camera's sensor, top-left 2x2 cell.
BAYER_PATTERN = "RGGB"

_TIME = re.compile(r"(\d{4})_(\d{2})_(\d{2})T(\d{2})_(\d{2})_(\d{2})")


def find_camera_image(capture_dir):
    """The RGB camera image of a capture directory, or None if it has none.

    Parameters
    ----------
    capture_dir : str or pathlib.Path

    Returns
    -------
    pathlib.Path or None
    """
    images = sorted(Path(capture_dir).glob("*-rgb/*.bmp"))
    return images[0] if images else None


def camera_image_time(path) -> datetime:
    """Acquisition time (UTC) from a camera image's file name."""
    match = _TIME.search(Path(path).name)
    if not match:
        raise ValueError(f"no time in camera image name {Path(path).name!r}")
    return datetime(*map(int, match.groups()))


def demosaic_bilinear(mosaic: np.ndarray, pattern: str = BAYER_PATTERN) -> np.ndarray:
    """Bilinear demosaic of a Bayer mosaic.

    Parameters
    ----------
    mosaic : numpy.ndarray
        2D raw values.
    pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        Colour filters of the top-left 2x2 cell, row by row.

    Returns
    -------
    numpy.ndarray
        ``(rows, cols, 3)`` float32 R, G, B.
    """
    if sorted(pattern) != ["B", "G", "G", "R"]:
        raise ValueError(f"not a Bayer pattern: {pattern!r}")
    mosaic = np.asarray(mosaic, dtype=np.float32)
    masks = {c: np.zeros(mosaic.shape, bool) for c in "RGB"}
    for (row, col), colour in zip(((0, 0), (0, 1), (1, 0), (1, 1)), pattern):
        masks[colour][row::2, col::2] = True
    # Weights of the neighbours: green on a quincunx grid, red and blue on a square one.
    green = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    red_blue = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = np.empty(mosaic.shape + (3,), np.float32)
    for i, colour in enumerate("RGB"):
        out[..., i] = convolve(np.where(masks[colour], mosaic, 0), green if colour == "G" else red_blue,
                               mode="mirror")
    return out


def read_camera_image(path, demosaic: bool = True, rotate: bool = True) -> xr.DataArray:
    """Read an RGB camera image.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.bmp`` (see :func:`find_camera_image`).
    demosaic : bool, default True
        Bilinear RGGB demosaic to R, G, B; False returns the raw mosaic.
    rotate : bool, default True
        Rotate 180 degrees, the orientation of the ground processing's
        demosaiced image.

    Returns
    -------
    xarray.DataArray
        ``(y, x, channel)`` float32 R, G, B in raw counts (0-255), or the
        ``(y, x)`` uint8 mosaic; attributes ``instrument`` ("RGB"),
        ``acquisition_time`` (ISO 8601, UTC), ``bayer_pattern``, ``source``.
    """
    from PIL import Image
    mosaic = np.asarray(Image.open(path))
    if mosaic.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel raw mosaic, got shape {mosaic.shape}")
    attrs = {"instrument": "RGB", "acquisition_time": camera_image_time(path).isoformat() + "Z",
             "bayer_pattern": BAYER_PATTERN, "source": Path(path).name,
             "orientation": "rotated 180 degrees" if rotate else "as stored"}
    if not demosaic:
        data = mosaic[::-1, ::-1] if rotate else mosaic
        return xr.DataArray(np.ascontiguousarray(data), dims=("y", "x"), attrs=attrs)
    rgb = demosaic_bilinear(mosaic)
    if rotate:
        rgb = rgb[::-1, ::-1]
    return xr.DataArray(np.ascontiguousarray(rgb), dims=("y", "x", "channel"), coords={"channel": ["R", "G", "B"]},
                        attrs=attrs)
