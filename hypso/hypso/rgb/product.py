"""RGB camera products: the camera image as a CF NetCDF file.

The camera is not a hypso.Product: that models a spectral cube with
wavelengths, calibration levels and SRFs, none of which a three-channel,
uncalibrated camera has. A camera product is an xarray Dataset with the same
conventions:

- ``R``, ``G``, ``B`` on ``(y, x)``: demosaiced counts in the camera's own
  frame (not resampled), stored as uint8 (the camera's 8 bits; the bilinear
  demosaic is rounded);
- ``latitude``, ``longitude`` per pixel and ``in_hsi_swath`` when registered
  (:func:`hypso.rgb.register_camera`); unregistered products have none;
- attributes: instrument ``RGB``, satellite, target, acquisition time,
  ``processing_level`` ``L1A`` (uncalibrated counts), the registration, CF/ACDD.

File names follow hypso.io.naming with the ``RGB`` instrument token:
``HYPSO2_RGB.<time>.<target>.L1A.nc``. Satpy reads them with the
``hypso_rgb_l1a`` reader (``true_color`` composite).
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.io.naming import product_name
from hypso.io.provenance import PRODUCT_FORMAT_VERSION, utc_now

CHANNELS = ("R", "G", "B")
_TARGET = re.compile(r"^(.+?)_\d{4}_\d{2}_\d{2}T")


def camera_product(camera: xr.DataArray, registration=None, hsi=None, sat_id: str = "HYPSO-2",
                   target: str = None) -> xr.Dataset:
    """A camera product from a camera image, georeferenced if registered.

    Parameters
    ----------
    camera : xarray.DataArray
        ``(y, x, channel)`` image from :func:`hypso.rgb.read_camera_image`.
    registration : hypso.rgb.CameraRegistration, optional
        From :func:`hypso.rgb.register_camera`; when ``ok``, latitude and
        longitude are added. A failed registration is recorded, without
        geolocation.
    hsi : hypso.Product, optional
        The capture's HSI product: target, satellite and ``in_hsi_swath``.
    sat_id : str, default "HYPSO-2"
        Used when no ``hsi`` is given (HYPSO-2 carries the camera).
    target : str, optional
        Capture target; default from ``hsi``, else from the image file name.

    Returns
    -------
    xarray.Dataset
    """
    from hypso.rgb.registration import georeference_camera
    if registration is not None and registration.ok:
        ds = georeference_camera(camera, registration, hsi)
    else:
        ds = xr.Dataset({c: (("y", "x"), np.asarray(camera.sel(channel=c).values, dtype=np.float32),
                             {"long_name": f"RGB camera {c} channel", "units": "1 (counts)"}) for c in CHANNELS},
                        attrs=dict(camera.attrs))
        if registration is not None:
            ds.attrs.update(registration.attrs())
    for c in CHANNELS:
        ds[c] = ds[c].round().clip(0, 255).astype(np.uint8)
        ds[c].attrs.update(long_name=f"RGB camera {c} channel (demosaiced counts)", units="1")
        if "latitude" in ds:
            ds[c].attrs["coordinates"] = "latitude longitude"
    if hsi is not None:
        sat_id = hsi.sat_id or sat_id
        target = target or hsi.target
    if target is None:
        match = _TARGET.match(str(camera.attrs.get("source", "")))
        target = match.group(1) if match else "unknown"
    ds.attrs.update(instrument="RGB", sat_id=sat_id, location_description=target, processing_level="L1A",
                    title=f"{sat_id} RGB camera image", Conventions="CF-1.10, ACDD-1.3",
                    product_format_version=PRODUCT_FORMAT_VERSION,
                    georeferenced=str("latitude" in ds).lower())
    return ds


def camera_filename(ds: xr.Dataset, label: str = None) -> str:
    """The product's file name, e.g. ``HYPSO2_RGB.20250404T022723.ariake.L1A.nc``
    (with the grid token for a :func:`hypso.rgb.regrid_camera` result)."""
    time = datetime.fromisoformat(str(ds.attrs["acquisition_time"]).rstrip("Z"))
    return product_name(ds.attrs["sat_id"], time, ds.attrs["location_description"], "L1A", instrument="RGB",
                        label=label, grid=ds.attrs.get("grid")).filename()


def write_camera(ds: xr.Dataset, path=None, out_dir=None, label: str = None) -> Path:
    """Write a camera product (default name from :func:`camera_filename` in
    ``out_dir``). Channels uint8, geolocation float32, zlib level 4."""
    path = Path(path) if path is not None else Path(out_dir or ".") / camera_filename(ds, label)
    out = ds.copy()
    out.attrs["date_created"] = utc_now()
    encoding = {name: {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": None} for name in out.data_vars}
    out.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def open_camera(path) -> xr.Dataset:
    """Read a camera product written by :func:`write_camera`."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        return ds.load()
