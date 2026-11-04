"""Capture parameters derived from a product tree's metadata/capture_config -
the values io/dispatch.py's set_hypso_attributes/check_capture_type put on a
capture object, computed the same way."""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from hypso.sensors import SensorProfile, get_sensor_profile


@dataclass(frozen=True)
class CaptureParams:
    sat_id: str
    sensor_profile: SensorProfile
    capture_type: str
    x_start: int
    x_stop: int
    y_start: int
    y_stop: int
    bin_factor: int
    sample_div: int
    background_value: int
    exposure: float          # seconds
    frame_count: int
    image_height: int
    image_width: int
    iso_time: str            # acquisition start, UTC (ISO 8601)
    capture_config: dict = field(repr=False)
    #: FWHM table ``(line wavelengths, FWHM)`` of the product's calibration
    #: (a release's ``fwhm`` version, kept in ``metadata/corrections``); None:
    #: the sensor profile's built-in table.
    fwhm_table: tuple = field(default=None, repr=False)

    @property
    def spatial_dimensions(self) -> tuple:
        return (self.frame_count, self.image_height)

    def fwhm(self, wavelengths) -> np.ndarray:
        """Per-band FWHM: the FWHM table (the calibration's, else the sensor
        profile's) interpolated linearly between its lines, constant outside
        them (production 11756523, from aria 368d16d9; it used the nearest
        line before)."""
        if self.fwhm_table is not None:
            lines, values = self.fwhm_table
        else:
            lines, values = self.sensor_profile.fwhm_lookup_wl, self.sensor_profile.fwhm_lookup_fwhm
        return interpolate_fwhm(lines, values, wavelengths)


def interpolate_fwhm(lines, values, wavelengths) -> np.ndarray:
    """FWHM at ``wavelengths`` from a table of lines: linear interpolation,
    constant outside the table (``numpy.interp``)."""
    lines = np.asarray(lines, dtype=float)
    order = np.argsort(lines)
    return np.interp(np.asarray(wavelengths, dtype=float), lines[order], np.asarray(values, dtype=float)[order])


def fwhm_table(tree: xr.DataTree):
    """The calibration's FWHM table from ``metadata/corrections``, or None."""
    if "metadata" not in tree.children or "corrections" not in tree["metadata"].children:
        return None
    corrections = tree["metadata/corrections"].dataset
    if "fwhm_table_wavelength" not in corrections:
        return None
    return corrections["fwhm_table_wavelength"].values, corrections["fwhm_table_fwhm"].values


def acquisition_time(global_attrs: dict) -> str:
    """io/dispatch.py's iso_time: the date_aquired (sic) or timestamp_acquired
    global attribute, as an ISO 8601 UTC string."""
    datestring = global_attrs.get("date_aquired", global_attrs.get("timestamp_acquired"))
    try:
        dt = datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        dt = datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S.%f%zZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(dt.timestamp(), tz=timezone.utc).isoformat()


def classify_capture_type(profile: SensorProfile, values: dict) -> str:
    """The imaging mode ("nominal", "wide", "moon", ...): the first of the
    sensor's capture_type_thresholds whose attribute (frame_count,
    image_height, ...) has its value in `values`; "custom" otherwise."""
    return next((name for name, attr, expected in profile.capture_type_thresholds
                 if values.get(attr) == expected), "custom")


def capture_params(tree: xr.DataTree) -> CaptureParams:
    """Capture parameters (dimensions, binning, timing, sensor profile) from a
    product tree's ``metadata/capture_config`` and global attributes."""
    attrs = dict(tree["metadata/capture_config"].attrs)
    if "framerate" not in attrs and "fps" in attrs:   # older files call it fps
        attrs["framerate"] = attrs["fps"]
    sat_id = tree.attrs.get("sat_id") or tree.attrs.get("instrument")
    profile = get_sensor_profile(sat_id)

    bin_factor = int(attrs["bin_factor"])
    sample_div = int(attrs.get("sample_div", 1))
    values = dict(
        x_start=int(attrs["aoi_x"]), x_stop=int(attrs["aoi_x"] + attrs["column_count"]),
        y_start=int(attrs["aoi_y"]), y_stop=int(attrs["aoi_y"] + attrs["row_count"]),
        bin_factor=bin_factor, sample_div=sample_div,
        background_value=8 * bin_factor,
        exposure=attrs["exposure"] / 1000,
        frame_count=int(attrs["frame_count"]),
        image_height=int(attrs["row_count"] / sample_div),
        image_width=int(attrs["column_count"] / bin_factor),
    )

    capture_type = classify_capture_type(profile, {**attrs, **values})
    return CaptureParams(sat_id=sat_id, sensor_profile=profile, capture_type=capture_type,
                         iso_time=acquisition_time(tree.attrs), capture_config=attrs,
                         fwhm_table=fwhm_table(tree), **values)
