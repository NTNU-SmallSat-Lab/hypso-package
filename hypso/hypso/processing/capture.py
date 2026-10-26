"""Capture parameters derived from a product tree's metadata/capture_config -
the values io/dispatch.py's set_hypso_attributes/check_capture_type put on a
capture object, computed the same way."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace

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

    @property
    def spatial_dimensions(self) -> tuple:
        return (self.frame_count, self.image_height)

    def fwhm(self, wavelengths) -> np.ndarray:
        """Per-band FWHM from the sensor's nearest-wavelength lookup table
        (capture_types._get_fwhm)."""
        lookup_wl = self.sensor_profile.fwhm_lookup_wl
        lookup_fwhm = self.sensor_profile.fwhm_lookup_fwhm
        return np.array([lookup_fwhm[np.argmin(np.abs(w - lookup_wl))] for w in np.asarray(wavelengths)],
                        dtype=float)


def acquisition_time(global_attrs: dict) -> str:
    """io/dispatch.py's iso_time: the date_aquired (sic) or timestamp_acquired
    global attribute, as an ISO 8601 UTC string."""
    datestring = global_attrs.get("date_aquired", global_attrs.get("timestamp_acquired"))
    try:
        dt = datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        dt = datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S.%f%zZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(dt.timestamp(), tz=timezone.utc).isoformat()


def capture_params(tree: xr.DataTree) -> CaptureParams:
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

    # check_capture_type: the first threshold whose attribute matches.
    lookup = SimpleNamespace(**{**attrs, **values})
    capture_type = next((name for name, attr, expected in profile.capture_type_thresholds
                         if getattr(lookup, attr, None) == expected), "custom")
    return CaptureParams(sat_id=sat_id, sensor_profile=profile, capture_type=capture_type,
                         iso_time=acquisition_time(tree.attrs), capture_config=attrs, **values)
