"""L1B -> L1C: georeferencing (pixel latitude/longitude) and viewing/solar
angles.

Same frame interpolation, direct georeferencing and angle computation as
hypso.georeferencing.geo (run_frame_interpolation, run_direct_georeferencing,
run_georeferencing, run_angles_geometry), taking and returning a product
tree. Two sources of pixel coordinates:

- indirect: latitude/longitude arrays you supply (e.g. the GCP-corrected
  processing-temp/latitudes_indirectgeoref.dat files), written as
  latitude/longitude;
- direct: from ADCS attitude/position telemetry alone, written as
  latitude_direct/longitude_direct.

Without indirect coordinates, the direct ones become latitude/longitude
too (the pipeline's fallback).
"""
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.geometry import compute_local_angles, direct_georeference, interpolate_at_frame_nc
from hypso.io.product import SPATIAL_DIMS
from hypso.io.provenance import add_step

from .capture import CaptureParams, capture_params

logger = logging.getLogger(__name__)

_ANGLES = ("sensor_zenith", "sensor_azimuth", "solar_zenith", "solar_azimuth", "relative_azimuth")


def _framepose(tree: xr.DataTree, params: CaptureParams):
    # run_frame_interpolation: per-frame pose from ADCS telemetry and line timestamps.
    adcs = tree["metadata/adcs"].to_dataset(inherit=False)
    timing = tree["metadata/timing"].to_dataset(inherit=False)
    timestamps = timing["timestamps_srv"] if "timestamps_srv" in timing else timing["timestamps"]
    return interpolate_at_frame_nc(adcs={name: adcs[name].values for name in adcs.data_vars},
                                   lines_timestamps=timestamps.values,
                                   framerate=params.capture_config["framerate"],
                                   exposure=params.capture_config["exposure"],
                                   verbose=False)


def _angles(framepose, latitudes, longitudes, params: CaptureParams) -> dict:
    # run_angles_geometry.
    samples = params.image_height
    indices = np.array([0, samples // 4 - 1, samples // 2 - 1, 3 * samples // 4 - 1, samples - 1], dtype="uint16")
    sun_azimuth, sun_zenith, sat_azimuth, sat_zenith = compute_local_angles(
        framepose_data=framepose, lats=latitudes, lons=longitudes, indices=indices, verbose=False)
    shape = params.spatial_dimensions
    solar_zenith = sun_zenith.reshape(shape)
    if np.nanmin(solar_zenith) >= 70.0:
        logger.warning("Solar zenith angle is >= 70 deg across the entire scene (min: %.1f deg) - standard "
                       "atmospheric correction methods (e.g. OC-SMART, default solz_limit=70) may find no "
                       "valid pixels for this capture.", np.nanmin(solar_zenith))
    solar_azimuth, sensor_azimuth = sun_azimuth.reshape(shape), sat_azimuth.reshape(shape)
    relative = np.abs(sensor_azimuth - solar_azimuth)
    relative = np.where(relative > 180, 360 - relative, relative)
    return {"sensor_zenith": sat_zenith.reshape(shape), "sensor_azimuth": sensor_azimuth,
            "solar_zenith": solar_zenith, "solar_azimuth": solar_azimuth, "relative_azimuth": relative}


def _read_coordinates(value, shape):
    """Array, or a path to a raw float32 file (the processing-temp .dat files)."""
    if value is None or isinstance(value, np.ndarray):
        return value
    return np.fromfile(Path(value), dtype=np.float32).reshape(shape)


def l1b_to_l1c(tree: xr.DataTree, latitudes=None, longitudes=None, direct: bool = True) -> xr.DataTree:
    """Georeference an L1B tree.

    latitudes/longitudes: indirect pixel coordinates, (lines, samples) arrays
        or paths to raw float32 files. Omit both to use direct
        georeferencing as the main geometry.
    direct: also compute direct georeferencing from ADCS telemetry
        (latitude_direct, longitude_direct, <angle>_direct). Skipped with a
        warning if the telemetry points off the Earth.
    """
    if tree.attrs.get("processing_level") != "L1B":
        raise ValueError(f"l1b_to_l1c needs an L1B tree, got {tree.attrs.get('processing_level')}")
    if (latitudes is None) != (longitudes is None):
        raise ValueError("give both latitudes and longitudes, or neither")
    params = capture_params(tree)
    shape = params.spatial_dimensions
    sources = {"latitudes": latitudes, "longitudes": longitudes}
    latitudes, longitudes = _read_coordinates(latitudes, shape), _read_coordinates(longitudes, shape)
    framepose = _framepose(tree, params)

    out = tree.to_dataset(inherit=False)
    if latitudes is not None:
        if latitudes.shape != shape or longitudes.shape != shape:
            raise ValueError(f"coordinates have shape {latitudes.shape}/{longitudes.shape}, capture is {shape}")
        out["latitude"] = (SPATIAL_DIMS, latitudes)
        out["longitude"] = (SPATIAL_DIMS, longitudes)
        for name, values in _angles(framepose, latitudes, longitudes, params).items():
            out[name] = (SPATIAL_DIMS, values)

    direct_done = False
    if direct or latitudes is None:
        lat, lon, _ = direct_georeference(framepose_data=framepose, image_height=params.image_height,
                                          aoi_offset=params.y_start, verbose=False)
        if isinstance(lat, int) and isinstance(lon, int):
            logger.warning("ADCS telemetry points parts or all of the image off the Earth's horizon; "
                           "no direct georeferencing for this capture.")
        else:
            lat, lon = lat.reshape(shape), lon.reshape(shape)
            angles = _angles(framepose, lat, lon, params)
            out["latitude_direct"] = (SPATIAL_DIMS, lat)
            out["longitude_direct"] = (SPATIAL_DIMS, lon)
            for name, values in angles.items():
                out[f"{name}_direct"] = (SPATIAL_DIMS, values)
            if latitudes is None:
                out["latitude"], out["longitude"] = out["latitude_direct"], out["longitude_direct"]
                for name in _ANGLES:
                    out[name] = out[f"{name}_direct"]
            direct_done = True
    if "latitude" not in out:
        raise ValueError("no coordinates: none given and direct georeferencing failed")
    out.attrs["processing_level"] = "L1C"

    result = tree.copy()
    result.dataset = out
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    parameters = {"georeferencing": "indirect" if latitudes is not None else "direct", "direct": direct_done}
    for name, value in sources.items():
        if value is not None and not isinstance(value, np.ndarray):
            parameters[f"{name}_file"] = Path(value).name
    add_step(result, "l1c", input_file=source or "L1B (in memory)", parameters=parameters)
    return result
