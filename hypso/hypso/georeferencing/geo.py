"""Georeferencing orchestration, extracted from HypsoCapture (self.geo composition -
part of the HypsoCapture breakup called for in the approved refactor plan, see
REFACTOR_PROGRESS.md). Bodies are moved verbatim from HypsoCapture.py, not rewritten -
same math, same behavior, just relocated - each function takes `satobj` explicitly
(matching the pattern already used by hypso.ac's free functions) rather than being
methods on a stored back-reference, since these read many HypsoCapture attributes
(image_height, spatial_dimensions, framepose, nc_*_vars, VERBOSE, ...) and taking
satobj as a parameter avoids either duplicating all of that state or coupling this
module tightly to HypsoCapture's internals via a stored reference.

HypsoCapture.run_georeferencing()/run_direct_georeferencing() are thin delegating
wrappers over this module's functions - kept as methods (not moved) since
run_direct_georeferencing() is called externally (hypso/ac/loading_acolite_output.py)
and run_georeferencing() by hypso-processing-pipeline. The private _run_* helpers
were only ever called from within HypsoCapture.py itself, so they moved here outright
with no wrapper needed.
"""
import logging

import numpy as np

from hypso.geometry import (
    interpolate_at_frame_nc,
    direct_georeference,
    compute_local_angles,
    compute_gsd,
    compute_bbox,
    compute_resolution,
)

from .geo_state import GeoAngles, TrackGeometry

logger = logging.getLogger(__name__)


def run_frame_interpolation(satobj) -> None:
    try:
        timing = satobj.metadata.timing.vars['timestamps_srv']
    except Exception:
        timing = satobj.metadata.timing.vars['timestamps']

    framepose_data = interpolate_at_frame_nc(adcs=satobj.metadata.adcs.vars,
                                          lines_timestamps=timing,
                                          framerate=satobj.metadata.capture_config.attrs['framerate'],
                                          exposure=satobj.metadata.capture_config.attrs['exposure'],
                                          verbose=satobj.VERBOSE
                                          )

    satobj.framepose = framepose_data

    return None


def run_track_geometry(satobj, latitudes: np.ndarray, longitudes: np.ndarray):
    logger.info("Running track geometry computations...")

    try:
        getattr(satobj, 'framepose')
    except Exception:
        run_frame_interpolation(satobj)

    bbox = compute_bbox(latitudes=latitudes, longitudes=longitudes)

    along_track_gsd, across_track_gsd = compute_gsd(frame_count=satobj.frame_count,
                                                              image_height=satobj.image_height,
                                                              latitudes=latitudes,
                                                              longitudes=longitudes,
                                                              verbose=satobj.VERBOSE)

    resolution = compute_resolution(along_track_gsd=along_track_gsd,
                                         across_track_gsd=across_track_gsd)

    if satobj.VERBOSE:
        logger.info("Track geometry computations done.")

    return bbox, resolution, along_track_gsd, across_track_gsd


def run_angles_geometry(satobj, latitudes: np.ndarray, longitudes: np.ndarray):
    logger.info("Running angles geometry computations...")

    try:
        getattr(satobj, 'framepose')
    except Exception:
        run_frame_interpolation(satobj)

    indices = np.array([0, satobj.samples // 4 - 1, satobj.samples // 2 - 1,
                        3 * satobj.samples // 4 - 1, satobj.samples - 1], dtype='uint16')

    sun_azimuth, sun_zenith, \
    sat_azimuth, sat_zenith = compute_local_angles(framepose_data=satobj.framepose,
                                                   lats=latitudes,
                                                   lons=longitudes,
                                                   indices=indices,
                                                   verbose=satobj.VERBOSE)

    solar_zenith_angles = sun_zenith.reshape(satobj.spatial_dimensions)
    solar_azimuth_angles = sun_azimuth.reshape(satobj.spatial_dimensions)
    sat_zenith_angles = sat_zenith.reshape(satobj.spatial_dimensions)
    sat_azimuth_angles = sat_azimuth.reshape(satobj.spatial_dimensions)

    relative_azimuth_angles = abs(sat_azimuth_angles - solar_azimuth_angles)

    relative_azimuth_angles = np.where(relative_azimuth_angles > 180,
                                       360 - relative_azimuth_angles,
                                       relative_azimuth_angles)

    if satobj.VERBOSE:
        logger.info("Angles geometry computations done.")

    return solar_zenith_angles, solar_azimuth_angles, sat_zenith_angles, sat_azimuth_angles, relative_azimuth_angles


def run_direct_georeferencing(satobj) -> None:
    if satobj.VERBOSE:
        logger.info("Running direct georeferencing...")

    try:
        getattr(satobj, 'framepose')
    except Exception:
        run_frame_interpolation(satobj)

    pixels_lat, pixels_lon, _ = direct_georeference(framepose_data=satobj.framepose,
                                                    image_height=satobj.image_height,
                                                    aoi_offset=satobj.y_start,
                                                    verbose=satobj.VERBOSE
                                                    )

    if type(pixels_lat) == int and type(pixels_lon) == int:
        if satobj.VERBOSE:
            logger.info("according to ADCS telemetry, parts or all of the image is pointing "
                       "off the earth's horizon. Cant georeference this image.")
        return None

    satobj.latitudes_direct = pixels_lat.reshape(satobj.spatial_dimensions)
    satobj.longitudes_direct = pixels_lon.reshape(satobj.spatial_dimensions)

    bbox, \
    resolution, \
    along_track_gsd, \
    across_track_gsd = run_track_geometry(satobj, latitudes=satobj.latitudes_direct,
                                          longitudes=satobj.longitudes_direct)

    satobj.track_direct = TrackGeometry(bbox=bbox, along_track_gsd=along_track_gsd,
                                        across_track_gsd=across_track_gsd)
    satobj.resolution_direct = resolution

    solar_zenith_angles_direct, \
    solar_azimuth_angles_direct, \
    sat_zenith_angles_direct, \
    sat_azimuth_angles_direct, \
    relative_azimuth_angles_direct = run_angles_geometry(satobj, latitudes=satobj.latitudes_direct,
                                                    longitudes=satobj.longitudes_direct)

    satobj.angles_direct = GeoAngles(sensor_zenith=sat_zenith_angles_direct,
                                     sensor_azimuth=sat_azimuth_angles_direct,
                                     solar_zenith=solar_zenith_angles_direct,
                                     solar_azimuth=solar_azimuth_angles_direct,
                                     relative_azimuth=relative_azimuth_angles_direct)

    return None


def run_georeferencing(satobj, latitudes: np.ndarray = None, longitudes: np.ndarray = None) -> None:
    if satobj.VERBOSE:
        logger.info("Running georeferencing...")

    if latitudes is not None and longitudes is not None:
        satobj.latitudes = latitudes
        satobj.longitudes = longitudes

    bbox, \
    resolution, \
    along_track_gsd, \
    across_track_gsd = run_track_geometry(satobj, latitudes=satobj.latitudes,
                                          longitudes=satobj.longitudes)

    satobj.track = TrackGeometry(bbox=bbox, along_track_gsd=along_track_gsd,
                                 across_track_gsd=across_track_gsd)
    satobj.resolution = resolution

    solar_zenith_angles, \
    solar_azimuth_angles, \
    sat_zenith_angles, \
    sat_azimuth_angles, \
    relative_azimuth_angles = run_angles_geometry(satobj, latitudes=satobj.latitudes,
                                                    longitudes=satobj.longitudes)

    satobj.angles = GeoAngles(sensor_zenith=sat_zenith_angles, sensor_azimuth=sat_azimuth_angles,
                              solar_zenith=solar_zenith_angles, solar_azimuth=solar_azimuth_angles,
                              relative_azimuth=relative_azimuth_angles)

    return None
