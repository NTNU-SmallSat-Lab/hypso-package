import numpy as np
import math as m
import pandas as pd
import datetime


import astropy.coordinates
import astropy.units
import astropy.time

import scipy.interpolate as si
from typing import Tuple, List

from .utils import mat_from_quat, \
                   rotate_axis_angle, \
                   ellipsoid_line_intersection, \
                   ecef_to_lat_lon_alt

from .time import get_julian_day_number, \
                  get_greenwich_mean_sidereal_time_seconds

# PARAMETERS ----------------------------------------------------------------------

# GRS80 (Geodetic Reference System 1980, https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980)
# R_eq = 6378137.0
# f = 1.0/298.257222100882711
# WGS84 (World Geodetic System 1984, https://en.wikipedia.org/wiki/World_Geodetic_System)
R_eq = 6378137.0
f = 1.0 / 298.257223563

e_2 = f * (2 - f)  # eccentricity squared
R_pl = 6356752.0

# ----------------------------------------------------------------------

hypso_height_sensor = 1216


def interpolate_at_frame_nc(adcs,
                            lines_timestamps: np.ndarray,
                            additional_time_offset: float=0.0, 
                            framerate: float=15,
                            exposure: float=25,
                            verbose=False) -> np.ndarray:

    pos_x = adcs["position_x"]
    pos_y = adcs["position_y"]
    pos_z = adcs["position_z"]
    pos = np.column_stack((pos_x, pos_y, pos_z))

    quat_s = adcs["quaternion_s"]
    quat_x = adcs["quaternion_x"]
    quat_y = adcs["quaternion_y"]
    quat_z = adcs["quaternion_z"]
    quat = np.column_stack((quat_s, quat_x, quat_y, quat_z))

    adcs_timestamps = adcs["timestamps"]

    # converting type "numpy.ma.MaskedArray" to "numpy.ndarray"
    pos = np.array(pos)
    quat = np.array(quat)
    adcs_timestamps = np.array(adcs_timestamps)
    lines_timestamps = np.array(lines_timestamps)

    return interpolate_at_frame(adcs_timestamps, pos, quat, lines_timestamps,
                                additional_time_offset, framerate, exposure, verbose)



def interpolate_at_frame(adcs_timestamps: np.ndarray, position: np.ndarray,
                         quaternion: np.ndarray, lines_timestamps: np.ndarray,
                         additional_time_offset: float=0.0, 
                         framerate: float=15,
                         exposure: float=25,
                         verbose=False) -> np.ndarray:
    """
    Function to interpolate at the frame based on the quaternion, position and timestamps

    :param adcs_timestamps: 
    :param position: 
    :param quaternion: 
    :param lines_timestamps: lines_Timestamps
    :param additional_time_offset: Tuning offset for timestamps
    :param framerate: framerate in fps
    :param exposure: Exposure time in milliseconds

    :return pd.DataFrame:
    """

    if adcs_timestamps.shape[0] != position.shape[0]:
        print('ERROR: ADCS timestamps size does not match ADCS positions size')
        return -1
    if adcs_timestamps.shape[0] != quaternion.shape[0]:
        print('ERROR: ADCS timestamps size does not match ADCS quaternions size')
        return -1

    frame_count = lines_timestamps.shape[0]

    if verbose:
        print('[INFO] ADCS samples:', adcs_timestamps.shape[0])

    # Workaround for making proper timestamps based on framerate and exposure time
    if framerate > 0.0:
        starttime = lines_timestamps[0]
        for i in range(lines_timestamps.shape[0]):
            lines_timestamps[i] = starttime - exposure / (2 * 1000.0) + i / framerate

    # Add offset
    lines_timestamps = lines_timestamps + additional_time_offset

    # ADCS data time boundary
    adcs_ts_start = adcs_timestamps[0]
    adcs_ts_end = adcs_timestamps[-1]

    # lines/data cube time boundary
    frame_ts_start = lines_timestamps[0]
    frame_ts_end = lines_timestamps[-1]

    if verbose:
        print(f'[INFO] ADCS time range: {adcs_ts_start:17.6f} to {adcs_ts_end:17.6f}')
        print(f'[INFO] Frame time range: {frame_ts_start:17.6f} to {frame_ts_end:17.6f}')

    # The ADCS time range needs to completely include the frames/lines time range
    if frame_ts_start < adcs_ts_start:
        print('ERROR: Frame timestamps begin earlier than ADCS data!')
        return -1
    if frame_ts_end > adcs_ts_end:
        print('ERROR: Frame timestamps end later than ADCS data!')
        return -1

    if verbose:
        # Printing how many samples there are in the 
        a = adcs_timestamps[:] > lines_timestamps[0]
        b = adcs_timestamps[:] < lines_timestamps[-1]
        print(f'[INFO] {np.sum(a & b)} sample(s) inside frame time range')
        print(f'[INFO] Interpolating {frame_count:d} frames')

    interpolation_method = 'linear'  # 'cubic' causes errors sometimes: "ValueError: Expect x to not have duplicates"

    posdata_eci_x_interp = si.griddata(adcs_timestamps, position[:,0], lines_timestamps, method=interpolation_method)
    posdata_eci_y_interp = si.griddata(adcs_timestamps, position[:,1], lines_timestamps, method=interpolation_method)
    posdata_eci_z_interp = si.griddata(adcs_timestamps, position[:,2], lines_timestamps, method=interpolation_method)

    quatdata_q0_interp = si.griddata(adcs_timestamps, quaternion[:,0], lines_timestamps, method=interpolation_method)
    quatdata_q1_interp = si.griddata(adcs_timestamps, quaternion[:,1], lines_timestamps, method=interpolation_method)
    quatdata_q2_interp = si.griddata(adcs_timestamps, quaternion[:,2], lines_timestamps, method=interpolation_method)
    quatdata_q3_interp = si.griddata(adcs_timestamps, quaternion[:,3], lines_timestamps, method=interpolation_method)

    flashindices = np.linspace(0, frame_count - 1, frame_count).astype(np.int32)

    framepose_data = np.column_stack((lines_timestamps, flashindices, posdata_eci_x_interp, posdata_eci_y_interp, posdata_eci_z_interp, quatdata_q0_interp, quatdata_q1_interp, quatdata_q2_interp, quatdata_q3_interp))

    return framepose_data 


def pixel_index_to_angle(index, aoi_offset, fov_full, pixels_full) -> float:
    """
    :param index:
    :param aoi_offset:
    :param fov_full: field of view of the sensor in degrees
    :param pixels_full: the count of pixels corresponding to full field of view

    :return: offset angle to pixel from the fov center in radians
    """

    full_index = index + aoi_offset

    # the angles
    #   -fov/4, 0, and fov/4
    # do *not* correspond to pixel nuber (starting at 1)
    #   171,    342,   and 513     (684/4, 684/2, and 3*684/4)
    # they correspond to
    #   171.75, 342.5, and 513.25

    # linear
    # angle = fov_full*(full_index/(pixels_full-1) - 0.5) * m.pi / 180.0

    # trigonometric
    pixel_position_decimal = (full_index / (pixels_full - 1) - 0.5)
    tan_fov = m.tan((fov_full/2.0) * (m.pi/180.0))
    angle = m.atan(2.0 * tan_fov * pixel_position_decimal)
    return angle


def direct_georeference(framepose_data: np.ndarray, image_height: int=684, aoi_offset: int=266,
                        verbose = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel coordinates (lat lon) using the framepose data

    :param framepose_data_path:
    :param image_height: pixels per line in a datacube

    :return: latitudes numpy arra, longitudes numpy array
    """

    # This time parameter is an offset to the time used in determining the 
    # to the rotation of the earth, effectively shifting the lotations east-west
    # The north south offset is in the "interpolate_at_frame()" function
    additional_time_offset = -650.0 * 0.0 + 0.0

    times = framepose_data[:, 0]
    pos_teme = framepose_data[:, 2:5]
    quat_teme = framepose_data[:, 5:9] # expresses rotation of body axes to TEME axes

    frame_count = times.shape[0]

    # Computing the coordinates for every single pixel in python takes a long time.
    # This method computes the latlons for only 5 pixels in every frame/line
    # and uses linear/cubic interpolation to determine the coords for the rest of the pixels.

    latlon_top = np.zeros([frame_count, 2])
    latlon_mid_top = np.zeros([frame_count, 2])
    latlon_mid = np.zeros([frame_count, 2])
    latlon_mid_bot = np.zeros([frame_count, 2])
    latlon_bot = np.zeros([frame_count, 2])

    campos_itrs = np.zeros([frame_count, 3])
    body_x_itrs = np.zeros([frame_count, 3])
    body_z_itrs = np.zeros([frame_count, 3])

    if verbose:
        print('[INFO] Computing pixel latitude and longitude coordinates ...')  # computing pixel lat,lon's

    # computing lat,lons using astropy
    pointing_off_earth_indicator = 0
    for i in range(frame_count):

        # converting frame/line unix timestamp to an astropy time type
        time_astropy = astropy.time.Time(times[i] + additional_time_offset, format='unix')

        # converting quaternion to rotation matrix
        mat = mat_from_quat(quat_teme[i, :])

        # Defining body axes, z is hsi pointing direction and x is usually the along track direction
        body_x_body = np.array([1.0, 0.0, 0.0])
        body_z_body = np.array([0.0, 0.0, 1.0])

        # rotating body axes to TEME
        body_x_teme = np.matmul(mat, body_x_body)
        body_z_teme = np.matmul(mat, body_z_body)

        # Converting axes and satellite position to an astropy TEME position type
        body_x_astropy_teme = astropy.coordinates.TEME(astropy.coordinates.CartesianRepresentation(body_x_teme * astropy.units.m), obstime=time_astropy)
        body_z_astropy_teme = astropy.coordinates.TEME(astropy.coordinates.CartesianRepresentation(body_z_teme * astropy.units.m), obstime=time_astropy)
        pos_astropy_teme = astropy.coordinates.TEME(astropy.coordinates.CartesianRepresentation(pos_teme[i, :] * astropy.units.m), obstime=time_astropy)

        # TEME to ITRS transformation 
        body_x_astropy_itrs = body_x_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))
        body_z_astropy_itrs = body_z_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))
        pos_astropy_itrs = pos_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))

        # Converting ITRS astropy to numpy arrays
        body_x_itrs[i, :] = np.array([body_x_astropy_itrs.earth_location.value[0], body_x_astropy_itrs.earth_location.value[1], body_x_astropy_itrs.earth_location.value[2]])
        body_z_itrs[i, :] = np.array([body_z_astropy_itrs.earth_location.value[0], body_z_astropy_itrs.earth_location.value[1], body_z_astropy_itrs.earth_location.value[2]])
        campos_itrs[i, :] = np.array([pos_astropy_itrs.earth_location.value[0], pos_astropy_itrs.earth_location.value[1], pos_astropy_itrs.earth_location.value[2]])

        # Comments about orientations and reference frames and stuff:
        # For dayside captures, the HYPSO-1 and 2 satellite is moving ca. north to south.
        # Looking at raw data, 
        # The lowest pixel index (0) ('bottom of frame') is west
        # corresponds to +y axis side of the slit
        # highest pixel index (683) ('top of frame') is east
        # corresponds to -y axis side of the slit

        # concerning direct georectification:
        # - get top pixel direction by rotating center view (ca. +z axis) around +x axis with positive angle
        # - get bottom pixel direction is rotation with negative angle

        center_fov_offset = 0.0 # shall be determined from geometric calibration at some point
        fov = 8.135 # degrees, theoretical 8.008 deg
        fov_half = fov / 2.0 # in degrees
        fov_half_half = fov / 4.0 # in degrees

        # Top == highest pixel index ; Bot == lowest pixel index
        # Top == -y axis side of slit ; Bot == +y axis side of slit
        # Getting angle offsets to the selection of pixel indices
        top_pixel_angle     = pixel_index_to_angle(  image_height     - 1, aoi_offset, fov, hypso_height_sensor)
        mid_top_pixel_angle = pixel_index_to_angle(3*image_height / 4 - 1, aoi_offset, fov, hypso_height_sensor)
        mid_pixel_angle     = pixel_index_to_angle(  image_height / 2 - 1, aoi_offset, fov, hypso_height_sensor)
        mid_bot_pixel_angle = pixel_index_to_angle(  image_height / 4 - 1, aoi_offset, fov, hypso_height_sensor)
        bot_pixel_angle     = pixel_index_to_angle(                     0, aoi_offset, fov, hypso_height_sensor)

        # Computing direction vectors pointing from the satellite body axes in the direction of the pixel
        view_dir_top     = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], top_pixel_angle)
        view_dir_mid_top = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_top_pixel_angle)
        view_dir_mid     = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_pixel_angle)
        view_dir_mid_bot = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_bot_pixel_angle)
        view_dir_bot     = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], bot_pixel_angle)

        # Given direction and position, draw a ray and find the closest intersection of
        # the ray with the earth ellipsoid.
        # Obtain the the intersection position in ITRS frame
        pos_itrs_viewpoint_top, t1     = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_top)
        pos_itrs_viewpoint_mid_top, t2 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_top)
        pos_itrs_viewpoint_mid, t3     = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid)
        pos_itrs_viewpoint_mid_bot, t4 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_bot)
        pos_itrs_viewpoint_bot, t5     = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_bot)

        # Convert the intersection position an astropy type and tell astropy that it is a position
        # given in ITRS frame at the given frame/line timestamp
        pos_itrs_viewpoint_top_astropy_certesian = astropy.coordinates.CartesianRepresentation(pos_itrs_viewpoint_top * astropy.units.m)
        pos_itrs_viewpoint_mid_top_astropy_certesian = astropy.coordinates.CartesianRepresentation(pos_itrs_viewpoint_mid_top * astropy.units.m)
        pos_itrs_viewpoint_mid_astropy_certesian = astropy.coordinates.CartesianRepresentation(pos_itrs_viewpoint_mid * astropy.units.m)
        pos_itrs_viewpoint_mid_bot_astropy_certesian = astropy.coordinates.CartesianRepresentation(pos_itrs_viewpoint_mid_bot * astropy.units.m)
        pos_itrs_viewpoint_bot_astropy_certesian = astropy.coordinates.CartesianRepresentation(pos_itrs_viewpoint_bot * astropy.units.m)
        pos_viewpoint_top_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_top_astropy_certesian, obstime=time_astropy)
        pos_viewpoint_mid_top_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_top_astropy_certesian, obstime=time_astropy)
        pos_viewpoint_mid_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_astropy_certesian, obstime=time_astropy)
        pos_viewpoint_mid_bot_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_bot_astropy_certesian, obstime=time_astropy)
        pos_viewpoint_bot_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_bot_astropy_certesian, obstime=time_astropy)

        # Read lat,lon from the astropy ITRS position variable
        latlon_top[i, :]     = np.array([pos_viewpoint_top_astropy_itrs.earth_location.geodetic.lat.value, pos_viewpoint_top_astropy_itrs.earth_location.geodetic.lon.value])
        latlon_mid_top[i, :] = np.array([pos_viewpoint_mid_top_astropy_itrs.earth_location.geodetic.lat.value, pos_viewpoint_mid_top_astropy_itrs.earth_location.geodetic.lon.value])
        latlon_mid[i, :]     = np.array([pos_viewpoint_mid_astropy_itrs.earth_location.geodetic.lat.value, pos_viewpoint_mid_astropy_itrs.earth_location.geodetic.lon.value])
        latlon_mid_bot[i, :] = np.array([pos_viewpoint_mid_bot_astropy_itrs.earth_location.geodetic.lat.value, pos_viewpoint_mid_bot_astropy_itrs.earth_location.geodetic.lon.value])
        latlon_bot[i, :]     = np.array([pos_viewpoint_bot_astropy_itrs.earth_location.geodetic.lat.value, pos_viewpoint_bot_astropy_itrs.earth_location.geodetic.lon.value])

        # Check if hypso was pointing off the earth
        five = int(np.sign(t1)) + int(np.sign(t2)) + int(np.sign(t3)) + int(np.sign(t4)) + int(np.sign(t5))
        # has to equal 5. if not, this frame/line was pointing off the earth
        if five == 5:
            pointing_off_earth_indicator += 1

    if pointing_off_earth_indicator != frame_count:
        print('[ERROR] At least one pixel was pointing beyond the earth\'s horizon!')
        return -1, -1, -1

    # COMPUTING SATELLITE TRACK TODO
    '''
    # Setting altitude to zero to compute satellite ground track from lat lon
    satpos_lat_lon_alt = ecef_to_lat_lon_alt(campos_itrs)
    satpos_lat_lon = satpos_lat_lon_alt.copy()
    satpos_lat_lon[:, 2] = np.zeros([satpos_lat_lon.shape[0], 1])[:, 0]
    # could also use "pos_astropy_itrs.earth_location.geodetic.lat.value"
    # and "pos_astropy_itrs.earth_location.geodetic.lon.value"
    # in the above for loop to get satellite ground track
    '''

    if verbose:
        print('[INFO] Interpolating pixel coordinate gaps...')

    pixels_lat = np.zeros([frame_count, image_height])
    pixels_lon = np.zeros([frame_count, image_height])

    subsample_pixels_indices = np.array([image_height, 3 * image_height / 4, image_height / 2, image_height / 4, 1])
    all_pixels_indices = np.linspace(1, image_height, image_height)

    interpolation_method = 'cubic'

    for i in range(frame_count):
        lats = np.array([latlon_top[i, 0], latlon_mid_top[i, 0], latlon_mid[i, 0], latlon_mid_bot[i, 0], latlon_bot[i, 0]])
        lons = np.array([latlon_top[i, 1], latlon_mid_top[i, 1], latlon_mid[i, 1], latlon_mid_bot[i, 1], latlon_bot[i, 1]])
        pixels_lat[i, :] = si.griddata(subsample_pixels_indices, lats, all_pixels_indices, method=interpolation_method)
        pixels_lon[i, :] = si.griddata(subsample_pixels_indices, lons, all_pixels_indices, method=interpolation_method)

    if verbose:
        print('[INFO] Direct georeferencing done')

    return pixels_lat, pixels_lon, [campos_itrs, body_x_itrs, body_z_itrs]


# https://pyorbital.readthedocs.io/en/latest/#computing-astronomical-parameters
# -> from pyorbital import astronomy
# Function verified using: https://gml.noaa.gov/grad/antuv/SolarCalc.jsp
def compute_local_angles(framepose_data: np.ndarray,
                         lats: np.ndarray, lons: np.ndarray,
                         indices: np.ndarray,
                         verbose=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Local Angles of the Capture (Second Variation)

    :param framepose_data:
    :param lats:
    :param lons:
    :param indices: neds to be an integer type

    :return: Returns array of local angles, and a list of numpy arrays of the satellite and sun zenith and azimuth pos
    """
    if verbose:
        print('[INFO] Computing local angles (sun and satellite azimuth and zenith angles) ...')

    times = framepose_data[:,0]
    pos_teme = framepose_data[:,2:5]

    if times.shape[0] != lats.shape[0]:
        print(f'[ERROR] Size of time array not the same as size of latlons!: {times.shape[0]} != {lats.shape[0]}')
        return None

    if lats.shape != lons.shape:
        print(f'[ERROR] Size of latitudes and longitudes array do not match!: {lats.shape} != {lons.shape}')
        return None

    frame_count = times.shape[0]

    sun_azi_samples = np.zeros([times.shape[0], len(indices)])
    sun_zen_samples = np.zeros([times.shape[0], len(indices)])
    sat_azi_samples = np.zeros([times.shape[0], len(indices)])
    sat_zen_samples = np.zeros([times.shape[0], len(indices)])

    times_astropy = astropy.time.Time(times, format='unix')
    poses_astropy_teme = astropy.coordinates.TEME(astropy.coordinates.CartesianRepresentation(pos_teme.transpose(1, 0) * astropy.units.m), obstime=times_astropy)

    if verbose:
        print('[INFO] Using astropy on a subsampling of pixels ... (TODO skyfield may be faster)')

    for i in range(frame_count):
        sun_pos = astropy.coordinates.get_sun(times_astropy[i])

        for j, k in enumerate(indices):
            pixel_location_astropy = astropy.coordinates.EarthLocation(lat=lats[i, k] * astropy.units.deg,
                                                                       lon=lons[i, k] * astropy.units.deg,
                                                                       height=0 * astropy.units.m)
            sun_altaz = sun_pos.transform_to(astropy.coordinates.AltAz(obstime=times_astropy[i], location=pixel_location_astropy))
            sat_altaz = poses_astropy_teme[i].transform_to(astropy.coordinates.AltAz(obstime=times_astropy[i], location=pixel_location_astropy))

            sun_azi_samples[i, j] = sun_altaz.az.value
            sun_zen_samples[i, j] = 90.0 - sun_altaz.alt.value
            sat_azi_samples[i, j] = sat_altaz.az.value
            sat_zen_samples[i, j] = 90.0 - sat_altaz.alt.value

    image_height = lats.shape[1]
    all_pixels_indices = np.linspace(1,image_height,image_height)

    sun_azi_full = np.zeros([frame_count, image_height])
    sun_zen_full = np.zeros([frame_count, image_height])
    sat_azi_full = np.zeros([frame_count, image_height])
    sat_zen_full = np.zeros([frame_count, image_height])

    if verbose:
        print('[INFO] Interpolating the rest of the pixels ...')

    interpolation_method = 'cubic'

    # Why do the indices for interpolation be 1 more than for computing the angles again?
    subsample_pixels_indices = indices + 1
    for i in range(frame_count):
        sun_azi_full[i,:] = si.griddata(subsample_pixels_indices, sun_azi_samples[i,:], all_pixels_indices, method=interpolation_method)
        sun_zen_full[i,:] = si.griddata(subsample_pixels_indices, sun_zen_samples[i,:], all_pixels_indices, method=interpolation_method)
        sat_azi_full[i,:] = si.griddata(subsample_pixels_indices, sat_azi_samples[i,:], all_pixels_indices, method=interpolation_method)
        sat_zen_full[i,:] = si.griddata(subsample_pixels_indices, sat_zen_samples[i,:], all_pixels_indices, method=interpolation_method)

    if verbose:
        print('[INFO] Computing local angles done')

    return sun_azi_full, sun_zen_full, sat_azi_full, sat_zen_full





def compute_elevation_angle(image_pos, sat_pos) -> float:
    """
    Computes the elevation angle given a location and a satellite position in ITRS (type of ECEF)

    :param image_pos:
    :param sat_pos:

    :return: Elevation angle in radians
    """
    viewpoint_mid_latlon_geocentric = np.array(
        [m.atan2(R_eq * image_pos[2], R_pl * m.sqrt(image_pos[0] ** 2 + image_pos[1] ** 2)),
         m.atan2(image_pos[1], image_pos[0])])
    lat = viewpoint_mid_latlon_geocentric[0]
    lon = viewpoint_mid_latlon_geocentric[1]
    norm_f = m.sqrt((R_eq * m.sin(lat)) ** 2 + (R_pl * m.cos(lon)) ** 2)
    # tangent along latitude
    north = np.array([-R_eq * m.sin(lat) * m.cos(lon), -R_eq * m.sin(lat) * m.sin(lon), R_pl * m.cos(lat)]) / norm_f
    # tangent along longitude
    east = np.array([-m.sin(lon), m.cos(lon), 0.0])
    vp_mid_normal = np.cross(east, north)

    pos_itrf_vp_to_sat = sat_pos - image_pos
    pos_ENU_vp_to_sat = np.array([np.dot(east, pos_itrf_vp_to_sat),
                                  np.dot(north, pos_itrf_vp_to_sat),
                                  np.dot(vp_mid_normal, pos_itrf_vp_to_sat)])
    elevation_angle = m.atan2(pos_ENU_vp_to_sat[2], m.sqrt(pos_ENU_vp_to_sat[0] ** 2 + pos_ENU_vp_to_sat[1] ** 2))

    return elevation_angle

def compute_off_nadir_angle(image_pos, sat_pos) -> float:
    """
    Compute the off-nadir angle between the image and the satellite position

    :param image_pos:
    :param sat_pos:

    :return: The off-nadir angle float value in radians
    """

    pos_itrf_vp_to_sat = sat_pos - image_pos
    off_nadir_angle = m.acos(np.dot(pos_itrf_vp_to_sat / m.sqrt((pos_itrf_vp_to_sat**2).sum()), sat_pos / m.sqrt((sat_pos**2).sum())))

    return off_nadir_angle

def get_wkt(top_latlon, bot_latlon) -> str:
    """
    Gets WKT

    :param top_latlon:
    :param bot_latlon:

    :return: a long string in Well known text representation of shapes format
    """

    string = 'LINESTRING('
    # print(f'LINESTRING(', end='')
    for i in range(top_latlon.shape[0]):
        string = string + f'{top_latlon[i, 1] * 180.0 / m.pi} {top_latlon[i, 0] * 180.0 / m.pi}, '
        # print(f'{top_latlon[i,1]*180.0/m.pi} {top_latlon[i,0]*180.0/m.pi}, ', end='')
    for i in range(bot_latlon.shape[0]):
        j = bot_latlon.shape[0] - i - 1
        string = string + f'{bot_latlon[j, 1] * 180.0 / m.pi} {bot_latlon[j, 0] * 180.0 / m.pi}, '
        # print(f'{bot_latlon[j,1]*180.0/m.pi} {bot_latlon[j,0]*180.0/m.pi}, ', end='')
    string = string + f'{top_latlon[0, 1] * 180.0 / m.pi} {top_latlon[0, 0] * 180.0 / m.pi})'
    # print(f'{top_latlon[0,1]*180.0/m.pi} {top_latlon[0,0]*180.0/m.pi})', end='')
    return string

def get_wkt_points(latlon_top, latlon_mid_top, latlon_mid, latlon_mid_bot, latlon_bot) -> str:
    """
    Get WKT Points in a string readable format

    :param latlon_top:
    :param latlon_mid_top:
    :param latlon_mid:
    :param latlon_mid_bot:
    :param latlon_bot:

    :return: String of the WKT points
    """
    string = ''
    for i in range(latlon_top.shape[0]):
        # string = '"POINT('
        string = string + f'"POINT({latlon_top[i, 1] * 180.0 / m.pi} {latlon_top[i, 0] * 180.0 / m.pi})",\n'
        string = string + f'"POINT({latlon_mid_top[i, 1] * 180.0 / m.pi} {latlon_mid_top[i, 0] * 180.0 / m.pi})",\n'
        string = string + f'"POINT({latlon_mid[i, 1] * 180.0 / m.pi} {latlon_mid[i, 0] * 180.0 / m.pi})",\n'
        string = string + f'"POINT({latlon_mid_bot[i, 1] * 180.0 / m.pi} {latlon_mid_bot[i, 0] * 180.0 / m.pi})",\n'
        string = string + f'"POINT({latlon_bot[i, 1] * 180.0 / m.pi} {latlon_bot[i, 0] * 180.0 / m.pi})",\n'
        # print(f'{bot_latlon[j,1]*180.0/m.pi} {bot_latlon[j,0]*180.0/m.pi}, ', end='')
    # string = string + f'{top_latlon[0,1]*180.0/m.pi} {top_latlon[0,0]*180.0/m.pi})'
    # print(f'{top_latlon[0,1]*180.0/m.pi} {top_latlon[0,0]*180.0/m.pi})', end='')
    return string

