import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import datetime
import os
from pathlib import Path

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




def interpolate_at_frame(adcs_pos_df: pd.DataFrame,
                         adcs_quat_df: pd.DataFrame,
                         timestamps_srv: np.ndarray,
                         frame_count: int,
                         additional_time_offset: float=0.0, 
                         fps: float=-1.0,
                         exposure: float=-1.0,
                         verbose=False) -> pd.DataFrame:
    """
    Function to interpolate at the frame based on the quaternion, position and timestamps

    :param adcs_pos_df: Formatted position information from ADCS
    :param adcs_quat_df: Formatted quaternion information from ADCS
    :param timestamps_srv: Timestamps_srv variable from timing
    :param frame_count:  Line/frames/rows in image. Frame_count attribute from capture_config
    :param additional_time_offset: Offset for timestamps
    :param fps: FPS or framerate attribute from capture_config
    :param exposure: Exposure attribute from capture_config

    :return pd.DataFrame:
    """

    # 1. Reading .csv file with ECI position info
    posdata = adcs_pos_df
    pos_time_col_index = 0
    eci_x_col_index = 1
    eci_y_col_index = 2
    eci_z_col_index = 3

    if verbose:
        print('ECI position samples:', posdata.shape[0])

    # 2. Reading .csv file with GPS info
    quatdata = adcs_quat_df
    quat_time_col_index = 0
    q0_col_index = 1
    q1_col_index = 2
    q2_col_index = 3
    q3_col_index = 4

    if verbose:
        print('Quaternion samples:', quatdata.shape[0])

    # 3. Reading frame timestamps

    if isinstance(timestamps_srv, np.ma.MaskedArray):
        flashtimes = timestamps_srv.data
    else:
        flashtimes = timestamps_srv

    # Add offset
    flashtimes = flashtimes + additional_time_offset

    ## workaround for making the timestamps smoother. Does not actually smooth out the data.
    if fps > 0.0:
        starttime = flashtimes[0]
        for i in range(flashtimes.shape[0]):
            flashtimes[i] = starttime - exposure / (2 * 1000.0) + i / fps

    # Inclusion of frame times in ADCS telemetry time series check

    # ADCS data time boundary
    adcs_ts_start = posdata.iloc[0, 0]
    adcs_ts_end = posdata.iloc[-1, 0]

    frame_ts_start = flashtimes[0]
    frame_ts_end = flashtimes[-1]

    if verbose:
        print(f'ADCS time range: {adcs_ts_start:17.6f} to {adcs_ts_end:17.6f}')
        print(f'Frame time range: {frame_ts_start:17.6f} to {frame_ts_end:17.6f}')

    if frame_ts_start < adcs_ts_start:
        print('ERROR: Frame timestamps begin earlier than ADCS data!')
        exit(-1)

    if frame_ts_end > adcs_ts_end:
        print('ERROR: Frame timestamps end later than ADCS data!')
        exit(-1)

    a = quatdata.values[:, 0] > flashtimes[0]
    b = quatdata.values[:, 0] < flashtimes[-1]

    if verbose:
        print(f'{np.sum(a & b)} sample(s) inside frame time range')
        print(f'Interpolating {frame_count:d} frames')

    posdata_time_unix = posdata.values[:, pos_time_col_index].astype(np.float64)

    posdata_eci_x = posdata.values[:, eci_x_col_index].astype(np.float64)
    posdata_eci_y = posdata.values[:, eci_y_col_index].astype(np.float64)
    posdata_eci_z = posdata.values[:, eci_z_col_index].astype(np.float64)

    # USE THIS IF TIME IS GIVEN AS A DATETIME STRING
    # quatdata_time_unix = np.zeros(quatdata.shape[0]).astype('float64')
    # for i, dto in enumerate(posdata.values[:,quat_time_col_index]):
    #    dt = pd.to_datetime(dto)
    #    dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
    #    quatdata_time_unix[i] = dt_utc.timestamp()

    # USE THIS IF TIME IS GIVEN AS A UNIX TIMESTAMP
    quatdata_time_unix = quatdata.values[:, quat_time_col_index].astype(np.float64)

    quatdata_q0 = quatdata.values[:, q0_col_index].astype(np.float64)
    quatdata_q1 = quatdata.values[:, q1_col_index].astype(np.float64)
    quatdata_q2 = quatdata.values[:, q2_col_index].astype(np.float64)
    quatdata_q3 = quatdata.values[:, q3_col_index].astype(np.float64)

    ##############################################################################

    interpolation_method = 'linear'  # 'cubic' causes errors sometimes: "ValueError: Expect x to not have duplicates"

    posdata_eci_x_interp = si.griddata(posdata_time_unix, posdata_eci_x, flashtimes, method=interpolation_method)
    posdata_eci_y_interp = si.griddata(posdata_time_unix, posdata_eci_y, flashtimes, method=interpolation_method)
    posdata_eci_z_interp = si.griddata(posdata_time_unix, posdata_eci_z, flashtimes, method=interpolation_method)

    quatdata_q0_interp = si.griddata(quatdata_time_unix, quatdata_q0, flashtimes, method=interpolation_method)
    quatdata_q1_interp = si.griddata(quatdata_time_unix, quatdata_q1, flashtimes, method=interpolation_method)
    quatdata_q2_interp = si.griddata(quatdata_time_unix, quatdata_q2, flashtimes, method=interpolation_method)
    quatdata_q3_interp = si.griddata(quatdata_time_unix, quatdata_q3, flashtimes, method=interpolation_method)

    # print(quatdata_q0.shape, quatdata_q0_interp.shape)

    flashindices = np.linspace(0, frame_count - 1, frame_count).astype(np.int32)

    data = {
        'timestamp': flashtimes,
        'frame index': flashindices,
        'eci x [m]': posdata_eci_x_interp,
        'eci y [m]': posdata_eci_y_interp,
        'eci z [m]': posdata_eci_z_interp,
        'q_eta': quatdata_q0_interp,
        'q_e1': quatdata_q1_interp,
        'q_e2': quatdata_q2_interp,
        'q_e3': quatdata_q3_interp
    }

    #This used to written to frametime-pose.csv
    frametime_pose = pd.DataFrame(data)

    framepose_data = frametime_pose 

    return framepose_data 

def numpy_norm(np_array) -> float:
    """
    Gets the norm of an array

    :param np_array: Numpy array to get the norm

    :return: Float of the norm of the array
    """
    summ = 0.0
    for i in range(np_array.shape[0]):
        summ += np_array[i] ** 2
    return m.sqrt(summ)

def compute_elevation_angle(image_pos, sat_pos) -> float:
    """
    Computes the elevation angle

    :param image_pos:
    :param sat_pos:

    :return: Float number of the elevation angle
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

    # print(viewpoint_mid_latlon_geocentric*180.0/m.pi)
    # print(pos_itrf_vp_to_sat)
    # print("north:", north)
    # print("east :", east)
    # print("up   :", vp_mid_normal)
    # print("pos ENU", pos_ENU_vp_to_sat)
    # print("pos ENU components ITRF:", east*pos_ENU_vp_to_sat[0])
    # print("pos ENU components ITRF:", north*pos_ENU_vp_to_sat[1])
    # print("pos ENU components ITRF:", vp_mid_normal*pos_ENU_vp_to_sat[2])
    # print("ELEVTION:", elevation_angle*180.0/m.pi)
    # off_nadir_angle = m.acos(np.dot(pos_itrf_vp_to_sat/numpy_norm(pos_itrf_vp_to_sat), sat_pos/numpy_norm(sat_pos)))
    # print("Off nadir angle:", off_nadir_angle*180.0/m.pi)

    return elevation_angle

def compute_off_nadir_angle(image_pos, sat_pos) -> float:
    """
    Compute the off-nadir angle between the image and the satellite position

    :param image_pos:
    :param sat_pos:

    :return: The off-nadir angle float value
    """
    pos_itrf_vp_to_sat = sat_pos - image_pos
    off_nadir_angle = m.acos(np.dot(pos_itrf_vp_to_sat / numpy_norm(pos_itrf_vp_to_sat), sat_pos / numpy_norm(sat_pos)))

    # print("Off nadir angle:", off_nadir_angle*180.0/m.pi)

    return off_nadir_angle

def pixel_index_to_angle(index, aoi_offset, fov_full, pixels_full) -> float:
    """
    Get the pixel to angle

    :param index:
    :param aoi_offset:
    :param fov_full:
    :param pixels_full:

    :return:
    """
    # return angle in radians units
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
    angle = m.atan(2.0 * m.tan((fov_full / 2.0) * (m.pi / 180.0)) * (full_index / (pixels_full - 1) - 0.5))
    return angle


# https://pyorbital.readthedocs.io/en/latest/#computing-astronomical-parameters
# -> from pyorbital import astronomy
# Function verifies using: https://gml.noaa.gov/grad/antuv/SolarCalc.jsp
def compute_local_angles(times, pos_teme, latlons, verbose=False) -> np.ndarray:
    """
    Compute the local angles on the capture

    :param times:
    :param pos_teme:
    :param latlons:

    :return: Numpy array with the local angles
    """
    if times.shape[0] != latlons.shape[0]:
        print(f'Size of time array not the same as size of latlons!: {times.shape[0]} != {latlons.shape[0]}')
        return None

    # index, time, zenith angle, lat, lon
    local_angles_and_more = np.zeros([times.shape[0], 8])
    for i in range(times.shape[0]):
        dt = datetime.datetime.fromtimestamp(times[i], tz=datetime.timezone.utc)
        local_angles_and_more[i, 0] = i
        local_angles_and_more[i, 1] = times[i]
        local_angles_and_more[i, 2] = latlons[i, 0] * 180 / m.pi
        local_angles_and_more[i, 3] = latlons[i, 1] * 180 / m.pi

        location_astropy = astropy.coordinates.EarthLocation(lat=latlons[i, 0] * 180 / m.pi * astropy.units.deg,
                                                             lon=latlons[i, 1] * 180 / m.pi * astropy.units.deg,
                                                             height=0 * astropy.units.m)
        # utcoffset = -4*u.hour  # Eastern Daylight Time
        # time = Time('2012-7-12 23:00:00') - utcoffset
        time_astropy = astropy.time.Time(times[i], format='unix')

        pos_astropy_teme = astropy.coordinates.TEME(
            astropy.coordinates.CartesianRepresentation(pos_teme[i, :] * astropy.units.m), obstime=time_astropy)

        sun_pos = astropy.coordinates.get_sun(time_astropy)
        sat_altaz = pos_astropy_teme.transform_to(
            astropy.coordinates.AltAz(obstime=time_astropy, location=location_astropy))
        sun_altaz = sun_pos.transform_to(astropy.coordinates.AltAz(obstime=time_astropy, location=location_astropy))

        # local_angles_and_more[i,4] = astronomy.sun_zenith_angle(dt, latlons[i,1]*180/m.pi, latlons[i,0]*180/m.pi)
        local_angles_and_more[i, 4] = 90.0 - sun_altaz.alt.value
        local_angles_and_more[i, 5] = sun_altaz.az.value  # solar azimuth
        local_angles_and_more[i, 6] = 90.0 - sat_altaz.alt.value  # satellite zenith angle (90-elevation)
        local_angles_and_more[i, 7] = sat_altaz.az.value  # satellite azimuth

    return local_angles_and_more


# https://pyorbital.readthedocs.io/en/latest/#computing-astronomical-parameters
# -> from pyorbital import astronomy
# Function verified using: https://gml.noaa.gov/grad/antuv/SolarCalc.jsp
def compute_local_angles_2(times, pos_teme, lats, lons, indices, verbose=False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the Local Angles of the Capture (Second Variation)

    :param times:
    :param pos_teme:
    :param lats:
    :param lons:
    :param indices:

    :return: Returns array of local angles, and a list of numpy arrays of the satellite and sun zenith and azimuth pos
    """
    if times.shape[0] != lats.shape[0]:
        print(f'Size of time array not the same as size of latlons!: {times.shape[0]} != {lats.shape[0]}')
        return None

    if lats.shape != lons.shape:
        print(f'Size of latitudes and longitudes array do not match!: {lats.shape[0]} != {lons.shape[0]}')
        return None

    hypso_height = lats.shape[1]
    mid_index = hypso_height // 2 - 1

    # index, time, zenith angle, lat, lon
    local_angles_and_more = np.zeros([times.shape[0], 8])

    sun_azi_full = np.zeros([times.shape[0], len(indices)])
    sun_zen_full = np.zeros([times.shape[0], len(indices)])
    sat_azi_full = np.zeros([times.shape[0], len(indices)])
    sat_zen_full = np.zeros([times.shape[0], len(indices)])
    times_astropy = astropy.time.Time(times, format='unix')
    poses_astropy_teme = astropy.coordinates.TEME(
        astropy.coordinates.CartesianRepresentation(pos_teme.transpose(1, 0) * astropy.units.m), obstime=times_astropy)
    for i in range(times.shape[0]):
        sun_pos = astropy.coordinates.get_sun(times_astropy[i])
        local_angles_and_more[i, 0] = i
        local_angles_and_more[i, 1] = times[i]
        local_angles_and_more[i, 2] = lats[i, mid_index]
        local_angles_and_more[i, 3] = lons[i, mid_index]
        for j, k in enumerate(indices):

            pixel_location_astropy = astropy.coordinates.EarthLocation(lat=lats[i, k] * astropy.units.deg,
                                                                       lon=lons[i, k] * astropy.units.deg,
                                                                       height=0 * astropy.units.m)

            sun_altaz = sun_pos.transform_to(
                astropy.coordinates.AltAz(obstime=times_astropy[i], location=pixel_location_astropy))
            sat_altaz = poses_astropy_teme[i].transform_to(
                astropy.coordinates.AltAz(obstime=times_astropy[i], location=pixel_location_astropy))

            sun_azi_full[i, j] = sun_altaz.az.value
            sun_zen_full[i, j] = 90.0 - sun_altaz.alt.value
            sat_azi_full[i, j] = sat_altaz.az.value
            sat_zen_full[i, j] = 90.0 - sat_altaz.alt.value

            if k == mid_index:
                # local_angles_and_more[i,4] = astronomy.sun_zenith_angle(dt, latlons[i,1]*180/m.pi, latlons[i,0]*180/m.pi)
                local_angles_and_more[i, 4] = 90.0 - sun_altaz.alt.value
                local_angles_and_more[i, 5] = sun_altaz.az.value  # solar azimuth
                local_angles_and_more[i, 6] = 90.0 - sat_altaz.alt.value  # satellite zenith angle (90-elevation)
                local_angles_and_more[i, 7] = sat_altaz.az.value  # satellite azimuth

    return local_angles_and_more, [sun_azi_full, sun_zen_full, sat_azi_full, sat_zen_full]


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


# TODO refactor this function
def geometry_computation(framepose_data: pd.DataFrame,
                         image_height: int=684, 
                         verbose = False) -> None:
    """
    Compute geometry using the framepose data

    :param framepose_data_path:
    :param image_height: Height of the Hypso capture

    :return: No return.
    """

    # This is a kind of "east-west offset"
    # the north south offset is in "interpolate_at_frame.py" (as of october 2022)
    additional_time_offset = -650.0 * 0.0 + 0.0

    times = framepose_data.values[:, 0]
    pos_teme = framepose_data.values[:, 2:5]
    quat_teme = framepose_data.values[:, 5:9]  # expresses rotation of body axes to TEME axes?

    frame_count = times.shape[0]

    # new method
    # computing five points of ground locations (geo codes) along each detector line 

    latlon_top = np.zeros([frame_count, 2])
    latlon_mid_top = np.zeros([frame_count, 2])
    latlon_mid = np.zeros([frame_count, 2])
    latlon_mid_bot = np.zeros([frame_count, 2])
    latlon_bot = np.zeros([frame_count, 2])

    impos_mid_itrs = np.zeros([frame_count, 3])

    campos_itrs = np.zeros([frame_count, 3])
    body_x_itrs = np.zeros([frame_count, 3])
    body_z_itrs = np.zeros([frame_count, 3])

    if verbose:
        print(f'Spatial dimensions: {frame_count} frames/lines, {image_height} pixels/samples')
        print('Computing pixel latitude and longitude coordinates...')  # computing pixel lat,lon's

    pointing_off_earth_indicator = 0
    for i in range(frame_count):
        # for i in range(3):
        time_astropy = astropy.time.Time(times[i] + additional_time_offset, format='unix')
        mat = mat_from_quat(quat_teme[i, :])

        body_x_body = np.array([1.0, 0.0, 0.0])
        body_z_body = np.array([0.0, 0.0, 1.0])

        body_x_teme = np.matmul(mat, body_x_body)  # up is +x for hypso
        body_z_teme = np.matmul(mat, body_z_body)

        body_x_astropy_teme = astropy.coordinates.TEME(
            astropy.coordinates.CartesianRepresentation(body_x_teme * astropy.units.m), obstime=time_astropy)
        
        body_z_astropy_teme = astropy.coordinates.TEME(
            astropy.coordinates.CartesianRepresentation(body_z_teme * astropy.units.m), obstime=time_astropy)
        
        pos_astropy_teme = astropy.coordinates.TEME(
            astropy.coordinates.CartesianRepresentation(pos_teme[i, :] * astropy.units.m), obstime=time_astropy)
        
        body_x_astropy_itrs = body_x_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))
        body_z_astropy_itrs = body_z_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))

        pos_astropy_itrs = pos_astropy_teme.transform_to(astropy.coordinates.ITRS(obstime=time_astropy))

        body_x_itrs[i, :] = np.array(
            [body_x_astropy_itrs.earth_location.value[0], body_x_astropy_itrs.earth_location.value[1],
             body_x_astropy_itrs.earth_location.value[2]])
        
        body_z_itrs[i, :] = np.array(
            [body_z_astropy_itrs.earth_location.value[0], body_z_astropy_itrs.earth_location.value[1],
             body_z_astropy_itrs.earth_location.value[2]])
        
        campos_itrs[i, :] = np.array(
            [pos_astropy_itrs.earth_location.value[0], pos_astropy_itrs.earth_location.value[1],
             pos_astropy_itrs.earth_location.value[2]])

        # Comments about orientations and reference frames and stuff:
        # for dayside captures, satellite is moving due south.
        # looking at raw data, 
        # lowest pixel index (0) ('bottom of frame') is west (corresponds to +y axis side of the slit)
        # highest pixel index (683) ('top of frame') is east (corresponds to -y axis side of the slit)
        # concerning direct georectification:
        # get top pixel direction by rotating center view (ca. +z axis) around +x axis with positive angle
        # get bottom pixel direction is rotation with negative angle

        fov = 8.4  # degrees, theoretical 8.008 deg

        top_pixel_angle = pixel_index_to_angle(image_height - 1, 266, fov, hypso_height_sensor)
        mid_top_pixel_angle = pixel_index_to_angle(3 * image_height / 4 - 1, 266, fov, hypso_height_sensor)
        mid_pixel_angle = pixel_index_to_angle(image_height / 2 - 1, 266, fov, hypso_height_sensor)
        mid_bot_pixel_angle = pixel_index_to_angle(image_height / 4 - 1, 266, fov, hypso_height_sensor)
        bot_pixel_angle = pixel_index_to_angle(0, 266, fov, image_height)

        view_dir_top = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], top_pixel_angle)
        view_dir_mid_top = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_top_pixel_angle)
        view_dir_mid = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_pixel_angle)
        view_dir_mid_bot = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_bot_pixel_angle)
        view_dir_bot = rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], bot_pixel_angle)

        pos_itrs_viewpoint_top, t1 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_top)
        pos_itrs_viewpoint_mid_top, t2 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_top)
        pos_itrs_viewpoint_mid, t3 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid)
        pos_itrs_viewpoint_mid_bot, t4 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_bot)
        pos_itrs_viewpoint_bot, t5 = ellipsoid_line_intersection(campos_itrs[i, :], view_dir_bot)

        pos_itrs_viewpoint_top_astropy_certesian = astropy.coordinates.CartesianRepresentation(
            pos_itrs_viewpoint_top * astropy.units.m)
        pos_itrs_viewpoint_mid_top_astropy_certesian = astropy.coordinates.CartesianRepresentation(
            pos_itrs_viewpoint_mid_top * astropy.units.m)
        pos_itrs_viewpoint_mid_astropy_certesian = astropy.coordinates.CartesianRepresentation(
            pos_itrs_viewpoint_mid * astropy.units.m)
        pos_itrs_viewpoint_mid_bot_astropy_certesian = astropy.coordinates.CartesianRepresentation(
            pos_itrs_viewpoint_mid_bot * astropy.units.m)
        pos_itrs_viewpoint_bot_astropy_certesian = astropy.coordinates.CartesianRepresentation(
            pos_itrs_viewpoint_bot * astropy.units.m)
        
        pos_viewpoint_top_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_top_astropy_certesian,
                                                                  obstime=time_astropy)
        pos_viewpoint_mid_top_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_top_astropy_certesian,
                                                                      obstime=time_astropy)
        pos_viewpoint_mid_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_astropy_certesian,
                                                                  obstime=time_astropy)
        pos_viewpoint_mid_bot_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_mid_bot_astropy_certesian,
                                                                      obstime=time_astropy)
        pos_viewpoint_bot_astropy_itrs = astropy.coordinates.ITRS(pos_itrs_viewpoint_bot_astropy_certesian,
                                                                  obstime=time_astropy)

        latlon_top[i, :] = np.array([pos_viewpoint_top_astropy_itrs.earth_location.geodetic.lat.value,
                                     pos_viewpoint_top_astropy_itrs.earth_location.geodetic.lon.value])
        
        latlon_mid_top[i, :] = np.array([pos_viewpoint_mid_top_astropy_itrs.earth_location.geodetic.lat.value,
                                         pos_viewpoint_mid_top_astropy_itrs.earth_location.geodetic.lon.value])
        
        latlon_mid[i, :] = np.array([pos_viewpoint_mid_astropy_itrs.earth_location.geodetic.lat.value,
                                     pos_viewpoint_mid_astropy_itrs.earth_location.geodetic.lon.value])
        
        latlon_mid_bot[i, :] = np.array([pos_viewpoint_mid_bot_astropy_itrs.earth_location.geodetic.lat.value,
                                         pos_viewpoint_mid_bot_astropy_itrs.earth_location.geodetic.lon.value])
        
        latlon_bot[i, :] = np.array([pos_viewpoint_bot_astropy_itrs.earth_location.geodetic.lat.value,
                                     pos_viewpoint_bot_astropy_itrs.earth_location.geodetic.lon.value])
        impos_mid_itrs[i, :] = pos_itrs_viewpoint_mid

        five = int(np.sign(t1)) + int(np.sign(t2)) + int(np.sign(t3)) + int(np.sign(t4)) + int(np.sign(t5))
        # has to be 5 if not pointing off earth
        if five == 5:
            pointing_off_earth_indicator += 1

        if i == -1:

            dt = datetime.datetime.fromtimestamp(times[i], tz=datetime.timezone.utc)
            if verbose:
                print(f'JD number: {get_julian_day_number(dt)} + {(3600 * dt.hour + 60 * dt.minute + dt.second) / ((24 * 3600))}')
                print(f'GMST radians: {2 * m.pi * get_greenwich_mean_sidereal_time_seconds(dt) / (24 * 3600)}')
                
                print(f'campos TEME: [{pos_teme[i, 0]}, {pos_teme[i, 1]}, {pos_teme[i, 2]}]')
                print(f'campos ITRF: [{campos_itrs[i, 0]}, {campos_itrs[i, 1]}, {campos_itrs[i, 2]}]')

                print(f'view dir top: [{view_dir_top[0]}, {view_dir_top[1]}, {view_dir_top[2]}]')
                print(f'view dir bot: [{view_dir_bot[0]}, {view_dir_bot[1]}, {view_dir_bot[2]}]')

                print(f'ellipsoid pos top: [{pos_itrs_viewpoint_top[0]}, {pos_itrs_viewpoint_top[1]}, {pos_itrs_viewpoint_top[2]}]')
                print(f'ellipsoid pos bot: [{pos_itrs_viewpoint_bot[0]}, {pos_itrs_viewpoint_bot[1]}, {pos_itrs_viewpoint_bot[2]}]')

                print(f'latlon top: {latlon_top[i, 0]}, {latlon_top[i, 1]}')
                print(f'latlon bot: {latlon_bot[i, 0]}, {latlon_bot[i, 1]}')

    # Setting altitude to zero to compute satellite ground track from lat lon
    satpos_lat_lon_alt = ecef_to_lat_lon_alt(campos_itrs)
    satpos_lat_lon = satpos_lat_lon_alt.copy()
    satpos_lat_lon[:, 2] = np.zeros([satpos_lat_lon.shape[0], 1])[:, 0]


    lat_center = latlon_mid[frame_count // 2, 0] * m.pi / 180.0
    lon_center = latlon_mid[frame_count // 2, 1] * m.pi / 180.0

    if pointing_off_earth_indicator != frame_count:

        print('[ERROR] At least one pixel was pointing beyond the earth\'s horizon!')
        exit(2)

    if verbose:
        print('Interpolating pixel coordinate gaps...')

    pixels_lat = np.zeros([frame_count, image_height])
    pixels_lon = np.zeros([frame_count, image_height])

    subsample_pixels_indices = np.array([image_height, 3 * image_height / 4, image_height / 2, image_height / 4, 1])
    # subsample_pixels_indices = np.array([ 1, hypso_height/4, hypso_height/2, 3*hypso_height/4, hypso_height])
    all_pixels_indices = np.linspace(1, image_height, image_height)

    # Determining (interpolating) latlon for all pixels (pixels  [0,1,...,682,683] )

    if verbose:
        print('Using geometry-computed latitude and longitude values')

    for i in range(frame_count):

        lats = np.array([latlon_top[i, 0], latlon_mid_top[i, 0], latlon_mid[i, 0], latlon_mid_bot[i, 0], latlon_bot[i, 0]])
        lons = np.array([latlon_top[i, 1], latlon_mid_top[i, 1], latlon_mid[i, 1], latlon_mid_bot[i, 1], latlon_bot[i, 1]])


        # TODO: override if self.latitudes and self.longitudes exist?
        pixels_lat[i, :] = si.griddata(subsample_pixels_indices, lats, all_pixels_indices, method='cubic')
        pixels_lon[i, :] = si.griddata(subsample_pixels_indices, lons, all_pixels_indices, method='cubic')




    if verbose:
        print('Computing local angles (sun and satellite azimuth and zenith angles)...')

    # must contain "hypso_height//2 - 1" so that local_angles_summary is computed properly
    subsample_pixels_indices = [0, image_height // 4 - 1, 
                                image_height // 2 - 1, 
                                3 * image_height // 4 - 1,
                                image_height - 1]

    local_angles_summary, allangles = compute_local_angles_2(times, 
                                                             pos_teme, 
                                                             pixels_lat, 
                                                             pixels_lon,
                                                             subsample_pixels_indices)
    
    local_angles = pd.DataFrame(local_angles_summary,
                                columns=['Frame index', 'Frame time', 'latitude [degrees]', 'longitude [degrees]',
                                        'Solar Zenith Angle [degrees]', 'Solar Azimuth Angle [degrees]',
                                        'Satellite Zenith Angle [degrees]', 'Satellite Azimuth Angle [degrees]'])
    
    local_angles['Frame index'] = local_angles['Frame index'].astype('uint16')


    subsample_pixels_indices = np.array([1, image_height / 4, image_height / 2, 3 * image_height / 4, image_height])
    all_pixels_indices = np.linspace(1, image_height, image_height)

    sun_azi_full = np.zeros([frame_count, image_height])
    sun_zen_full = np.zeros([frame_count, image_height])
    sat_azi_full = np.zeros([frame_count, image_height])
    sat_zen_full = np.zeros([frame_count, image_height])

    for i in range(frame_count):

        sun_azi = np.array([allangles[0][i, 0], allangles[0][i, 1], allangles[0][i, 2], allangles[0][i, 3], allangles[0][i, 4]])
        sun_zen = np.array([allangles[1][i, 0], allangles[1][i, 1], allangles[1][i, 2], allangles[1][i, 3], allangles[1][i, 4]])
        sat_azi = np.array([allangles[2][i, 0], allangles[2][i, 1], allangles[2][i, 2], allangles[2][i, 3], allangles[2][i, 4]])
        sat_zen = np.array([allangles[3][i, 0], allangles[3][i, 1], allangles[3][i, 2], allangles[3][i, 3], allangles[3][i, 4]])
        
        sun_azi_full[i, :] = si.griddata(subsample_pixels_indices, sun_azi, all_pixels_indices, method='cubic')
        sun_zen_full[i, :] = si.griddata(subsample_pixels_indices, sun_zen, all_pixels_indices, method='cubic')
        sat_azi_full[i, :] = si.griddata(subsample_pixels_indices, sat_azi, all_pixels_indices, method='cubic')
        sat_zen_full[i, :] = si.griddata(subsample_pixels_indices, sat_zen, all_pixels_indices, method='cubic')


    wkt_linestring_footprint = get_wkt(latlon_top * m.pi / 180.0, latlon_bot * m.pi / 180.0)

    prj_file_contents = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    
    elevation_angle = compute_elevation_angle(impos_mid_itrs[frame_count // 2, :], campos_itrs[frame_count // 2, :])
    off_nadir_angle = compute_off_nadir_angle(impos_mid_itrs[frame_count // 2, :], campos_itrs[frame_count // 2, :])

    if verbose:
        print(f'Image Center (lat,lon): ({lat_center * 180.0 / m.pi:08.5f}\t{lon_center * 180.0 / m.pi:09.5f})')
        print(f'Image Center elevation angle: {elevation_angle * 180.0 / m.pi:08.5f}')
        print(f'Image Center off-nadir angle: {off_nadir_angle * 180.0 / m.pi:08.5f}')
    
    geometric_meta_info ={}
    geometric_meta_info['image_center_lat'] = lat_center * 180.0 / m.pi
    geometric_meta_info['image_center_lon'] = lon_center * 180.0 / m.pi
    geometric_meta_info['image_center_elevation_angle'] = elevation_angle * 180.0 / m.pi
    geometric_meta_info['image_center_off_nadir_angle'] = off_nadir_angle * 180.0 / m.pi


    sun_azimuth = sun_azi_full
    sun_zenith = sun_zen_full
    sat_azimuth = sat_azi_full
    sat_zenith = sat_zen_full


    return wkt_linestring_footprint, \
           prj_file_contents, \
           local_angles, \
           geometric_meta_info, \
           pixels_lat, \
           pixels_lon, \
           sun_azimuth, \
           sun_zenith, \
           sat_azimuth, \
           sat_zenith


