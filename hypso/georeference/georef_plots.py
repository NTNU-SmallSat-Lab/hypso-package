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
from hypso.georeference import georef as gref

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
def compute_local_angles(times, pos_teme, latlons) -> np.ndarray:
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
def compute_local_angles_2(times, pos_teme, lats, lons, indices) -> Tuple[np.ndarray, List[np.ndarray]]:
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


# PLOTTING FUNCTIONS

def add_grid(axes, lat_center, lon_center, seg_lat, seg_lon, size_lat_deg, size_lon_deg) -> None:
    """
    Adds a grid for plotting based on the lat and lon

    :param axes:
    :param lat_center:
    :param lon_center:
    :param seg_lat:
    :param seg_lon:
    :param size_lat_deg:
    :param size_lon_deg:

    :return: No return.
    """
    grid_size_lat_rad = size_lat_deg * m.pi / 180;
    grid_latmin = m.floor(lat_center / grid_size_lat_rad) * grid_size_lat_rad - grid_size_lat_rad;
    grid_latmax = grid_latmin + 3 * grid_size_lat_rad;

    grid_size_lon_rad = size_lon_deg * m.pi / 180;
    grid_lonmin = m.floor(lon_center / grid_size_lon_rad) * grid_size_lon_rad - grid_size_lon_rad;
    grid_lonmax = grid_lonmin + 3 * grid_size_lon_rad;

    grid_lat_bounds = [grid_latmin, grid_latmax];
    grid_lon_bounds = [grid_lonmin, grid_lonmax];

    grid_lats = np.linspace(grid_lat_bounds[0], grid_lat_bounds[1], seg_lat + 1)
    grid_lons = np.linspace(grid_lon_bounds[0], grid_lon_bounds[1], seg_lon + 1)

    zdir = None

    for i in range(seg_lon + 1):
        x_grid_lat = R_eq * np.cos(grid_lats) * m.cos(grid_lons[i])
        y_grid_lat = R_eq * np.cos(grid_lats) * m.sin(grid_lons[i])
        z_grid_lat = R_pl * np.sin(grid_lats)
        axes.plot3D(x_grid_lat, y_grid_lat, z_grid_lat, color='#DD7777')
        if i != 0:
            axes.text(x_grid_lat[0], y_grid_lat[0], z_grid_lat[0], f'{grid_lons[i] * 180 / m.pi:4.1f}', zdir,
                      color='#DD7777')

    for i in range(seg_lat + 1):
        x_grid_lon = R_eq * m.cos(grid_lats[i]) * np.cos(grid_lons)
        y_grid_lon = R_eq * m.cos(grid_lats[i]) * np.sin(grid_lons)
        z_grid_lon = R_pl * m.sin(grid_lats[i]) * (1.0 + 0.0 * np.sin(grid_lons))
        axes.plot3D(x_grid_lon, y_grid_lon, z_grid_lon, color='#7777DD')
        axes.text(x_grid_lon[0], y_grid_lon[0], z_grid_lon[0], f'{grid_lats[i] * 180 / m.pi:4.1f}', zdir,
                  color='#7777DD')


def geometry_computation(framepose_data_path, hypso_height=684) -> None:
    """
    Compute geometry using the framepose data

    :param framepose_data_path:
    :param hypso_height: Height of the Hypso capture

    :return: No return.
    """
    print('Geometric Computations')

    # print('  This script requires:')
    # print('    a path to a \'frametime pose file\'')
    # print('    optional the data height in pixels (default 684)')
    # This is a kind of "east-west offset"
    # the north south offset is in "interpolate_at_frame.py" (as of october 2022)
    additional_time_offset = -650.0 * 0.0 + 0.0

    framepose_data = pd.read_csv(framepose_data_path)

    parent_dir = framepose_data_path.parent.absolute()

    # gis_vector_layers_dir = Path(parent_dir, 'gis')
    # output_path_wkt = Path(parent_dir, 'footprint.txt')
    output_path_wkt_csv = Path(parent_dir, 'footprint.csv')
    output_path_prj = os.path.join(parent_dir, 'footprint.prj')

    output_path_local_angles = Path(parent_dir, 'local-angles.csv')

    output_path_plots_png = Path(parent_dir, 'plots-png')
    output_path_plots_svg = Path(parent_dir, 'plots-svg')
    output_path_plots_svg.mkdir(parents=True, exist_ok=True)
    output_path_plots_png.mkdir(parents=True, exist_ok=True)

    output_path_geo_info = Path(parent_dir, 'geometric-meta-info.txt')

    output_path_pxls_lat = Path(parent_dir, 'latitudes.dat')
    output_path_pxls_lon = Path(parent_dir, 'longitudes.dat')

    output_path_sun_azi = Path(parent_dir, 'sun-azimuth.dat')
    output_path_sun_zen = Path(parent_dir, 'sun-zenith.dat')
    output_path_sat_azi = Path(parent_dir, 'sat-azimuth.dat')
    output_path_sat_zen = Path(parent_dir, 'sat-zenith.dat')

    # output_path_band_tif_base = Path(parent_dir, 'geotiff/band_')
    # output_path_rgb_tif = Path(parent_dir, 'geotiff/rgb.tif')

    times = framepose_data.values[:, 0]
    pos_teme = framepose_data.values[:, 2:5]
    quat_teme = framepose_data.values[:, 5:9]  # expresses rotation of body axes to TEME axes?

    frame_count = times.shape[0]

    # old method
    # Compute satellite position in terms of lat lon alt
    # pos_lat_lon_alt = gref.eci_to_lat_lon_alt(pos_teme, times, additional_time_offset)
    # Compute satellite position in terms of a cartesian ECEF frame
    # pos_ecef = lat_lon_alt_to_ecef(pos_lat_lon_alt)
    # Compute satellite body axes in an ECEF frame
    # body_x_ecef, body_z_ecef = xz_axes_ecef(times, quat_teme, additional_time_offset)
    # compute view directions of top, mid and bottom pixel of each frame/detector line
    # view_dirs_top, view_dirs_bot, view_dirs_mid = compute_pixel_view_dirs(body_z_ecef, body_x_ecef)
    # compute pixel locations
    # top_latlon = pos_view_to_lat_lon(pos_ecef, view_dirs_top)
    # bot_latlon = pos_view_to_lat_lon(pos_ecef, view_dirs_bot)
    # mid_latlon = pos_view_to_lat_lon(pos_ecef, view_dirs_mid)

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

    print(f'Spatial dimensions: {frame_count} frames/lines, {hypso_height} pixels/samples')

    print('  Pixel coordinates ...')  # computing pixel lat,lon's

    pointing_off_earth_indicator = 0
    for i in range(frame_count):
        # for i in range(3):
        time_astropy = astropy.time.Time(times[i] + additional_time_offset, format='unix')
        mat = gref.mat_from_quat(quat_teme[i, :])

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

        center_fov_offset = 0.0  # shall be determined from geometric calibration at some point
        fov = 8.4  # degrees, theoretical 8.008 deg
        fov_half = fov / 2.0  # in degrees
        fov_half_half = fov / 4.0  # in degrees

        # Top == highest pixel index ; Bot == lowest pixel index
        # Top == -y axis side of slit ; Bot == +y axis side of slit
        # top_pixel_angle     = (   fov_half      * (hypso_height/hypso_height_sensor) + center_fov_offset) * m.pi/180.0
        # mid_top_pixel_angle = (   fov_half_half * (hypso_height/hypso_height_sensor) + center_fov_offset) * m.pi/180.0
        # mid_pixel_angle     = (                                                        center_fov_offset) * m.pi/180.0
        # mid_bot_pixel_angle = ( - fov_half_half * (hypso_height/hypso_height_sensor) + center_fov_offset) * m.pi/180.0
        # bot_pixel_angle     = ( - fov_half      * (hypso_height/hypso_height_sensor) + center_fov_offset) * m.pi/180.0

        top_pixel_angle = pixel_index_to_angle(hypso_height - 1, 266, fov, hypso_height_sensor)
        mid_top_pixel_angle = pixel_index_to_angle(3 * hypso_height / 4 - 1, 266, fov, hypso_height_sensor)
        mid_pixel_angle = pixel_index_to_angle(hypso_height / 2 - 1, 266, fov, hypso_height_sensor)
        mid_bot_pixel_angle = pixel_index_to_angle(hypso_height / 4 - 1, 266, fov, hypso_height_sensor)
        bot_pixel_angle = pixel_index_to_angle(0, 266, fov, hypso_height_sensor)

        # print(top_pixel_angle, bot_pixel_angle)
        # if i > 4:
        #    exit(3)

        # print(body_x_astropy_itrs.earth_location.value)
        # print(body_z_astropy_itrs.earth_location.value)
        # print(body_x_itrs)
        # print(body_z_itrs)
        # print(np.array([body_x_itrs[0], body_x_itrs[1], body_x_itrs[2]]))
        # print(np.array([body_z_itrs[0], body_z_itrs[1], body_z_itrs[2]]))

        view_dir_top = gref.rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], top_pixel_angle)
        view_dir_mid_top = gref.rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_top_pixel_angle)
        view_dir_mid = gref.rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_pixel_angle)
        view_dir_mid_bot = gref.rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], mid_bot_pixel_angle)
        view_dir_bot = gref.rotate_axis_angle(body_z_itrs[i, :], body_x_itrs[i, :], bot_pixel_angle)

        pos_itrs_viewpoint_top, t1 = gref.ellipsoid_line_intersection(campos_itrs[i, :], view_dir_top)
        pos_itrs_viewpoint_mid_top, t2 = gref.ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_top)
        pos_itrs_viewpoint_mid, t3 = gref.ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid)
        pos_itrs_viewpoint_mid_bot, t4 = gref.ellipsoid_line_intersection(campos_itrs[i, :], view_dir_mid_bot)
        pos_itrs_viewpoint_bot, t5 = gref.ellipsoid_line_intersection(campos_itrs[i, :], view_dir_bot)
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
            print(
                f'JD number: {gref.get_julian_day_number(dt)} + {(3600 * dt.hour + 60 * dt.minute + dt.second) / ((24 * 3600))}')
            print(f'GMST radians: {2 * m.pi * gref.get_greenwich_mean_sidereal_time_seconds(dt) / (24 * 3600)}')
            print(f'campos TEME: [{pos_teme[i, 0]}, {pos_teme[i, 1]}, {pos_teme[i, 2]}]')
            print(f'campos ITRF: [{campos_itrs[i, 0]}, {campos_itrs[i, 1]}, {campos_itrs[i, 2]}]')

            print(f'view dir top: [{view_dir_top[0]}, {view_dir_top[1]}, {view_dir_top[2]}]')
            print(f'view dir bot: [{view_dir_bot[0]}, {view_dir_bot[1]}, {view_dir_bot[2]}]')

            print(
                f'ellipsoid pos top: [{pos_itrs_viewpoint_top[0]}, {pos_itrs_viewpoint_top[1]}, {pos_itrs_viewpoint_top[2]}]')
            print(
                f'ellipsoid pos bot: [{pos_itrs_viewpoint_bot[0]}, {pos_itrs_viewpoint_bot[1]}, {pos_itrs_viewpoint_bot[2]}]')

            print(f'latlon top: {latlon_top[i, 0]}, {latlon_top[i, 1]}')
            print(f'latlon bot: {latlon_bot[i, 0]}, {latlon_bot[i, 1]}')

        '''
        # compute local tangent and bitangen, or normal at the center latlon point
        x = Req*cos(lon)*cos(lat)
        y = Req*sin(lon)*cos(lat)
        z = Rpol*sin(lat)

        # tangent long
        x = Req*(-sin(lon)*cos(lat))
        y = Req*(cos(lon)*cos(lat))
        z = Rpol*sin(lat)
        # tangent lat
        x = Req*(cos(lon)*-sin(lat))
        y = Req*(sin(lon)*-sin(lat))
        z = Rpol*cos(lat)
        '''

    print('  Plotting computations ...')
    # Setting altitude to zero to compute satellite ground track from lat lon
    satpos_lat_lon_alt = gref.ecef_to_lat_lon_alt(campos_itrs)
    satpos_lat_lon = satpos_lat_lon_alt.copy()
    satpos_lat_lon[:, 2] = np.zeros([satpos_lat_lon.shape[0], 1])[:, 0]
    satpos_ecef_ground_track = gref.lat_lon_alt_to_ecef(satpos_lat_lon)

    # lat_bounds_satellite_track = [np.min(satpos_lat_lon_alt[:,0]), np.max(satpos_lat_lon_alt[:,0])];
    # lon_bounds_satellite_track = [np.min(satpos_lat_lon_alt[:,1]), np.max(satpos_lat_lon_alt[:,1])];
    # lat_center = (lat_bounds_satellite_track[0] + lat_bounds_satellite_track[1])/2;
    # lon_center = (lon_bounds_satellite_track[0] + lon_bounds_satellite_track[1])/2;
    lat_center = latlon_mid[frame_count // 2, 0] * m.pi / 180.0
    lon_center = latlon_mid[frame_count // 2, 1] * m.pi / 180.0

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    add_grid(ax, lat_center, lon_center, 3, 3, 1.6, 1.6)

    # def add_view_dirs(ax, campos_itrs, body_z_itrs, body_x_itrs):
    #    '''
    #    plot3Ds corner view directions
    #    returns cartesian corner positions
    #    '''
    plot_view_line_length = 530 * 1000

    fov_half = 8.0 / 2.0

    view_dir_0_top = gref.rotate_axis_angle(body_z_itrs[0, :], body_x_itrs[0, :],
                                            -fov_half * (hypso_height / hypso_height_sensor) * m.pi / 180.0)
    view_dir_0_bot = gref.rotate_axis_angle(body_z_itrs[0, :], body_x_itrs[0, :],
                                            fov_half * (hypso_height / hypso_height_sensor) * m.pi / 180.0)
    view_start_0 = campos_itrs[0, :]
    view_end_0_top = campos_itrs[0, :] + plot_view_line_length * view_dir_0_top
    view_end_0_bot = campos_itrs[0, :] + plot_view_line_length * view_dir_0_bot

    view_dir_N_top = gref.rotate_axis_angle(body_z_itrs[-1, :], body_x_itrs[-1, :],
                                            -fov_half * (hypso_height / hypso_height_sensor) * m.pi / 180.0)
    view_dir_N_bot = gref.rotate_axis_angle(body_z_itrs[-1, :], body_x_itrs[-1, :],
                                            fov_half * (hypso_height / hypso_height_sensor) * m.pi / 180.0)
    view_start_N = campos_itrs[-1, :]
    view_end_N_top = campos_itrs[-1, :] + plot_view_line_length * view_dir_N_top
    view_end_N_bot = campos_itrs[-1, :] + plot_view_line_length * view_dir_N_bot

    p1, t = gref.ellipsoid_line_intersection(view_start_0, view_dir_0_top)
    p2, t = gref.ellipsoid_line_intersection(view_start_0, view_dir_0_bot)
    p3, t = gref.ellipsoid_line_intersection(view_start_N, view_dir_N_top)
    p4, t = gref.ellipsoid_line_intersection(view_start_N, view_dir_N_bot)

    ax.plot3D([p1[0], p2[0], p4[0], p3[0], p1[0]], [p1[1], p2[1], p4[1], p3[1], p1[1]],
              [p1[2], p2[2], p4[2], p3[2], p1[2]], linewidth=0.8)

    # ax.plot3D([view_start_0[0], view_end_0_top[0]],[view_start_0[1], view_end_0_top[1]],[view_start_0[2], view_end_0_top[2]], color='#222222', linewidth=0.8)
    # ax.plot3D([view_start_0[0], view_end_0_bot[0]],[view_start_0[1], view_end_0_bot[1]],[view_start_0[2], view_end_0_bot[2]], color='#222222', linewidth=0.8)
    # ax.plot3D([view_start_N[0], view_end_N_top[0]],[view_start_N[1], view_end_N_top[1]],[view_start_N[2], view_end_N_top[2]], color='#222222', linewidth=0.8)
    # ax.plot3D([view_start_N[0], view_end_N_bot[0]],[view_start_N[1], view_end_N_bot[1]],[view_start_N[2], view_end_N_bot[2]], color='#222222', linewidth=0.8)

    leng = body_z_itrs.shape[0]
    lines = 5
    for i in range(0, lines + 1):
        index = (i * (leng - 1)) // (lines)

        view_start = campos_itrs[index, :]
        view_end = view_start + plot_view_line_length * body_z_itrs[index, :]
        ax.plot3D([view_start[0], view_end[0]], [view_start[1], view_end[1]], [view_start[2], view_end[2]],
                  color='#222222', linewidth=0.8)

    ax.plot3D(campos_itrs[:, 0], campos_itrs[:, 1], campos_itrs[:, 2])
    ax.plot3D(satpos_ecef_ground_track[:, 0], satpos_ecef_ground_track[:, 1], satpos_ecef_ground_track[:, 2])
    image_corner_positions = np.array([p1, p2, p3, p4])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    try:
        os.mkdir(output_path_plots_png)
    except FileExistsError:
        pass

    try:
        os.mkdir(output_path_plots_svg)
    except FileExistsError:
        pass

    elevation_plot = 20
    for az in range(0, 360, 20):
        ax.view_init(elev=elevation_plot, azim=az)
        # ax.axis('equal')
        # plt.tight_layout()
        svg_save_path = str(Path(output_path_plots_svg, f'view_geometry_az{az}_el{elevation_plot}.svg'))
        png_save_path = str(Path(output_path_plots_png, f'view_geometry_az{az}_el{elevation_plot}.png'))
        fig.savefig(svg_save_path)
        fig.savefig(png_save_path, dpi=300)

    if pointing_off_earth_indicator != frame_count:
        print('At least one pixel was pointing beyond the earth\'s horizon!')
        exit(2)

    try:
        os.mkdir(parent_dir)
    except FileExistsError:
        pass

    print('  Interpolating pixel coordinate gaps ...')
    pixels_lat = np.zeros([frame_count, hypso_height])
    pixels_lon = np.zeros([frame_count, hypso_height])

    subsample_pixels_indices = np.array([hypso_height, 3 * hypso_height / 4, hypso_height / 2, hypso_height / 4, 1])
    # subsample_pixels_indices = np.array([ 1, hypso_height/4, hypso_height/2, 3*hypso_height/4, hypso_height])
    all_pixels_indices = np.linspace(1, hypso_height, hypso_height)

    # Determining (interpolating) latlon for all pixels (pixels  [0,1,...,682,683] )

    for i in range(frame_count):
        lats = np.array(
            [latlon_top[i, 0], latlon_mid_top[i, 0], latlon_mid[i, 0], latlon_mid_bot[i, 0], latlon_bot[i, 0]])
        lons = np.array(
            [latlon_top[i, 1], latlon_mid_top[i, 1], latlon_mid[i, 1], latlon_mid_bot[i, 1], latlon_bot[i, 1]])
        pixels_lat[i, :] = si.griddata(subsample_pixels_indices, lats, all_pixels_indices, method='cubic')
        pixels_lon[i, :] = si.griddata(subsample_pixels_indices, lons, all_pixels_indices, method='cubic')

    print('  Local angles (sun and satellite azimuth and zenith angles) ...')

    # must contain "hypso_height//2 - 1" so that local_angles_summary is computed properly
    subsample_pixels_indices = [0, hypso_height // 4 - 1, hypso_height // 2 - 1, 3 * hypso_height // 4 - 1,
                                hypso_height - 1]

    local_angles_summary, allangles = compute_local_angles_2(times, pos_teme, pixels_lat, pixels_lon,
                                                             subsample_pixels_indices)
    df = pd.DataFrame(local_angles_summary,
                      columns=['Frame index', 'Frame time', 'latitude [degrees]', 'longitude [degrees]',
                               'Solar Zenith Angle [degrees]', 'Solar Azimuth Angle [degrees]',
                               'Satellite Zenith Angle [degrees]', 'Satellite Azimuth Angle [degrees]'])
    df['Frame index'] = df['Frame index'].astype('uint16')
    df.to_csv(output_path_local_angles, index=None)

    # [sun_azi_full, sun_zen_full, sat_azi_full, sat_zen_full]

    subsample_pixels_indices = np.array([1, hypso_height / 4, hypso_height / 2, 3 * hypso_height / 4, hypso_height])
    all_pixels_indices = np.linspace(1, hypso_height, hypso_height)
    sun_azi_full = np.zeros([frame_count, hypso_height])
    sun_zen_full = np.zeros([frame_count, hypso_height])
    sat_azi_full = np.zeros([frame_count, hypso_height])
    sat_zen_full = np.zeros([frame_count, hypso_height])
    for i in range(frame_count):
        sun_azi = np.array(
            [allangles[0][i, 0], allangles[0][i, 1], allangles[0][i, 2], allangles[0][i, 3], allangles[0][i, 4]])
        sun_zen = np.array(
            [allangles[1][i, 0], allangles[1][i, 1], allangles[1][i, 2], allangles[1][i, 3], allangles[1][i, 4]])
        sat_azi = np.array(
            [allangles[2][i, 0], allangles[2][i, 1], allangles[2][i, 2], allangles[2][i, 3], allangles[2][i, 4]])
        sat_zen = np.array(
            [allangles[3][i, 0], allangles[3][i, 1], allangles[3][i, 2], allangles[3][i, 3], allangles[3][i, 4]])
        sun_azi_full[i, :] = si.griddata(subsample_pixels_indices, sun_azi, all_pixels_indices, method='cubic')
        sun_zen_full[i, :] = si.griddata(subsample_pixels_indices, sun_zen, all_pixels_indices, method='cubic')
        sat_azi_full[i, :] = si.griddata(subsample_pixels_indices, sat_azi, all_pixels_indices, method='cubic')
        sat_zen_full[i, :] = si.griddata(subsample_pixels_indices, sat_zen, all_pixels_indices, method='cubic')

    sun_azi_full.astype('float32').tofile(output_path_sun_azi)
    sun_zen_full.astype('float32').tofile(output_path_sun_zen)
    sat_azi_full.astype('float32').tofile(output_path_sat_azi)
    sat_zen_full.astype('float32').tofile(output_path_sat_zen)

    wkt_linestring_footprint = get_wkt(latlon_top * m.pi / 180.0, latlon_bot * m.pi / 180.0)

    # with open(output_path_wkt, 'w') as f:
    #    f.write(wkt_linestring_footprint)
    #    f.write('\n')

    with open(output_path_wkt_csv, 'w') as f:
        f.write('WKT,name\n"')
        f.write(wkt_linestring_footprint)
        f.write('",\n')

    prj_file_contents = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    with open(output_path_prj, 'w') as f:
        f.write(prj_file_contents)
        f.write('\n')

    pixels_lat.astype('float32').tofile(output_path_pxls_lat)
    pixels_lon.astype('float32').tofile(output_path_pxls_lon)

    elevation_angle = compute_elevation_angle(impos_mid_itrs[frame_count // 2, :], campos_itrs[frame_count // 2, :])
    off_nadir_angle = compute_off_nadir_angle(impos_mid_itrs[frame_count // 2, :], campos_itrs[frame_count // 2, :])

    print(f'  Image Center (lat,lon): ({lat_center * 180.0 / m.pi:08.5f}\t{lon_center * 180.0 / m.pi:09.5f})')
    print(f'  Image Center elevation angle: {elevation_angle * 180.0 / m.pi:08.5f}')
    print(f'  Image Center off-nadir angle: {off_nadir_angle * 180.0 / m.pi:08.5f}')
    with open(output_path_geo_info, 'a') as f:
        f.write(f'Image Center \'lat lon\': {lat_center * 180.0 / m.pi:08.5f} {lon_center * 180.0 / m.pi:09.5f}\n')
        f.write(f'Image Center elevation angle: {elevation_angle * 180.0 / m.pi:08.5f}\n')
        f.write(f'Image Center off-nadir angle: {off_nadir_angle * 180.0 / m.pi:08.5f}\n')

#    wkt_points_pixels = get_wkt_points(latlon_top*m.pi/180.0, latlon_mid_top*m.pi/180.0, latlon_mid*m.pi/180.0, latlon_mid_bot*m.pi/180.0, latlon_bot*m.pi/180.0)
#    with open(output_path_wkt_px_csv, 'w') as f:
#        f.write('WKT,name\n')
#        f.write(wkt_points_pixels)
#        f.write(',\n')
#
#    #prj_file_contents = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
#    with open(output_path_px_prj, 'w') as f:
#        f.write(prj_file_contents)
#        f.write('\n')
