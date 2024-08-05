import math as m
import numpy as np
import scipy.spatial as ss
# from .time_process import get_greenwich_mean_sidereal_time_seconds
import hypso.georeference.georef as gref
import datetime
from typing import Tuple


# source: https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
def minimum_bounding_rectangle(points) -> np.ndarray:
    """
    Find the smallest bounding rectangle for a set of points. Rval is a 4x2 matrix of bounding box corner coordinates

    :param points: nx2 matrix of coordinates

    :return: a set of points representing the corners of the bounding box.
    """
    # from scipy.ndimage.interpolation import rotate
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ss.ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    #  rotations = np.vstack([
    #      np.cos(angles),
    #      -np.sin(angles),
    #      np.sin(angles),
    #      np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval  # corner points of bbox


def mat_from_quat(quat) -> np.ndarray:
    """
    Matrix from Quaternion

    :param quat: must be a four element list of numbers or 4 element nump array

    :return:a 3x3 numpy array containing the rotation matrix
    """

    mag = m.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
    quat[0] /= mag
    quat[1] /= mag
    quat[2] /= mag
    quat[3] /= mag

    w2 = quat[0] * quat[0]
    x2 = quat[1] * quat[1]
    y2 = quat[2] * quat[2]
    z2 = quat[3] * quat[3]

    wx = quat[0] * quat[1]
    wy = quat[0] * quat[2]
    wz = quat[0] * quat[3]
    xy = quat[1] * quat[2]
    xz = quat[1] * quat[3]
    zy = quat[3] * quat[2]

    mat = np.zeros([3, 3])

    mat[0, 0] = w2 + x2 - y2 - z2
    mat[1, 0] = 2.0 * (xy + wz)
    mat[2, 0] = 2.0 * (xz - wy)
    mat[0, 1] = 2.0 * (xy - wz)
    mat[1, 1] = w2 - x2 + y2 - z2
    mat[2, 1] = 2.0 * (zy + wx)
    mat[0, 2] = 2.0 * (xz + wy)
    mat[1, 2] = 2.0 * (zy - wx)
    mat[2, 2] = w2 - x2 - y2 + z2

    return mat


def rotate_axis_angle(vec, axis, angle) -> np.ndarray:
    """
    Rotates vector vec around axis axis by angle angle Both vec, axis, and vec_rot are column vectors. axis is a
    unit vector, angle is in radians

    :param vec:
    :param axis:
    :param angle:

    :return:
    """

    dot = np.dot(vec, axis)
    cross = np.cross(axis, vec)
    vec_rot = vec * m.cos(angle) + axis * (dot) * (1.0 - m.cos(angle)) + cross * m.sin(angle)
    return vec_rot


# Positions coordinates conversion functions

# PARAMETERS -----------------------------------------------

# ETRS89 (European Terrestrial Reference System 1989, https://en.wikipedia.org/wiki/European_Terrestrial_Reference_System_1989)
# R_eq = 6378137.0
# f = 1.0 / 298.257222101

# ED50
# R_eq = 6378388.0
# f = 1.0 / 297

# ITRFxx
# R_eq = ?
# f_1 = ?

# GRS80 (Geodetic Reference System 1980, https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980)
# R_eq = 6378137.0
# f = 1.0 / 298.257222100882711

# WGS84 (World Geodetic System 1984, https://en.wikipedia.org/wiki/World_Geodetic_System)
R_eq = 6378137.0
f = 1.0 / 298.257223563

e_2 = f * (2 - f)  # eccentricity squared
R_pl = 6356752.0


# ---------------------------------------------------------


def ecef_to_lat_lon_alt(pos) -> np.ndarray:
    """
    Input pos has to be Nx3, and units in meters

    :param pos:

    :return: lat lon (with respect to ECI axes) and altitude in degrees and meters
    """

    if len(pos.shape) != 2:
        print('Error, wrong shape 1', pos.shape)
        return []
    if pos.shape[1] != 3:
        print('Error, wrong shape 2', pos.shape)
        return []

    output_lat_lon_alt = np.zeros(pos.shape)
    for i in range(pos.shape[0]):

        norm = m.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
        norm_xy = m.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2)
        lat_i = m.asin(pos[i, 2] / norm)
        lat_ip1 = 0

        max_iterations = 50
        iterations = 0
        threshold = 1e-9
        error = 1
        while error > threshold:
            slat = m.sin(lat_i)
            N = R_eq / m.sqrt(1.0 - e_2 * slat ** 2)
            lat_ip1 = m.atan2((N * e_2 * slat + pos[i, 2]), norm_xy)
            error = abs(lat_ip1 - lat_i)
            lat_i = lat_ip1
            iterations = iterations + 1

        slat = m.sin(lat_ip1)
        clat = m.cos(lat_ip1)
        N = R_eq / m.sqrt(1.0 - e_2 * slat ** 2)
        output_lat_lon_alt[i, 0] = lat_ip1
        output_lat_lon_alt[i, 1] = m.atan2(pos[i, 1], pos[i, 0])
        output_lat_lon_alt[i, 2] = norm_xy / clat - N
    return output_lat_lon_alt


def eci_to_lat_loneci_alt(pos) -> np.ndarray:
    """
    ECI to latitude and longitude and altitude

    :param pos:

    :return:
    """
    return ecef_to_lat_lon_alt(pos)


def eci_lon_to_ecef_lon(datetime_utc, lon, time_offset) -> float:
    """
    ECI to longitude to ECEF Longitude conversion
    :param datetime_utc:
    :param lon:
    :param time_offset:

    :return:
    """
    s_sdrl = gref.get_greenwich_mean_sidereal_time_seconds(datetime_utc) + time_offset
    sdrl_rad = s_sdrl * 2.0 * m.pi / (24 * 3600)
    return lon - sdrl_rad


def eci_to_lat_lon_alt(pos, times, time_offset) -> np.ndarray:
    """
    ECI to Lat/Lon Altitude

    :param pos: needs to be Nx3
    :param times: needs to be N
    :param time_offset: is a float. is an additional time offset to adjust georeferencing

    :return: Numpy array with pseudo latitude and longitude and altitude
    """

    if len(pos.shape) != 2:
        return []
    if pos.shape[1] != 3:
        return []

    if times.shape[0] != pos.shape[0]:
        return []

    pseudo_lat_lon_alt = eci_to_lat_loneci_alt(pos)

    for i in range(pseudo_lat_lon_alt.shape[0]):
        dt = datetime.datetime.fromtimestamp(times[i], tz=datetime.timezone.utc)
        pseudo_lat_lon_alt[i, 1] = (eci_lon_to_ecef_lon(dt, pseudo_lat_lon_alt[i, 1], time_offset) + m.pi) % (
                2 * m.pi) - m.pi
        pseudo_lat_lon_alt[i, 0] = pseudo_lat_lon_alt[i, 0]
    return pseudo_lat_lon_alt


def pos_ecef_to_lat_lon_alt(pos) -> np.ndarray:
    """
    Coverts POS ECEF to latitude and longitude and altitude
    :param pos:
    :return:
    """
    return eci_to_lat_loneci_alt(pos)


def lat_lon_alt_to_ecef(lla) -> np.ndarray:
    """
    Converts latitude and longitude ALT to ECEF
    :param lla:
    :return:
    """
    pos_ecef = np.zeros(lla.shape)
    for i in range(lla.shape[0]):
        slat = m.sin(lla[i][0])
        clat = m.cos(lla[i][0])
        slon = m.sin(lla[i][1])
        clon = m.cos(lla[i][1])
        N = R_eq / m.sqrt(1.0 - e_2 * slat ** 2)
        pos_ecef[i][0] = (N + lla[i][2]) * clat * clon
        pos_ecef[i][1] = (N + lla[i][2]) * clat * slon
        pos_ecef[i][2] = ((1 - e_2) * N + lla[i][2]) * slat
    return pos_ecef


def ellipsoid_line_intersection(point, direction) -> Tuple[float, float]:
    """
    Calculates the ellipsodie and line intersection

    :param point:
    :param direction:

    :return:
    """
    # x^2 + y^2 +f_*z^2 = a^2

    f_ = 1.0 / ((1.0 - f) ** 2)
    # Projecting ray onto spheroid
    h = 1 - (1 - f_) * direction[2] ** 2
    c1 = (direction[0] * point[0] + direction[1] * point[1] + f_ * direction[2] * point[2]) / h
    c2 = (point[0] ** 2 + point[1] ** 2 + f_ * point[2] ** 2 - R_eq * R_eq) / h

    discr = c1 * c1 - c2
    if discr < 0:  # no intersection case
        t = -1.0
        surfacePoint = point
    else:
        t = -c1 - m.sqrt(discr)
        surfacePoint = point + t * direction
        # print(point, direction, t, surfacePoint)

    return surfacePoint, t
