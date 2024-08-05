import datetime
from typing import Tuple
import datetime


def get_julian_day_number(date_time) -> float:
    """
    Get Julian Day Number from Date Time

    :param date_time: Date Time

    :return: Julian day as float
    """
    Y = date_time.year
    M = date_time.month
    D = date_time.day
    if M < 3:
        Y = Y - 1
        M = M + 12
    A = int(Y / 100)
    B = 2 - A + int(A / 4)
    C = int(365.25 * (Y + 4716))
    E = int(30.60001 * (M + 1))
    JD = C + E + D + B - 1524.5
    return JD


def get_greenwich_mean_sidereal_time_seconds(date_time) -> float:
    """
    Gets Greenwich mean sidereal time in seconds

    :param date_time:

    :return: seconds since last midnight of a sidereal day
    """

    JD = get_julian_day_number(date_time)
    # T julian centuries since J2000
    T = (JD - 2451545) / 36525;
    # sidereal time at midnight universal time (UT1) at the day corresponding to julian day JD
    # 1982 coefficients
    s0h = 24110.54841 + 8640184.812866 * T + 0.093104 * T * T - 0.0000062 * T * T * T;
    while s0h < 0.0:
        s0h = s0h + 24 * 3600;
    while s0h > 24 * 3600:
        s0h = s0h - 24 * 3600;
    # sidereal time at time s seconds UT since midnight at julian day JD
    s_sdrl = s0h + 1.00273790935 * (3600 * date_time.hour + 60 * date_time.minute + date_time.second);
    while s_sdrl > 24 * 3600:
        s_sdrl = s_sdrl - 24 * 3600;

    return s_sdrl

