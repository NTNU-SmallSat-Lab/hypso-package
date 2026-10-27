"""The Sun as the reflectance step needs it: reference solar spectral
irradiance (TSIS-1 HSRS or Thuillier 2002) and the Earth-Sun distance."""
import logging
from importlib.resources import files

import numpy as np
from dateutil import parser

logger = logging.getLogger(__name__)

TSIS_FILE = "hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"
THUILLIER_FILE = "Solar_irradiance_Thuillier_2002.csv"


def load_solar_irradiance(source: str = "tsis"):
    """A reference solar irradiance spectrum.

    Parameters
    ----------
    source : {"tsis", "thuillier"}, default "tsis"
        TSIS-1 HSRS v2 (2022-11-30, 0.005 nm sampling) or Thuillier (2002).

    Returns
    -------
    irradiance : numpy.ndarray
        In mW m-2 nm-1.
    wavelengths : numpy.ndarray
        In nm.
    """
    data = files("hypso.srf").joinpath("data")
    if source == "tsis":
        spectrum = np.load(str(data.joinpath(TSIS_FILE)))
        return spectrum["solar_y"] * 1000, spectrum["solar_x"]   # W -> mW
    if source == "thuillier":
        import pandas as pd
        logger.warning("Using Thuillier SSI reference spectrum for ToA reflectance processing!")
        table = pd.read_csv(str(data.joinpath(THUILLIER_FILE)))
        return np.array(table["mW/m2/nm"]), np.array(table["nm"])
    raise ValueError(f"Unknown solar irradiance source: {source!r} (use 'tsis' or 'thuillier')")


def earth_sun_distance(iso_time) -> float:
    """Earth-Sun distance in AU for a date.

    ``1 - 0.01672 cos(0.9856 (doy - 4))`` with the cosine argument in degrees:
    0.9856 deg/day is the mean orbital motion, day 4 is perihelion. (Until
    2026-10-23 the argument was passed as radians, a date-dependent TOA
    reflectance error of up to about 6%.)

    Parameters
    ----------
    iso_time : str or datetime
        Acquisition time.

    Returns
    -------
    float
        Distance over the mean distance (R / R_0).
    """
    julian_day = parser.isoparse(iso_time).timetuple().tm_yday
    return float(1 - 0.01672 * np.cos(np.radians(0.9856 * (julian_day - 4))))
