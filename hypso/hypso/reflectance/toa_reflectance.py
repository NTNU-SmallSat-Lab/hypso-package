
from importlib.resources import files
from dateutil import parser

import numpy as np
import pandas as pd
import xarray as xr



def compute_toa_reflectance(srf,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            ) -> xr.DataArray:

    
    # Get Local variables
    #srf = self.srf
    #toa_radiance = self.l1b_cube.to_numpy()

    #iso_time = self.iso_time
    #solar_zenith_angles = self.solar_zenith_angles

    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday


    # Read Solar Data
    solar_data_path = str(files('hypso.reflectance').joinpath("Solar_irradiance_Thuillier_2002.csv"))
    solar_df = pd.read_csv(solar_data_path)

    # Create new solar X with a new delta
    solar_array = np.array(solar_df)
    current_num = solar_array[0, 0]
    delta = 0.01
    new_solar_x = [solar_array[0, 0]]
    while current_num <= solar_array[-1, 0]:
        current_num = current_num + delta
        new_solar_x.append(current_num)

    # Interpolate for Y with original solar data
    new_solar_y = np.interp(new_solar_x, solar_array[:, 0], solar_array[:, 1])

    # Replace solar Dataframe
    solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

    # Estimation of TOA Reflectance
    band_number = 0
    toa_reflectance = np.empty_like(toa_radiance)

    for single_wl, single_srf in srf:
        # Resample HYPSO SRF to new solar wavelength
        resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
        weights_srf = resamp_srf / np.sum(resamp_srf)
        ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py

        # Earth-Sun distance (from day of year) using julian date
        # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
        distance_sun = 1 - 0.01672 * np.cos(0.9856 * (
                julian_day - 4))

        # Get toa_reflectance
        solar_angle_correction = np.cos(np.radians(solar_zenith_angles))
        multiplier = (ESUN * solar_angle_correction) / (np.pi * distance_sun ** 2)
        toa_reflectance[:, :, band_number] = toa_radiance[:, :, band_number] / multiplier

        band_number = band_number + 1

    #toa_reflectance = -toa_reflectance

    return toa_reflectance