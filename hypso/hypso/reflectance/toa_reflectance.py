
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

    if True:
        compute_toa_reflectance_thuillier_2002(srf=srf, 
                                          toa_radiance=toa_radiance, 
                                          iso_time=iso_time, 
                                          solar_zenith_angles=solar_zenith_angles)

    # Get Local variables
    #srf = self.srf
    #toa_radiance = self.l1b_cube.to_numpy()

    #iso_time = self.iso_time
    #solar_zenith_angles = self.solar_zenith_angles

    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday


    # Load the NetCDF file
    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc"))
    ds = xr.open_dataset(solar_data_path)
    #print(ds)
    #print(ds.data_vars)
    #print(ds.attrs)

    solar_x = ds["Vacuum Wavelength"].values
    solar_y = ds["SSI"].values * 1000 # convert to milliwatts

    ds.close()


    # Create new solar X with a new delta
    current_num = solar_x[0]
    delta = 0.01
    new_solar_x = [solar_x[0]]
    while current_num <= solar_x[-1]:
        current_num = current_num + delta
        new_solar_x.append(current_num)

    new_solar_x = np.array(new_solar_x)

    # Interpolate for Y with original solar data
    new_solar_y = np.interp(new_solar_x, solar_x[:], solar_y[:])

    import pickle
    with open('ssi.pkl', 'wb') as file:
        pickle.dump(new_solar_y, file)

    with open('ssi_wl.pkl', 'wb') as file:
        pickle.dump(new_solar_x, file)


    ssi_values = new_solar_y
    # Replace solar Dataframe
    #solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

    # Estimation of TOA Reflectance
    band_number = 0
    toa_reflectance = np.empty_like(toa_radiance)

    ESUN_hypso = []

    for single_wl, single_srf in srf:
        # Resample HYPSO SRF to new solar wavelength
        resamp_srf = np.interp(new_solar_x, single_wl, single_srf, left=0, right=0)
        #resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
        resamp_srf_sum = np.sum(resamp_srf)
        weights_srf = resamp_srf / resamp_srf_sum

        #import matplotlib.pyplot as plt
        #plt.plot(range(0,len(resamp_srf)), resamp_srf)
        #plt.savefig('resamp_srf.png')
        #plt.close()

        #plt.plot(range(0,len(weights_srf)), weights_srf)
        #plt.savefig('weights_srf.png')
        #plt.close()

        #ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py
        ESUN = np.sum(ssi_values * weights_srf)  # units matche HYPSO from device.py
        ESUN_hypso.append(ESUN)
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

    #with open('esun_hypso.pkl', 'wb') as file:
    #    pickle.dump(np.array(ESUN_hypso), file)

    return toa_reflectance







def compute_toa_reflectance_thuillier_2002(srf,
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

    ESUN_hypso = []

    for single_wl, single_srf in srf:
        # Resample HYPSO SRF to new solar wavelength
        resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
        weights_srf = resamp_srf / np.sum(resamp_srf)
        ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py
        #print(ESUN)
        ESUN_hypso.append(ESUN)
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

    #import pickle
    #with open('esun_hypso_thuillier.pkl', 'wb') as file:
    #    pickle.dump(np.array(ESUN_hypso), file)

    return toa_reflectance