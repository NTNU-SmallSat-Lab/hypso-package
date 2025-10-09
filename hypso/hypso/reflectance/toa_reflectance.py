
from importlib.resources import files
from dateutil import parser

import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline



def compute_toa_reflectance(srf,
                            wavelengths,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            ) -> xr.DataArray:

    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday


    # Load the NetCDF file
    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc"))
    ds = xr.open_dataset(solar_data_path)

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

    ssi_values = new_solar_y

    # Estimation of TOA Reflectance
    band_number = 0
    toa_reflectance = np.empty_like(toa_radiance)

    ESUN_hypso = []
    
    for i in range(len(srf)):
        single_wl = wavelengths
        single_srf = np.array(srf[i])

        # check if the srf goes outside the wavelength range of hypso, if so
        # extend the wavelength range with a cubic spline
        if (len(single_srf) // 2 > i): 
            # check if srf wants to fetch values lower than the lowest wavelength of hypso
            x_orig = np.arange(0, len(wavelengths))
            interpolator = CubicSpline(x_orig, wavelengths, extrapolate=True)        
            x_new = np.arange(-np.abs(i - len(single_srf) // 2), len(wavelengths))
            single_wl = interpolator(x_new)
        elif (i + (len(single_srf)//2) >= len(wavelengths)):
            # check if srf wants to fetch values higher than the highest wavelength of hypso
            x_orig = np.arange(0, len(wavelengths))
            interpolator = CubicSpline(x_orig, wavelengths, extrapolate=True)        
            x_new = np.arange(0, i + (len(single_srf)//2) + 1)
            single_wl = interpolator(x_new)
        # put the srf funciton on the wavelength axis, assuming srf function is symmetric
        k = 0 
        single_srf_on_wl_axis = np.zeros_like(single_wl)
        for j in range(-(len(single_srf)//2), len(single_srf)//2+1):
            center_idx = np.argwhere(single_wl == wavelengths[i])
            single_srf_on_wl_axis[center_idx + j] = single_srf[k]
            k += 1
        
        # for single_wl, single_srf in (wavelengths, srf):
        # Resample HYPSO SRF to new solar wavelength
        resamp_srf = np.interp(new_solar_x, single_wl, single_srf_on_wl_axis, left=0, right=0)
        resamp_srf_sum = np.sum(resamp_srf)
        weights_srf = resamp_srf / resamp_srf_sum

        #ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py
        ESUN = np.sum(ssi_values * weights_srf)  # units matche HYPSO from device.py
        ESUN_hypso.append(ESUN)
        # Earth-Sun distance scaler (from day of year) using julian date
        # (R/R_0) earth-sun distance divided by average earth-sun distance
        # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
        sun_distance_scaler = 1 - 0.01672 * np.cos(0.9856 * (
                julian_day - 4)) # 4 is when earth reaches perihelion, day 4 for 2025 

        # Get toa_reflectance
        # equation for "Normalized reflectances" found here:
        # https://oceanopticsbook.info/view/atmospheric-correction/normalized-reflectances 
        solar_angle_correction = np.cos(np.radians(solar_zenith_angles))
        multiplier = (ESUN * solar_angle_correction) / (np.pi * sun_distance_scaler ** 2)
        toa_reflectance[:, :, band_number] = toa_radiance[:, :, band_number] / multiplier

        band_number = band_number + 1

    return toa_reflectance