
from importlib.resources import files
from dateutil import parser
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline





def compute_toa_reflectance(sensor_wavelengths,
                            sensor_fwhm,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            ) -> xr.DataArray:


    ssi, solar_wavelengths = load_ssi()

    srf_list, srf_ssi_list = compute_srf(ssi=ssi,
                                         solar_wavelengths=solar_wavelengths,
                                         sensor_wavelengths=sensor_wavelengths,
                                         sensor_fwhm=sensor_fwhm)
    
    esun_list = compute_esun(srf_list=srf_list, 
                             srf_ssi_list=srf_ssi_list)

    scene_date = parser.isoparse(iso_time)
    julian_day = scene_date.timetuple().tm_yday


    toa_reflectance = np.empty_like(toa_radiance)

    for band, esun in enumerate(esun_list):

        # Earth-Sun distance scaler (from day of year) using julian date
        # (R/R_0) earth-sun distance divided by average earth-sun distance
        # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
        # 4 is when earth reaches perihelion, day 4 for 2025
        sun_distance_scaler = 1 - 0.01672 * np.cos(0.9856 * (julian_day - 4))  

        # Get toa_reflectance
        # equation for "Normalized reflectances" found here:
        # https://oceanopticsbook.info/view/atmospheric-correction/normalized-reflectances 
        solar_angle_correction = np.cos(np.radians(solar_zenith_angles))
        multiplier = (esun * solar_angle_correction) / (np.pi * sun_distance_scaler ** 2)
        
        toa_reflectance[:, :, band] = toa_radiance[:, :, band] / multiplier



    if True:
        import csv
        import matplotlib.pyplot as plt

        esun_array = np.array(esun_list)

        with open('esun_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'ESUN'])  # Header
            for wl, computed_esun in zip(sensor_wavelengths, esun_array):
                writer.writerow([wl, computed_esun])


        plt.title('HYPSO ESUN w/ TSIS-1 (2022) SSI')
        #plt.plot(ssi_wl, ssi, label='TSIS-1 SSI')
        #plt.plot(esun_wl, esun, label='HYPSO ESUN')
        plt.plot(solar_wavelengths, ssi, label='TSIS-1 SSI')
        plt.plot(sensor_wavelengths, np.array(esun_array), label='HYPSO ESUN')

        plt.xlim(350, 850)
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Solar Irradiance')
        plt.tight_layout()
        plt.savefig('spectrum_tsis.png')

        plt.close



    return toa_reflectance, srf_list, esun_list






def compute_srf(ssi,
                solar_wavelengths,
                sensor_wavelengths,
                sensor_fwhm):


    # Create SRFs
    fwhm_nm = sensor_fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    
    sensor_band_indices = [np.abs(solar_wavelengths - w).argmin() for w in sensor_wavelengths]

    srf_list = []
    srf_ssi_list = []
    
    for i, sensor_band_index in enumerate(sensor_band_indices):

        center_lambda_nm = solar_wavelengths[sensor_band_index]

        #center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm[i]), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm[i]), 4)

        srf_wl = [center_lambda_nm]
        lower_wl = []
        upper_wl = []

        # get the elements in the wavelength array that are 3Sigma above and below center_w
        # if the center_w is closer to one end than 3Sigma then one of the arrays will be shorter 
        # than the other. This needs to be corrected for so that the center is still kept
        # when making the gaussian, therefore len_diff is found.

        #for ele in solar_x:
        #    if start_lambda_nm < ele < center_lambda_nm:
        #        lower_wl.append(ele)
        #    elif center_lambda_nm < ele < soft_end_lambda_nm:
        #        upper_wl.append(ele)

        # Faster if done based on array indexing
        start_srf_index = np.abs(solar_wavelengths - start_lambda_nm).argmin()
        end_srf_index = np.abs(solar_wavelengths - soft_end_lambda_nm).argmin()

        srf_x = solar_wavelengths[start_srf_index:end_srf_index+1]

        center_srf_index = np.abs(srf_x - center_lambda_nm).argmin()

        lower_wl = srf_x[0:center_srf_index]

        upper_wl = srf_x[center_srf_index+1:]


        lower_wl = list(lower_wl)
        upper_wl = list(upper_wl)

        while len(lower_wl) > len(upper_wl):
            lower_wl.pop(0)
        while len(upper_wl) > len(lower_wl):
            upper_wl.pop(-1)
        len_diff = 0
            
        srf_wl = lower_wl + srf_wl + upper_wl


        # Delta based on Hypso Sampling (Wavelengths)
        gx = None
        if len(srf_wl) == 1:
            gx = [0]
        else:
            # len_diff is added to make up for the missing elements because of clipping
            # at the ends mentioned above. this replaces the clipped elements and makes sure
            # gaussian has correct width
            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wl) + len_diff)

        gaussian_srf = np.exp(-(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        srf_list.append(gaussian_srf)


        ssi_start_index = sensor_band_index - len(lower_wl)
        ssi_end_index = sensor_band_index + len(upper_wl)

        srf_ssi_list.append(ssi[ssi_start_index:ssi_end_index+1])


    return srf_list, srf_ssi_list




def compute_esun(srf_list, srf_ssi_list):

    esun_list = []

    for i, gaussian_srf in enumerate(srf_list):


        srf_ssi = srf_ssi_list[i]

        gaussian_srf_sum = np.sum(gaussian_srf)
        srf_weights = gaussian_srf / gaussian_srf_sum

        esun_value = np.sum(srf_ssi * srf_weights)  # units matche HYPSO from device.py

        esun_list.append(esun_value)

    return esun_list






def load_ssi():

    # Load the NetCDF file
    #solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc"))
    #ds = xr.open_dataset(solar_data_path)
    #solar_x = ds["Vacuum Wavelength"].values
    #solar_y = ds["SSI"].values * 1000 # convert to milliwatts
    #ds.close()

    # Load .npz file containing the pre-processed solar spectrum irradiance (SSI). It is generating using the 'write_ssi_npz.py' script. 
    # The SSI is truncated to the visible spectrum range covering the HYPSO-1 & -2 bands
    # The SSI is from the TSIS-1 SSI v2 file 'hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc'. A copy is in the Git repository.
    # Source & download: https://lasp.colorado.edu/lisird/data/tsis1_hsrs_p1nm
    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"))
    ds = np.load(solar_data_path)

    solar_wavelengths = ds["solar_x"]
    ssi = ds["solar_y"] * 1000 # convert to milliwatts


    return ssi, solar_wavelengths









'''
def compute_toa_reflectance_interpolation(srf,
                            wavelengths,
                            toa_radiance: np.ndarray,
                            iso_time,
                            solar_zenith_angles,
                            fwhm = None,
                            srf_cameron = None,
                            srf_original = None
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

        print(i)

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
        
        print(single_srf_on_wl_axis.shape)
        print(single_srf_on_wl_axis)


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


    if True:
        import csv
        import matplotlib.pyplot as plt

        with open('esun_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Wavelength', 'ESUN'])  # Header
            for wl, esun in zip(wavelengths, ESUN_hypso):
                writer.writerow([wl, esun])


        plt.title('HYPSO ESUN w/ TSIS-1 (2022) SSI')
        #plt.plot(ssi_wl, ssi, label='TSIS-1 SSI')
        #plt.plot(esun_wl, esun, label='HYPSO ESUN')
        plt.plot(new_solar_x, new_solar_y, label='TSIS-1 SSI')
        plt.plot(wavelengths, np.array(ESUN_hypso), label='HYPSO ESUN')

        plt.xlim(350, 850)
        plt.legend(loc='upper right')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Solar Irradiance')
        plt.tight_layout()
        plt.savefig('spectrum_tsis.png')

        plt.close

    return toa_reflectance

'''




'''
# Old ToA reflectance code
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

'''




'''
# Updated function to process fwhm vector of length number of bands
def get_spectral_response_function(wavelengths, fwhm: np.array) -> None:
    """
    Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
    used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

    :return: None.
    """

    fwhm_nm = fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    srf = []
    for i, band in enumerate(wavelengths):

        center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm[i]), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm[i]), 4)

        srf_wl = [center_lambda_nm]
        lower_wl = []
        upper_wl = []


        # get the elements in the wavelength array that are 3Sigma above and below center_w
        # if the center_w is closer to one end than 3Sigma then one of the arrays will be shorter 
        # than the other. This needs to be corrected for so that the center is still kept
        # when making the gaussian, therefore len_diff is found.
        for ele in wavelengths:
            if start_lambda_nm < ele < center_lambda_nm:
                lower_wl.append(ele)
            elif center_lambda_nm < ele < soft_end_lambda_nm:
                upper_wl.append(ele)

        # Make symmetric
        if (len(wavelengths) - i) <= len(lower_wl):
            # Close to highest wavelength, find how many wavelengths missed because 
            # 3 Sigma is out of wavelength bounds, skip symmetry 
            len_diff = len(lower_wl) - len(upper_wl)
        elif i < len(upper_wl):
            # Close to lowest wavelength, find how many wavelengths missed because 
            # 3 Sigma is out of wavelength bounds, skip symmetry 
            len_diff = len(upper_wl) - len(lower_wl)
        else:
            # Close to neither the highest nor lowest wavelength, enforce symmetry
            # correcting for one beign one element longer than the other.
            # correcting for one beign one element longer than the other.
            while len(lower_wl) > len(upper_wl):
                lower_wl.pop(0)
            while len(upper_wl) > len(lower_wl):
                upper_wl.pop(-1)
            len_diff = 0

        srf_wl = lower_wl + srf_wl + upper_wl

        # Delta based on Hypso Sampling (Wavelengths)
        gx = None
        if len(srf_wl) == 1:
            gx = [0]
        else:
            # len_diff is added to make up for the missing elements because of clipping
            # at the ends mentioned above. this replaces the clipped elements and makes sure
            # gaussian has correct width
            # len_diff is added to make up for the missing elements because of clipping
            # at the ends mentioned above. this replaces the clipped elements and makes sure
            # gaussian has correct width
            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wl) + len_diff)

        gaussian_srf = np.exp(-(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        srf.append(gaussian_srf)



    return srf


'''



'''

# Updated function to process fwhm vector of length number of bands
def get_spectral_response_function_old(wavelengths, fwhm: np.array) -> None:
    """
    Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
    used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

    :return: None.
    """

    fwhm_nm = fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    srf = []
    for i, band in enumerate(wavelengths):


        #if i == 119:
        #    print("119")

        center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm[i]), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm[i]), 4)

        srf_wl = [center_lambda_nm]
        lower_wl = []
        upper_wl = []

        for j, ele in enumerate(wavelengths):
            if start_lambda_nm < ele < center_lambda_nm:
                lower_wl.append(ele)
            elif center_lambda_nm < ele < soft_end_lambda_nm:
                upper_wl.append(ele)

        #print(upper_wl)
        #print(lower_wl)

        # Make symmetric
        if (len(wavelengths) - i) <= len(lower_wl):
            # Close to highest wavelength, skip symmetry 
            print(i)
            print("within upper limit")
            len_diff = len(lower_wl) - len(upper_wl)
            pass
        elif i < len(upper_wl):
            # Close to lowest wavelength, skip symmetry 
            print(i)
            print("within lower limit")
            len_diff = len(upper_wl) - len(lower_wl)
            pass
        else:
            # Close to neither the highest nor lowest wavelength, enforce symmetry
            while len(lower_wl) > len(upper_wl):
                lower_wl.pop(0)
            while len(upper_wl) > len(lower_wl):
                upper_wl.pop(-1)
            len_diff = 0

        srf_wl = lower_wl + srf_wl + upper_wl

        good_idx = [(True if ele in srf_wl else False) for ele in wavelengths]

        # Delta based on Hypso Sampling (Wavelengths)
        gx = None
        if len(srf_wl) == 1:
            gx = [0]
        else:
            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wl) + len_diff)
        gaussian_srf = np.exp(
            -(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        # Get final wavelength and SRF
        srf_wl_single = wavelengths
        srf_single = np.zeros_like(srf_wl_single)
        
        
        N = len(gaussian_srf)
        M = len(srf_single)
        half_N = N // 2


        #start_main = max(i - half_N, 0)
        #end_main = min(i + half_N + 1, M)

        #start_kernel = start_main - (i - half_N)
        #end_kernel = start_kernel + (end_main - start_main)

        #start_kernel = max(half_N - i, 0)
        #end_kernel = min(M - i, N)

        if i < half_N:
            start_idx = half_N - i
        else:
            start_idx = 0


        if (M - i) <= half_N:
            end_idx = half_N + (M - i)
        else:
            end_idx = N

        print(start_idx)
        print(end_idx)
        print(good_idx)

        gaussian_srf_subset = gaussian_srf[start_idx:end_idx]

        srf_single[good_idx] = gaussian_srf_subset

        srf.append([srf_wl_single, srf_single])


    #with open('srf_new.pkl', 'wb') as file:
    #    pickle.dump(srf, file)

    return srf


'''



'''

# Old function for Thuillier spectrum
def get_spectral_response_function_thuillier_2002(wavelengths, fwhm: np.array) -> None:
    """
    Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
    used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

    :return: None.
    """

    fwhm_nm = fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    srf = []
    for i, band in enumerate(wavelengths):
        center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm[i]), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm[i]), 4)

        srf_wl = [center_lambda_nm]
        lower_wl = []
        upper_wl = []
        for ele in wavelengths:
            if start_lambda_nm < ele < center_lambda_nm:
                lower_wl.append(ele)
            elif center_lambda_nm < ele < soft_end_lambda_nm:
                upper_wl.append(ele)

        # Make symmetric
        while len(lower_wl) > len(upper_wl):
            lower_wl.pop(0)
        while len(upper_wl) > len(lower_wl):
            upper_wl.pop(-1)

        srf_wl = lower_wl + srf_wl + upper_wl

        good_idx = [(True if ele in srf_wl else False) for ele in wavelengths]

        # Delta based on Hypso Sampling (Wavelengths)
        gx = None
        if len(srf_wl) == 1:
            gx = [0]
        else:
            gx = np.linspace(-3 * sigma_nm[i], 3 * sigma_nm[i], len(srf_wl))
        gaussian_srf = np.exp(
            -(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        # Get final wavelength and SRF
        srf_wl_single = wavelengths
        srf_single = np.zeros_like(srf_wl_single)
        srf_single[good_idx] = gaussian_srf

        srf.append([srf_wl_single, srf_single])

    return srf

'''