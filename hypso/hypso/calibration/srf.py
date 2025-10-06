import numpy as np
import pickle


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


# Updated function to process fwhm vector of length number of bands
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

