import numpy as np


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
        gaussian_srf = np.exp(
            -(gx / sigma_nm[i]) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        srf.append(gaussian_srf)

    return srf

