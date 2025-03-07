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








'''def get_spectral_response_function(wavelengths, fwhm: float = 3.3) -> None:
    """
    Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
    used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

    :return: None.
    """

    fwhm_nm = fwhm
    sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

    srf = []
    for band in wavelengths:
    # for i, band in enumerate(wavelengths):
        center_lambda_nm = band
        start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm), 4)
        soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm), 4)

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
            gx = np.linspace(-3 * sigma_nm, 3 * sigma_nm, len(srf_wl))
        gaussian_srf = np.exp(
            -(gx / sigma_nm) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

        # Get final wavelength and SRF
        srf_wl_single = wavelengths
        srf_single = np.zeros_like(srf_wl_single)
        srf_single[good_idx] = gaussian_srf

        srf.append([srf_wl_single, srf_single])

    return srf'''

