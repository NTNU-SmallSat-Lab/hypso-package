from importlib.resources import files

def get_hypso2_calibration_files(capture_type=None, coeff_type=None) -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """



    # TODO should one of the other two spectral calibration files be used?
    # npz_file_spectral = "h2_spectral_calibration_matrix.npz"
    # npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"


    if coeff_type == 'moved':
        # old versions commented out
        # Moon calibrated, without adjusting gain based on PACE
        # npz_file_radiometric = "h2_radiometric_calibration_matrix_adjusted_weighted_final.npz"
        # Moon calibrated, with adjusting gain based on PACE
        # npz_file_radiometric = "h2_rad_coeff_adjusted_full.npz"
        # Adjusted based on moon dip I think
        # npz_file_spectral = "h2_spectral_coeff_adjusted_full.npz"

        print("[INFO] - Using moved calibration coefficients for HYPSO-2.")
        # Radiometric coefficients only moved
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full_moved.npz"
        # Adjusted using polynomial fit, using whole spectrum, static offset
        npz_file_spectral = "spectral_array_calibrated_poly_full.npz"

    elif coeff_type == 'adjusted':
        print("[INFO] - Using adjusted calibration coefficients for HYPSO-2.")
        npz_file_radiometric = "h2_radiometric_calibration_matrix_adjusted_v11.npz"
        npz_file_spectral = "spectral_array_calibrated_poly_full.npz"

    elif coeff_type == 'original':
        print("[INFO] - Using original calibration coefficients for HYPSO-2.")
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full.npz"
        npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"
    elif coeff_type == 'moved_v3':

        print("[INFO] - Using v3 moved calibration coefficients for HYPSO-2.")
        # Radiometric coefficients only moved
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full_moved_v3.npz"
        # Adjusted using polynomial fit, using whole spectrum, static offset
        npz_file_spectral = "spectral_array_calibrated_poly_full_v3.npz"
    elif coeff_type == 'adjusted_v3':
        # in-orbit calibration chain (h2-onboard-calibration): smile (Sobel) -> spectral (moon/LIME) -> SRF -> radiometric (moon/LIME)
        print("[INFO] - Using v3 (in-orbit) calibration coefficients for HYPSO-2.")
        if capture_type == "wide":
            # derived from binned wide data (aoi_x=428, aoi_y=62, bin 9), used as is
            npz_file_radiometric = "h2_radiometric_calibration_matrix_wide_v3.npz"
        else:
            npz_file_radiometric = "h2_radiometric_calibration_matrix_full_v3.npz"
        npz_file_spectral = "spectral_array_calibrated_poly_full_v3.npz"
    elif coeff_type == 'abs_v3':
        # adjusted_v3 scaled to the absolute level of HYPSO-1 relative to PACE over Libya-4
        # (h2-onboard-calibration/pace_hypso_rad_comparison/abs_radiometric_correction.py)
        print("[INFO] - Using v3 (in-orbit) calibration coefficients for HYPSO-2, scaled to the level of HYPSO-1.")
        if capture_type == "wide":
            npz_file_radiometric = "h2_radiometric_calibration_matrix_wide_v3_abs.npz"
        else:
            npz_file_radiometric = "h2_radiometric_calibration_matrix_full_v3_abs.npz"
        npz_file_spectral = "spectral_array_calibrated_poly_full_v3.npz"
    else:
        raise ValueError(f"Invalid coeff_type: {coeff_type}. Must be 'moved', 'adjusted', 'original', 'moved_v3', 'adjusted_v3', or 'abs_v3'.")

    if 'v3' in coeff_type:
        # full-frame smile matrix (Sobel), cropped and binned to the capture AOI when read
        npz_file_smile = "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz"
    else:
        # smile_correction_matrix_HYPSO-2_full.npz is spatially flipped, and was never applied
        # (the smile file failed to load), so it is kept off for the older coefficient types
        npz_file_smile = None
    npz_file_destriping = None

    if npz_file_radiometric:
        rad_coeff_file = files('hypso2_calibration').joinpath(f'data/{npz_file_radiometric}')
    else:
        rad_coeff_file = None

    if npz_file_smile:
        smile_coeff_file = files('hypso2_calibration').joinpath(f'data/{npz_file_smile}')
    else:
        smile_coeff_file = None

    if npz_file_destriping:
        destriping_coeff_file = files('hypso2_calibration').joinpath(f'data/{npz_file_destriping}')
    else:
        destriping_coeff_file = None

    if npz_file_spectral:
        spectral_coeff_file = files('hypso2_calibration').joinpath(f'data/{npz_file_spectral}')
    else:
        spectral_coeff_file = None

    calibration_files = {
        "radiometric": rad_coeff_file,
        "smile": smile_coeff_file,
        "destriping": destriping_coeff_file,
        "spectral": spectral_coeff_file
    }

    return calibration_files
