from importlib.resources import files

def get_hypso1_calibration_files(capture_type, new_coeff=False) -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """
    if new_coeff:
        print("[INFO] - Using new calibration coefficients for HYPSO-1.")
        # without adjusting gain based on PACE
        npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1_adjusted_weighted_final.npz"
        # with adjusting gain based on PACE
        # npz_file_radiometric = "h1_rad_coeff_adjusted_full.npz"
        npz_file_spectral = "h1_spectral_coeff_adjusted_full.npz"
        match capture_type:

            case "custom":
                npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"  
                npz_file_destriping = None
            case "nominal":
                npz_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"
            case "wide":
                npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"
            case _:
                npz_file_radiometric = None
                npz_file_smile = None
                npz_file_destriping = None
                npz_file_spectral = None
    else: 
        print("[INFO] - Using original calibration coefficients for HYPSO-1.")
        npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.npz"
        npz_file_spectral = "spectral_calibration_wavelengths_center_row_HYPSO-1.npz"
        match capture_type:

            case "custom":
                npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"  
                npz_file_destriping = None

            case "nominal":
                npz_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"
            case "wide":
                npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"
            case _:
                npz_file_radiometric = None
                npz_file_smile = None
                npz_file_destriping = None
                npz_file_spectral = None

    if npz_file_radiometric:
        rad_coeff_file = files('hypso1_calibration').joinpath(f'data/{npz_file_radiometric}')
    else:
        rad_coeff_file = None

    if npz_file_smile:
        smile_coeff_file = files('hypso1_calibration').joinpath(f'data/{npz_file_smile}')
    else:
        smile_coeff_file = None

    if npz_file_destriping:
        destriping_coeff_file = files('hypso1_calibration').joinpath(f'data/{npz_file_destriping}')
    else:
        destriping_coeff_file = None

    if npz_file_spectral:
        spectral_coeff_file = files('hypso1_calibration').joinpath(f'data/{npz_file_spectral}')
    else:
        spectral_coeff_file = None

    calibration_files = {
        "radiometric": rad_coeff_file,
        "smile": smile_coeff_file,
        "destriping": destriping_coeff_file,
        "spectral": spectral_coeff_file
    }

    return calibration_files