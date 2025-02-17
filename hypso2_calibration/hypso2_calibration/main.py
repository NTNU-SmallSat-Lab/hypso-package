from importlib.resources import files

def get_hypso2_calibration_files(capture_type) -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """

    match capture_type:

        # TODO should one of the other two spectral calibration files be used?
        # npz_file_spectral = "h2_spectral_calibration_matrix.npz"
        # npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"
        
        case "custom":
            npz_file_radiometric = "h2_radiometric_calibration_matrix_full.npz"
            npz_file_smile = None  
            npz_file_destriping = None
            npz_file_spectral = "spectral_bands_HYPSO-2.npz"

        case "nominal":
            # TODO nominal matrix should probably also be constructed for hypso-2
            npz_file_radiometric = None
            npz_file_smile = None
            npz_file_destriping = None
            npz_file_spectral = "spectral_bands_HYPSO-2.npz"

        case "wide":
            # TODO the correctness of the wide radiometric matrix should be confirmed
            npz_file_radiometric = "h2_radiometric_calibration_matrix_wide.npz"
            npz_file_smile = None
            npz_file_destriping = None
            npz_file_spectral = "spectral_bands_HYPSO-2.npz"

        case _:
            npz_file_radiometric = None
            npz_file_smile = None
            npz_file_destriping = None
            npz_file_spectral = None

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




