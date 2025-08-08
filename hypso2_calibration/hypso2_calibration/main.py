from importlib.resources import files
import numpy as np

def get_hypso2_calibration_files(new_coeff=False) -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """



    # TODO should one of the other two spectral calibration files be used?
    # npz_file_spectral = "h2_spectral_calibration_matrix.npz"
    # npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"
    

    if new_coeff:
        print("[INFO] - Using new calibration coefficients for HYPSO-2.")
        # Moon calibrated, without adjusting gain based on PACE 
        # npz_file_radiometric = "h2_radiometric_calibration_matrix_adjusted_weighted_final.npz"
        # Moon calibrated, with adjusting gain based on PACE
        # npz_file_radiometric = "h2_rad_coeff_adjusted_full.npz"
        # Radiometric coefficients only moved
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full_moved.npz"
        npz_file_smile = "smile_correction_matrix_HYPSO-2_full.npz"
        npz_file_destriping = None
        # Adjusted based on moon dip I think
        # npz_file_spectral = "h2_spectral_coeff_adjusted_full.npz"
        # Adjusted using polynomial fit, using whole spectrum, static offset
        npz_file_spectral = "spectral_array_calibrated_poly_full.npz"
    else:
        print("[INFO] - Using original calibration coefficients for HYPSO-2.")
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full.npz"
        npz_file_smile = "smile_correction_matrix_HYPSO-2_full.npz"  
        npz_file_destriping = None
        npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"

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




