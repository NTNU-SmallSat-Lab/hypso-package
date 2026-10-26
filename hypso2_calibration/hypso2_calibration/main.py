from importlib.resources import files
import numpy as np

def get_hypso2_calibration_files(capture_type=None, coeff_type='moved') -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """



    # TODO should one of the other two spectral calibration files be used?
    # npz_file_spectral = "h2_spectral_calibration_matrix.npz"
    # npz_file_spectral = "h2_spectral_calibration_wavelengths_center_row.npz"

    # V3 calibration (2026-10). The pre-V3 'moved', 'adjusted' and 'original'
    # coefficient sets were removed from the package.
    if coeff_type == 'moved':
        print("[INFO] Using 'moved' (V3) calibration coefficients for HYPSO-2. This is the default. The other option 'abs' (V3 absolute) can be passed using the 'coeff_type' keyword argument.")
        npz_file_radiometric = "h2_radiometric_calibration_matrix_full_moved_v3.npz"

    elif coeff_type == 'abs':
        # V3 scaled to the absolute level of HYPSO-1 relative to PACE over Libya-4
        # (h2-onboard-calibration/pace_hypso_rad_comparison/abs_radiometric_correction.py):
        # about 11% lower radiance than 'moved' V3 (median ratio 0.887, 5-95%
        # 0.857-0.921), with no spectral shift between the two matrices.
        # Wide captures use the wide matrix, derived from binned data and used
        # as is (read_coeffs_from_file); about 1% below the binned full matrix.
        print("[INFO] Using 'abs' (V3, scaled to the level of HYPSO-1) calibration coefficients for HYPSO-2.")
        if capture_type == "wide":
            npz_file_radiometric = "h2_radiometric_calibration_matrix_wide_v3_abs.npz"
        else:
            npz_file_radiometric = "h2_radiometric_calibration_matrix_full_v3_abs.npz"

    else:
        raise ValueError(f"Invalid coeff_type: {coeff_type}. Must be 'moved' or 'abs'.")

    # Adjusted using polynomial fit, using whole spectrum.
    # For now the pre-V3 spectral calibration: the V3 one
    # (spectral_array_calibrated_poly_full_v3.npz, shipped) shifts band centers
    # by +0.3 to +1.7 nm, which renames the integer-wavelength bands that
    # POLYMER band settings and OC-SMART's HYPSO band table select by name.
    npz_file_spectral = "spectral_array_calibrated_poly_full.npz"

    npz_file_smile = "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz"
    npz_file_destriping = None

    npz_file_spectral_full_frame = "spectral_array_calibrated_poly_full.npz"

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

    if npz_file_spectral_full_frame:
        spectral_full_frame_coeff_file = files('hypso2_calibration').joinpath(f'data/{npz_file_spectral_full_frame}')
    else:
        spectral_full_frame_coeff_file = None

    calibration_files = {
        "radiometric": rad_coeff_file,
        "smile": smile_coeff_file,
        "destriping": destriping_coeff_file,
        "spectral": spectral_coeff_file,
        "spectral_full_frame": spectral_full_frame_coeff_file
    }

    return calibration_files
