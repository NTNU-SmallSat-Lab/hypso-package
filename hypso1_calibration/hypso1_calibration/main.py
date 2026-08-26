from importlib.resources import files

def get_hypso1_calibration_files(capture_type="custom", coeff_type='moved') -> None:
    """
    Get the absolute path for the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """

    if coeff_type == 'moved':
        # old versions commented out
        # Moon calibrated, without adjusting gain based on PACE
        # npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1_adjusted_weighted_final.npz"
        # Moon calibrated, with adjusting gain based on PACE
        # npz_file_radiometric = "h1_rad_coeff_adjusted_full.npz"
        # Adjusted based on moon dip I think
        # npz_file_spectral = "h1_spectral_coeff_adjusted_full.npz"

        print("[INFO] Using 'moved' calibration coefficients for HYPSO-1. This is the default for radiometric and spectral coefficients. Other options 'original' or 'adjusted' can be passed using the 'coeff_type' keyword argument.")
        # Radiometric coefficients only moved
        npz_file_radiometric = "h1_radiometric_calibration_matrix_full_moved.npz"
        # Adjusted using polynomial fit, using whole spectrum, static offset
        npz_file_spectral = "spectral_array_calibrated_poly_full.npz"
    elif coeff_type == 'adjusted':
        print("[INFO] Using 'adjusted' calibration coefficients for HYPSO-1.")
        npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1_adjusted_v11.npz"
        npz_file_spectral = "spectral_array_calibrated_poly_full.npz"

    elif coeff_type == 'original': 
        print("[INFO] Using 'original' calibration coefficients for HYPSO-1.")
        npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.npz"
        npz_file_spectral = "spectral_calibration_wavelengths_center_row_HYPSO-1.npz"
    else: 
        raise ValueError(f"Invalid coeff_type: {coeff_type}. Must be 'moved', 'adjusted', or 'original.'")

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

        case "moon":
            # Previously unhandled - fell through to case _ below, silently
            # zeroing out ALL FOUR coefficient files (radiometric included)
            # for every moon capture. radiometric_calibration_matrix_HYPSO-1_
            # moon.npz already shipped in data/ unused, which is what
            # surfaced this - a moon case was clearly intended but never
            # wired up here. No moon-specific smile/destriping matrix exists
            # (moon captures don't share nominal/wide's frame geometry), so
            # smile falls back to the same full-frame matrix "custom" uses
            # below (read_coeffs_from_file crops it to this capture's actual
            # AOI/bin_factor - see calibration/correction.py's 'smile' case,
            # which only skips that crop for a path containing 'wide' or
            # 'nominal'), and destriping is skipped, exactly like "custom".
            npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_moon.npz"
            npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"
            npz_file_destriping = None

        case _:
            npz_file_radiometric = None
            npz_file_smile = None
            npz_file_destriping = None
            npz_file_spectral = None

    npz_file_spectral_full_frame = "spectral_array_calibrated_poly_full.npz"

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

    if npz_file_spectral_full_frame:
        spectral_full_frame_coeff_file = files('hypso1_calibration').joinpath(f'data/{npz_file_spectral_full_frame}')
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