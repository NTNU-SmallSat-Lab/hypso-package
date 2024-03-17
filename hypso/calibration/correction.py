import numpy as np
from scipy import interpolate
from pathlib import Path

def crop_and_bin_matrix(matrix, x_start, x_stop, y_start, y_stop, bin_x=1, bin_y=1) -> np.ndarray:
    """ Crops matrix to AOI. Bins matrix so that the average value in the bin_x
    number of pixels is stored.
    """
    # Crop to selected AOI
    width_binned = None
    new_matrix = matrix[y_start:y_stop, x_start:x_stop]
    height, width = new_matrix.shape

    # If bin is set to 0 or negative we assume this means no binning, aka bin=1
    if bin_x < 1:
        bin_x = 1
    if bin_y < 1:
        bin_y = 1

    # Bin spectral direction
    if bin_x != 1:
        width_binned = int(width / bin_x)
        matrix_cropped_and_binned = np.zeros((height, width_binned))
        for i in range(width_binned):
            this_pixel_sum = 0
            for j in range(bin_x):
                this_pixel_value = new_matrix[:, i * bin_x + j]
                this_pixel_sum += this_pixel_value
            average_pixel_value = this_pixel_sum / bin_x
            matrix_cropped_and_binned[:, i] = average_pixel_value
        new_matrix = matrix_cropped_and_binned

    # Bin spatial direction
    if bin_y != 1:
        height_binned = int(height / bin_y)
        matrix_binned_spatial = np.zeros((height_binned, width_binned))
        for i in range(height_binned):
            this_pixel_sum = 0
            for j in range(bin_y):
                this_pixel_value = new_matrix[i * bin_y + j, :]
                this_pixel_sum += this_pixel_value
            average_pixel_value = this_pixel_sum / bin_y
            matrix_binned_spatial[i, :] = average_pixel_value / bin_y
        new_matrix = matrix_binned_spatial

    return new_matrix


def get_coefficients_from_file(coeff_path: str) -> np.ndarray:
    """
    Get correction coefficients from file

    :param coeff_path: Coefficient path to read (.csv)

    :return: 2D array of coefficients
    """
    coefficients = None
    try:
        # Processing should be Float 32
        coefficients = np.genfromtxt(
            coeff_path, delimiter=',', dtype="float64")
        # coefficients = readCsvFile(coeff_path)
    except BaseException:
        coefficients = None
        raise ValueError("Could not read coefficients file.")

    return coefficients


def get_coefficients_from_dict(coeff_dict: dict, satobj) -> dict:
    """
    Get the coefficients from the csv files contained in a dictionary.

    :param coeff_dict: Dictionary containing the paths of the csv files to read
    :param satobj: Hypso satellite object

    :return: Dictionary containing the 2D coefficients read from the csv files
    """

    coeffs = coeff_dict.copy()
    for k in coeff_dict:
        # Coefficients Custom (needs trimming)
        if "full" in str(coeff_dict[k]):
            bin_x = satobj.info["bin_factor"]
            full_coeff = get_coefficients_from_file(coeff_dict[k])
            coeffs[k] = crop_and_bin_matrix(
                full_coeff,
                satobj.info["x_start"],
                satobj.info["x_stop"],
                satobj.info["y_start"],
                satobj.info["y_stop"],
                bin_x)
        # Just read coefficients
        elif "nominal" in str(coeff_dict[k]) or "wide" in str(coeff_dict[k]):
            coeffs[k] = get_coefficients_from_file(coeff_dict[k])

        else:
            coeff_dict[k] = None

    return coeffs


def calibrate_cube(info_sat: dict, raw_cube: np.ndarray, correction_coefficients_dict: dict) -> np.ndarray:
    """
    Radiometrically Calibrate the Raw Cube (digital counts) to Radiance

    :param info_sat: Dictionary containing capture information
    :param raw_cube: Numpy array containing digital countes 3-channel cube
    :param correction_coefficients_dict: Dictionary containing the 2D coefficients for correction

    :return: Corrected 3-channel cube
    """

    DEBUG = False

    if correction_coefficients_dict["radiometric"] is None:
        return raw_cube.copy()

    print("Radiometric Correction Ongoing")

    background_value = info_sat['background_value']
    exp = info_sat['exp']
    image_height = info_sat['image_height']
    image_width = info_sat['image_width']

    # Radiometric calibration
    num_frames = info_sat["frame_count"]
    cube_calibrated = np.zeros([num_frames, image_height, image_width])

    if DEBUG:
        print("F:", num_frames, "H:", image_height, "W:", image_width)
        print("Radioshape: ",
              correction_coefficients_dict["radiometric"].shape)

    for i in range(num_frames):
        frame = raw_cube[i, :, :]
        # Radiometric Calibration
        frame_calibrated = apply_radiometric_calibration(
            frame, exp, background_value, correction_coefficients_dict["radiometric"])

        cube_calibrated[i, :, :] = frame_calibrated

    l1b_cube = cube_calibrated

    return l1b_cube


def apply_radiometric_calibration(
        frame: np.ndarray,
        exp: float,
        background_value: float,
        radiometric_calibration_coefficients: np.ndarray):
    """
    Applies radiometric calibration. Assumes input is 12-bit values, and that the radiometric calibration
    coefficients are the same size as the input image. Note: radiometric calibration coefficients have original
    size (684,1080), matching the "normal" AOI of the HYPSO-1 data (with no binning).

    :param frame: 2D frame to apply radiometric calibration
    :param exp: Exposure value
    :param background_value: Background value
    :param radiometric_calibration_coefficients: 2D array of radio calibration coefficients

    :return: Calibrated frame 2D array
    """

    frame = frame - background_value
    frame_calibrated = frame * radiometric_calibration_coefficients / exp

    return frame_calibrated


def smile_correction_one_row(row, w, w_ref) -> np.ndarray:
    """
    Applies smile correction. Use cubic spline interpolation to resample one row onto the correct
    wavelengths/bands from a reference wavelength/band array to correct for the smile effect.

    :param row: Data row to apply smile correction
    :param w:
    :param w_ref:

    :return: Row corrected for smile effect
    """

    row_interpolated = interpolate.splrep(w, row)
    row_corrected = interpolate.splev(w_ref, row_interpolated)
    # Set values for wavelengths below 400 nm to zero
    for i in range(len(w_ref)):
        w = w_ref[i]
        if w < 400:
            row_corrected[i] = 0
        else:
            break
    return row_corrected


def smile_correction_one_frame(frame, spectral_band_matrix) -> np.ndarray:
    """
    Run smile correction on each row in a frame, using the center row as the reference wavelength/band for
    smile correction.

    :param frame: 2D frame on which to apply smile correction
    :param spectral_band_matrix: Spectral coefficients (Wavelength)

    :return: Corrected frame after smile correction
    """

    image_height, image_width = frame.shape
    center_row_no = int(image_height / 2)
    w_ref = spectral_band_matrix[center_row_no]
    frame_smile_corrected = np.zeros([image_height, image_width])
    for i in range(image_height):  # For each row
        this_w = spectral_band_matrix[i]
        this_row = frame[i]
        # Correct row
        row_corrected = smile_correction_one_row(this_row, this_w, w_ref)
        frame_smile_corrected[i, :] = row_corrected
    return frame_smile_corrected


def smile_correct_cube(cube, correction_coefficients_dict: dict) -> np.ndarray:
    """
    Run smile correction on each frame in a cube, using the center row in the frame as the reference wavelength/band
    for smile correction.

    :param cube: 3-channel spectral cube
    :param correction_coefficients_dict: Dictionary containing the coefficients for smile correction
    :return:
    """

    if correction_coefficients_dict["smile"] is None:
        return cube.copy()

    print("Smile Correction Ongoing")

    spectral_band_matrix = correction_coefficients_dict["smile"]
    num_frames, image_height, image_width = cube.shape
    cube_smile_corrected = np.zeros([num_frames, image_height, image_width])
    for i in range(num_frames):
        this_frame = cube[i, :, :]
        frame_smile_corrected = smile_correction_one_frame(
            this_frame, spectral_band_matrix)
        cube_smile_corrected[i, :, :] = frame_smile_corrected
    return cube_smile_corrected


def destriping_correct_cube(cube, correction_coefficients_dict) -> np.ndarray:
    """
    Apply destriping correction matrix.

    :param cube: 3-channel spectral cube
    :param correction_coefficients_dict: Dictionary containing the 2D coefficients for destriping

    :return: 3-channel array for destriping correction
    """


    if correction_coefficients_dict["destriping"] is None:
        return cube.copy()

    print("Destriping Correction Ongoing")

    destriping_correction_matrix = correction_coefficients_dict["destriping"]
    # print(destriping_correction_matrix.shape)
    # print(cube.shape)
    # cube_delined = copy.deepcopy(cube)
    # cube_delined[:, 1:] -= destriping_correction_matrix[:-1]
    cube_delined = cube * destriping_correction_matrix
    return cube_delined
