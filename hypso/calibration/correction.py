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


def read_coeffs_from_file(coeff_path: str) -> np.ndarray:
    """
    Read correction coefficients from file

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


'''
def get_coeffs_from_file(coeff_file: str, 
                        capture_type: str = "nominal",
                        x_start: int = None,
                        x_stop: int = None,
                        y_start: int = None,
                        y_stop: int = None,
                        bin_x: int = None) -> np.ndarray:

    match capture_type:

        case 'custom':

            if any([x_start, x_stop, y_start, y_stop, bin_x]):

                if not all([x_start, x_stop, y_start, y_stop, bin_x]):

                    coeffs = None

                else:
                    full_coeffs = read_coeffs_from_file(coeff_file)

                    coeffs = crop_and_bin_matrix(
                                    matrix=full_coeffs,
                                    x_start=x_start,
                                    x_stop=x_stop,
                                    y_start=y_start,
                                    y_stop=y_stop,
                                    bin_x=bin_x,
                                    bin_y=1)

            else:
                coeffs = None

        case 'nominal':
            coeffs = read_coeffs_from_file(coeff_file)

        case 'wide':
            coeffs = read_coeffs_from_file(coeff_file)

        case _:
            coeffs = None

    return coeffs
'''


def run_radiometric_calibration(cube: np.ndarray, 
                                background_value: int,
                                exp: int,
                                image_height: int,
                                image_width: int,
                                frame_count: int,
                                rad_coeffs: np.ndarray) -> np.ndarray:
    """
    Radiometrically Calibrate the Raw Cube (digital counts) to Radiance

    :param cube: Numpy array containing digital countes 3-channel cube
    :param background_value: Integer representing capture the background value level.
    :param exp: Integer representing the capture exposure level.
    :param image_height: Integer representing image height. Equivalent to row_count
    :param image_width: Integer representing image width. Equivalent to column_count divided by bin_factor (or bin_x)
    :param frame_count: Integer representing number of frames.
    :param correction_coefficients_dict: Dictionary containing the 2D coefficients for correction

    :return: Corrected 3-channel cube
    """

    # Radiometric calibration
    cube_rad_calibrated = np.zeros([frame_count, image_height, image_width])

    for i in range(frame_count):

        frame = cube[i, :, :]

        frame_rad_calibrated = run_radiometric_calibration_one_frame(frame, 
                                                                     exp, 
                                                                     background_value, 
                                                                     rad_coeffs)

        cube_rad_calibrated[i, :, :] = frame_rad_calibrated

    return cube_rad_calibrated

def run_radiometric_calibration_one_frame(frame: np.ndarray,
                                          exp: float,
                                          background_value: float,
                                          rad_coeffs: np.ndarray) -> np.ndarray:
    
    """
    Applies radiometric calibration. Assumes input is 12-bit values, and that the radiometric calibration
    coefficients are the same size as the input image. Note: radiometric calibration coefficients have original
    size (684,1080), matching the "normal" AOI of the HYPSO-1 data (with no binning).

    :param frame: 2D frame to apply radiometric calibration
    :param exp: Exposure value
    :param background_value: Background value
    :param rad_coeffs: 2D array of radio calibration coefficients

    :return: Calibrated frame 2D array
    """

    frame = frame - background_value

    frame_rad_calibrated = frame * rad_coeffs / exp

    return frame_rad_calibrated


def run_smile_correction(cube: np.ndarray, 
                         smile_coeffs: np.ndarray) -> np.ndarray:
    """
    Run smile correction on each frame in a cube, using the center row in the frame as the reference wavelength/band
    for smile correction.

    :param cube: 3-channel spectral cube
    :param correction_coefficients_dict: Dictionary containing the coefficients for smile correction
    :return:
    """

    #if smile_coeffs is None:
    #    return cube.copy()

    #spectral_band_matrix = smile_coeffs

    num_frames, image_height, image_width = cube.shape

    cube_smile_corrected = np.zeros([num_frames, image_height, image_width])

    for i in range(num_frames):

        frame = cube[i, :, :]

        frame_smile_corrected = run_smile_correction_one_frame(frame, smile_coeffs)

        cube_smile_corrected[i, :, :] = frame_smile_corrected

    return cube_smile_corrected


def run_smile_correction_one_frame(frame: np.ndarray,
                                   smile_coeffs: np.ndarray) -> np.ndarray:
    """
    Run smile correction on each row in a frame, using the center row as the reference wavelength/band for
    smile correction.

    :param frame: 2D frame on which to apply smile correction
    :param smile_coeffs: Spectral coefficients (Wavelength)

    :return: Corrected frame after smile correction
    """

    image_height, image_width = frame.shape

    center_row_no = int(image_height / 2)

    wavelength_ref = smile_coeffs[center_row_no]

    frame_smile_corrected = np.zeros([image_height, image_width])

    for i in range(image_height):  # For each row

        wavelength = smile_coeffs[i]

        row = frame[i]

        # Correct row
        row_corrected = run_smile_correction_one_row(row, 
                                                     wavelength, 
                                                     wavelength_ref)

        frame_smile_corrected[i, :] = row_corrected

    return frame_smile_corrected



def run_smile_correction_one_row(row: np.ndarray, 
                                 wavelength: np.ndarray, 
                                 wavelength_ref: np.ndarray) -> np.ndarray:
    """
    Applies smile correction. Use cubic spline interpolation to resample one row onto the correct
    wavelengths/bands from a reference wavelength/band array to correct for the smile effect.

    :param row: Data row to apply smile correction
    :param wavelength:
    :param wavelength_ref:

    :return: Row corrected for smile effect
    """

    row_interpolated = interpolate.splrep(wavelength, row)

    row_corrected = interpolate.splev(wavelength_ref, row_interpolated)

    # Set values for wavelengths below 400 nm to zero
    for i in range(len(wavelength_ref)):
        wavelength = wavelength_ref[i]
        if wavelength < 400:
            row_corrected[i] = 0
        else:
            break

    return row_corrected




def run_destriping_correction(cube, correction_coefficients_dict) -> np.ndarray:
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
