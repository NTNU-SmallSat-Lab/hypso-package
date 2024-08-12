import numpy as np


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
