import numpy as np
import copy
import os
from scipy import interpolate
import csv


def get_coefficients_from_file(coeff_path: str) -> np.ndarray:
    coefficients = None
    try:
        coefficients = np.genfromtxt(
            coeff_path, delimiter=',')
        # coefficients = readCsvFile(coeff_path)
    except BaseException:
        coefficients = None
        raise ValueError("Could not read coefficients file.")

    return coefficients


def get_coefficients_from_dict(coeff_dict: str) -> np.ndarray:
    """Get the coefficients from the csv file.

    Args:
        path (str, optional): Path to the radiometric coefficients csv file. Defaults to None.
        sets the rad_file attribute to the path.
        if no path is given, the rad_file path is used.

    Returns:
        np.ndarray: The coefficients.

    """
    coeffs = coeff_dict.copy()
    for k in coeff_dict:
        coeffs[k] = get_coefficients_from_file(coeff_dict[k])

    return coeffs


def calibrate_cube(info_sat: dict, raw_cube: np.ndarray, correction_coefficients_dict: dict) -> np.ndarray:
    """Calibrate the raw data cube."""
    DEBUG = False

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
        frame,
        exp,
        background_value,
        radiometric_calibration_coefficients):
    ''' Assumes input is 12-bit values, and that the radiometric calibration
    coefficients are the same size as the input image.

    Note: radiometric calibration coefficients have original size (684,1080),
    matching the "normal" AOI of the HYPSO-1 data (with no binning).'''

    frame = frame - background_value
    frame_calibrated = frame * radiometric_calibration_coefficients / exp

    return frame_calibrated


def smile_correction_one_row(row, w, w_ref):
    ''' Use cubic spline interpolation to resample one row onto the correct
    wavelengths/bands from a reference wavelength/band array to correct for
    the smile effect.
    '''
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


def smile_correction_one_frame(frame, spectral_band_matrix):
    ''' Run smile correction on each row in a frame, using the center row as 
    the reference wavelength/band for smile correction.
    '''
    image_height, image_width = frame.shape
    center_row_no = int(image_height/2)
    w_ref = spectral_band_matrix[center_row_no]
    frame_smile_corrected = np.zeros([image_height, image_width])
    for i in range(image_height):  # For each row
        this_w = spectral_band_matrix[i]
        this_row = frame[i]
        # Correct row
        row_corrected = smile_correction_one_row(this_row, this_w, w_ref)
        frame_smile_corrected[i, :] = row_corrected
    return frame_smile_corrected


def smile_correct_cube(cube, correction_coefficients_dict: dict):
    ''' Run smile correction on each frame in a cube, using the center row in 
    the frame as the reference wavelength/band for smile correction.
    '''
    spectral_band_matrix = correction_coefficients_dict["smile"]
    num_frames, image_height, image_width = cube.shape
    cube_smile_corrected = np.zeros([num_frames, image_height, image_width])
    for i in range(num_frames):
        this_frame = cube[i, :, :]
        frame_smile_corrected = smile_correction_one_frame(
            this_frame, spectral_band_matrix)
        cube_smile_corrected[i, :, :] = frame_smile_corrected
    return cube_smile_corrected


def destriping_correct_cube(cube, correction_coefficients_dict):
    ''' Apply destriping correction matrix. '''
    destriping_correction_matrix = correction_coefficients_dict["destriping"]
    # print(destriping_correction_matrix.shape)
    # print(cube.shape)
    # cube_delined = copy.deepcopy(cube)
    # cube_delined[:, 1:] -= destriping_correction_matrix[:-1]
    cube_delined = cube * destriping_correction_matrix
    return cube_delined
