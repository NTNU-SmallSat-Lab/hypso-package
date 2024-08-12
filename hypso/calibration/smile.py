import numpy as np
from scipy import interpolate

def run_smile_correction(cube: np.ndarray, 
                         smile_coeffs: np.ndarray) -> np.ndarray:
    """
    Run smile correction on each frame in a cube, using the center row in the frame as the reference wavelength/band
    for smile correction.

    :param cube: 3-channel spectral cube
    :param smile_coeffs: Dictionary containing the coefficients for smile correction
    :return:
    """

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
