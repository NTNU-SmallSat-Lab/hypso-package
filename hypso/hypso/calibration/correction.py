import logging

import numpy as np
from scipy import interpolate
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class CalibrationShapeMismatchError(ValueError):
    """Raised when a calibration coefficient file declared "as_is" (pre-baked
    for one specific bin_factor/AOI - see SensorProfile.capture_mode_
    crop_modes) doesn't actually match the shape this capture's own AOI/
    bin_factor implies. Means this capture's geometry doesn't really match
    the standard configuration its capture_type's pre-baked file was built
    for - see REFACTOR_PROGRESS.md's capture-dimensions plan (Limit D)."""


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

def read_coeffs_from_file(coeff_path: str, coeff_type: str, x_start: int=None,  x_stop: int=None,
                          y_start: int=None, y_stop: int=None, bin_factor: int=None,
                          crop_mode: str = "crop_and_bin") -> np.ndarray:
    """
    Read correction coefficients from file

    :param coeff_path: Coefficient path to read (.csv or .npz)
    :param crop_mode: only meaningful for coeff_type in ("smile", "destriping").
        "crop_and_bin" (default) crops the loaded array to [y_start:y_stop,
        x_start:x_stop] and bins bin_factor columns together - the file is
        assumed to cover the sensor's full native resolution. "as_is" skips
        that entirely and uses the file exactly as loaded, since it's
        already pre-baked for one specific bin_factor/AOI (e.g. HYPSO-1's
        nominal/wide smile/destriping files) - in that case the loaded
        shape is checked against what this capture's own AOI/bin_factor
        would imply, raising CalibrationShapeMismatchError on a mismatch
        rather than silently using a wrong-shaped array. Comes from
        SensorProfile.capture_mode_crop_modes (see calibration/pipeline.py's
        set_calibration_coeff_files), not inferred from the filename.

    :return: 2D array of coefficients
    """
    coefficients = None
    try:
        if coeff_path is None:
            coefficients = None
        else:

            # Processing should be Float 32
            if coeff_path.suffix == ".npz":
                coefficients = np.load(coeff_path)
                key = list(coefficients.keys())[0]
                match coeff_type:
                    # TODO smile and destriping is not defined for HYPSO-2 yet, which is why we can keep the simple definitions
                    # later this needs to be updated

                    case 'radiometric':
                        coefficients = coefficients[key][y_start:y_stop, x_start: x_stop].reshape(y_stop-y_start, -1, bin_factor).mean(axis=2).reshape(y_stop-y_start, -1) # reshape full coeff matrix based on values in config
                    case 'spectral':
                        coefficients = coefficients[key][x_start:x_stop].reshape(-1, bin_factor).mean(axis=1).reshape(-1)
                    case 'smile':
                        if crop_mode == "as_is":
                            coefficients = coefficients[key]
                            expected_shape = (y_stop - y_start, (x_stop - x_start) // bin_factor)
                            if coefficients.shape != expected_shape:
                                raise CalibrationShapeMismatchError(
                                    f"smile calibration file {coeff_path} is pre-baked for shape "
                                    f"{coefficients.shape}, but this capture's AOI/bin_factor implies "
                                    f"{expected_shape}."
                                )
                        else:
                            coefficients = coefficients[key][y_start:y_stop, x_start: x_stop].reshape(y_stop-y_start, -1, bin_factor).mean(axis=2).reshape(y_stop-y_start, -1)
                    case 'destriping':
                        if crop_mode == "as_is":
                            coefficients = coefficients[key]
                            expected_shape = (y_stop - y_start, (x_stop - x_start) // bin_factor)
                            if coefficients.shape != expected_shape:
                                raise CalibrationShapeMismatchError(
                                    f"destriping calibration file {coeff_path} is pre-baked for shape "
                                    f"{coefficients.shape}, but this capture's AOI/bin_factor implies "
                                    f"{expected_shape}."
                                )
                        else:
                            # Not exercised by any file shipped today (every
                            # destriping file hypso1_calibration provides is
                            # "as_is"), but wired through symmetrically with
                            # smile so a future imaging mode with a raw,
                            # croppable destriping matrix needs only a schema
                            # entry (SensorProfile.capture_mode_crop_modes),
                            # not new code here.
                            coefficients = coefficients[key][y_start:y_stop, x_start: x_stop].reshape(y_stop-y_start, -1, bin_factor).mean(axis=2).reshape(y_stop-y_start, -1)
                    case _:
                        raise ValueError('Coefficient type ' + coeff_type + ' does not exist.')

            elif coeff_path.suffix == ".csv": # TODO do we ever use this? should I account for it?
                coefficients = np.genfromtxt(coeff_path, delimiter=',', dtype="float64")
            else:
                coefficients = None

    except CalibrationShapeMismatchError:
        raise
    except BaseException as ex:
        logger.exception("Could not read coefficients file.")
        coefficients = None
        raise ValueError("Could not read coefficients file.") from ex

    return coefficients


def make_overexposed_mask(cube, over_exposed_lim=4094):
    ''' Makes mask for spatial image, so that all good values (not masked) are 
    not overexposed for all wavelengths. 
    
    1 in mask = good pixel
    0 in mask = bad pixel (overexposed)
    
    To apply the mask, just multiply each spatial frame with the mask.
    '''
    num_frames, image_height, image_width = cube.shape
    mask = np.ones([num_frames, image_height])
    for i in range(image_width):
        this_spatial_im = cube[:,:,i]
        mask = np.where(np.array(this_spatial_im) > over_exposed_lim, 0, mask)

    return mask


def make_mask(cube, sat_val_scale=0.25, plot=False):
    ''' Mask values based on all values in cube. Used with destriping.

    For water mask: sat_val_scale=0.25
    For overexposed mask: sat_val_scale=0.9 
    '''
    cube_sum = np.sum(cube, axis=2)#/num_frames
    sat_value = cube_sum.max()*sat_val_scale
    mask = cube_sum > sat_value

    return mask