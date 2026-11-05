"""Reading calibration coefficient files and the mask helpers used with destriping."""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationShapeMismatchError(ValueError):
    """Raised when a calibration coefficient file declared "as_is" (pre-baked
    for one specific bin_factor/AOI - see SensorProfile.capture_mode_
    crop_modes) doesn't actually match the shape this capture's own AOI/
    bin_factor implies. Means this capture's geometry doesn't really match
    the standard configuration its capture_type's pre-baked file was built
    for."""


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
                          crop_mode: str = "crop_and_bin", row_step: int = 1) -> np.ndarray:
    """
    Read correction coefficients from file

    Parameters
    ----------
    coeff_path
        Coefficient path to read (.csv or .npz)
    coeff_type
        "radiometric", "smile", "destriping" or "spectral".
    x_start, x_stop, y_start, y_stop
        The capture's area of interest on the full detector (columns, rows).
    bin_factor
        Spectral binning factor.
    crop_mode
        only meaningful for coeff_type in ("radiometric", "smile", "destriping").
        "crop_and_bin" (default) crops the loaded array to [y_start:y_stop,
        x_start:x_stop] and bins bin_factor columns together - the file is
        assumed to cover the sensor's full native resolution. "as_is" skips
        that entirely and uses the file exactly as loaded, since it's
        already pre-baked for one specific bin_factor/AOI (e.g. HYPSO-1's
        nominal/wide smile/destriping files) - in that case the loaded
        shape is checked against what this capture's own AOI/bin_factor
        would imply, raising CalibrationShapeMismatchError on a mismatch
        rather than silently using a wrong-shaped array. Comes from
        SensorProfile.capture_mode_crop_modes or, for a specific file, the
        calibration package's catalog (data/coefficients.yaml ``crop``), not
        inferred from the filename.
    row_step
        Spatial subsampling (the capture's ``sample_div``): readout kept every
        ``row_step``-th detector row from ``y_start``, so the 2D matrices keep
        the same rows (``[y_start:y_stop:row_step]``). An "as_is" matrix must
        already have that many rows.

    Returns
    -------
    out
        2D array of coefficients
    """
    if coeff_path is None:
        return None
    coeff_path = Path(coeff_path)
    try:
        if coeff_path.suffix == ".csv":
            # Coefficients registered by users (hypso.calibration.registry) may be CSV; they are used as
            # stored, without cropping or binning. The shipped coefficients are all .npz.
            return np.genfromtxt(coeff_path, delimiter=",", dtype="float64")
        if coeff_path.suffix != ".npz":
            return None
        with np.load(coeff_path) as npz:
            array = npz[list(npz.keys())[0]]
        if coeff_type == "spectral":
            # band centres along the detector's spectral axis: crop to the AOI, average bin_factor columns
            return array[x_start:x_stop].reshape(-1, bin_factor).mean(axis=1).reshape(-1)
        if coeff_type in ("radiometric", "smile", "destriping"):
            return _matrix_for_capture(array, coeff_type, coeff_path, x_start, x_stop, y_start, y_stop,
                                       bin_factor, crop_mode, row_step)
        raise ValueError(f"Coefficient type {coeff_type} does not exist.")
    except CalibrationShapeMismatchError:
        raise
    except BaseException as ex:
        logger.exception("Could not read coefficients file.")
        raise ValueError("Could not read coefficients file.") from ex


def _matrix_for_capture(array, coeff_type, coeff_path, x_start, x_stop, y_start, y_stop, bin_factor,
                        crop_mode, row_step=1):
    """A 2D (rows, spectral columns) coefficient matrix for one capture.

    "crop_and_bin": the file covers the full detector; crop it to the AOI
    and average bin_factor columns. "as_is": the file is pre-baked for one
    AOI and bin_factor (e.g. a wide-capture matrix); use it unchanged, after
    checking its shape against this capture's. With spatial subsampling
    (``row_step`` > 1) only every ``row_step``-th AOI row is kept.
    """
    rows = len(range(y_start, y_stop, row_step))
    columns = (x_stop - x_start) // bin_factor
    if crop_mode == "as_is":
        # pre-baked for the AOI's detector rows (thinned like the readout), or already thinned
        if row_step > 1 and array.shape == (y_stop - y_start, columns):
            return array[::row_step]
        if array.shape != (rows, columns):
            raise CalibrationShapeMismatchError(
                f"{coeff_type} calibration file {coeff_path} is pre-baked for shape {array.shape}, but this "
                f"capture's AOI/bin_factor implies {(rows, columns)}.")
        return array
    return array[y_start:y_stop:row_step, x_start:x_stop].reshape(rows, -1, bin_factor).mean(axis=2).reshape(rows, -1)


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