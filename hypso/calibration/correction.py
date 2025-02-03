import numpy as np
from scipy import interpolate
from pathlib import Path
import copy

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

    :param coeff_path: Coefficient path to read (.csv or .npz)

    :return: 2D array of coefficients
    """
    coefficients = None
    try:

        # Processing should be Float 32
        if coeff_path.suffix == ".npz":
            coefficients = np.load(coeff_path)
            key = list(coefficients.keys())[0]
            coefficients = coefficients[key]

        elif coeff_path.suffix == ".csv":
            coefficients = np.genfromtxt(coeff_path, delimiter=',', dtype="float64")
        else:
            coefficients = None

    except BaseException:
        coefficients = None
        raise ValueError("[ERROR] Could not read coefficients file.")

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