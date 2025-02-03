import numpy as np
import skimage.morphology as morph
import copy

def run_destriping_correction(cube: np.ndarray, 
                              destriping_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply destriping correction matrix.

    :param cube: 3-channel spectral cube
    :param destriping_coeffs: Dictionary containing the 2D coefficients for destriping

    :return: 3-channel array for destriping correction
    """

    # From https://github.com/NTNU-SmallSat-Lab/cal-char-corr/
    #cube_destriped = copy.deepcopy(cube)
    #cube_destriped[:,1:] -= destriping_coeffs[:-1]
    #return cube_destriped

    # From previous hypso-package
    cube_destriped= np.multiply(cube, destriping_coeffs)
    return cube_destriped


def run_destriping_correction_with_computed_matrix(cube: np.ndarray, 
                              destriping_coeffs: np.ndarray) -> np.ndarray:
    ''' Apply destriping correction matrix. '''
    cube_delined = copy.deepcopy(cube)
    cube_delined[:,1:] -= destriping_coeffs[:-1]
    return cube_delined


def get_destriping_correction_matrix(cube, water_mask, plot=False, plot_min_val=-1, plot_max_val=1):
    ''' Use masked ocean cube to create a destriping correction matrix 
    (cumulative correction frame). Must be calibrated with the same calibration
    coefficients (and smile correction or other correction steps) that will be
    used before the destriping correction is used on any other cube.

    Based on Joe's jupyter code.
    '''
    num_frames, image_height, image_width = cube.shape
    # Determine correction
    wm = morph.dilation(water_mask, morph.square(7))
    diff = cube[:,1:,100] - cube[:,:-1,100] 
    wm_t = wm[:,:-1]
    wm_t[wm[:,1:]] = 1 
    diff[wm_t]=0
    diff = np.zeros((num_frames,image_height-1,image_width))
    diff[:] = cube[:,1:,:] - cube[:,:-1,:]
    diff[wm_t]=0
    corrections = np.zeros((image_height,image_width))
    for i in range(0,image_height-1):
        corrections[i,:] = np.median(diff[:,i][~wm_t[:,i]], axis=0)
    corrections[:] -= np.mean(corrections, axis=0)
    cumulative = np.zeros((image_height, image_width))
    cumulative[0] = corrections[0]
    for i in range(1,image_height):
        cumulative[i] = corrections[i] + cumulative[i-1]

    return cumulative