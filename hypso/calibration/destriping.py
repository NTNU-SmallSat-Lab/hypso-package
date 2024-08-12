import numpy as np

def run_destriping_correction(cube: np.ndarray, 
                              destriping_coeffs: np.ndarray) -> np.ndarray:
    """
    Apply destriping correction matrix.

    :param cube: 3-channel spectral cube
    :param destriping_coeffs: Dictionary containing the 2D coefficients for destriping

    :return: 3-channel array for destriping correction
    """

    #print("Destriping Correction Ongoing")

    # print(destriping_correction_matrix.shape)
    # print(cube.shape)
    # cube_destriped = copy.deepcopy(cube)
    # cube_destriped[:, 1:] -= destriping_correction_matrix[:-1]

    # From https://github.com/NTNU-SmallSat-Lab/cal-char-corr/
    #cube_destriped = copy.deepcopy(cube)
    #cube_destriped[:,1:] -= destriping_coeffs[:-1]
    #return cube_destriped

    # From previous hypso-package
    cube_destriped= np.multiply(cube, destriping_coeffs)
    return cube_destriped
