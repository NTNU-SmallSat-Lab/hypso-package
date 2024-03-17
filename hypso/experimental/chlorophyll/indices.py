import numpy as np
from typing import Union, Tuple


def TBVI(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Two Band Vegetation Index

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    return (wl_1 - wl_2) / (wl_1 + wl_2)


def BR(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Band Ratio Feature

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    return wl_1 / wl_2


def FBM(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Four Band Model

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    return (wl_1 - wl_2) / (wl_3 + wl_4)


def SINGLE(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Single Band (No features purely reflectance)

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    return wl_1


def BDRATIO(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Band Difference Ratio

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]
    return (wl_1 - wl_2) / (wl_3 - wl_4)


def BR_LOG(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Log of Band Ratio

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    return np.log(wl_1 / wl_2)


def BDIFF(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Band Difference

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    return wl_1 - wl_2


def BDIFF_LOG(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Log of Band Difference

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    return np.log(wl_1 - wl_2)


def BR_DIFF(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Band Ratio Difference

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]
    return (wl_1 / wl_2) - (wl_3 / wl_4)


def BR_DIFF_LOG(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Log of Band Ratio Difference

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    return np.log((wl_1 / wl_2) - (wl_3 / wl_4))


def TBM(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Three Band Model

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return wl_3 * ((1 / wl_1) - (1 / wl_2))


def EXG(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Excess Green Index

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return (2 * wl_1) - wl_2 - wl_3


def GLI(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Green Leaf Index

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return ((2 * wl_1) - wl_2 - wl_3) / ((2 * wl_1) + wl_2 + wl_3)


def OCVI(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Optimized Chlorophyll Vegetation Index

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return (wl_1 / wl_2) * ((wl_3 / wl_2) ** 0.64)


def EVI(refl: np.ndarray) -> Union[np.ndarray, float]:
    """
    Enhanced Vegetation Index

    :param refl: Numpy array with reflectances per column

    :return: Features calculated either a single value or a 1D array
    """
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    L = 1
    c1 = 6
    c2 = 7.5
    return wl_1 * (wl_2 - wl_3) / (wl_2 + (c1 * wl_3) - (c2 * wl_4) + L)
