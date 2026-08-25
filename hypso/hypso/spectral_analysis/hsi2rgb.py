"""HSI-cube-to-RGB rendering via CIE color matching functions (Magnusson et
al., IGARSS 2020 - see the docstring's citation). Moved here from the old
utils/utils_file.py grab-bag: rendering a color-accurate RGB from spectra is
spectral analysis, not file utilities. Currently has no in-package callers -
kept (rather than deleted with the rest of the grab-bag) because it is the
natural comparison tool for the planned HYPSO-2 RGB camera support
(REFACTOR_PROGRESS.md TODO): a rendered-from-HSI RGB to hold up against the
real RGB camera image. The D_illuminants.mat data file lives in this
package's data/ directory."""
from bisect import bisect
from importlib.resources import files
from typing import Literal

import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator


def HSI2RGB(wY, HSI, d: Literal[50, 55, 65, 75] = 65, threshold=0.002) -> np.ndarray:
    """
    Calculate a colore accurate RGB image with HSI values. Original function used for the paper
    "Creating RGB Images from Hyperspectral Images Using a Color Matching Function" by
    M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson, H. Deborah and J. R. Sveinsson,

    If you use this method, please cite the following paper:
    @INPROCEEDINGS{hsi2rgb,
    author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson}
    and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}},
    booktitle={IEEE International Geoscience and Remote Sensing Symposium},
    title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function},
    year={2020}, volume={}, number={}, pages={}}

    Paper is available at
    https://www.researchgate.net/profile/Jakob_Sigurdsson

    :param wY: Wavelengths in nm
    :param HSI: Hyperspectral Image as a (#pixels x #bands) matrix,
    :param d: Determines the illuminant used, if in doubt use d65. Values of  50, 55, 65, 75
    :param threshold: True if RGB thesholding should be done to increase contrast

    :return: RGB Image array
    """

    (ydim, xdim, zdim) = HSI.shape
    HSI = np.reshape(HSI, [-1, zdim]) / np.nanmax(HSI)

    # Load reference illuminant
    illuminant_path = files('hypso.spectral_analysis').joinpath('data/D_illuminants.mat')
    D = spio.loadmat(str(illuminant_path))
    w = D['wxyz'][:, 0]
    x = D['wxyz'][:, 1]
    y = D['wxyz'][:, 2]
    z = D['wxyz'][:, 3]
    D = D['D']

    i = {50: 2,
         55: 3,
         65: 1,
         75: 4}
    wI = D[:, 0]
    I = D[:, i[d]]

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(wY)  # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w, x, extrapolate=True)(wY)  # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w, y, extrapolate=True)(wY)  # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w, z, extrapolate=True)(wY)  # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i = bisect(wY, 780)
    HSI = HSI[:, 0:i] / HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute k
    k = 1 / np.trapezoid(y * I, wY)

    # Compute X,Y & Z for image
    X = k * np.trapezoid(HSI @ np.diag(I * x), wY, axis=1)
    Z = k * np.trapezoid(HSI @ np.diag(I * z), wY, axis=1)
    Y = k * np.trapezoid(HSI @ np.diag(I * y), wY, axis=1)

    XYZ = np.array([X, Y, Z])

    # Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    sRGB = M @ XYZ

    # Gamma correction
    """Convert sRGB values to physically linear ones. The transformation is
           uniform in RGB, so *srgb* can be of any shape.

           *srgb* values should range between 0 and 1, inclusively.

        """
    gamma = ((sRGB + 0.055) / 1.055) ** 2.4
    scale = sRGB / 12.92
    sRGB = np.where(sRGB > 0.04045, gamma, scale)

    # gamma_map = sRGB > 0.0031308
    # sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
    # sRGB[np.invert(gamma_map)] = 12.92 * sRGB[np.invert(gamma_map)]

    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB[sRGB > 1] = 1
    sRGB[sRGB < 0] = 0

    if threshold:
        for idx in range(3):
            y = sRGB[idx, :]
            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            th = b[0]
            i = a < threshold
            if i.any():
                th = b[i][-1]
            y = y - th
            y[y < 0] = 0

            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            i = a > 1 - threshold
            th = b[i][0]
            y[y > th] = th
            y = y / th
            sRGB[idx, :] = y

    R = np.reshape(sRGB[0, :], [ydim, xdim])
    G = np.reshape(sRGB[1, :], [ydim, xdim])
    B = np.reshape(sRGB[2, :], [ydim, xdim])

    return np.dstack((R, G, B))
