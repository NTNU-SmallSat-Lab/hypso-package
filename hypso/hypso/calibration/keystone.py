"""Keystone correction from a detector distortion model (groundwork, unreviewed).

The model (Lawrence 2003, as used by the HYPSO calibration team) maps each
full-frame detector pixel - ``x`` spectral column, ``y`` spatial row - to where
that pixel's signal really is, with second-order polynomials over the terms
``[1, x, y, xy, x**2, y**2]``: ``A_hat`` for the spectral position (smile),
``B_hat`` for the spatial position (keystone). The calibration team's
``transform()`` corrects a full frame by resampling it at those positions.

Here only the keystone part is applied, to a cropped and binned cube: each
band (a group of ``bin_factor`` full-frame columns, evaluated at its centre) is
resampled along the spatial axis at ``B(x, y)``; smile has its own correction
matrix. Off by default - ``calibrate(keystone=True)`` - until reviewed
(REFACTOR_PROGRESS.md TODO; HYPSO-2's model is ``hypso2_calibration``'s
optional ``keystone`` V1).
"""
import numpy as np
from scipy.ndimage import map_coordinates


def model_terms(x, y) -> np.ndarray:
    """The model's polynomial terms ``[1, x, y, xy, x**2, y**2]`` for arrays x, y."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=-1)


def keystone_positions(b_hat, n_samples: int, n_bands: int, x_start: int, y_start: int,
                       bin_factor: int) -> np.ndarray:
    """Source sample position, in the capture's own sample coordinates, of each
    ``(sample, band)`` pixel: ``B(x, y) - y_start``, with ``x`` the full-frame
    column at the band's centre and ``y`` the full-frame row of the sample.

    Returns
    -------
    numpy.ndarray
        ``(n_samples, n_bands)``.
    """
    x = x_start + np.arange(n_bands) * bin_factor + (bin_factor - 1) / 2
    y = y_start + np.arange(n_samples)
    xx, yy = np.meshgrid(x, y)
    return model_terms(xx, yy) @ np.asarray(b_hat, dtype=np.float64) - y_start


def correct_keystone(cube: np.ndarray, b_hat, x_start: int, y_start: int, bin_factor: int,
                     order: int = 1) -> np.ndarray:
    """Keystone-correct a ``(lines, samples, bands)`` cube.

    Parameters
    ----------
    cube : numpy.ndarray
        Cropped and binned cube.
    b_hat : array_like
        The model's 6 keystone (spatial) coefficients.
    x_start, y_start : int
        Full-frame column and row of the capture's first band column and
        first sample (the AOI offsets).
    bin_factor : int
        Spectral binning factor.
    order : int, default 1
        Spline order of the resampling (1 = linear, no ringing; the
        calibration team's ``transform()`` used scipy's default, 3).

    Returns
    -------
    numpy.ndarray
        The corrected cube, same shape and dtype; samples whose source falls
        outside the capture are 0, as ``map_coordinates`` fills them.
    """
    lines, samples, bands = cube.shape
    positions = keystone_positions(b_hat, samples, bands, x_start, y_start, bin_factor)
    out = np.empty_like(cube)
    line_index = np.arange(lines, dtype=np.float64)[:, None]
    for band in range(bands):
        coords = np.broadcast_arrays(line_index, positions[None, :, band])
        out[:, :, band] = map_coordinates(cube[:, :, band], coords, order=order, mode="constant", cval=0.0)
    return out
