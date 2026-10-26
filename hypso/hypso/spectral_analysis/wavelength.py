from typing import Union

import numpy as np


def get_closest_wavelength_index(product, wavelength: Union[float, int]) -> int:
    """Index of the band whose centre is nearest to `wavelength` (nm).
    `product`: a hypso.Product or an array of band wavelengths."""
    wavelengths = np.asarray(getattr(product, "wavelengths", product), dtype=float)
    return int(np.argmin(np.abs(wavelengths - wavelength)))
