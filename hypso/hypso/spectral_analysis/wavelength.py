from typing import Union
import numpy as np

def get_closest_wavelength_index(satobj, wavelength: Union[float, int]) -> int:

    wavelengths = np.array(satobj.wavelengths)
    differences = np.abs(wavelengths - wavelength)
    closest_index = np.argmin(differences)

    return closest_index