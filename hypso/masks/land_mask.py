import numpy as np
from global_land_mask import globe
from hypso.classification import ndwi_watermask, threshold_watermask

def run_global_land_mask(spatial_dimensions: tuple, 
                        latitudes: np.ndarray,
                        longitudes: np.ndarray
                        ) -> np.ndarray:

    land_mask = np.zeros(spatial_dimensions, dtype=bool)

    rows = spatial_dimensions[0]
    cols = spatial_dimensions[1]

    for x in range(0,rows):

        for y in range(0,cols):

            lat = latitudes[x][y]
            lon = longitudes[x][y]

            land_mask[x][y] = globe.is_land(lat, lon)

    return land_mask


def run_ndwi_land_mask(cube: np.ndarray, wavelengths: np.ndarray, verbose=False):

    water_mask = ndwi_watermask(cube=cube, verbose=verbose)

    land_mask = ~water_mask

    return land_mask


def run_threshold_land_mask(cube: np.ndarray, wavelengths: np.ndarray, verbose=False) -> np.ndarray:

    water_mask = threshold_watermask(cube=cube,
                                     wavelengths=wavelengths,
                                     verbose=verbose)
    
    land_mask = ~water_mask

    return land_mask
    