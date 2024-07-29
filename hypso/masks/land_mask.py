import numpy as np
from global_land_mask import globe

def run_global_land_mask(spatial_dimensions: tuple, 
                        latitudes: np.ndarray,
                        longitudes: np.ndarray
                        ) -> np.ndarray:

    land_mask = np.zeros(spatial_dimensions, dtype=bool)

    rows = spatial_dimensions[0]
    cols = spatial_dimensions[0]

    for y in range(0,rows):

        for x in range(0,cols):

            lat = latitudes[y][x]
            lon = longitudes[y][x]

            land_mask[x][y] = globe.is_land(lat, lon)

    return land_mask


def run_ndwi_land_mask():

    pass
    # ndwi_watermask