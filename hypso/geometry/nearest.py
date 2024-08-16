import numpy as np
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info

def get_nearest_pixel(self, target_latitude: float, target_longitude: float, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[int, int]:
    """
    Find the nearest pixel in an array of latitude and longitude values given a target latitude and longitude.

    Parameters:
    - latitude: Target latitude
    - longitude: Target longitude
    - latitudes: Array of latitudes from scene
    - longitudes: Array of longitudes from scene

    Returns:
    - (i, j): Indices of the nearest pixel in the swath definition
    """
    # Wrap target coordinates in arrays

    target_latitude = np.array([target_latitude])
    target_longitude = np.array([target_longitude])
    
    source_swath_def = SwathDefinition(lons=longitudes, lats=latitudes)
    target_swath_def = SwathDefinition(lons=target_longitude, lats=target_latitude)

    # Find nearest neighbor info
    valid_input_index, valid_output_index, index_array, distance_array = get_neighbour_info(
        source_swath_def, target_swath_def, radius_of_influence=np.inf, neighbours=1
    )

    if len(valid_input_index) > 0:
        nearest_index = np.unravel_index(index_array[0], source_swath_def.shape)
        return nearest_index
    else:
        return None

# TODO check that this works and does image flip need to apply?
def haversine(self, lat1, lon1, lat2, lon2):
    """
    WARNING: ChatGPT wrote this... ()

    Calculate the great-circle distance between two points 
    on the Earth using the Haversine formula.
    """
    R = 6371.0  # Radius of the Earth in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def get_nearest_pixel_haversine(self, latitude: float, longitude: float, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[int, int]:
    """
    Find the nearest index in 2D latitude and longitude matrices
    to the given target latitude and longitude.

    Parameters:
    - target_lat: Target latitude
    - target_lon: Target longitude

    Returns:
    - (i, j): Tuple of indices in the matrices closest to the target coordinate
    """

    lat_matrix = self.latitudes
    lon_matrix = self.longitudes

    distances = self._haversine(lat_matrix, lon_matrix, latitude, longitude)
    nearest_index = np.unravel_index(np.argmin(distances), distances.shape)
    return nearest_index
