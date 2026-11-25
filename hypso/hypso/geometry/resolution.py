
import numpy as np

def compute_resolution(along_track_gsd, across_track_gsd) -> float:
    """The coarser of the mean along- and across-track ground sampling
    distances (m) - the capture's resolution; 0 if neither is available."""

    distances = [np.mean(along_track_gsd), 
                 np.mean(across_track_gsd)]

    filtered_distances = [d for d in distances if d is not None]

    try:
        resolution = max(filtered_distances)
    except ValueError:
        resolution = 0

    return resolution

