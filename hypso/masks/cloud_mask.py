# File Manipulation
import numpy as np
import xarray as xr


def run_cloud_mask():

    return None

def run_fmask_cloud_mask():

    return None

def run_quantile_threshold_cloud_mask(cube: np.ndarray, quantile: float=0.075) -> xr.DataArray:
    
    cloud_mask = np.sum(cube, axis=2)**2

    #cloud_mask_threshold = 0.075e8
    cloud_mask_threshold = np.quantile(cloud_mask, quantile)

    cloud_mask = np.sum(cube, axis=2)**2 > cloud_mask_threshold

    return cloud_mask