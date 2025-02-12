# File Manipulation
import numpy as np
import xarray as xr


'''
# Cloud mask functions
    



# Public cloud mask methods

# TODO
def load_cloud_mask(self, path: str) -> None:

    return None

def generate_cloud_mask(self, cloud_mask_name: CLOUD_MASK_PRODUCTS = DEFAULT_CLOUD_MASK_PRODUCT, **kwargs):

    self._run_cloud_mask(cloud_mask_name=cloud_mask_name, **kwargs)

    return None

# TODO
def write_cloud_mask(self, path: str) -> None:

    return None



def _run_cloud_mask(self, cloud_mask_name: str="quantile_threshold", **kwargs) -> None:

    cloud_mask_name = cloud_mask_name.lower()

    match cloud_mask_name:
        case "quantile_threshold":

            if self.VERBOSE:
                print("[INFO] Running quantile threshold cloud mask generation...")

            self.cloud_mask = self._run_quantile_threshold_cloud_mask(**kwargs)


        case "saturated_pixel":

            if self.VERBOSE:
                print("[INFO] Running saturated pixel cloud mask generation...")

            self.cloud_mask = self._run_saturated_pixel_cloud_mask(**kwargs)


        case _:
            print("[WARNING] No such cloud mask supported!")
            return None

    return None

def _run_quantile_threshold_cloud_mask(self, quantile: float = 0.075) -> None:

    cloud_mask = run_quantile_threshold_cloud_mask(cube=self.l1b_cube,
                                                    quantile=quantile)

    cloud_mask = self._format_cloud_mask_dataarray(cloud_mask)
    cloud_mask.attrs['method'] = "quantile threshold"
    cloud_mask.attrs['quantile'] = quantile

    return cloud_mask 

def _run_saturated_pixel_cloud_mask(self, threshold: float = 35000):

    sat = np.max(self.l1a_cube.to_numpy(), axis=-1) > threshold

    cloud_mask = self._format_cloud_mask_dataarray(sat)
    cloud_mask.attrs['method'] = "saturated pixel"
    cloud_mask.attrs['threshold'] = threshold

    return cloud_mask 
'''



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