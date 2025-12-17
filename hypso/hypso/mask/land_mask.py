import numpy as np
from global_land_mask import globe
from hypso.classification import ndwi_watermask, threshold_watermask





'''
# Land mask functions




# Public land mask methods

# TODO
def load_land_mask(self, path: str) -> None:

    return None

def generate_land_mask(self, land_mask_name: LAND_MASK_PRODUCTS = DEFAULT_LAND_MASK_PRODUCT, **kwargs) -> None:

    self._run_land_mask(land_mask_name=land_mask_name, **kwargs)

    return None



# TODO
def write_land_mask(self, path: str) -> None:

    return None




def _run_land_mask(self, land_mask_name: str="global", **kwargs) -> None:

    land_mask_name = land_mask_name.lower()

    match land_mask_name:
        case "global":
            self.land_mask = self._run_global_land_mask(**kwargs)
        case "ndwi":
            self.land_mask = self._run_ndwi_land_mask(**kwargs)
        case "threshold":
            self.land_mask = self._run_threshold_land_mask(**kwargs)

        case _:

            print("[WARNING] No such land mask supported!")
            return None

    return None

def _run_global_land_mask(self) -> np.ndarray:

    if self.VERBOSE:
        print("[INFO] Running global land mask generation...")

    land_mask = run_global_land_mask(spatial_dimensions=self.spatial_dimensions,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes
                                    )
    
    land_mask = self._format_land_mask_dataarray(land_mask)
    land_mask.attrs['method'] = "global"

    return land_mask

def _run_ndwi_land_mask(self) -> np.ndarray:

    if self.VERBOSE:
        print("[INFO] Running NDWI land mask generation...")

    cube = self.l1b_cube.to_numpy()

    land_mask = run_ndwi_land_mask(cube=cube, 
                                    wavelengths=self.wavelengths,
                                    verbose=self.VERBOSE)

    land_mask = self._format_land_mask_dataarray(land_mask)
    land_mask.attrs['method'] = "ndwi"

    return land_mask

def _run_threshold_land_mask(self) -> xr.DataArray:

    if self.VERBOSE:
        print("[INFO] Running threshold land mask generation...")

    cube = self.l1b_cube.to_numpy()

    land_mask = run_threshold_land_mask(cube=cube,
                                        wavelengths=self.wavelengths,
                                        verbose=self.VERBOSE)
    
    land_mask = self._format_land_mask_dataarray(land_mask)
    land_mask.attrs['method'] = "threshold"

    return land_mask
    
def _format_land_mask(self, land_mask: Union[np.ndarray, xr.DataArray]) -> None:

    land_mask_attributes = {
                            'description': "Land mask"
                            }
    
    v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

    data = v.validate(data=land_mask)
    data = data.assign_attrs(land_mask_attributes)

    return data
'''


















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
    