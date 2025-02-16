
from typing import Union

from ..hypso import Hypso
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2


import numpy as np
import xarray as xr


from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler







# TODO: make more effient by replacing the for loop and using deepcopy or list to assemble datacube
def resample_dataarray(area_def: AreaDefinition, 
                        data: xr.DataArray,
                        latitudes: np.ndarray,
                        longitudes: np.ndarray,
                        dims_2d: list = ['y', 'x'],
                        dims_3d: list = ['y', 'x', 'band']
                        ) -> xr.DataArray:

    #dims = satobj.dim_names_2d

    latitudes_xr = xr.DataArray(latitudes, dims=dims_2d)
    longitudes_xr = xr.DataArray(longitudes, dims=dims_2d)

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)

    brs = XArrayBilinearResampler(source_geo_def=swath_def, target_geo_def=area_def, radius_of_influence=50000)
    #brs = KDTreeNearestXarrayResampler(source_geo_def=swath_def, target_geo_def=area_def)


    # Calculate bilinear neighbour info and generate pre-computed resampling LUTs
    brs.get_bil_info()

    if data.ndim == 2:
        resampled_data = brs.resample(data=data[:,:], fill_value=np.nan)

    elif data.ndim == 3:

        num_bands = data.shape[2]

        resampled_data = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
        resampled_data = xr.DataArray(resampled_data, dims=dims_3d)
        resampled_data.attrs.update(data.attrs)

        for band in range(0,num_bands):
            
            # Resample using pre-computed resampling LUTs
            resampled_data[:,:,band] = brs.get_sample_from_bil_info(data=data[:,:,band], 
                                                                    fill_value=np.nan, 
                                                                    output_shape=area_def.shape)

            #resampled_data[:,:,band] = brs.resample(data=data[:,:,band], fill_value=np.nan, radius_of_influence=50000)
            #resampled_data = brs.resample(data=data, fill_value=np.nan, radius_of_influence=50000)

    else:
        return None
    
    return resampled_data



'''

def _generate_swath_definition()
    latitudes_xr = xr.DataArray(latitudes, dims=satobj.dim_names_2d)
    longitudes_xr = xr.DataArray(longitudes, dims=satobj.dim_names_2d)

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)


def resample_l1a_cube(self, area_def) -> xr.DataArray:

    return self._resample_dataarray(area_def=area_def, data=self.l1a_cube)

def resample_l1b_cube(self, area_def) -> xr.DataArray:

    return self._resample_dataarray(area_def=area_def, data=self.l1b_cube)

def resample_l2a_cube(self, area_def) -> xr.DataArray:

    return self._resample_dataarray(area_def=area_def, data=self.l2a_cube)

def resample_toa_reflectance_cube(self, area_def) -> xr.DataArray:

    return self._resample_dataarray(area_def=area_def, data=self.toa_reflectance_cube)

def resample_chlorophyll_estimates(self, area_def) -> xr.DataArray:

    resampled_chl = DataArrayDict(dims_shape=area_def.shape, 
                                    attributes=self.chl.attributes, 
                                    dims_names=self.dim_names_2d,
                                    num_dims=2
                                    )

    for key, chl in self.chl.items():

        resampled_chl[key] = self._resample_dataarray(area_def=area_def, data=chl)

    return resampled_chl

def resample_products(self, area_def) -> xr.DataArray:

    resampled_products = DataArrayDict(dims_shape=area_def.shape, 
                                    attributes=self.products.attributes, 
                                    dims_names=self.dim_names_2d,
                                    num_dims=2
                                    )

    for key, product in self.products.items():

        resampled_products[key] = self._resample_dataarray(area_def=area_def, data=product)

    return resampled_products

'''