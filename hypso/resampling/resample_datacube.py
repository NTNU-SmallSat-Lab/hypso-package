
from typing import Union

from ..DataArrayDict import DataArrayDict

from ..hypso import Hypso
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2


import numpy as np
import xarray as xr


from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler


from .resamplers import resample_dataarray_bilinear, \
                        resample_dataarray_kd_tree_nearest



def resample_l1a_cube(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:

    datacube = satobj.l1a_cube
    resolution = satobj.resolution

    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  


    resampled_data = resample_dataarray_kd_tree_nearest(area_def = area_def, 
                                                        data = datacube,
                                                        latitudes = latitudes,
                                                        longitudes = longitudes,
                                                        radius_of_influence=resolution)
    
    return resampled_data








# TODO
def resample_products(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:


    products = satobj.products
    resolution = satobj.resolution
    dim_names = satobj.dim_names_2d


    if use_indirect_georef:
        latitudes = satobj.latitudes_indirect
        longitudes = satobj.longitudes_indirect
        resolution = satobj.resolution_indirect
    else:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        resolution = satobj.resolution  


    resampled_products = DataArrayDict(dims_shape=area_def.shape, 
                                    attributes=products.attributes, 
                                    dims_names=dim_names,
                                    num_dims=2
                                    )

    for key, product in products.items():

        resampled_data = resample_dataarray_kd_tree_nearest(area_def = area_def, 
                                                            data = products,
                                                            latitudes = latitudes,
                                                            longitudes = longitudes,
                                                            radius_of_influence=resolution)

        resampled_products[key] = resampled_data

    return resampled_products





'''


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



'''