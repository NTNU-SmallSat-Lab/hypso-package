
from ..hypso1 import Hypso1
from ..hypso2 import Hypso2

#import numpy as np
import xarray as xr
from typing import Union
#from ..DataArrayDict import DataArrayDict

from pyresample.geometry import SwathDefinition, AreaDefinition
#from pyresample.bilinear.xarr import XArrayBilinearResampler 
#from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler


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
    
    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    return resampled_data, resampled_latitudes, resampled_longitudes


def resample_l1b_cube(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:

    datacube = satobj.l1b_cube
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
    
    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    return resampled_data, resampled_latitudes, resampled_longitudes


def resample_l1c_cube(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:

    datacube = satobj.l1c_cube
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
    
    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    return resampled_data, resampled_latitudes, resampled_longitudes


def resample_l1d_cube(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:

    datacube = satobj.l1d_cube
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
    
    resampled_longitudes, resampled_latitudes = area_def.get_lonlats()

    return resampled_data, resampled_latitudes, resampled_longitudes


# TODO
def resample_products(satobj: Union[Hypso1, Hypso2], 
                      area_def: Union[SwathDefinition, AreaDefinition],
                      use_indirect_georef: bool = False
                      ) -> xr.DataArray:

    print('[ERROR] This function has not yet been implemented.')

    '''
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
    
    return None
    