import numpy as np
import xarray as xr

from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler


def resample_dataarray_bilinear(area_def: AreaDefinition, 
                        data: xr.DataArray,
                        latitudes: np.ndarray,
                        longitudes: np.ndarray,
                        dims_2d: list = ['y', 'x'],
                        dims_3d: list = ['y', 'x', 'band'],
                        radius_of_influence: float = 50000
                        ) -> xr.DataArray:

    latitudes_xr = xr.DataArray(latitudes, dims=dims_2d)
    longitudes_xr = xr.DataArray(longitudes, dims=dims_2d)

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)

    brs = XArrayBilinearResampler(source_geo_def=swath_def, target_geo_def=area_def, radius_of_influence=radius_of_influence)

    # Calculate bilinear neighbour info and generate pre-computed resampling LUTs
    brs.get_bil_info()

    if data.ndim == 2:
        resampled_data = brs.resample(data=data[:,:], fill_value=np.nan)

    elif data.ndim == 3:

        num_bands = data.shape[2]

        resampled_data = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
        resampled_data = xr.DataArray(resampled_data, dims=dims_3d)
        resampled_data.attrs.update(data.attrs)

        # TODO: make more effient by replacing the for loop and using deepcopy or list to assemble datacube
        for band in range(0,num_bands):
            
            # Resample using pre-computed resampling LUTs
            resampled_data[:,:,band] = brs.get_sample_from_bil_info(data=data[:,:,band], 
                                                                    fill_value=np.nan, 
                                                                    output_shape=area_def.shape)

    else:
        return None
    
    return resampled_data









def resample_dataarray_kd_tree_nearest(area_def: AreaDefinition, 
                        data: xr.DataArray,
                        latitudes: np.ndarray,
                        longitudes: np.ndarray,
                        dims_2d: list = ['y', 'x'],
                        dims_3d: list = ['y', 'x', 'band'],
                        radius_of_influence: float = 500
                        ) -> xr.DataArray:

    latitudes_xr = xr.DataArray(latitudes, dims=dims_2d)
    longitudes_xr = xr.DataArray(longitudes, dims=dims_2d)

    swath_def = SwathDefinition(lons=longitudes_xr, lats=latitudes_xr)

    kdtn = KDTreeNearestXarrayResampler(source_geo_def=swath_def, target_geo_def=area_def)


    if data.ndim == 2:
        resampled_data = kdtn.resample(data=data[:,:], fill_value=np.nan, radius_of_influence=radius_of_influence)

    elif data.ndim == 3:

        num_bands = data.shape[2]

        resampled_data = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
        resampled_data = xr.DataArray(resampled_data, dims=dims_3d)
        resampled_data.attrs.update(data.attrs)

        for band in range(0,num_bands):
            
            resampled_data[:,:,band] = kdtn.resample(data=data[:,:,band], fill_value=np.nan, radius_of_influence=radius_of_influence)


    else:
        return None
    
    return resampled_data

