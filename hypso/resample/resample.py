
'''

# TODO: make more effient by replacing the for loop and using deepcopy or list to assemble datacube
def _resample_dataarray(self, area_def, data: xr.DataArray) -> xr.DataArray:

    swath_def = self._generate_swath_definition()

    brs = XArrayBilinearResampler(source_geo_def=swath_def, target_geo_def=area_def, radius_of_influence=50000)
    #brs = KDTreeNearestXarrayResampler(source_geo_def=swath_def, target_geo_def=area_def)


    # Calculate bilinear neighbour info and generate pre-computed resampling LUTs
    brs.get_bil_info()

    if data.ndim == 2:
        resampled_data = brs.resample(data=data[:,:], fill_value=np.nan)

    elif data.ndim == 3:

        num_bands = data.shape[2]

        resampled_data = np.zeros((area_def.shape[0], area_def.shape[1], num_bands))
        resampled_data = xr.DataArray(resampled_data, dims=self.dim_names_3d)
        resampled_data.attrs.update(data.attrs)

        for band in range(0,num_bands):
            
            # Resample using pre-computed resampling LUTs
            resampled_data[:,:,band] = brs.get_sample_from_bil_info(data=data[:,:,band], 
                                                                    fill_value=np.nan, 
                                                                    output_shape=area_def.shape)

            #resampled_data[:,:,band] = brs.resample(data=data[:,:,band], fill_value=np.nan)

    else:
        return None
    
    return resampled_data

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