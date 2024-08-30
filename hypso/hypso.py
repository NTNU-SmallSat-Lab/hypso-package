from pathlib import Path
from typing import Union
import xarray as xr
from .DataArrayValidator import DataArrayValidator
import numpy as np

class Hypso:

    def __init__(self, path: Union[str, Path] = None):

        """
        Initialization of HYPSO Class.

        :param path: Absolute path to NetCDF file

        """


        self.path = Path(path).absolute()

        # Initialize platform and sensor names
        self.platform = None
        self.sensor = None

        # Initialize capture name and target
        self.capture_name = None
        self.capture_target = None

        # Initialize directory and file info
        self.capture_dir = None
        self.parent_dir = None
        self.l1a_nc_file = None
        self.l1b_nc_file = None
        self.l2a_nc_file = None


        # Initialize datacubes
        #self.l1a_cube = None
        #self.l1b_cube = None
        #self.l2a_cube = None
        self._l1a_cube = None
        self._l1b_cube = None
        self._l2a_cube = None

        # Initialize top of atmpshere (TOA) reflectance
        self.toa_reflectance = None

        # Initialize metadata dictionaries
        self.capture_config = {}
        self.timing = {}
        self.adcs = {}

        # Initialize timing info
        self.capture_datetime = None
        self.start_timestamp_capture = None
        self.end_timestamp_capture = None
        self.start_timestamp_adcs = None
        self.end_timestamp_adcs = None
        self.unixtime = None
        self.iso_time = None


        # Initialize dimensions
        self.capture_type = None
        self.spatial_dimensions = (956, 684)  # 1092 x variable
        self.standard_dimensions = {
            "nominal": 956,  # Along frame_count
            "wide": 1092  # Along image_height (row_count)
        }

        self.x_start = None
        self.x_stop = None
        self.y_start = None
        self.y_stop = None
        self.bin_factor = 8
        self.row_count = None
        self.frame_count = None
        self.column_count = None
        self.image_height = None
        self.image_width = None
        self.im_size = None
        self.bands = None
        self.lines = None
        self.samples = None

        # Misc metadata
        self.background_value = None
        self.exposure = None

        # Initialize georeferencing info
        self.projection_metadata = None
        self.latitudes = None
        self.longitudes = None
        self.datacube_flipped = False
        self.along_track_gsd = None
        self.across_track_gsd = None
        self.along_track_mean_gsd = None
        self.across_track_mean_gsd = None
        self.resolution = None
        self.bbox = None

       # Initialize calibration file paths:
        self.rad_coeff_file = None
        self.smile_coeff_file = None
        self.destriping_coeff_file = None
        self.spectral_coeff_file = None

        # Initialize calibration coefficients
        self.rad_coeffs = None
        self.smile_coeffs = None
        self.destriping_coeffs = None
        self.spectral_coeffs = None

        # Initialize wavelengths
        self.wavelengths = None
        self.wavelengths_units = r'$nm$'

        # Initialize spectral response function
        self.srf = None

        # Initilize land mask dict
        #self.land_mask = None
        self._land_mask = None

        # Initilize cloud mask dict
        #self.cloud_mask = None
        self._cloud_mask = None

        # Intialize unified mask
        #self.unified_mask = None
        self._unified_mask = None

        # Initialize chlorophyll estimates dict
        #self.chl = {}

        # Initialize products dict
        #self.products = {}

        # Initialize ADCS data
        self.adcs = None
        self.adcs_pos_df = None
        self.adcs_quat_df = None

        # Initialize geometry data
        self.framepose_df = None
        self.wkt_linestring_footprint = None
        self.prj_file_contents = None
        self.local_angles = None
        self.geometric_meta_info = None
        self.solar_zenith_angles = None
        self.solar_azimuth_angles = None
        self.sat_zenith_angles = None
        self.sat_azimuth_angles = None
        self.latitudes_original = None
        self.longitudes_original = None

        # Other
        self.dim_names_3d = ["y", "x", "band"]
        self.dim_names_2d = ["y", "x"]

        # DEBUG
        self.DEBUG = False
        self.VERBOSE = False
    

    def _update_dataarray_attrs(self, data: xr.DataArray, attrs: dict) -> xr.DataArray:

        for key, value in attrs.items():
            if key not in data.attrs:
                data.attrs[key] = value

        return data

    def _format_l1a_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1a",
                      'units': "a.u.",
                      'description': "Raw sensor value"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data
    


    def _format_l1b_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1b",
                      'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
                      'description': "Radiance (L)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data




    def _format_l2a_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L2a",
                      'units': r"sr^{-1}",
                      'description': "Reflectance (Rrs)",
                      'correction': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_land_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {
                      'description': "Land mask",
                      'method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_cloud_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {
                      'description': "Cloud mask",
                      'method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data




    def _format_unified_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {
                      'description': "Unified mask",
                      'land_mask_method': None,
                      'cloud_mask_method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data



    @property
    def l1a_cube(self):
        return self._l1a_cube   

    @l1a_cube.setter
    def l1a_cube(self, value):
        self._l1a_cube = self._format_l1a_dataarray(value)


    @property
    def l1b_cube(self):
        return self._l1b_cube   

    @l1b_cube.setter
    def l1b_cube(self, value):
        self._l1b_cube = self._format_l1b_dataarray(value)

    @property
    def l2a_cube(self):
        return self._l2a_cube   

    @l2a_cube.setter
    def l2a_cube(self, value):
        self._l2a_cube = self._format_l2a_dataarray(value)


    @property
    def land_mask(self):
        return self._land_mask 

    @land_mask.setter
    def land_mask(self, value):
        self._land_mask = self._format_land_mask_dataarray(value)
        


    @property
    def cloud_mask(self):
        return self._cloud_mask   

    @cloud_mask.setter
    def cloud_mask(self, value):
        self._cloud_mask = self._format_cloud_mask_dataarray(value)
        


    @property
    def unified_mask(self):

        if self.land_mask is None and self.cloud_mask is None:
            return None
        
        elif self.land_mask is None:

            unified_mask = self.cloud_mask.to_numpy()
            self._unified_mask = self._format_unified_mask_dataarray(unified_mask)
            self._unified_mask.attrs['cloud_mask_method'] = self.cloud_mask.attrs['method']

        elif self.cloud_mask is None:
            
            unified_mask = self.land_mask.to_numpy()
            self._unified_mask = self._format_unified_mask_dataarray(unified_mask)
            self._unified_mask.attrs['land_mask_method'] = self.land_mask.attrs['method']
        
        else:
            unified_mask = self.land_mask.to_numpy() | self.cloud_mask.to_numpy()
            self._unified_mask = self._format_unified_mask_dataarray(unified_mask)
            self._unified_mask.attrs['land_mask_method'] = self.land_mask.attrs['method']
            self._unified_mask.attrs['cloud_mask_method'] = self.cloud_mask.attrs['method']

        return self._unified_mask  

    @unified_mask.setter
    def unified_mask(self, value):
        self._unified_mask = self._format_unified_mask_dataarray(value)

