from pathlib import Path
from typing import Union
import xarray as xr
import copy
#from .DataArrayValidator import DataArrayValidator
import numpy as np
from datetime import datetime, timezone
from trollsift import Parser
import sys


from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction


from hypso.geometry import interpolate_at_frame_nc, \
                           direct_georeference, \
                           compute_local_angles, \
                           compute_gsd, \
                           compute_bbox, \
                           compute_resolution

from hypso.georeferencing import Georeferencer, \
                                check_star_tracker_orientation

from hypso.load import load_l1a_nc, \
                        load_l1b_nc, \
                        load_l1c_nc, \
                        load_l1d_nc, \
                        load_ocsmart_h5, \
                        load_acolite_l2r_nc, \
                        load_acolite_l2w_nc

from hypso.reflectance import compute_toa_reflectance

from hypso.utils import find_file

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict

import netCDF4 as nc


class HypsoBase:

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
        self.l1c_nc_file = None
        self.l1d_nc_file = None

        # Initialize datacubes
        self._l1a_cube = None
        self._l1b_cube = None
        self._l1c_cube = None
        self._l1d_cube = None


        # Initialize dimensions
        #self.capture_type = None
        #self.spatial_dimensions = (956, 684)  # 1092 x variable
        #self.standard_dimensions = {
        #    "nominal": 956,  # Along frame_count
        #    "wide": 1092  # Along image_height (row_count)
        #}

        # Initialize masks
        self._land_mask = None
        self._cloud_mask = None

        # Initialize latitude and longitude
        # TODO: store latitude and longitude as xarray
        self.latitudes = None
        self.longitudes = None
        self.latitudes_direct = None
        self.longitudes_direct = None

        # Other
        self.dim_names_3d = ["y", "x", "band"]
        self.dim_names_2d = ["y", "x"]

        # Products dictionary
        self._products = DataArrayDict()

        # Constants
        self.UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here
        self.AVERAGE_FWHM = 3.33 #8.2 
        
        # Atmospheric Correction
        self.ocsmart_dir = None
        self.acolite_dir = None

        # DEBUG
        self.DEBUG = False
        self.VERBOSE = False


        # Level-2 datacubes

        l2_attributes = {'level': "L2",
                    'units': r"sr^{-1}",
                    'description': "Bottom of Atmosphere Reflectance (Rrs)",
                    'l2_variable_name': "rrs"
                    }

        self._l2_cubes = DataArrayDict(attributes=l2_attributes, num_dims=3, key_attribute='correction')


    @property
    def l2_cube(self):

        self._l2_cubes.dim_shape = self.spatial_dimensions
        self._l2_cubes.dim_names = self.dim_names_3d
        self._l2_cubes.num_dims = 3

        return self._l2_cubes   

    @l2_cube.setter
    def l2_cubes(self, value):
        raise AttributeError("[ERROR] Use \"l2_cubes[key] = value\" to set items.")



    

    def _update_dataarray_attrs(self, data: xr.DataArray, attrs: dict) -> xr.DataArray:

        for key, value in attrs.items():
            if key not in data.attrs:
                data.attrs[key] = value

        return data

    def _format_l1a_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1a",
                      'units': "counts",
                      'description': "Digital Number (DN)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data
    
    def _format_l1b_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1b",
                      'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
                      'description': "Top-of-Atmosphere Radiance (Lt)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data

    def _format_l1c_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1c",
                      'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
                      'description': "Top-of-Atmosphere Radiance (Lt)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data

    def _format_l1d_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1d",
                      'units': r"sr^{-1}",
                      'description': "Top-of-Atmosphere Reflectance (Rhot)",
                      'correction': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_land_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {
                      'description': "Land mask",
                      'method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_cloud_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {
                      'description': "Cloud mask",
                      'method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_2d, num_dims=2)

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
    def l1c_cube(self):
        # Return l1b cube since it is the same as the l1c cube
        cube = copy.deepcopy(self._l1b_cube)
        cube.attrs['level'] = 'L1c'
        return cube 


    @l1c_cube.setter
    def l1c_cube(self, value):
        self._l1c_cube = self._format_l1c_dataarray(value)


    @property
    def l1d_cube(self):
        return self._l1d_cube   

    @l1d_cube.setter
    def l1d_cube(self, value):
        self._l1d_cube = self._format_l1d_dataarray(value)


    @property
    def land_mask(self):
        return self._land_mask 

    @land_mask.setter
    def land_mask(self, value):
        if value is not None:
            self._land_mask = self._format_land_mask_dataarray(value)
        else:
            self._land_mask = None


    @property
    def cloud_mask(self):
        return self._cloud_mask   

    @cloud_mask.setter
    def cloud_mask(self, value):
        if value is not None:
            self._cloud_mask = self._format_cloud_mask_dataarray(value)
        else:
            self._cloud_mask = None


    @property
    def masked_l1a_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1a_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1a_cube   
        

    @property
    def masked_l1b_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1b_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1b_cube   


    @property
    def masked_l1c_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1c_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1c_cube   


    @property
    def masked_l1d_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1d_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1d_cube           


    def _unified_mask(self) -> xr.DataArray:
        if self._land_mask is not None and self._cloud_mask is not None:
            unified_mask = self._land_mask | self._cloud_mask
        elif self._land_mask is not None:
            unified_mask = self._land_mask
        elif self._cloud_mask is not None:
            unified_mask = self._cloud_mask
        else:
            return None
        
        return unified_mask


    @property
    def products(self):

        self._products.dim_shape = self.spatial_dimensions
        self._products.dim_names = self.dim_names_2d
        self._products.num_dims = 2

        return self._products   

    @products.setter
    def products(self, value):
        raise AttributeError("[ERROR] Use \"products[key] = value\" to set items.")
























    def _compose_capture_name(self, fields: dict) -> str:


        if hasattr(self, '_use_old_filename_format'):
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}") # Old filename format
        else:
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}") # New filename format

        capture_name = p.compose(fields)

        return capture_name


    def _parse_filename(self, path: str) -> dict:

        path = Path(path).absolute()
        field = None

        try:
            # New filename format
            #aegean_2024-08-22T08-41-46Z-l1a.nc
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        except:
            # Old filename format
            setattr(self, '_use_old_filename_format', True)
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        
        return fields


    def _load_capture_file(self, path: Path) -> None:

        path = Path(path).absolute()



        fields = self._parse_filename(path=path)

        for key, value in fields.items():
            setattr(self, key, value)

        capture_name = self._compose_capture_name(fields=fields)

        self.capture_name = capture_name

        self.capture_dir = Path(path.parent.absolute(), capture_name + "_tmp")
        self.parent_dir = Path(path.parent.absolute())

        self.l1a_nc_file = Path(path.parent, capture_name + "-l1a.nc")
        self.l1b_nc_file = Path(path.parent, capture_name + "-l1b.nc")
        self.l1c_nc_file = Path(path.parent, capture_name + "-l1c.nc")
        self.l1d_nc_file = Path(path.parent, capture_name + "-l1d.nc")

        match fields['product_level']:
            case "l1a":
                if self.VERBOSE: print('[INFO] Loading L1a capture ' + self.capture_name)

                load_func = load_l1a_nc
                cube_name = "l1a_cube"
                
            case "l1b":
                if self.VERBOSE: print('[INFO] Loading L1b capture ' + self.capture_name)

                load_func = load_l1b_nc
                cube_name = "l1b_cube"

            case "l1c":
                if self.VERBOSE: print('[INFO] Loading L1c capture ' + self.capture_name)

                load_func = load_l1c_nc
                cube_name = "l1b_cube" # L1c cube is the same as the L1b cube

            case "l1d":
                if self.VERBOSE: print('[INFO] Loading L1d capture ' + self.capture_name)

                load_func = load_l1d_nc
                cube_name = "l1d_cube"

            case _:
                print("[ERROR] Unsupported product level.")
                return None

        # TODO: find a better method to pass all of this information
        nc_metadata_vars, \
        nc_metadata_attrs, \
        nc_navigation_vars, \
        nc_navigation_attrs, \
        nc_gcp_vars, \
        nc_gcp_attrs, \
        nc_global_metadata, \
        nc_cube_attrs, \
        nc_cube = load_func(nc_file_path=path)

        setattr(self, "nc_adcs_vars", nc_metadata_vars["adcs"])
        setattr(self, "nc_capture_config_vars", nc_metadata_vars["capture_config"])
        setattr(self, "nc_corrections_vars", nc_metadata_vars["corrections"])
        setattr(self, "nc_database_vars", nc_metadata_vars["database"])
        setattr(self, "nc_logfiles_vars", nc_metadata_vars["logfiles"])
        setattr(self, "nc_temperature_vars", nc_metadata_vars["temperature"])
        setattr(self, "nc_timing_vars", nc_metadata_vars["timing"])

        setattr(self, "nc_adcs_attrs", nc_metadata_attrs["adcs"])
        setattr(self, "nc_capture_config_attrs", nc_metadata_attrs["capture_config"])
        setattr(self, "nc_corrections_attrs", nc_metadata_attrs["corrections"])
        setattr(self, "nc_database_attrs", nc_metadata_attrs["database"])
        setattr(self, "nc_logfiles_attrs", nc_metadata_attrs["logfiles"])
        setattr(self, "nc_temperature_attrs", nc_metadata_attrs["temperature"])
        setattr(self, "nc_timing_attrs", nc_metadata_attrs["timing"])
 
        setattr(self, "nc_navigation_vars", nc_navigation_vars)
        setattr(self, "nc_navigation_attrs", nc_navigation_attrs)

        setattr(self, "nc_gcp_vars", nc_gcp_vars)
        setattr(self, "nc_gcp_attrs", nc_gcp_attrs)

        setattr(self, "nc_dimensions", nc_global_metadata["dimensions"])
        setattr(self, "nc_attrs", nc_global_metadata["ncattrs"])

        setattr(self, "nc_cube_attrs", nc_cube_attrs)

        # TODO: pass the dicts returned by load_func to _set_hypso_attributes()
        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_hypso_attributes()
        self._check_capture_type()

        setattr(self, cube_name, nc_cube)
        


        self.ocsmart_l1d_input_nc_file = Path(path.parent, str(self.sensor).upper() + "_" + str(capture_name) + "-l1d.nc")
        self.ocsmart_l2a_output_h5_file = Path(path.parent, str(self.sensor).upper() + "_" + str(capture_name) + "-l1d_L2_OCSMART.h5") #HYPSO2_HSI_aeronetvenice_2025-06-22T10-46-15Z-l1d_L2_OCSMART


        dt = datetime.fromtimestamp(self.unixtime, tz=timezone.utc)
        self.acolite_l2r_output_nc_file =  Path(self.capture_dir, f"{self.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2R.nc")
        self.acolite_l2w_output_nc_file =  Path(self.capture_dir, f"{self.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2W.nc")


        return None


    # TODO: Clean up this function. Use setattr, hasattr, getattr for setting class variables?
    def _set_hypso_attributes(self) -> None:

        # Capture config related attributes
        for attr in self.nc_capture_config_attrs.keys():
            setattr(self, attr, self.nc_capture_config_attrs[attr])
        # FPS has been renamed to framerate. Need to support both since old .nc files may still use FPS
        try:
            self.nc_capture_config_attrs['fps'] = self.nc_capture_config_attrs['framerate']
        except:
            self.nc_capture_config_attrs['framerate'] = self.nc_capture_config_attrs['fps']
            
        self.background_value = 8 * self.nc_capture_config_attrs["bin_factor"]
        self.exposure = self.nc_capture_config_attrs["exposure"] / 1000  # in seconds


        # Capture dimensions attributes
        self.x_start = self.nc_capture_config_attrs["aoi_x"]
        self.x_stop = self.nc_capture_config_attrs["aoi_x"] + self.nc_capture_config_attrs["column_count"]
        self.y_start = self.nc_capture_config_attrs["aoi_y"]
        self.y_stop = self.nc_capture_config_attrs["aoi_y"] + self.nc_capture_config_attrs["row_count"]
        self.bin_factor = self.nc_capture_config_attrs["bin_factor"]
        # Try/except here since not all captures have sample_div
        try:
            self.sample_div = self.nc_capture_config_attrs['sample_div']
        except:
            self.sample_div = 1
        self.row_count = self.nc_capture_config_attrs["row_count"]
        self.frame_count = self.nc_capture_config_attrs["frame_count"]
        self.column_count = self.nc_capture_config_attrs["column_count"]
        self.image_height = int(self.nc_capture_config_attrs["row_count"] / self.sample_div)
        self.image_width = int(self.nc_capture_config_attrs["column_count"] / self.nc_capture_config_attrs["bin_factor"])
        self.im_size = self.image_height * self.image_width
        self.bands = self.image_width
        self.lines = self.nc_capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols
        self.spatial_dimensions = (self.nc_capture_config_attrs["frame_count"], self.image_height)
        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")


        # Calibration related atrributes
        self.rad_coeffs = self.nc_corrections_vars['rad_matrix']

        try:
            self.spectral_coeffs = self.nc_corrections_vars['spec_coeffs']
        except KeyError:
            self.spectral_coeffs = self.nc_corrections_vars['wavelengths']

        if not hasattr(self, 'wavelengths'):
            if ('wavelengths' in self.nc_cube_attrs.keys()):
                self.wavelengths = self.nc_cube_attrs['wavelengths']
            else:
                self.wavelengths = np.array(range(0, self.image_width))

        if not hasattr(self, 'fwhm'):
            if 'fwhm' in self.nc_cube_attrs.keys():
                self.fwhm = self.nc_cube_attrs['fwhm']
            else:
                self.fwhm = [self.AVERAGE_FWHM] * self.bands



        # Navigation atrributes
        for key, value in self.nc_navigation_vars.items():
            if key == 'unixtime':
                continue
            elif key == 'latitude':
                setattr(self, 'latitudes', value)
            elif key == 'longitude':
                setattr(self, 'longitudes', value)
            elif key == 'latitude_indirect':
                setattr(self, 'latitudes_indirect', value)
            elif key == 'longitude_indirect':
                setattr(self, 'longitudes_indirect', value)


            elif key == 'sensor_zenith':
                setattr(self, 'sat_zenith_angles', value)
            elif key == 'sensor_azimuth':
                setattr(self, 'sat_azimuth_angles', value)
            elif key == 'sensor_zenith_indirect':
                setattr(self, 'sat_zenith_angles_indirect', value)
            elif key == 'sensor_azimuth_indirect':
                setattr(self, 'sat_azimuth_angles_indirect', value)


            elif key == 'solar_zenith':
                setattr(self, 'solar_zenith_angles', value)
            elif key == 'solar_azimuth':
                setattr(self, 'solar_azimuth_angles', value)
            elif key == 'solar_zenith_indirect':
                setattr(self, 'solar_zenith_angles_indirect', value)
            elif key == 'solar_azimuth_indirect':
                setattr(self, 'solar_azimuth_angles_indirect', value)


            elif key == 'relative_azimuth':
                setattr(self, 'relative_azimuth_angles', value)
            elif key == 'relative_azimuth_indirect':
                setattr(self, 'relative_azimuth_angles_indirect', value)

            else:
                setattr(self, key, value)



        # Capture timing attributes
        try:
            self.start_timestamp_capture = int(self.timing['capture_start_unix']) + self.UNIX_TIME_OFFSET
        except:
            try:
                datestring = self.nc_attrs['date_aquired']
            except:
                datestring = self.nc_attrs['timestamp_acquired']


            try:
                dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            except ValueError:
                dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%f%zZ').replace(tzinfo=timezone.utc)

            self.start_timestamp_capture = dt.timestamp()

        #self.start_timestamp_capture = int(self.nc_timing_attrs['capture_start_unix']) + self.UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if framerate and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.nc_capture_config_attrs["frame_count"] / self.nc_capture_config_attrs["framerate"] + self.nc_capture_config_attrs["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] Framerate or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.nc_capture_config_attrs["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0
        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end
        self.unixtime = self.start_timestamp_capture

        #self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()
        self.iso_time = datetime.fromtimestamp(self.unixtime, tz=timezone.utc).isoformat()

        return None


    def _check_capture_type(self):

        #self.spatial_dimensions = (956, 684)  # 1092 x variable
        #self.standard_dimensions = {
        #    "nominal": 956,  # Along frame_count
        #    "wide": 1092  # Along image_height (row_count)
        #}

        if self.nc_capture_config_attrs["frame_count"] == 956:
        #if self.nc_capture_config_attrs["frame_count"] == self.standard_dimensions["nominal"]:
            self.capture_type = "nominal"

        elif self.image_height == 1092:
        #elif self.image_height == self.standard_dimensions["wide"]:
            self.capture_type = "wide"
        else:
            # EXPERIMENTAL_FEATURES
            if self.VERBOSE:
                print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
            self.capture_type = "custom"

        if self.VERBOSE:
            print(f"[INFO] Capture capture type: {self.capture_type}")


    def _run_calibration(self, 
                         radiometric: bool = True,
                         smile: bool = True,
                         destripe: bool = True,
                         spectral: bool = True,
                         set_coeffs: bool = True,
                         **kwargs) -> np.ndarray:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')

        # TODO: move this function call
        if set_coeffs:
            self._set_calibration_coeff_files()

        self._load_calibration_coeff_files()

        calibrated_cube = self.l1a_cube.to_numpy()

        if self.rad_coeffs is not None:
            if radiometric:

                if self.VERBOSE:
                    print("[INFO] Running radiometric calibration...")

                calibrated_cube = run_radiometric_calibration(cube=calibrated_cube, 
                                                background_value=self.background_value,
                                                exp=self.exposure,
                                                image_height=self.image_height,
                                                image_width=self.image_width,
                                                frame_count=self.frame_count,
                                                bin_factor=self.bin_factor,
                                                rad_coeffs=self.rad_coeffs
                                                )

        if self.smile_coeffs is not None:
            if smile:

                if self.VERBOSE:
                    print("[INFO] Running smile correction...")

                calibrated_cube = run_smile_correction(cube=calibrated_cube, 
                                                smile_coeffs=self.smile_coeffs)

        if self.destriping_coeffs is not None:
            if destripe:

                if self.VERBOSE:
                    print("[INFO] Running destriping correction...")

                calibrated_cube = run_destriping_correction(cube=calibrated_cube, 
                                                    destriping_coeffs=self.destriping_coeffs)

        if self.spectral_coeffs is not None:
            if spectral:
                if self.VERBOSE:
                    print("[INFO] Running spectral correction...")

                self.wavelengths = self.spectral_coeffs

        return calibrated_cube


    def _load_calibration_coeff_files(self) -> None:
        """
        Load the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        try:
            self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file, 'radiometric', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.rad_coeffs = None

        try:
            self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file, 'smile', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.smile_coeffs = None

        try:
            self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file, 'destriping', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.destriping_coeffs = None

        try:
            self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.spectral_coeffs = None

        return None
    

    def _run_toa_reflectance(self, use_indirect_georef=False) -> np.ndarray:

        if self.l1b_cube is not None:
            toa_radiance = self.l1b_cube
        elif self.l1c_cube is not None:
            toa_radiance = self.l1c_cube
        else:
            self.generate_l1b_cube()
            toa_radiance = self.l1b_cube

        
        if use_indirect_georef and hasattr(self, 'solar_zenith_angles_indirect'):

            if self.VERBOSE:
                print('[WARNING] Computing TOA reflectance using INDIRECT georeferencing geometry.')

            solar_zenith_angles=self.solar_zenith_angles_indirect

        else:

            if self.VERBOSE:
                print('[WARNING] Computing TOA reflectance using DIRECT georeferencing geometry.')

            solar_zenith_angles=self.solar_zenith_angles


        toa_reflectance, srf, esun = compute_toa_reflectance(sensor_wavelengths=self.wavelengths,
                                                             sensor_fwhm=self.fwhm,
                                                             toa_radiance=toa_radiance,
                                                             iso_time=self.iso_time,
                                                             solar_zenith_angles=solar_zenith_angles
                                                            )

        return toa_reflectance, srf, esun

    def run_direct_georeferencing(self) -> None: 

        if self.VERBOSE:
            print("[INFO] Running direct georeferencing...")

        try:
            getattr(self, 'framepose')
        except:
            self._run_frame_interpolation()

        pixels_lat, pixels_lon, _ = direct_georeference(framepose_data=self.framepose,
                                                        image_height=self.image_height,
                                                        aoi_offset=self.y_start,
                                                        verbose=self.VERBOSE
                                                        )
        
        if type(pixels_lat) == int and type(pixels_lon) == int:
            if self.VERBOSE:
                print('[INFO] according to ADCS telemetry, parts or all of the image is pointing')
                print('[INFO] off the earth\'s horizon. Cant georeference this image.')
            return None

        self.latitudes = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes = pixels_lon.reshape(self.spatial_dimensions)

        bbox, \
        resolution, \
        along_track_gsd, \
        across_track_gsd = self._run_track_geometry(latitudes=self.latitudes,
                                                    longitudes=self.longitudes)

        setattr(self, 'bbox', bbox)
        setattr(self, 'along_track_gsd', along_track_gsd)
        setattr(self, 'across_track_gsd', across_track_gsd)
        setattr(self, 'resolution', resolution)

        solar_zenith_angles, \
        solar_azimuth_angles, \
        sat_zenith_angles, \
        sat_azimuth_angles, \
        relative_azimuth_angles = self._run_angles_geometry(latitudes=self.latitudes,
                                                        longitudes=self.longitudes)

        setattr(self, 'solar_zenith_angles', solar_zenith_angles)
        setattr(self, 'solar_azimuth_angles', solar_azimuth_angles)
        setattr(self, 'sat_zenith_angles', sat_zenith_angles)
        setattr(self, 'sat_azimuth_angles', sat_azimuth_angles)
        setattr(self, 'relative_azimuth_angles', relative_azimuth_angles)

        return None


    def run_indirect_georeferencing(self, 
                          points_file_path: Union[str, Path] = None, 
                          latitudes: np.ndarray = None,
                          longitudes: np.ndarray = None,
                          image_mode: str = None, 
                          origin_mode: str = 'cube',
                          flip: bool = False,
                          ) -> None:
        

        if self.VERBOSE:
            print('[INFO] Running indirect georeferencing...')
        

        if latitudes is not None and longitudes is not None:
            self.latitudes_indirect = latitudes
            self.longitudes_indirect = longitudes    

        else:
            points_file_path = Path(points_file_path).absolute()

            if not origin_mode:
                origin_mode = 'cube'

            gr = Georeferencer(filename=points_file_path,
                                                cube_height=self.spatial_dimensions[0],
                                                cube_width=self.spatial_dimensions[1],
                                                image_mode=image_mode,
                                                origin_mode=origin_mode)

            if self.VERBOSE:
                print("[INFO] Does check_star_tracker_orientation() indicate image flip?")
                print(check_star_tracker_orientation(self.nc_adcs_vars))

            #datacube_flipped = check_star_tracker_orientation(self.nc_adcs_vars)

            if flip:
                self.latitudes_indirect = gr.latitudes[:,::-1]
                self.longitudes_indirect = gr.longitudes[:,::-1]
            else:
                self.latitudes_indirect = gr.latitudes[:,:]
                self.longitudes_indirect = gr.longitudes[:,:]
    
        # Check if direct and indirect georeferencing have the same lat/lon orientations
        if (self.latitudes_indirect[-1,-1] - self.latitudes_indirect[-1,0]) * (latitudes[-1,-1] - latitudes[-1,0]) < 0:
            raise ValueError("Latitude of indirect georeferencing is flipped with respect to direct georeferencing. Check if flip paramater is set correctly")
        elif (self.longitudes_indirect[-1,-1] - self.longitudes_indirect[-1,0]) * (longitudes[-1,-1] - longitudes[-1,0]) < 0:
            raise ValueError("Longitude of indirect georeferencing is flipped with respect to direct georeferencing. Check if flip paramater is set correctly")


    
        bbox, \
        resolution, \
        along_track_gsd, \
        across_track_gsd = self._run_track_geometry(latitudes=self.latitudes_indirect,
                                                    longitudes=self.longitudes_indirect)

        setattr(self, 'bbox_indirect', bbox)
        setattr(self, 'along_track_gsd_indirect', along_track_gsd)
        setattr(self, 'across_track_gsd_indirect', across_track_gsd)
        setattr(self, 'resolution_indirect', resolution)

        solar_zenith_angles, \
        solar_azimuth_angles, \
        sat_zenith_angles, \
        sat_azimuth_angles, \
        relative_azimuth_angles = self._run_angles_geometry(latitudes=self.latitudes_indirect,
                                                        longitudes=self.longitudes_indirect)

        setattr(self, 'solar_zenith_angles_indirect', solar_zenith_angles)
        setattr(self, 'solar_azimuth_angles_indirect', solar_azimuth_angles)
        setattr(self, 'sat_zenith_angles_indirect', sat_zenith_angles)
        setattr(self, 'sat_azimuth_angles_indirect', sat_azimuth_angles)
        setattr(self, 'relative_azimuth_angles_indirect', relative_azimuth_angles)

        return None
    





    def _run_custom_georeferencing(self, 
                          latitudes: np.ndarray,
                          longitudes: np.ndarray
                          ) -> None:
        
        if self.VERBOSE:
            print('[INFO] Running custom georeferencing...')
        

        self.latitudes = latitudes
        self.longitudes = longitudes
    

        bbox, \
        resolution, \
        along_track_gsd, \
        across_track_gsd = self._run_track_geometry(latitudes=self.latitudes,
                                                    longitudes=self.longitudes)

        setattr(self, 'bbox', bbox)
        setattr(self, 'along_track_gsd', along_track_gsd)
        setattr(self, 'across_track_gsd', across_track_gsd)
        setattr(self, 'resolution', resolution)

        solar_zenith_angles, \
        solar_azimuth_angles, \
        sat_zenith_angles, \
        sat_azimuth_angles, \
        relative_azimuth_angles = self._run_angles_geometry(latitudes=self.latitudes,
                                                        longitudes=self.longitudes)

        setattr(self, 'solar_zenith_angles', solar_zenith_angles)
        setattr(self, 'solar_azimuth_angles', solar_azimuth_angles)
        setattr(self, 'sat_zenith_angles', sat_zenith_angles)
        setattr(self, 'sat_azimuth_angles', sat_azimuth_angles)
        setattr(self, 'relative_azimuth_angles', relative_azimuth_angles)

        return None






    def _run_frame_interpolation(self) -> None:

        try:
            timing = self.nc_timing_vars['timestamps_srv']
        except:
            timing = self.nc_timing_vars['timestamps']
        
        framepose_data = interpolate_at_frame_nc(adcs=self.nc_adcs_vars,
                                              lines_timestamps=timing,
                                              framerate=self.nc_capture_config_attrs['framerate'],
                                              exposure=self.nc_capture_config_attrs['exposure'],
                                              verbose=self.VERBOSE
                                              )
        
        setattr(self, "framepose", framepose_data)


        return None


    def _run_track_geometry(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None: 

        print("[INFO] Running track geometry computations...")

        try:
            getattr(self, 'framepose')
        except:
            self._run_frame_interpolation()

        bbox = compute_bbox(latitudes=latitudes, longitudes=longitudes)

        along_track_gsd, across_track_gsd = compute_gsd(frame_count=self.frame_count, 
                                                                  image_height=self.image_height, 
                                                                  latitudes=latitudes, 
                                                                  longitudes=longitudes,
                                                                  verbose=self.VERBOSE)

        resolution = compute_resolution(along_track_gsd=along_track_gsd, 
                                             across_track_gsd=across_track_gsd)


        if self.VERBOSE:
            print("[INFO] Track geometry computations done.")

        return bbox, resolution, along_track_gsd, across_track_gsd


    def _run_angles_geometry(self,  latitudes: np.ndarray, longitudes: np.ndarray) -> None: 

        print("[INFO] Running angles geometry computations...")

        try:
            getattr(self, 'framepose')
        except:
            self._run_frame_interpolation()

        indices = np.array([ 0, self.samples//4 - 1, self.samples//2 - 1, 3*self.samples//4 - 1, self.samples - 1], dtype='uint16')

        sun_azimuth, sun_zenith, \
        sat_azimuth, sat_zenith = compute_local_angles(framepose_data=self.framepose,
                                                       lats=latitudes, 
                                                       lons=longitudes,
                                                       indices=indices,
                                                       verbose=self.VERBOSE)
        
        solar_zenith_angles = sun_zenith.reshape(self.spatial_dimensions)
        solar_azimuth_angles = sun_azimuth.reshape(self.spatial_dimensions)
        sat_zenith_angles = sat_zenith.reshape(self.spatial_dimensions)
        sat_azimuth_angles = sat_azimuth.reshape(self.spatial_dimensions)

        relative_azimuth_angles = abs(sat_azimuth_angles - solar_azimuth_angles)

        relative_azimuth_angles = np.where(relative_azimuth_angles > 180, 
                                           360 - relative_azimuth_angles, 
                                           relative_azimuth_angles)

        if self.VERBOSE:
            print("[INFO] Angles geometry computations done.")

        return solar_zenith_angles, solar_azimuth_angles, sat_zenith_angles, sat_azimuth_angles, relative_azimuth_angles


    def generate_l1b_cube(self, **kwargs) -> None:

        if self.l1a_cube is None:
            return None

        self.l1b_cube = self._run_calibration(**kwargs)

        return None



    def generate_l1c_cube(self) -> None:
        
        if self.l1b_cube is None:
            self.generate_l1b_cube()
        
        self.run_direct_georeferencing()
        
        return None



    def generate_l1d_cube(self, use_indirect_georef=False) -> None:

        try:
            self.l1d_cube, self.srf, self.esun = self._run_toa_reflectance(use_indirect_georef=use_indirect_georef)

        except:
            self.generate_l1c_cube()
            self.l1d_cube, self.srf, self.esun = self._run_toa_reflectance(use_indirect_georef=use_indirect_georef)

        return None





    def ac_ocsmart_stage_input(self):

        """
        Stages OC-SMART input file to the L1B directory located in the OC-SMART installation directory. The L1d file is copied and renamed to the L1B directory.

        :return: None
        """


        if self.ocsmart_dir is not None:
            try:
                
                dst_dir = Path(self.ocsmart_dir, "L1B/")
                dst_dir.mkdir(parents=True, exist_ok=True)

                src_file = self.l1d_nc_file
                dst_file = Path(dst_dir, self.ocsmart_l1d_input_nc_file.name)

                self.ocsmart_l1d_input_nc_file = dst_file

                import shutil
                shutil.copy2(src_file, dst_file)

                print("[INFO] Successfully staged OC-SMART input file to " + str(dst_file))

            except Exception as ex:
                print("[ERROR] Unable to stage OC-SMART input. An error occured.")
                print(ex)

        else:
            print("[ERROR] OC-SMART directory is not configured. The 'ocsmart_dir' attribute is empty.")

        return None


    def ac_ocsmart_run_correction(self):
        """
        Execute 'OCSMART.py' as a subprocess.

        :return: None
        """

        print("[INFO] Running OC-SMART atmospheric correction as a subprocess.")

        import subprocess
        ocsmart_run_script = Path(self.ocsmart_dir, "OCSMART.py")
        subprocess.run(["python3", ocsmart_run_script], cwd=self.ocsmart_dir, check=True)

        print("[INFO] Removing staged OC-SMART input file " + str(self.ocsmart_l1d_input_nc_file))
        self.ocsmart_l1d_input_nc_file.unlink(missing_ok=True)

        print("[INFO] OC-SMART atmospheric correction complete.")

        return None



    def ac_ocsmart_open_output(self, h5_file_path: Path = None):
        """
        Open and read OC-SMART atmospheric correction HDF5 output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2_cube' dictionary.

        :param h5_file_path: Path to the OC-SMART HDF5 file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the HDF5 and stored as xarray DataArrays.
        """


        if h5_file_path is not None:
            h5_file_path = Path(h5_file_path).absolute()
        else:
            ocsmart_output_dir = Path(self.ocsmart_dir, "L2/")
            h5_file_path = Path(ocsmart_output_dir, self.ocsmart_l2a_output_h5_file.name)


        if h5_file_path.is_file():
            print("[INFO] Opening OC-SMART output file " + str(h5_file_path))
            datasets = load_ocsmart_h5(h5_file_path = h5_file_path)

        else:
            print("[ERROR] OC-SMART output file " + str(h5_file_path) + " does not exist.")
            return None

        try:
            key = "Rrs"
            inferred_wavelengths = datasets[key].band.to_numpy()

            # Map inferred OC-SMART wavelengths to HYPSO wavelengths
            wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

            '''
            l2_cube_wavelengths = inferred_wavelengths

            A = np.array(l2_cube_wavelengths, dtype=float)
            B = np.array(self.wavelengths, dtype=float)

            index_map = {}
            indices_unique = []

            for a in A:
                ix = np.argmin(np.abs(B - a))
                if ix not in index_map: # ensure uniqueness
                    index_map[ix] = a
                    indices_unique.append(ix)
                else:
                    print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

            ocsmart_dataset_indices = np.array(indices_unique, dtype=int)

            wl_band_map = ocsmart_dataset_indices
            '''
            
            # Create empty cube with standard HYPSO cube dims
            shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
            cube = np.full(shape=shape, fill_value=np.nan)
            cube[:,:,wl_band_map] = datasets[key]

            self.l2_cube["ocsmart"] = cube
            self.l2_cube["ocsmart"].attrs['l2_variable_name'] = key

        except Exception as ex:
            print("[ERROR] Unable to load OC-SMART L2 Rrs dataset.")

        return datasets


    


    def ac_acolite_run_correction(self, settings_file: Path = None, 
                                  input_product_level: str = 'l1c',
                                  EARTHDATA_u: str = None,
                                  EARTHDATA_p: str = None
                                  ):
        

        

        acolite_path = Path(self.acolite_dir).absolute()
        
        print("[INFO] Running ACOLITE atmospheric correction installed in " + str(acolite_path))

        sys.path.append(str(acolite_path))
        #print(sys.path)

        import acolite as ac
        from acolite.acolite.settings import load
        from acolite.acolite import acolite_run
        
        # optional file with processing settings
        # if set to None defaults will be used

        # import settings
        #settings = ac.acolite.settings.load(settings_file)
        settings = load(settings_file)

        if EARTHDATA_u is not None and EARTHDATA_p is not None:
            settings['EARTHDATA_u'] = EARTHDATA_u
            settings['EARTHDATA_p'] = EARTHDATA_p
            settings['ancillary_data'] = True

        # set settings provided above

        if input_product_level.upper() == 'L1D':
            print("[INFO] Using L1d NetCDF as ACOLITE input.")
            settings['inputfile'] = str(self.l1d_nc_file) # L1d reflectance
        else:
            print("[INFO] Using L1c NetCDF as ACOLITE input.")
            settings['inputfile'] = str(self.l1c_nc_file) # default L1c (radiance)


        print("[INFO] Writing ACOLITE output to " + str(self.capture_dir))
        settings['output'] = str(self.capture_dir)

        settings['polygon'] = None
        settings['rgb_rhot'] = True
        settings['rgb_rhos'] = True
        settings['map_l2w'] = False #produces blank .pngs
        settings['l2w_mask'] = False
        settings['l2w_mask_threshold'] = 0.2

        settings['l2w_parameters'] = ['Rrs_*', \
                                    'spm_nechad2010', \
                                    'spm_nechad2016', \
                                    'chl_re_mishra',\
                                    'chl_oc2', \
                                    'chl_oc3', \
                                    'chl_re_moses3b', \
                                    'chl_re_moses3b740', \
                                    'fai', \
                                    'fai_rhot', \
                                    'fait', \
                                    'ndci']


        processed = acolite_run(settings=settings)

        #acolite_l2_file = processed[0]['l2r'][0]

        print("[INFO] ACOLITE atmospheric correction complete.")

        return None
    



    def ac_acolite_open_output(self, acolite_l2r_output_nc_file: Path = None, acolite_l2w_output_nc_file: Path = None):
        
        """
        Open and read ACOLITE atmospheric correction L2R and L2W NetCDF output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2_cube' dictionary.

        :param h5_file_path: Path to the ACOLITE NetCDF file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the NetCDFs and stored as xarray DataArrays.
        """


        if acolite_l2r_output_nc_file is not None:
            acolite_l2r_output_nc_file = Path(acolite_l2r_output_nc_file).absolute()
        else:
            acolite_l2r_output_nc_file = Path(self.acolite_l2r_output_nc_file).absolute()

        if acolite_l2w_output_nc_file is not None:
            acolite_l2w_output_nc_file = Path(acolite_l2w_output_nc_file).absolute()
        else:
            acolite_l2w_output_nc_file = Path(self.acolite_l2w_output_nc_file).absolute()




        if acolite_l2r_output_nc_file.is_file():
            print("[INFO] Opening ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file))
            l2r_datasets = load_acolite_l2r_nc(acolite_l2r_output_nc_file)

            try:
                key = "rhos"
                inferred_wavelengths = l2r_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2r_datasets[key]

                self.l2_cube["acolite_l2r"] = cube
                self.l2_cube["acolite_l2r"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2R dataset.")
                l2r_datasets = None

        else:
            print("[ERROR] ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file) + " does not exist.")
            l2r_datasets = None


        if acolite_l2w_output_nc_file.is_file():
            print("[INFO] Opening ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file))
            l2w_datasets = load_acolite_l2w_nc(acolite_l2w_output_nc_file)

            try:
                key = "Rrs"
                inferred_wavelengths = l2w_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2w_datasets[key]

                self.l2_cube["acolite_l2w"] = cube
                self.l2_cube["acolite_l2w"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2W dataset.")
                l2w_datasets = None

        else:
            print("[ERROR] ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file) + " does not exist.")
            l2w_datasets = None


        return l2r_datasets, l2w_datasets


    def _get_inferred_wavelength_band_map(self, inferred_wavelengths):

        # Map inferred ACOLITE wavelengths to HYPSO wavelengths
        A = np.array(inferred_wavelengths, dtype=float)
        B = np.array(self.wavelengths, dtype=float)

        index_map = {}
        indices_unique = []

        for a in A:
            ix = np.argmin(np.abs(B - a))
            if ix not in index_map: # ensure uniqueness
                index_map[ix] = a
                indices_unique.append(ix)
            else:
                print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

        wl_band_map = np.array(indices_unique, dtype=int)


        return wl_band_map





    '''
    def _get_fwhm(self, wavelengths) -> None:
        
        self.fwhm = [8.2] * self.bands

        fwhm = copy.deepcopy(self.wavelengths)

        start_wl = self.wavelengths[0]
        end_wl = self.wavelengths[-1]

        for i in range(0,len(fwhm)):

            if start_wl <= fwhm[i] < 430:
                fwhm[i] = 9.6
            elif 430 <= fwhm[i] < 480:
                fwhm[i] = 9.6
            elif 480 <= fwhm[i] < 530:
                fwhm[i] = 6.6
            elif 530 <= fwhm[i] < 580:
                fwhm[i] = 8.2
            elif 580 <= fwhm[i] < 630:
                fwhm[i] = 5.8
            elif 630 <= fwhm[i] < 680:
                fwhm[i] = 5.8
            elif 680 <= fwhm[i] < 730:
                fwhm[i] = 4.1
            elif 730 <= fwhm[i] < 780:
                fwhm[i] = 4.0
            elif 780 <= fwhm[i] < end_wl:
                fwhm[i] = 4.0
            else:
                fwhm[i] = 8.2

        self.fwhm = fwhm

        return None
    '''
