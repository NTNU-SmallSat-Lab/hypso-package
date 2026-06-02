from pathlib import Path
from typing import Union, Literal
import xarray as xr
import copy
#from .DataArrayValidator import DataArrayValidator
import numpy as np
from datetime import datetime, timezone
from trollsift import Parser
import sys
import re


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
                        load_l2a_nc, \
                        load_ocsmart_h5, \
                        load_acolite_l2r_nc, \
                        load_acolite_l2w_nc, \
                        load_polymer_l2_v1_nc, \
                        load_polymer_l2_v2_nc

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
        self.UNBINNED_BAND_COUNT = 1936
        
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

        self._l2a_cubes = DataArrayDict(attributes=l2_attributes, num_dims=3, key_attribute='correction')


    @property
    def l2a_cube(self):

        self._l2a_cubes.dim_shape = self.spatial_dimensions
        self._l2a_cubes.dim_names = self.dim_names_3d
        self._l2a_cubes.num_dims = 3

        return self._l2a_cubes   

    @l2a_cube.setter
    def l2a_cubes(self, value):
        raise AttributeError("[ERROR] Use \"l2a_cubes[key] = value\" to set items.")

    def l2a_name(self, coeff_type: str = None, atmospheric_correction: str = None):

        if coeff_type:
            coeff_type = "-" + str(coeff_type)
        elif hasattr(self, "coeff_type"):
            coeff_type = "-" + str(getattr(self, "coeff_type"))
        else:
            coeff_type = ""


        if atmospheric_correction:
            atmospheric_correction = "-" + str(atmospheric_correction)
        elif hasattr(self, "atmospheric_correction"):
            atmospheric_correction = "-" + str(getattr(self, "atmospheric_correction"))
        else:
            atmospheric_correction = ""


        #aeronetvenice_2025-07-22T09-57-52Z-moved-l2a-polymer
        l2a_name = self.capture_name + coeff_type + "-l2a" + atmospheric_correction + ".nc" 


        return l2a_name
    

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
        filename = path.name

        pattern = re.compile(
            r"""
            (?P<capture_target>.+?)_
            (?P<capture_datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)
            -
            (?:(?P<coeff_type>[^-]+)-)?
            (?P<product_level>l\d[a-z])
            (?:-(?P<atmospheric_correction>[^.]+))?
            \.
            (?P<file_type>\w+)
            """,
            re.VERBOSE,
        )

        match = pattern.fullmatch(filename)

        if not match:
            raise ValueError(f"Could not parse filename: {filename}")

        fields = match.groupdict()

        fields["capture_datetime"] = datetime.strptime(
            fields["capture_datetime"],
            "%Y-%m-%dT%H-%M-%SZ",
        )

        return fields


    '''
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
    '''

    def _load_capture_file(self, path: Path, load_cube: bool = True) -> None:

        path = Path(path).absolute()



        fields = self._parse_filename(path=path)

        for key, value in fields.items():
            setattr(self, key, value)

        capture_name = self._compose_capture_name(fields=fields)

        self.capture_name = capture_name

        self.capture_dir = Path(path.parent.absolute())
        self.parent_dir = Path(path.parent.absolute())


        if self.label is not None:
            label = "-" + str(self.label)
        else:
            label = "" 


        self.l1a_name = capture_name + label + "-l1a"
        self.l1b_name = capture_name + label + "-l1b"
        self.l1c_name = capture_name + label + "-l1c"
        self.l1d_name = capture_name + label + "-l1d"
        #self.l2a_name = capture_name + label + "-l2a"

        self.l1a_nc_file = Path(path.parent, self.l1a_name + ".nc")
        self.l1b_nc_file = Path(path.parent, self.l1b_name + ".nc")
        self.l1c_nc_file = Path(path.parent, self.l1c_name + ".nc")
        self.l1d_nc_file = Path(path.parent, self.l1d_name + ".nc")




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

            case "l2a":
                if self.VERBOSE: print('[INFO] Loading L2a capture ' + self.capture_name)

                ac = getattr(self, 'atmospheric_correction', None)

                if ac is not None:
                    print("[INFO] L2a Detected atmospheric correction: " + str(ac))
                else:
                    print("[WARNING] No L2a atmospheric correction detected.")
                    setattr(self, "atmospheric_correction", "default")

                load_func = load_l2a_nc
                cube_name = "l2a_cube"
                


            case _:
                print("[ERROR] Unsupported product level.")
                print(fields['product_level'])
                return None

        # TODO: find a better method to pass all of this information
        nc_metadata_vars, \
        nc_metadata_attrs, \
        nc_geometry_vars, \
        nc_geometry_attrs, \
        nc_gcp_vars, \
        nc_gcp_attrs, \
        nc_global_metadata, \
        nc_cube_attrs, \
        nc_cube = load_func(nc_file_path=path, load_cube=load_cube)

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
 
        setattr(self, "nc_geometry_vars", nc_geometry_vars)
        setattr(self, "nc_geometry_attrs", nc_geometry_attrs)

        setattr(self, "nc_gcp_vars", nc_gcp_vars)
        setattr(self, "nc_gcp_attrs", nc_gcp_attrs)

        setattr(self, "nc_dimensions", nc_global_metadata["dimensions"])
        setattr(self, "nc_attrs", nc_global_metadata["ncattrs"])

        setattr(self, "nc_cube_attrs", nc_cube_attrs)

        # TODO: pass the dicts returned by load_func to _set_hypso_attributes()
        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_hypso_attributes()
        self._check_capture_type()

        if load_cube:
            if self.product_level.lower() == "l2a":
                self.l2a_cubes[self.atmospheric_correction] = nc_cube
            else:
                setattr(self, cube_name, nc_cube)
        
        else:
            print("[WARNING] Datacube is not loaded!")


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
                #self.fwhm = [self.AVERAGE_FWHM] * self.bands
                self.fwhm = [self.AVERAGE_FWHM] * self.UNBINNED_BAND_COUNT



        # Geometry atrributes
        for key, value in self.nc_geometry_vars.items():
            if key == 'unixtime':
                continue
            elif key == 'latitude':
                setattr(self, 'latitudes', value)
            elif key == 'longitude':
                setattr(self, 'longitudes', value)

            elif key == 'latitude_direct':
                setattr(self, 'latitudes_direct', value)
            elif key == 'longitude_direct':
                setattr(self, 'longitudes_direct', value)


            elif key == 'sensor_zenith':
                setattr(self, 'sat_zenith_angles', value)
            elif key == 'sensor_azimuth':
                setattr(self, 'sat_azimuth_angles', value)

            elif key == 'sensor_zenith_direct':
                setattr(self, 'sat_zenith_angles_direct', value)
                if getattr(self, 'sat_zenith_angles', None) is None:
                    setattr(self, 'sat_zenith_angles', value)

            elif key == 'sensor_azimuth_direct':
                setattr(self, 'sat_azimuth_angles_direct', value)
                if getattr(self, 'sat_azimuth_angles', None) is None:
                    setattr(self, 'sat_azimuth_angles', value)

            elif key == 'solar_zenith':
                setattr(self, 'solar_zenith_angles', value)
            elif key == 'solar_azimuth':
                setattr(self, 'solar_azimuth_angles', value)

            elif key == 'solar_zenith_direct':
                setattr(self, 'solar_zenith_angles_direct', value)
                if getattr(self, 'solar_zenith_angles', None) is None:
                    setattr(self, 'solar_zenith_angles', value)

            elif key == 'solar_azimuth_direct':
                setattr(self, 'solar_azimuth_angles_direct', value)
                if getattr(self, 'solar_azimuth_angles', None) is None:
                    setattr(self, 'solar_azimuth_angles', value)

            elif key == 'relative_azimuth':
                setattr(self, 'relative_azimuth_angles', value)

            elif key == 'relative_azimuth_direct':
                setattr(self, 'relative_azimuth_angles_direct', value)
                if getattr(self, 'relative_azimuth_angles', None) is None:
                    setattr(self, 'relative_azimuth_angles', value)
                
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
        elif self.nc_capture_config_attrs["frame_count"] == 106:
                    self.capture_type = "moon"
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
                         coeff_type: str = None,
                         **kwargs) -> np.ndarray:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')


        if coeff_type is None:
            try:
                coeff_type = self.nc_corrections_attrs['radiometric_coefficients_version']
            except:
                pass
        else:
            self.nc_corrections_attrs['radiometric_coefficients_version'] = str(coeff_type).lower()
            

        # TODO: move this function call
        if set_coeffs:
            self._set_calibration_coeff_files(coeff_type=coeff_type, **kwargs)

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
                    print("[INFO] Running spectral correction (binned)...")

                self.wavelengths = self.spectral_coeffs

        if self.spectral_coeffs_unbinned is not None:
            if spectral:
                if self.VERBOSE:
                    print("[INFO] Running spectral correction (unbinned)...")

                self.wavelengths_unbinned = self.spectral_coeffs_unbinned

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

        try:
            self.spectral_coeffs_unbinned = read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, 1)
        except:
            self.spectral_coeffs_unbinned = None

        return None
    



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

        self.latitudes_direct = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes_direct = pixels_lon.reshape(self.spatial_dimensions)

        bbox, \
        resolution, \
        along_track_gsd, \
        across_track_gsd = self._run_track_geometry(latitudes=self.latitudes_direct,
                                                    longitudes=self.longitudes_direct)

        setattr(self, 'bbox_direct', bbox)
        setattr(self, 'along_track_gsd_direct', along_track_gsd)
        setattr(self, 'across_track_gsd_direct', across_track_gsd)
        setattr(self, 'resolution_direct', resolution)

        solar_zenith_angles_direct, \
        solar_azimuth_angles_direct, \
        sat_zenith_angles_direct, \
        sat_azimuth_angles_direct, \
        relative_azimuth_angles_direct = self._run_angles_geometry(latitudes=self.latitudes_direct,
                                                        longitudes=self.longitudes_direct)

        setattr(self, 'solar_zenith_angles_direct', solar_zenith_angles_direct)
        setattr(self, 'solar_azimuth_angles_direct', solar_azimuth_angles_direct)
        setattr(self, 'sat_zenith_angles_direct', sat_zenith_angles_direct)
        setattr(self, 'sat_azimuth_angles_direct', sat_azimuth_angles_direct)
        setattr(self, 'relative_azimuth_angles_direct', relative_azimuth_angles_direct)

        return None


    def run_georeferencing(self,
                            latitudes: np.ndarray = None,
                            longitudes: np.ndarray = None
                            ) -> None:
        

        if self.VERBOSE:
            print('[INFO] Running georeferencing...')
    
        if latitudes is not None and longitudes is not None:
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


    def generate_l1b_cube(self, coeff_type: str = None, **kwargs) -> None:

        print("[INFO] Generating L1b cube")
        if self.l1a_cube is None:
            return None

        self.l1b_cube = self._run_calibration(coeff_type=coeff_type, **kwargs)

        return None



    def generate_l1c_cube(self, coeff_type: str = None, **kwargs) -> None:
        
        print("[INFO] Generating L1c cube")
        if self.l1b_cube is None:
            self.generate_l1b_cube(coeff_type=coeff_type, **kwargs)
        
        self.run_georeferencing()
        
        return None



    def generate_l1d_cube(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True, generate_figures=False) -> None:

        print("[INFO] Generating L1d cube")
        self._get_fwhm()
        self._get_fwhm_unbinned()
        

        if self.l1b_cube is not None:
            toa_radiance = self.l1b_cube
        elif self.l1c_cube is not None:
            toa_radiance = self.l1c_cube
        else:
            self.generate_l1b_cube()
            toa_radiance = self.l1b_cube

        if use_direct_georef and hasattr(self, 'solar_zenith_angles_direct'):

            if self.VERBOSE:
                print('[WARNING] Computing TOA reflectance using DIRECT georeferencing geometry.')

            solar_zenith_angles=self.solar_zenith_angles_direct
        else:
            solar_zenith_angles=self.solar_zenith_angles

        if use_unbinned:
            sensor_wavelengths = self.wavelengths_unbinned
            sensor_fwhm = self.fwhm_unbinned
            sensor_bin_factor = self.bin_factor
        else:
            sensor_wavelengths = self.wavelengths
            sensor_fwhm = self.fwhm
            sensor_bin_factor = 1


        toa_reflectance, \
        effective_fwhm, \
        srf, \
        srf_ssi, \
        srf_ssi_wl, \
        esun, \
        esun_wl = compute_toa_reflectance(sensor_wavelengths=sensor_wavelengths,
                                            sensor_fwhm=sensor_fwhm,
                                            bin_factor = sensor_bin_factor,
                                            toa_radiance=toa_radiance,
                                            iso_time=self.iso_time,
                                            solar_zenith_angles=solar_zenith_angles,
                                            use_thuillier = use_thuillier,
                                            generate_figures=generate_figures
                                            )

        self.l1d_cube = toa_reflectance
        
        self.srf = srf
        self.srf_ssi = srf_ssi
        self.srf_ssi_wl = srf_ssi_wl
        self.esun = esun
        self.esun_wl = esun_wl
        self.effective_fwhm = effective_fwhm

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
        Open and read OC-SMART atmospheric correction HDF5 output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

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
            l2a_cube_wavelengths = inferred_wavelengths

            A = np.array(l2a_cube_wavelengths, dtype=float)
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

            self.l2a_cube["ocsmart"] = cube
            self.l2a_cube["ocsmart"].attrs['l2_variable_name'] = key

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
        Open and read ACOLITE atmospheric correction L2R and L2W NetCDF output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

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

                self.l2a_cube["acolite_l2r"] = cube
                self.l2a_cube["acolite_l2r"].attrs['l2_variable_name'] = key

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

                self.l2a_cube["acolite_l2w"] = cube
                self.l2a_cube["acolite_l2w"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2W dataset.")
                l2w_datasets = None

        else:
            print("[ERROR] ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file) + " does not exist.")
            l2w_datasets = None


        return l2r_datasets, l2w_datasets


    def ac_polymer_get_id_sensor(self):

        sensor_version = "_" + str(self.coeff_type)

        # combine sensor name ("HYPSO-1" or "HYPSO-2") with coefficients version
        # Polymer expects format like "HYPSO-2_moved"
        id_sensor = str(self.sat_id) + sensor_version 

        return id_sensor


    def ac_polymer_get_srf_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        srf_nc_file = id_sensor + "_srf.nc"
        srf_nc_path = Path(self.parent_dir, srf_nc_file )

        self.srf_nc_file = srf_nc_file
        self.srf_nc_path = srf_nc_path

        return srf_nc_file, srf_nc_path
    

    def ac_polymer_get_ssi_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        ssi_nc_file = id_sensor + "_ssi.nc"
        ssi_nc_path = Path(self.parent_dir, ssi_nc_file )

        self.ssi_nc_file = ssi_nc_file
        self.ssi_nc_path = ssi_nc_path

        return ssi_nc_file, ssi_nc_path
    

    def ac_polymer_get_esun_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        esun_nc_file = id_sensor + "_esun.nc"
        esun_nc_path = Path(self.parent_dir, esun_nc_file )

        self.ssi_nc_file = esun_nc_file
        self.ssi_nc_path = esun_nc_path

        return esun_nc_file, esun_nc_path










    def ac_polymer_generate_srf_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, srf_nc_path = self.ac_polymer_get_srf_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'Spectral response functions for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        for idx, wl in enumerate(self.wavelengths):
            
            # Construct band ID            
            bid = "Band_" + str(idx)

            # Read ith SRF and convert from CSR sparse array
            srf = self.srf[idx,:].toarray().flatten()
            srf_wavelengths = self.srf_ssi_wl

            # Find where SRF is non-zero
            nonzero_mask = srf > 0
            
            # Extract non-zero portion of SRF and SRF wavelength array
            if np.any(nonzero_mask):
                srf_nonzero = srf[nonzero_mask]
                srf_wavelengths_nonzero = srf_wavelengths[nonzero_mask]
            else:
                srf_nonzero = srf
                srf_wavelengths_nonzero = srf_wavelengths

            # Add band entry to dataset
            ds[bid] = xr.DataArray(
                srf_nonzero,
                coords={f"wav_{bid}": srf_wavelengths_nonzero},
                attrs={
                    "band_info": bid,
                    "band_wavelength": wl,
                    "index": idx,
                    "effective_fwhm": self.effective_fwhm[idx],
                    "center_fwhm": self.fwhm[idx]
                },
            )
            ds[f"wav_{bid}"].attrs["units"] = "nm"
            
        # Sort dataarrays within dataset based on index
        ds = ds[sorted(ds, key=lambda x: ds[x].attrs['index'])]



        ds.to_netcdf(srf_nc_path)

        return srf_nc_path        






    def ac_polymer_generate_ssi_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, ssi_nc_path = self.ac_polymer_get_ssi_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'TSIS-1 solar spectral irradiance for {id_sensor} (0.005 nm spectral resolution)'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        ds["ssi"] = xr.DataArray(
            self.srf_ssi,
            coords={f"wav_ssi": self.srf_ssi_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_ssi"].attrs["units"] = "nm"


        ds.to_netcdf(ssi_nc_path)

        return ssi_nc_path        





    def ac_polymer_generate_esun_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, esun_nc_path = self.ac_polymer_get_esun_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'ESUN for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        #ds.attrs["ssi"] = self.srf_ssi
        #ds.attrs["ssi_wavelengths"] = self.srf_ssi_wl
        #ds.attrs["ssi_units"] = "mW m-2 nm-1"

        #ds.attrs["esun"] = self.esun
        #ds.attrs["esun_wavlengths"] = self.esun_wl
        #ds.attrs["esun_units"] = "mW m-2 nm-1"


        ds["esun"] = xr.DataArray(
            self.esun,
            coords={f"wav_esun": self.esun_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_esun"].attrs["units"] = "nm"


        ds.to_netcdf(esun_nc_path)

        return esun_nc_path    








    def ac_polymer_run_correction(self, 
                                  polymer_base_path: str,
                                  polymer_path: str = None, 
                                  eoread_path: str = None, 
                                  eotools_path: str = None, 
                                  core_path: str = None, 
                                  input_product_level: str = "l1c",
                                  #coeff_type: str = None,
                                  optional_output_datasets: list = ["SPM"],
                                  if_exists: str = "overwrite"):

        #polymer_path = Path(self.polymer_dir).absolute()

        if polymer_path is not None:
            polymer_path = str(Path(polymer_path).absolute())
            sys.path.insert(0, polymer_path)

        if eotools_path is not None:
            eotools_path = str(Path(eotools_path).absolute())
            sys.path.insert(0, eotools_path)

        if eoread_path is not None:
            eoread_path = str(Path(eoread_path).absolute())
            sys.path.insert(0, eoread_path)

        if core_path is not None:
            core_path = str(Path(core_path).absolute())
            sys.path.insert(0, core_path)

        sys.path.insert(0, polymer_base_path)



        # TODO
        srf_nc_path, srf_nc_path = self.ac_polymer_get_srf_nc_path()

        run_polymer_kwargs = {"srf_getter": "hypso.ac.ac_polymer_srf_getter",
                                "srf_getter_arg": srf_nc_path}


        from eoread.hypso import Level1_HYPSO
        from polymer.main_v5 import run_polymer, run_polymer_dataset, default_output_datasets


        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        match input_product_level.lower():
            
            case "l1c":
                polymer_l1_input_nc_file = Path(self.parent_dir, self.l1c_nc_file)
                polymer_l2_output_nc_file = Path(self.parent_dir, str(self.l1c_name) + ".polymer.nc")
            case "l1d":
                polymer_l1_input_nc_file = Path(self.parent_dir, self.l1d_nc_file)
                polymer_l2_output_nc_file = Path(self.parent_dir, str(self.l1d_name) + ".polymer.nc")
            case _:
                return None
            
        

        #import os
        #cwd = os.getcwd()
        #os.chdir(polymer_path)

        # This is from the Feb 2026 version of Polymer
        #from polymer.level1 import Level1
        #from polymer.level2 import Level2
        #from eoread.hypso import Level1_HYPSO
        #from polymer.main_v5 import run_polymer, run_polymer_dataset
        #from core.files.fileutils import mdir
        #polymer_output_file = run_polymer(Level1_HYPSO(polymer_input_file), dir_out=mdir(polymer_output_dir), split_bands=False)

        # Run Polymer
        if True:
            output_file = run_polymer(
                Level1_HYPSO(polymer_l1_input_nc_file),
                dir_out=str(self.parent_dir),
                output_datasets=default_output_datasets + optional_output_datasets,
                if_exists = if_exists,
                srf_getter = "hypso.ac.ac_polymer_srf_getter",
                srf_getter_arg = srf_nc_path

            )

        try:
            polymer_l2_output_nc_file = Path(output_file).rename(polymer_l2_output_nc_file)
        except FileNotFoundError:
            print("[WARNING] Polymer L2 NetCDF output file has already been renamed.")
            pass

        print(output_file)
        print(polymer_l2_output_nc_file)

        return Path(polymer_l2_output_nc_file)


    





    def ac_polymer_open_output(self, 
                               polymer_l2_output_nc_file: Path = None, 
                               input_product_level="l1c",
                               version = "v1" 
                               #coeff_type: str = None
                               ):
        
        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        if polymer_l2_output_nc_file is not None:
            polymer_l2_output_nc_file = Path(polymer_l2_output_nc_file)
        else:
            match input_product_level.lower():

                case "l1c":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1c product.")
                    polymer_l2_output_nc_file = Path(self.parent_dir, str(self.l1c_name)+ ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1c.nc.polymer.nc

                case "l1d":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1d product.")
                    polymer_l2_output_nc_file = Path(self.parent_dir, str(self.l1d_name) + ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1d.nc.polymer.nc
            

        polymer_l2_output_nc_file = polymer_l2_output_nc_file.absolute()
        

        if polymer_l2_output_nc_file.is_file():

            if version == "v1":
                polymer_datasets = load_polymer_l2_v1_nc(polymer_l2_output_nc_file)

                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    self.l2a_cube["polymer"] = cube
                    self.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.") 

            elif version == "v2":

                polymer_datasets = load_polymer_l2_v2_nc(polymer_l2_output_nc_file)
            
                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    self.l2a_cube["polymer"] = cube
                    self.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.")

        else:
            print("[ERROR] Polymer L2 NetCDF output file " + str(polymer_l2_output_nc_file) + " does not exist.")
            polymer_datasets = None

        
        return polymer_datasets










    def _get_inferred_wavelength_band_map(self, inferred_wavelengths):

        # Map inferred wavelengths to HYPSO wavelengths
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


    def _get_fwhm(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths: 
            idx = np.argmin(np.abs(band - self.srf_wl))
            fwhm_per_band.append(self.srf_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm = fwhm
        
        return None


    def _get_fwhm_unbinned(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths_unbinned: 
            idx = np.argmin(np.abs(band - self.srf_wl))
            fwhm_per_band.append(self.srf_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm_unbinned = fwhm
        
        return None




