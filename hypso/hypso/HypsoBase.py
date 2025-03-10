from pathlib import Path
from typing import Union
import xarray as xr
import copy
#from .DataArrayValidator import DataArrayValidator
import numpy as np
from datetime import datetime, timezone
from trollsift import Parser


from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              get_spectral_response_function


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
                        load_l1d_nc

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
        #self.l2a_nc_file = None

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
        self.AVERAGE_FWHM = 8.2 #3.33

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

        self.l1a_nc_file = Path(path.parent, capture_name + "-l1a.nc")
        self.l1b_nc_file = Path(path.parent, capture_name + "-l1b.nc")
        self.l1c_nc_file = Path(path.parent, capture_name + "-l1c.nc")
        self.l1d_nc_file = Path(path.parent, capture_name + "-l1d.nc")

        self.capture_dir = Path(path.parent.absolute(), capture_name + "_tmp")
        self.parent_dir = Path(path.parent.absolute())

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

        setattr(self, "nc_dimensions", nc_global_metadata["dimensions"])
        setattr(self, "nc_attrs", nc_global_metadata["ncattrs"])

        setattr(self, "nc_cube_attrs", nc_cube_attrs)

        # TODO: pass the dicts returned by load_func to _set_hypso_attributes()
        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_hypso_attributes()
        self._check_capture_type()

        setattr(self, cube_name, nc_cube)
        

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
                self.wavelengths = self.nc_cube_attrs['fwhm']
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
        self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()

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
                                                rad_coeffs=self.rad_coeffs)

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

        #self.srf = get_spectral_response_function(wavelengths=self.wavelengths, fwhm=self.fwhm)

        return calibrated_cube


    def _load_calibration_coeff_files(self) -> None:
        """
        Load the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        try:
            self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        except:
            self.rad_coeffs = None

        try:
            self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        except:
            self.smile_coeffs = None

        try:
            self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        except:
            self.destriping_coeffs = None

        try:
            self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)
        except:
            self.spectral_coeffs = None

        return None
    

    # def _run_toa_reflectance(self) -> np.ndarray:
    def _run_toa_reflectance(self, use_indirect_georef=False) -> np.ndarray:

        if not hasattr(self, "srf"):
            self.srf = get_spectral_response_function(wavelengths=self.wavelengths, fwhm=self.fwhm)

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


        return compute_toa_reflectance(srf=self.srf,
                                        toa_radiance=toa_radiance,
                                        iso_time=self.iso_time,
                                        solar_zenith_angles=solar_zenith_angles,
                                        )


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
                          flip: bool = None,
                          ) -> None:
        

        if self.VERBOSE:
            print('[INFO] Running indirect georeferencing...')
        

        if latitudes is not None and longitudes is not None:
            self.latitudes_indirect = latitudes
            self.longitudes_indirect = longitudes    

        else:

            if flip is None:
                raise ValueError("[ERROR] The 'flip' variable must be provided. Can be found in processing-temp/geometric-meta-info.txt as 'Filp RGB:' ")

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


            # TODO: flip lat/lon matrices?
            if not flip:
                self.latitudes_indirect = gr.latitudes[:,::-1]
                self.longitudes_indirect = gr.longitudes[:,::-1]
            else:
                self.latitudes_indirect = gr.latitudes[:,:]
                self.longitudes_indirect = gr.longitudes[:,:]
    

    
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
            self.l1d_cube = self._run_toa_reflectance(use_indirect_georef=use_indirect_georef)

        except:
            self.generate_l1c_cube()
            self.l1d_cube = self._run_toa_reflectance(use_indirect_georef=use_indirect_georef)

        return None


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