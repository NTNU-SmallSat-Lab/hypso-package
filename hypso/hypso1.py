import datetime
from dateutil import parser
from importlib.resources import files
from pathlib import Path
from typing import Literal, Union
import copy

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyproj as prj
import xarray as xr

import hypso
from hypso import Hypso

from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              make_mask, \
                              make_overexposed_mask, \
                              get_destriping_correction_matrix, \
                              run_destriping_correction_with_computed_matrix, \
                              get_spectral_response_function


from hypso.geometry import interpolate_at_frame_nc, \
                           direct_georeference, \
                           compute_local_angles, \
                           get_nearest_pixel, \
                           compute_gsd, \
                           compute_bbox, \
                           compute_resolution

from hypso.georeferencing import georeferencing
from hypso.georeferencing.utils import check_star_tracker_orientation

from hypso.loading import load_l1a_nc, \
                          load_l1b_nc, \
                          load_l2a_nc

from hypso.reflectance import compute_toa_reflectance

from hypso.writing import l1b_nc_writer, \
                          l1c_nc_writer, \
                          l1d_nc_writer

from hypso.utils import find_file

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict



from trollsift import Parser

UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here

# TODO: store latitude and longitude as xarray
# TODO: setattr, hasattr, getattr for setting class variables, especially those from geometry

class Hypso1(Hypso):

    def __init__(self, path: Union[str, Path], verbose=False) -> None:
        
        """
        Initialization of HYPSO-1 Class.

        :param path: Absolute path to NetCDF file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(path=path)

        # General -----------------------------------------------------
        self.platform = 'hypso1'
        self.sensor = 'hypso1_hsi'
        self.VERBOSE = verbose

        self._load_capture_file(path=path)

        product_attributes = {}

        products = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                      attributes=product_attributes, 
                                      dims_names=self.dim_names_2d,
                                      num_dims=2
                                      )

        setattr(self, "products", products)

        return None

        

    # TODO: move into hypso class
    def _set_capture_config(self, capture_config_attrs: dict) -> None:


        for attr in capture_config_attrs.keys():
            setattr(self, attr, capture_config_attrs[attr])

        # FPS has been renamed to framerate. Need to support both since old .nc files may still use FPS
        try:
            capture_config_attrs['fps'] = capture_config_attrs['framerate']
        except:
            capture_config_attrs['framerate'] = capture_config_attrs['fps']

        self.background_value = 8 * self.capture_config_attrs["bin_factor"]
        self.exposure = self.capture_config_attrs["exposure"] / 1000  # in seconds

        self.x_start = self.capture_config_attrs["aoi_x"]
        self.x_stop = self.capture_config_attrs["aoi_x"] + self.capture_config_attrs["column_count"]
        self.y_start = self.capture_config_attrs["aoi_y"]
        self.y_stop = self.capture_config_attrs["aoi_y"] + self.capture_config_attrs["row_count"]

        self.bin_factor = self.capture_config_attrs["bin_factor"]

        # Try/except here since not all captures have sample_div
        try:
            self.sample_div = self.capture_config_attrs['sample_div']
        except:
            self.sample_div = 1

        self.row_count = self.capture_config_attrs["row_count"]
        self.frame_count = self.capture_config_attrs["frame_count"]
        self.column_count = self.capture_config_attrs["column_count"]

        self.image_height = int(self.capture_config_attrs["row_count"] / self.sample_div)
        self.image_width = int(self.capture_config_attrs["column_count"] / self.capture_config_attrs["bin_factor"])
        self.im_size = self.image_height * self.image_width

        self.bands = self.image_width
        self.lines = self.capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols

        self.spatial_dimensions = (self.capture_config_attrs["frame_count"], self.image_height)

        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")

        try:
            self.start_timestamp_capture = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET
        except:
            try:
                datestring = self.ncattrs['date_aquired']
            except:
                datestring = self.ncattrs['timestamp_acquired']

            dt = datetime.datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=datetime.timezone.utc)
            self.start_timestamp_capture = dt.timestamp()

        self.start_timestamp_capture = int(self.timing_attrs['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if framerate and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config_attrs["frame_count"] / self.capture_config_attrs["framerate"] + self.capture_config_attrs["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] Framerate or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config_attrs["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0

        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end

        self.unixtime = self.start_timestamp_capture
        self.iso_time = datetime.datetime.utcfromtimestamp(self.unixtime).isoformat()


        if self.capture_config_attrs["frame_count"] == self.standard_dimensions["nominal"]:
            self.capture_type = "nominal"

        elif self.image_height == self.standard_dimensions["wide"]:
            self.capture_type = "wide"
        else:
            # EXPERIMENTAL_FEATURES
            if self.VERBOSE:
                print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
            self.capture_type = "custom"

        if self.VERBOSE:
            print(f"[INFO] Capture capture type: {self.capture_type}")

        return None


    # Filename parsing and composing

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
        self.l2a_nc_file = Path(path.parent, capture_name + "-l2a.nc")

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
                # TODO
                print("[ERROR] L1c loading is not yet implemented.")
                
                #load_func = load_l1c_nc
                #cube_name = "l1c_cube"

            case "l2a":
                if self.VERBOSE: print('[INFO] Loading L2a capture ' + self.capture_name)

                load_func = load_l2a_nc
                cube_name = "l2a_cube"

            case _:
                print("[ERROR] Unsupported product level.")
                return None

        metadata_vars, metadata_attrs, global_metadata, cube = load_func(nc_file_path=path)

        setattr(self, "adcs_vars", metadata_vars["adcs"])
        setattr(self, "capture_config_vars", metadata_vars["capture_config"])
        setattr(self, "corrections_vars", metadata_vars["corrections"])
        setattr(self, "database_vars", metadata_vars["database"])
        setattr(self, "logfiles_vars", metadata_vars["logfiles"])
        setattr(self, "temperature_vars", metadata_vars["temperature"])
        setattr(self, "timing_vars", metadata_vars["timing"])

        setattr(self, "adcs_attrs", metadata_attrs["adcs"])
        setattr(self, "capture_config_attrs", metadata_attrs["capture_config"])
        setattr(self, "corrections_attrs", metadata_attrs["corrections"])
        setattr(self, "database_attrs", metadata_attrs["database"])
        setattr(self, "logfiles_attrs", metadata_attrs["logfiles"])
        setattr(self, "temperature_attrs", metadata_attrs["temperature"])
        setattr(self, "timing_attrs", metadata_attrs["timing"])
 

        setattr(self, "dimensions", global_metadata["dimensions"])
        setattr(self, "ncattrs", global_metadata["ncattrs"])

        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_capture_config(capture_config_attrs=self.capture_config_attrs)

        setattr(self, cube_name, cube)

        return None





    def _run_calibration(self, 
                         radiometric: bool = True,
                         smile: bool = True,
                         destripe: bool = True,
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

        if set_coeffs:
            self._set_calibration_coeff_files()

        self._load_calibration_coeff_files()

        self.srf = get_spectral_response_function(wavelengths=self.wavelengths)

        calibrated_cube = self.l1a_cube.to_numpy()

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


        if smile:

            if self.VERBOSE:
                print("[INFO] Running smile correction...")

            calibrated_cube = run_smile_correction(cube=calibrated_cube, 
                                            smile_coeffs=self.smile_coeffs)

        if destripe:
            if self.VERBOSE:
                print("[INFO] Running destriping correction...")

            calibrated_cube = run_destriping_correction(cube=calibrated_cube, 
                                                destriping_coeffs=self.destriping_coeffs)


        return calibrated_cube

    # TODO: split into function to set calibration coeffs and function to load calibration coeffs
    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        match self.capture_type:

            case "custom":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.npz"
                npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"  
                npz_file_destriping = None
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "nominal":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "wide":
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.npz"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case _:
                npz_file_radiometric = None
                npz_file_smile = None
                npz_file_destriping = None
                npz_file_spectral = None

        self.rad_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_radiometric}')
        self.smile_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_smile}')
        self.destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_destriping}')
        self.spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_spectral}')


    def _load_calibration_coeff_files(self) -> None:
        """
        Load the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """


        try:
            self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        except:
            self.rad_coeff_file = None

        try:
            self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        except:
            self.smile_coeff_file = None

        try:
            self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        except:
            self.destriping_coeff_file = None

        try:
            self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)
            self.wavelengths = self.spectral_coeffs
        except:
            self.spectral_coeff_file = None
            self.wavelengths = range(0,120)

        return None










































    def _run_framepose(self) -> None:

        framepose_data = interpolate_at_frame_nc(adcs=self.adcs_vars,
                                              lines_timestamps=self.timing_vars['timestamps_srv'],
                                              framerate=self.capture_config_attrs['framerate'],
                                              exposure=self.capture_config_attrs['exposure'],
                                              verbose=self.VERBOSE
                                              )
        

        setattr(self, "framepose", framepose_data)

        return None






    def run_direct_georeferencing(self) -> None: 

        if self.VERBOSE:
            print("[INFO] Running direct georeferencing...")

        try:
            getattr(self, 'framepose')
        except:
            self._run_framepose()

        pixels_lat, pixels_lon = direct_georeference(framepose_data=self.framepose,
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

        self._run_geometry()

        return None



    def run_indirect_georeferencing(self, 
                          points_file_path: Union[str, Path], 
                          image_mode: str = None, 
                          origin_mode: str = 'qgis'
                          ) -> None:

        if self.VERBOSE:
            print('[INFO] Running indirect georeferencing...')
        
        points_file_path = Path(points_file_path).absolute()

        if not origin_mode:
            origin_mode = 'qgis'

        gr = georeferencing.Georeferencer(filename=points_file_path,
                                            cube_height=self.spatial_dimensions[0],
                                            cube_width=self.spatial_dimensions[1],
                                            image_mode=image_mode,
                                            origin_mode=origin_mode)
        
        self.latitudes_indirect = gr.latitudes
        self.longitudes_indirect = gr.longitudes
    
        self._run_geometry(indirect=True)

        return None



    def _run_geometry(self, indirect=False) -> None: 

        print("[INFO] Running geometry computations...")

        if indirect:
            modifier = "_indirect"
        else:
            modifier = ""
            
        try:
            getattr(self, 'framepose')
        except:
            self._run_framepose()

        latitudes = getattr(self, 'latitudes' + modifier)
        longitudes = getattr(self, 'longitudes' + modifier)

        bbox = compute_bbox(latitudes=latitudes, longitudes=longitudes)

        along_track_gsd, across_track_gsd = compute_gsd(frame_count=self.frame_count, 
                                                                  image_height=self.image_height, 
                                                                  latitudes=latitudes, 
                                                                  longitudes=longitudes,
                                                                  verbose=self.VERBOSE)

        resolution = compute_resolution(along_track_gsd=along_track_gsd, 
                                             across_track_gsd=across_track_gsd)

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

        setattr(self, 'bbox' + modifier, bbox)
        setattr(self, 'along_track_gsd' + modifier, along_track_gsd)
        setattr(self, 'across_track_gsd' + modifier, across_track_gsd)
        setattr(self, 'resolution' + modifier, resolution)

        setattr(self, 'solar_zenith_angles' + modifier, solar_zenith_angles)
        setattr(self, 'solar_azimuth_angles' + modifier, solar_azimuth_angles)
        setattr(self, 'sat_zenith_angles' + modifier, sat_zenith_angles)
        setattr(self, 'sat_azimuth_angles' + modifier, sat_azimuth_angles)

        if self.VERBOSE:
            print("[INFO] Geometry computations done")

        return None












    def _run_toa_reflectance(self) -> np.ndarray:

        return compute_toa_reflectance(srf=self.srf,
                                        toa_radiance=self.l1b_cube,
                                        iso_time=self.iso_time,
                                        solar_zenith_angles=self.solar_azimuth_angles,
                                        )
        


    def generate_l1b_cube(self, **kwargs) -> None:

        self.l1b_cube = self._run_calibration(**kwargs)

        return None

    def write_l1b_nc_file(self, overwrite: bool = False, **kwargs) -> None:

        if Path(self.l1b_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l1b_nc_writer(satobj=self, 
                      dst_nc=self.l1b_nc_file, 
                      **kwargs)

        return None




    



    def generate_l1c_cube(self) -> None:

        self.generate_l1b_cube()
        self.run_direct_georeferencing()
        
        return None

    def write_l1c_nc_file(self, overwrite: bool = False, **kwargs) -> None:

        if Path(self.l1c_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1c NetCDF file has already been generated. Skipping.")

            return None

        l1c_nc_writer(satobj=self, 
                      dst_nc=self.l1c_nc_file, 
                      **kwargs)

        return None







    def generate_l1d_cube(self) -> None:

        self.generate_l1c_cube()
        self.l1d_cube = self._run_toa_reflectance()

        return None

    # TODO
    def write_l1d_nc_file(self, overwrite: bool = False, **kwargs) -> None:
        
        if Path(self.l1d_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1c NetCDF file has already been generated. Skipping.")

            return None

        l1d_nc_writer(satobj=self, 
                      dst_nc=self.l1d_nc_file, 
                      **kwargs)

        return None







