from datetime import datetime
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


from hypso.geometry import interpolate_at_frame, \
                           geometry_computation, \
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
                          l2a_nc_writer

from hypso.utils import find_file

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict

from pyresample.geometry import SwathDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler

from trollsift import Parser

UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here

# TODO: store latitude and longitude as xarray
# TODO: setattr, hasattr, getattr for setting class variables, especially those from geometry

class Hypso1(Hypso):

    def __init__(self, path: Union[str, Path], points_path: Union[str, Path, None] = None, verbose=False) -> None:
        
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
        self._load_points_file(path=points_path)

        chl_attributes = {'model': None}
        product_attributes = {}

        chl = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                 attributes=chl_attributes, 
                                 dims_names=self.dim_names_2d,
                                 num_dims=2
                                 )
        
        products = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                      attributes=product_attributes, 
                                      dims_names=self.dim_names_2d,
                                      num_dims=2
                                      )

        setattr(self, "chl", chl)
        setattr(self, "products", products)


        return None

        


    def _set_capture_config(self,
                                 capture_config,
                                 ) -> None:

        # FPS has been renamed to framerate. Need to support both since old .nc files may still use FPS
        try:
            capture_config['fps'] = capture_config['framerate']
        except:
            capture_config['framerate'] = capture_config['fps']

        self.background_value = 8 * self.capture_config["bin_factor"]
        self.exposure = self.capture_config["exposure"] / 1000  # in seconds

        self.x_start = self.capture_config["aoi_x"]
        self.x_stop = self.capture_config["aoi_x"] + self.capture_config["column_count"]
        self.y_start = self.capture_config["aoi_y"]
        self.y_stop = self.capture_config["aoi_y"] + self.capture_config["row_count"]

        self.bin_factor = self.capture_config["bin_factor"]

        # Try/except here since not all captures have sample_div
        try:
            self.sample_div = self.capture_config['sample_div']
        except:
            self.sample_div = 1

        self.row_count = self.capture_config["row_count"]
        self.frame_count = self.capture_config["frame_count"]
        self.column_count = self.capture_config["column_count"]

        self.image_height = int(self.capture_config["row_count"] / self.sample_div)
        self.image_width = int(self.capture_config["column_count"] / self.capture_config["bin_factor"])
        self.im_size = self.image_height * self.image_width

        self.bands = self.image_width
        self.lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols

        self.spatial_dimensions = (self.capture_config["frame_count"], self.image_height)

        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")


        self.start_timestamp_capture = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if framerate and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / self.capture_config["framerate"] + self.capture_config["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] Framerate or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0

        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end

        self.unixtime = self.start_timestamp_capture
        self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()


        if self.capture_config["frame_count"] == self.standard_dimensions["nominal"]:
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
        setattr(self, "capture_name", capture_name)

        setattr(self, "l1a_nc_file", Path(path.parent, capture_name + "-l1a.nc"))
        setattr(self, "l1b_nc_file", Path(path.parent, capture_name + "-l1b.nc"))
        setattr(self, "l2a_nc_file", Path(path.parent, capture_name + "-l2a.nc"))

        setattr(self, "capture_dir", Path(path.parent.absolute(), capture_name + "_tmp"))
        setattr(self, "parent_dir", Path(path.parent.absolute()))

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

        capture_config, \
        timing, \
        target_coords, \
        adcs, \
        dimensions, \
        navigation, \
        database, \
        corrections, \
        logfiles, \
        temperature, \
        ncattrs, \
        cube = load_func(nc_file_path=path)

        setattr(self, "capture_config", capture_config)
        setattr(self, "timing", timing)
        setattr(self, "adcs", adcs)
        setattr(self, "dimensions", dimensions)
        setattr(self, "navigation", navigation)
        setattr(self, "database", database)
        setattr(self, "corrections", corrections)
        setattr(self, "logfiles", logfiles)
        setattr(self, "temperature", temperature)
        setattr(self, "ncattrs", ncattrs)

        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_capture_config(capture_config=capture_config)

        setattr(self, cube_name, cube)

        return None



    # TODO refactor into two functions
    def _load_points_file(self, 
                          path: str, 
                          image_mode: str = None, 
                          origin_mode: str = 'qgis',
                          flip_lats: bool = False,
                          flip_lons: bool = False) -> None:


        if path:
            path = Path(path).absolute()
        else:
            if self.VERBOSE:
                print('[INFO] No georeferencing .points file provided. Skipping georeferencing.')
            return None


        if not origin_mode:
            origin_mode = 'qgis'

        # Compute latitude and longitudes arrays if a points file is available

        if self.VERBOSE:
            print('[INFO] Running georeferencing...')

        gr = georeferencing.Georeferencer(filename=path,
                                            cube_height=self.spatial_dimensions[0],
                                            cube_width=self.spatial_dimensions[1],
                                            image_mode=image_mode,
                                            origin_mode=origin_mode)
        

        self.latitudes = gr.latitudes
        self.longitudes = gr.longitudes
        
        datacube_flipped = check_star_tracker_orientation(self.adcs)

        if not datacube_flipped:

            if self.l1a_cube is not None:
                self.l1a_cube = self.l1a_cube[:, ::-1, :]

            if self.l1b_cube is not None:  
                self.l1b_cube = self.l1b_cube[:, ::-1, :]
                
            if self.l2a_cube is not None:  
                self.l2a_cube = self.l2a_cube[:, ::-1, :]


        self.datacube_flipped = datacube_flipped

        self.bbox = compute_bbox(latitudes=self.latitudes, 
                                 longitudes=self.longitudes)

        self.along_track_gsd, self.across_track_gsd = compute_gsd(frame_count=self.frame_count, 
                                                                  image_height=self.image_height, 
                                                                  latitudes=self.latitudes, 
                                                                  longitudes=self.longitudes,
                                                                  verbose=self.VERBOSE)

        self.resolution = compute_resolution(along_track_gsd=self.along_track_gsd, 
                                             across_track_gsd=self.across_track_gsd)


        if flip_lons:
            self.latitudes = self.latitudes[:,::-1]
            self.longitudes = self.longitudes[:,::-1]

        if flip_lats:
            self.latitudes = self.latitudes[::-1,:]
            self.longitudes = self.longitudes[::-1,:]

        return None
        
    def _run_calibration(self, 
                         overwrite: bool = False,
                         rad_cal = True,
                         smile_corr = True,
                         destriping_corr = True,
                         **kwargs) -> None:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')

        self._load_calibration_coeff_files()

        self.srf = get_spectral_response_function(wavelengths=self.wavelengths)

        l1a_cube = self.l1a_cube.to_numpy()

        if self.VERBOSE:
            print("[INFO] Running radiometric calibration...")

        l1b_cube = run_radiometric_calibration(cube=l1a_cube, 
                                           background_value=self.background_value,
                                           exp=self.exposure,
                                           image_height=self.image_height,
                                           image_width=self.image_width,
                                           frame_count=self.frame_count,
                                           rad_coeffs=self.rad_coeffs)

        if self.VERBOSE:
            print("[INFO] Running smile correction...")

        l1b_cube = run_smile_correction(cube=l1b_cube, 
                                        smile_coeffs=self.smile_coeffs)

        if self.VERBOSE:
            print("[INFO] Running destriping correction...")

        l1b_cube = run_destriping_correction(cube=l1b_cube, 
                                             destriping_coeffs=self.destriping_coeffs)

        self.l1b_cube = l1b_cube

        return None

    def _load_calibration_coeff_files(self) -> None:
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


        try:
            self.rad_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_radiometric}')
            self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        except:
            self.rad_coeff_file = None

        try:
            self.smile_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_smile}')
            self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        except:
            self.smile_coeff_file = None

        try:
            self.destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_destriping}')
            self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        except:
            self.destriping_coeff_file = None

        try:
            self.spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_spectral}')
            self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)
            self.wavelengths = self.spectral_coeffs
        except:
            self.spectral_coeff_file = None
            self.wavelengths = range(0,120)


        return None


    # TODO: Move to hypso parent class
    def _run_geometry(self, overwrite: bool = False) -> None: 

        if self.VERBOSE:
            print("[INFO] Running geometry computation...")

        framepose_data = interpolate_at_frame_nc(adcs=self.adcs,
                                                 timestamps_srv=self.timing['timestamps_srv'],
                                                 frame_count=self.capture_config['frame_count'],
                                                 framerate=self.capture_config['framerate'],
                                                 exposure=self.capture_config['exposure'],
                                                 verbose=self.VERBOSE
                                                 )
        self.framepose_np = framepose_data


        pixels_lat, pixels_lon = direct_georeference(framepose_data=framepose_data,
                                                     image_height=self.image_height,
                                                     aoi_offset=self.y_start,
                                                     verbose=self.VERBOSE
                                                     )
        if pixels_lat == -1 and pixels_lon == -1:            
            if self.VERBOSE:
                print('[INFO] according to ADCS telemetry, parts or all of the image is pointing')
                print('[INFO] off the earth\'s horizon. Cant georeference this image.')
            return None

        self.latitudes_original = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes_original = pixels_lon.reshape(self.spatial_dimensions)

        sun_azimuth, sun_zenith, \
        sat_azimuth, sat_zenith = compute_local_angles(framepose_data=framepose_data,
                                                       lats=pixels_lat, lons=pixels_lon,
                                                       indices=np.array([ 0, hypso_height/4 - 1, hypso_height/2 - 1, 3*hypso_height/4 - 1, hypso_height - 1]),
                                                       verbose=self.VERBOSE)
        self.solar_zenith_angles = sun_zenith.reshape(self.spatial_dimensions)
        self.solar_azimuth_angles = sun_azimuth.reshape(self.spatial_dimensions)
        self.sat_zenith_angles = sat_zenith.reshape(self.spatial_dimensions)
        self.sat_azimuth_angles = sat_azimuth.reshape(self.spatial_dimensions)
    
        #self.wkt_linestring_footprint = wkt_linestring_footprint
        #self.prj_file_contents = prj_file_contents
        #self.local_angles = local_angles
        #self.geometric_meta_info = geometric_meta_info

        return None


    # Atmospheric correction functions

    def _run_atmospheric_correction(self, product_name: str) -> None:

        try:
            match product_name.lower():
                case "6sv1":
                    self.l2a_cube = self._run_6sv1_atmospheric_correction()
                case "acolite":
                    self.l2a_cube = self._run_acolite_atmospheric_correction()
                case "machi":
                    self.l2a_cube = self._run_machi_atmospheric_correction() 
                case _:
                    print("[ERROR] No such atmospheric correction product supported!")
                    return None
        except:
            print("[ERROR] Unable to generate L2a datacube.")

        return None



    def _run_toa_reflectance(self) -> None:

        toa_reflectance_cube = compute_toa_reflectance(srf=self.srf,
                                                    toa_radiance=self.l1b_cube,
                                                    iso_time=self.iso_time,
                                                    solar_zenith_angles=self.solar_azimuth_angles,
                                                    )
        
        #self.l1c_cube = xr.DataArray(toa_reflectance, dims=("y", "x", "band"))
        #self.l1c_cube.attrs['units'] = "sr^-1"
        #self.l1c_cube.attrs['description'] = "Top of atmosphere (TOA) reflectance"
        #toa_reflectance_cube = xr.DataArray(toa_reflectance, dims=("y", "x", "band"))

        self.l1c_cube = toa_reflectance_cube

        return None

    def generate_geometry(self, overwrite: bool = False) -> None:

        self._run_geometry(overwrite=overwrite)

        return None

    def get_closest_wavelength_index(self, wavelength: Union[float, int]) -> int:

        wavelengths = np.array(self.wavelengths)
        differences = np.abs(wavelengths - wavelength)
        closest_index = np.argmin(differences)

        return closest_index




    def generate_l1b_cube(self, **kwargs) -> None:

        self._run_calibration(**kwargs)

        return None

    def write_l1b_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1b_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l1b_nc_writer(satobj=self, 
                      dst_l1b_nc_file=self.l1b_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        return None

    def generate_l1c_cube(self) -> None:

        self._run_geometry()
        self._run_toa_reflectance()

        return None

    # TODO
    def write_l1c_nc_file(self, path: str) -> None:
        
        return None

    # TODO
    def generate_l2a_cube(self, product_name: str = "machi") -> None:

        self._run_geometry()
        #self._run_atmospheric_correction(product_name=product_name)

        return None

    def write_l2a_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1b_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l2a_nc_writer(satobj=self, 
                      dst_l2a_nc_file=self.l2a_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        return None






