from datetime import datetime
from dateutil import parser
from importlib.resources import files
from pathlib import Path
from typing import Literal, Union
import copy

import hypso.atmospheric
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyproj as prj
import xarray as xr

import hypso
from hypso import Hypso


from hypso.utils import find_file

from hypso.atmospheric import run_py6s, run_acolite, run_machi
from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              make_mask, \
                              make_overexposed_mask, \
                              get_destriping_correction_matrix, \
                              run_destriping_correction_with_computed_matrix, \
                              get_spectral_response_function

from hypso.chlorophyll import run_tuned_chlorophyll_estimation, \
                              run_band_ratio_chlorophyll_estimation, \
                              validate_tuned_model

from hypso.geometry import interpolate_at_frame, \
                           geometry_computation, \
                           get_nearest_pixel, \
                           compute_gsd, \
                           compute_bbox, \
                           compute_resolution

from hypso.georeferencing import georeferencing
from hypso.georeferencing.utils import check_star_tracker_orientation
from hypso.masks import run_global_land_mask, \
                        run_ndwi_land_mask, \
                        run_threshold_land_mask

from hypso.masks import run_cloud_mask, \
                        run_quantile_threshold_cloud_mask

from hypso.reading import load_l1a_nc_cube, \
                          load_l1a_nc_metadata, \
                          load_l1b_nc_cube, \
                          load_l1b_nc_metadata, \
                          load_l2a_nc_cube, \
                          load_l2a_nc_metadata

from hypso.writing import l1a_nc_writer, l1b_nc_writer, l2a_nc_writer

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange

from pyresample.geometry import SwathDefinition
from pyresample.bilinear.xarr import XArrayBilinearResampler 
from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler

from trollsift import Parser

SUPPORTED_PRODUCT_LEVELS = ["l1a", "l1b", "l2a"]

ATM_CORR_PRODUCTS = ["6sv1", "acolite", "machi"]
CHL_EST_PRODUCTS = Literal["band_ratio", "6sv1_aqua", "acolite_aqua"]
LAND_MASK_PRODUCTS = Literal["global", "ndwi", "threshold"]
CLOUD_MASK_PRODUCTS = Literal["default"]

DEFAULT_ATM_CORR_PRODUCT = "6sv1"
DEFAULT_CHL_EST_PRODUCT = "band_ratio"
DEFAULT_LAND_MASK_PRODUCT = "global"
DEFAULT_CLOUD_MASK_PRODUCT = "default"

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

        self.load_file(path=path)
        self.load_points_file(path=points_path)

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

        


    def _set_metadata_attributes(self,
                                 capture_config,
                                 timing,
                                 adcs,
                                 navigation) -> None:


        try:
            capture_config['fps'] = capture_config['framerate']
        except:
            capture_config['framerate'] = capture_config['fps']

        setattr(self, "capture_config", capture_config)
        setattr(self, "timing", timing)
        setattr(self, "adcs", adcs)
        setattr(self, "navigation", navigation)


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






    # Loading functions

    def load_file(self, path: Path) -> None:

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

                capture_config, \
                timing, \
                target_coords, \
                adcs, \
                dimensions, \
                navigation = load_l1a_nc_metadata(nc_file_path=path)

                self._set_metadata_attributes(capture_config=capture_config,
                                              timing=timing, 
                                              adcs=adcs, 
                                              navigation=navigation)

                self.l1a_cube = load_l1a_nc_cube(nc_file_path=path)
                
            case "l1b":
                if self.VERBOSE: print('[INFO] Loading L1b capture ' + self.capture_name)

                capture_config, \
                timing, \
                target_coords, \
                adcs, \
                dimensions, \
                navigation = load_l1b_nc_metadata(nc_file_path=path)

                self._set_metadata_attributes(capture_config=capture_config,
                                              timing=timing, 
                                              adcs=adcs, 
                                              navigation=navigation)

                self.l1b_cube = load_l1b_nc_cube(nc_file_path=path)

            case "l2a":
                if self.VERBOSE: print('[INFO] Loading L2a capture ' + self.capture_name)

                capture_config, \
                timing, \
                target_coords, \
                adcs, \
                dimensions, \
                navigation = load_l2a_nc_metadata(nc_file_path=path)

                self._set_metadata_attributes(capture_config=capture_config,
                                              timing=timing, 
                                              adcs=adcs, 
                                              navigation=navigation)

                self.l2a_cube = load_l2a_nc_cube(nc_file_path=path)
            case _:
                print("[ERROR] Unsupported product level.")

        return None





    # Georeferencing functions

    # TODO refactor
    def load_points_file(self, 
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

        self.resolution = compute_resolution(latitudes=self.latitudes, 
                                             longitudes=self.longitudes)


        if flip_lons:
            self.latitudes = self.latitudes[:,::-1]
            self.longitudes = self.longitudes[:,::-1]

        if flip_lats:
            self.latitudes = self.latitudes[::-1,:]
            self.longitudes = self.longitudes[::-1,:]

        self.latitudes_original = self.latitudes
        self.longitudes_original = self.longitudes


        return None




    # Calibration functions
        
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

        self._set_calibration_coeff_files()


        self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        #self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)

        if self.spectral_coeffs is not None:
            self.wavelengths = self.spectral_coeffs
        else:
            self.wavelengths = range(0,120)

        
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




    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """


        if rad_coeff_file:
            self.rad_coeff_file = rad_coeff_file
            return None

        if smile_coeff_file:
            self.smile_coeff_file = smile_coeff_file
            return None

        if destriping_coeff_file:
            self.destriping_coeff_file = destriping_coeff_file
            return None
        
        if spectral_coeff_file:
            self.spectral_coeff_file = spectral_coeff_file
            return None

        match self.capture_type:

            case "custom":
                #csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.csv"
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.npz"

                #csv_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.csv" 
                npz_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.npz"  

                #csv_file_destriping = None
                npz_file_destriping = None

                #csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "nominal":
                #csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv"
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.npz"

                #csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.npz"

                #csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"
                npz_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.npz"

                #csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case "wide":
                #csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.csv"
                npz_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.npz"

                #csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"
                npz_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.npz"

                #csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"
                npz_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.npz"

                #csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"
                npz_file_spectral = "spectral_bands_HYPSO-1_v1.npz"

            case _:
                npz_file_radiometric = None
                npz_file_smile = None
                npz_file_destriping = None

        if npz_file_radiometric:
            rad_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_radiometric}')
        else: 
            rad_coeff_file = None

        self.rad_coeff_file = rad_coeff_file


        if npz_file_smile:
            smile_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_smile}')
        else:
            smile_coeff_file = npz_file_smile

        self.smile_coeff_file = smile_coeff_file


        if npz_file_destriping:
            destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_destriping}')
        else:
            destriping_coeff_file = None

        self.destriping_coeff_file = destriping_coeff_file

        if npz_file_spectral:
            spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{npz_file_spectral}')
        else:
            spectral_coeff_file = None

        self.spectral_coeff_file = spectral_coeff_file

        return None




    # Geometry computation functions


    # Move to hypso parent class
    def _run_geometry(self, overwrite: bool = False) -> None:

        if self.VERBOSE:
            print("[INFO] Running geometry computation...")


        framepose_data = interpolate_at_frame_nc(adcs=self.adcs,
                                              timestamps_srv=self.timing['timestamps_srv'],
                                              framerate=self.capture_config['framerate'],
                                              exposure=self.capture_config['exposure'],
                                              verbose=self.VERBOSE
                                              )


        # TODO: split into tow functions: one to compute latitude and longitudes, one to compute solar angles

        wkt_linestring_footprint, \
           prj_file_contents, \
           local_angles, \
           geometric_meta_info, \
           pixels_lat, \
           pixels_lon, \
           sun_azimuth, \
           sun_zenith, \
           sat_azimuth, \
           sat_zenith = geometry_computation(framepose_data=framepose_data,
                                             image_height=self.image_height,
                                             verbose=self.VERBOSE
                                             )

        #self.framepose_df = framepose_data

        self.wkt_linestring_footprint = wkt_linestring_footprint
        self.prj_file_contents = prj_file_contents
        self.local_angles = local_angles
        self.geometric_meta_info = geometric_meta_info

        self.solar_zenith_angles = sun_zenith.reshape(self.spatial_dimensions)
        self.solar_azimuth_angles = sun_azimuth.reshape(self.spatial_dimensions)

        self.sat_zenith_angles = sat_zenith.reshape(self.spatial_dimensions)
        self.sat_azimuth_angles = sat_azimuth.reshape(self.spatial_dimensions)

        #self.lat_original = pixels_lat.reshape(self.spatial_dimensions)
        #self.lon_original = pixels_lon.reshape(self.spatial_dimensions)

        self.latitudes_original = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes_original = pixels_lon.reshape(self.spatial_dimensions)

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

    def _run_6sv1_atmospheric_correction(self, **kwargs) -> xr.DataArray:

        # Py6S Atmospheric Correction
        # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
        # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
        # atmos_dict = {
        #     'aot550': 0.01,
        #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
        # }
        # AOT550 parameter gotten from: https://giovanni.gsfc.nasa.gov/giovanni/

        if self.VERBOSE: 
            print("[INFO] Running 6SV1 atmospheric correction")

        # TODO: which values should we use?
        if self.latitudes is None:
            latitudes = self.latitudes_original # fall back on geometry computed values
        else:
            latitudes = self.latitudes

        if self.longitudes is None:
            longitudes = self.longitudes_original # fall back on geometry computed values
        else:
            longitudes = self.longitudes

        py6s_dict = {
            'aot550': 0.0580000256
        }

        time_capture = parser.parse(self.iso_time)

        cube = self.l1b_cube.to_numpy()

        cube = run_py6s(wavelengths=self.wavelengths, 
                        hypercube_L1=cube, 
                        lat_2d_array=latitudes,
                        lon_2d_array=longitudes,
                        solar_azimuth_angles=self.solar_azimuth_angles,
                        solar_zenith_angles=self.solar_zenith_angles,
                        sat_azimuth_angles=self.sat_azimuth_angles,
                        sat_zenith_angles=self.sat_zenith_angles,
                        iso_time=self.iso_time,
                        py6s_dict=py6s_dict, 
                        time_capture=time_capture,
                        srf=self.srf)
        
        cube = self._format_l2a_dataarray(cube)
        cube.attrs['correction'] = "6sv1"

        return cube


    def _run_acolite_atmospheric_correction(self) -> xr.DataArray:

        if hasattr(self, 'acolite_path'):

            nc_file_path = str(self.l1b_nc_file)
            acolite_path = str(self.acolite_path)

            cube = run_acolite(acolite_path=acolite_path, 
                               output_path=self.capture_dir, 
                               nc_file_path=nc_file_path)

            cube = self._format_l2a_dataarray(cube)
            cube.attrs['correction'] = "acolite"

            return cube
        else:
            print("[ERROR] Please set path to ACOLITE source code before generating ACOLITE L2a datacube using \"set_acolite_path()\"")
            print("[INFO] The ACOLITE source code can be downloaded from https://github.com/acolite/acolite")
            return None


    # TODO
    def _run_machi_atmospheric_correction(self) -> xr.DataArray:

        #print("[WARNING] Minimal Atmospheric Compensation for Hyperspectral Imagers (MACHI) atmospheric correction has not been enabled.")
        #return None




        if self.VERBOSE: 
            print("[INFO] Running MACHI atmospheric correction")

        # start working with ToA reflectance, so we don't have to worry about the solar spectrum
        cube = self.toa_reflectance_cube.to_numpy()
        
        T, S, objs = hypso.atmospheric.atm_correction(cube.reshape(-1,120), 
                                                      solar=np.ones(120), 
                                                      verbose=True,
                                                      tol=0.01, 
                                                      est_min_R=0.05)

        # normalize the whole cube
        cube_norm = (cube - S) /T
        cube_norm[self.cloud_mask] = np.nan

        cube = self._format_l2a_dataarray(cube_norm)
        cube.attrs['correction'] = "machi"

        return cube
    

    # Top of atmosphere reflectance functions

    # TODO: run product through validator, add proprty
    # TODO: move code to atmospheric module
    # TODO: add set functions
    def _run_toa_reflectance(self) -> None:

        
        # Get Local variables
        srf = self.srf
        toa_radiance = self.l1b_cube.to_numpy()

        scene_date = parser.isoparse(self.iso_time)
        julian_day = scene_date.timetuple().tm_yday
        solar_zenith = self.solar_zenith_angles

        # Read Solar Data
        solar_data_path = str(files('hypso.atmospheric').joinpath("Solar_irradiance_Thuillier_2002.csv"))
        solar_df = pd.read_csv(solar_data_path)

        # Create new solar X with a new delta
        solar_array = np.array(solar_df)
        current_num = solar_array[0, 0]
        delta = 0.01
        new_solar_x = [solar_array[0, 0]]
        while current_num <= solar_array[-1, 0]:
            current_num = current_num + delta
            new_solar_x.append(current_num)

        # Interpolate for Y with original solar data
        new_solar_y = np.interp(new_solar_x, solar_array[:, 0], solar_array[:, 1])

        # Replace solar Dataframe
        solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

        # Estimation of TOA Reflectance
        band_number = 0
        toa_reflectance = np.empty_like(toa_radiance)
        for single_wl, single_srf in srf:
            # Resample HYPSO SRF to new solar wavelength
            resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
            weights_srf = resamp_srf / np.sum(resamp_srf)
            ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py

            # Earth-Sun distance (from day of year) using julian date
            # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
            distance_sun = 1 - 0.01672 * np.cos(0.9856 * (
                    julian_day - 4))

            # Get toa_reflectance
            solar_angle_correction = np.cos(np.radians(solar_zenith))
            multiplier = (ESUN * solar_angle_correction) / (np.pi * distance_sun ** 2)
            toa_reflectance[:, :, band_number] = toa_radiance[:, :, band_number] / multiplier

            band_number = band_number + 1

        self.toa_reflectance_cube = xr.DataArray(toa_reflectance, dims=("y", "x", "band"))
        #self.toa_reflectance_cube.attrs['units'] = "sr^-1"
        #self.toa_reflectance_cube.attrs['description'] = "Top of atmosphere (TOA) reflectance"

        return None







    # Public L1a methods

    def get_l1a_cube(self) -> xr.DataArray:

        return self.l1a_cube

    def get_l1a_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> xr.DataArray:

        if self.l1a_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]

        return spectrum

    def plot_l1a_spectrum(self, 
                         latitude=None, 
                         longitude=None,
                         x: int = None,
                         y: int = None,
                         save: bool = False
                         ) -> None:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]
        bands = range(0, len(spectrum))
        units = spectrum.attrs["units"]

        output_file = Path(self.parent_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.xlabel("Band number")
        plt.title(f"L1a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            plt.imsave(output_file)
        else:
            plt.show()

        return None

    def write_l1a_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1a_nc_file).is_file() and not overwrite:

            original_path = self.l1a_nc_file

            file_name = original_path.name
            modified_file_name = file_name.replace('-l1a', '-l1a-modified')
            modified_path = original_path.with_name(modified_file_name)

            dst_l1a_nc_file = modified_path

            if self.VERBOSE:
                print("[INFO] L1a NetCDF file has already been generated. Writing L1a data to: " + str(dst_l1a_nc_file))

        else:
            dst_l1a_nc_file = self.l1a_nc_file


        l1a_nc_writer(satobj=self, 
                      dst_l1a_nc_file=dst_l1a_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        return None


    # Public L1b methods

    def generate_l1b_cube(self, **kwargs) -> None:

        self._run_calibration(**kwargs)

        return None

    def get_l1b_cube(self) -> xr.DataArray:

        return self.l1b_cube

    def get_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:

        if self.l1b_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]

        return spectrum

    def plot_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None,
                        save: bool = False
                        ) -> None:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]
        bands = self.wavelengths
        units = spectrum.attrs["units"]

        output_file = Path(self.parent_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.xlabel("Wavelength (nm)")
        plt.title(f"L1b (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            # TODO: TypeError: imsave() missing 1 required positional argument: 'arr'
            plt.imsave(output_file)
        else:
            plt.show()

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


    # Public L2a methods

    def generate_l2a_cube(self, product_name: str = "machi") -> None:

        self._run_atmospheric_correction(product_name=product_name)

        return None

    def get_l2a_cube(self) -> xr.DataArray:

        return self.l2a_cube

    def get_l2a_spectrum(self,
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> xr.DataArray:


        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        try:
            spectrum = self.l2a_cube[idx[0], idx[1], :]
        except KeyError:
            return None

        return spectrum

    def plot_l2a_spectrum(self,
                         latitude=None, 
                         longitude=None,
                         x: int = None,
                         y: int = None,
                         save: bool = False
                         ) -> np.ndarray:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        try:
            spectrum = self.l2a_cube[idx[0], idx[1], :]
        except KeyError:
            return None

        bands = self.wavelengths
        units = spectrum.attrs["units"]

        output_file = Path(self.parent_dir, self.capture_name + '_l2a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.ylim([0, 1])
        plt.xlabel("Wavelength (nm)")
        plt.title(f"L2a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            plt.imsave(output_file)
        else:
            plt.show()

    def write_l2a_nc_file(self, overwrite: bool = False) -> None:

        correction = self.l2a_cube.attrs['correction']

        original_path = self.l2a_nc_file

        file_name = original_path.name
        modified_file_name = file_name.replace('-l2a', '-l2a-' + correction)
        modified_path = original_path.with_name(modified_file_name)

        dst_l2a_nc_file = modified_path
        src_l1a_nc_file = self.l1a_nc_file

        if Path(dst_l2a_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l2a_nc_writer(satobj=self, 
                      dst_l2a_nc_file=dst_l2a_nc_file, 
                      src_l1a_nc_file=src_l1a_nc_file)

        return None

    def set_acolite_path(self, path: str) -> None:

        self.acolite_path = Path(path).absolute()

        return None







    def generate_geometry(self, overwrite: bool = False) -> None:

        self._run_geometry(overwrite=overwrite)

        return None







    # Public land mask methods

    # TODO
    def load_land_mask(self, path: str) -> None:

        return None

    def generate_land_mask(self, land_mask_name: LAND_MASK_PRODUCTS = DEFAULT_LAND_MASK_PRODUCT, **kwargs) -> None:

        self._run_land_mask(land_mask_name=land_mask_name, **kwargs)

        return None

    def get_land_mask(self) -> xr.DataArray:

        return self.land_mask

    # TODO
    def write_land_mask(self, path: str) -> None:

        return None


    # Public cloud mask methods

    # TODO
    def load_cloud_mask(self, path: str) -> None:

        return None

    def generate_cloud_mask(self, cloud_mask_name: CLOUD_MASK_PRODUCTS = DEFAULT_CLOUD_MASK_PRODUCT, **kwargs):

        self._run_cloud_mask(cloud_mask_name=cloud_mask_name, **kwargs)

        return None

    def get_cloud_mask(self) -> xr.DataArray:

        return self.cloud_mask
    
    # TODO
    def write_cloud_mask(self, path: str) -> None:

        return None


    # Public unified mask methods

    def get_unified_mask(self) -> xr.DataArray:

        return self.unified_mask


    # Public chlorophyll methods

    # TODO
    def load_chlorophyll_estimates(self, path: str) -> None:

        return None

    def generate_chlorophyll_estimates(self, 
                                       product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                       model: Union[str, Path] = None,
                                       factor: float = 0.1
                                       ) -> None:

        self._run_chlorophyll_estimation(product_name=product_name, model=model, factor=factor)

    def get_chlorophyll_estimates(self, product_name: str = DEFAULT_CHL_EST_PRODUCT,
                                 ) -> np.ndarray:

        key = product_name.lower()

        return self.chl[key]


    # TODO
    def write_chlorophyll_estimates(self, path: str) -> None:

        return None
    


    # Public custom products methods
    
    # TODO
    def load_products(self, path: str) -> None:

        return None

    # TODO
    def write_products(self, path: str) -> None:

        return None



    # Public L1c (top of atmosphere reflectance) methods

    # TODO
    def load_toa_reflectance(self, path: str) -> None:
        
        return None

    def generate_toa_reflectance(self) -> None:

        self._run_toa_reflectance()

        return None

    def get_toa_reflectance(self) -> xr.DataArray:
        """
        Convert Top Of Atmosphere (TOA) Radiance to TOA Reflectance.

        :return: Array with TOA Reflectance.
        """

        return self.toa_reflectance_cube


    # TODO
    def write_toa_reflectance(self, path: str) -> None:
        
        return None


    def get_l1a_satpy_scene(self) -> Scene:

        return self._generate_l1a_satpy_scene()

    def get_l1b_satpy_scene(self) -> Scene:

        return self._generate_l1b_satpy_scene()

    def get_l2a_satpy_scene(self) -> Scene:

        return self._generate_l2a_satpy_scene()
    
    def get_toa_reflectance_satpy_scene(self) -> Scene:

        return self._generate_toa_reflectance_satpy_scene()

    def get_chlorophyll_estimates_satpy_scene(self) -> Scene:

        return self._generate_chlorophyll_satpy_scene()

    def get_products_satpy_scene(self) -> Scene:

        return self._generate_products_satpy_scene()

    def get_bbox(self) -> tuple:
        
        return self.bbox
    
    def get_closest_wavelength_index(self, wavelength: Union[float, int]) -> int:

        wavelengths = np.array(self.wavelengths)
        differences = np.abs(wavelengths - wavelength)
        closest_index = np.argmin(differences)

        return closest_index


