from typing import Union
from osgeo import gdal, osr
import numpy as np
import pandas as pd
from importlib.resources import files
import rasterio
from pathlib import Path
from dateutil import parser
import netCDF4 as nc
import pyproj as prj
import matplotlib.pyplot as plt

from pyresample import SwathDefinition
from pyresample import geometry
from pyresample.kd_tree import get_neighbour_info

from typing import Literal, Union
from datetime import datetime

from hypso.georeferencing import georeferencing
from hypso.georeferencing.utils import check_star_tracker_orientation
from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction


from hypso.geometry import interpolate_at_frame, \
                           geometry_computation

from hypso.georeference import start_coordinate_correction, generate_full_geotiff, generate_rgb_geotiff
from hypso.masks import run_global_land_mask, run_ndwi_land_mask, run_threshold_land_mask, run_cloud_mask
from hypso.reading import load_l1a_nc_cube, load_l1a_nc_metadata
from hypso.utils import find_dir, find_file, find_all_files
from hypso.atmospheric import run_py6s, run_acolite, run_machi
from hypso.chlorophyll import run_tuned_chlorophyll_estimation, run_band_ratio_chlorophyll_estimation


EXPERIMENTAL_FEATURES = True
SUPPORTED_PRODUCT_LEVELS = ["l1a", "l1b", "l2a"]
SUPPORTED_ATM_CORR_PRODUCTS = ["6sv1", "acolite", "machi"]
SUPPORTED_CHL_EST_PRODUCTS = ["band_ratio", "6sv1_aqua", "acolite_aqua"]

UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here

class Hypso:

    def __init__(self, hypso_path: str, points_path: Union[str, None] = None):

        """
        Initialization of HYPSO Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """
        # Set NetCDF and .points file paths
        self._set_hypso_path(hypso_path=hypso_path)
        self._set_points_path(points_path=points_path)


        # Initialize platform and sensor names
        self.platform = None
        self.sensor = None


        # Initialize capture name and target
        self.capture_name = None
        self.capture_region = None


        # Initialize directory and file info
        self.tmp_dir = None
        self.nc_dir = None

        self.nc_file = None
        self.nc_name = None

        self.l1a_nc_file = None
        self.l1b_nc_file = None
        self.l2a_nc_file = None

        self.l1a_nc_name = None
        self.l1b_nc_name = None
        self.l2a_nc_name = None

        # Initialize datasets
        # TODO: store all products, data, etc in dataset object/dict?
        self.datasets = {}

        # Initialize datacubes
        self.l1a_cube = None
        self.l1b_cube = None
        self.l2a_cubes = {}


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
        #self.target_area = None
        self.background_value = None
        self.exposure = None


        # Initialize georeferencing info
        self.projection_metadata = None
        self.latitudes = None
        self.longitudes = None
        self.datacube_flipped = False


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

        # Initialize units
        self.units = r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$'
        self.l1a_units = "Raw sensor values (a.u.)"
        self.l1b_units = r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$'
        self.l2a_units = "Rrs [0,1] (a.u.)"


        # Initilize land mask dict
        self.land_masks = {}
        self.active_land_mask = None
        self.active_land_mask_name = None
        

        # Initilize cloud mask dict
        self.cloud_masks = {}
        self.active_cloud_mask = None
        self.active_cloud_mask_name = None


        # Intialize active mask
        self.active_mask = None


        # Initialize products dict
        self.products = {}


        # Initialize chlorophyll estimates dict
        self.chl = {}


        # Initialize ADCS data
        self.adcs = None
        self.adcs_pos_df = None
        self.adcs_quat_df = None


        # Initialize geometry data
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


        # DEBUG
        self.DEBUG = False
        self.verbose = False

    def _set_hypso_path(self, hypso_path=None) -> None:

        if hypso_path is None:
            hypso_path = self.hypso_path

        # Make NetCDF file path an absolute path
        self.hypso_path = Path(hypso_path).absolute()

        return None

    def _set_points_path(self, points_path=None) -> None:

        # Make .points file path an absolute path (if possible)
        if points_path is not None:
            self.points_path = Path(points_path).absolute()
        else:
            self.points_path = None

        return None
        


class Hypso1(Hypso):

    def __init__(self, hypso_path: str, points_path: Union[str, None] = None, verbose=False) -> None:
        
        """
        Initialization of HYPSO-1 Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(hypso_path=hypso_path, points_path=points_path)

        # General -----------------------------------------------------
        self._set_platform()
        self._set_sensor()
        self._set_verbose(verbose=verbose)

        # Booleans to check if certain processes have been run
        self.georeferencing_has_run = False
        self.calibration_has_run = False
        self.geometry_computation_has_run = False
        self.write_l1b_nc_file_has_run = False
        self.atmospheric_correction_has_run = False
        self.land_mask_has_run = False
        self.cloud_mask_has_run = False
        self.chlorophyll_estimation_has_run = False

        # L1a: raw
        # L1b: radiance
        # L1c: radiance
        # L2a: Rrs


        # Load L1a file -----------------------------------------------------
        self._load_l1a_file()

        # Georeferencing -----------------------------------------------------
        self._run_georeferencing()

        # Calibration -----------------------------------------------------
        #self._run_calibration()

        # Atmospheric correction -----------------------------------------------------
        #self._run_geometry_computation()
        #self._run_atmospheric_correction(product='6SV1')

        # Land Mask -----------------------------------------------------
        # self._run_land_mask()

        # Cloud Mask -----------------------------------------------------
        #self._run_cloud_mask()

        # Chlorophyll -----------------------------------------------------
        # self.chl

        # Products
        # TODO
        #self.products['chl'] = None
        #self.products['tsm'] = None
        #self.products['pca'] = None
        #self.products['ica'] = None

        return None


    # Setters

    def _set_verbose(self, verbose=False) -> None:

        self.verbose = verbose

        return None

    def _set_platform(self) -> None:

        self.platform = 'hypso1'

        return None

    def _set_sensor(self) -> None:

        self.sensor = 'hypso1_hsi'

        return None

    # TODO
    def _set_capture_datetime(self) -> None:

        self.capture_datetime = None

        return None


    def _set_capture_name(self) -> None:

        capture_name = self.hypso_path.stem

        for pl in SUPPORTED_PRODUCT_LEVELS:

            if "-" + pl in capture_name:
                capture_name = capture_name.replace("-" + pl, "")

        self.capture_name = capture_name

        return None

    def _set_capture_region(self) -> None:

        self._set_capture_name()
        self.capture_region = self.capture_name.split('_')[0].strip('_')

        return None

    def _set_capture_type(self) -> None:

        if self.capture_config["frame_count"] == self.standard_dimensions["nominal"]:

            self.capture_type = "nominal"

        elif self.image_height == self.standard_dimensions["wide"]:

            self.capture_type = "wide"
        else:

            if EXPERIMENTAL_FEATURES:
                print("Number of Rows (AKA frame_count) Is Not Standard")
                self.capture_type = "custom"
            else:
                raise Exception("Number of Rows (AKA frame_count) Is Not Standard")

        if self.verbose:
            print(f"[INFO] Capture capture type: {self.capture_type}")

        return None



    def _set_adcs_dataframes(self) -> None:

        self._set_adcs_pos_dataframe()
        self._set_adcs_quat_dataframe()

        return None

    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_pos_dataframe(self) -> None:

        
        position_headers = ["timestamp", "eci x [m]", "eci y [m]", "eci z [m]"]
        
        timestamps = self.adcs["timestamps"]
        pos_x = self.adcs["position_x"]
        pos_y = self.adcs["position_y"]
        pos_z = self.adcs["position_z"]

        pos_array = np.column_stack((timestamps, pos_x, pos_y, pos_z))
        pos_df = pd.DataFrame(pos_array, columns=position_headers)

        self.adcs_pos_df = pos_df

        return None

    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_quat_dataframe(self) -> None:

        
        quaternion_headers = ["timestamp", "quat_0", "quat_1", "quat_2", "quat_3", "Control error [deg]"]

        timestamps = self.adcs["timestamps"]
        quat_s = self.adcs["quaternion_s"]
        quat_x = self.adcs["quaternion_x"]
        quat_y = self.adcs["quaternion_y"]
        quat_z = self.adcs["quaternion_z"]
        control_error = self.adcs["control_error"]

        quat_array = np.column_stack((timestamps, quat_s, quat_x, quat_y, quat_z, control_error))
        quat_df = pd.DataFrame(quat_array, columns=quaternion_headers)

        self.adcs_quat_df = quat_df

        return None

    def _set_nc_files(self) -> None:

        self.nc_file = self.hypso_path
        self.nc_name = self.hypso_path.stem

        file_path = str(self.hypso_path)

        self.l1a_nc_file = Path(file_path)
        self.l1b_nc_file = Path(file_path.replace("-l1a", "-l1b"))
        self.l2a_nc_file = Path(file_path.replace("-l1a", "-l2a"))

        self.l1a_nc_name = self.l1a_nc_file.stem
        self.l1b_nc_name = self.l1b_nc_file.stem
        self.l2a_nc_name = self.l2a_nc_file.stem

        return None

    def _set_tmp_dir(self) -> None:

        self.tmp_dir = Path(self.nc_file.parent.absolute(), self.nc_name.replace("-l1a", "") + "_tmp")

        return None
    
    def _set_nc_dir(self) -> None:

        self.nc_dir = Path(self.nc_file.parent.absolute())

        return None

    # TODO: is this used for anything?
    '''
    def _set_target_area(self) -> None:

        if all([self.target_coords.get('latc'), self.target_coords.get('lonc')]):
            self.target_area = self.target_coords['latc'] + ' ' + self.target_coords['lonc']
        else:
            self.target_area = None

        return None
    '''

    def _set_background_value(self) -> None:

        self.background_value = 8 * self.capture_config["bin_factor"]

        return None

    def _set_exposure(self) -> None:

        self.exposure = self.capture_config["exposure"] / 1000  # in seconds

        return None

    def _set_aoi(self) -> None:

        self.x_start = self.capture_config["aoi_x"]
        self.x_stop = self.capture_config["aoi_x"] + self.capture_config["column_count"]
        self.y_start = self.capture_config["aoi_y"]
        self.y_stop = self.capture_config["aoi_y"] + self.capture_config["row_count"]

    def _set_dimensions(self) -> None:

        self.bin_factor = self.capture_config["bin_factor"]

        self.row_count = self.capture_config["row_count"]
        self.frame_count = self.capture_config["frame_count"]
        self.column_count = self.capture_config["column_count"]

        self.image_height = self.capture_config["row_count"]
        self.image_width = int(self.capture_config["column_count"] / self.capture_config["bin_factor"])
        self.im_size = self.image_height * self.image_width

        self.bands = self.image_width
        self.lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols

        self.spatial_dimensions = (self.capture_config["frame_count"], self.image_height)

        if self.verbose:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")

        return None

    def _set_timestamps(self) -> None:

        self.start_timestamp_capture = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if fps and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / self.capture_config["fps"] + self.capture_config["exposure"] / 1000.0
        except:
            if self.verbose:
                print("[WARNING] FPS or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0

        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end

        self.unixtime = self.start_timestamp_capture
        self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()

        return None


    # Loading L1a

    def _load_l1a_file(self) -> None:

        self._check_l1a_file_format()

        self._set_capture_name()
        self._set_capture_region()
        self._set_capture_datetime()

        self._load_l1a_cube()
        self._load_l1a_metadata()

        self._set_nc_files()
        self._set_tmp_dir()
        self._set_nc_dir()

        self._set_background_value()
        self._set_exposure()
        self._set_aoi()
        self._set_dimensions()
        self._set_timestamps()
        self._set_capture_type()
        self._set_adcs_dataframes()
        #self._set_target_area()

        return None

    def _check_l1a_file_format(self) -> None:

        # Check that hypso_path file is a NetCDF file:
        #if not self.hypso_path.suffix == '.nc':
        #    raise Exception("Incorrect HYPSO Path. Only .nc files supported")

        match self.hypso_path.suffix:
            case '.nc':
                return None
            case '.bip':
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
            case _:
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
    
        return None

    def _load_l1a_cube(self) -> None:

        self._load_l1a_nc_cube()
        #self._load_l1a_bip_cube()
        
        return None

    def _load_l1a_metadata(self) -> None:
        
        self._load_l1a_nc_metadata()
        self._load_l1a_bip_metadata()

        return None

    def _load_l1a_nc_cube(self) -> None:

        self.l1a_cube = load_l1a_nc_cube(self.hypso_path)

        return None

    def _load_l1a_nc_metadata(self) -> None:

        self.capture_config, \
            self.timing, \
            target_coords, \
            self.adcs, \
            dimensions = load_l1a_nc_metadata(self.hypso_path)
        
        return None

    def _load_l1a_bip_cube(self) -> None:
        
        return None

    def _load_l1a_bip_metadata(self) -> None:
        
        return None


    # Loading L1b

    # TODO
    def _load_l1b_file(self) -> None:
        return None

    # TODO
    def _load_l1b_cube(self) -> None:
        return None
    
    # TODO
    def _load_l1b_metadata(self) -> None:
        return None


    # Georeferencing functions

    def _run_georeferencing(self, overwrite: bool = False) -> None:

        if self._check_georeferencing_has_run() and not overwrite:

            if self.verbose:
                    print("[INFO] Georeferencing has already been run. Skipping.")

            return None

        # Compute latitude and longitudes arrays if a points file is available
        if self.points_path is not None:

            if self.verbose:
                print('[INFO] Running georeferencing...')

            gr = georeferencing.Georeferencer(filename=self.points_path,
                                              cube_height=self.spatial_dimensions[0],
                                              cube_width=self.spatial_dimensions[1],
                                              image_mode=None,
                                              origin_mode='qgis')
            
            # Update latitude and longitude arrays with computed values from Georeferencer
            self.latitudes = gr.latitudes[:, ::-1]
            self.longitudes = gr.longitudes[:, ::-1]
            
            self._run_georeferencing_orientation()

            self.latitudes_original = self.latitudes
            self.longitudes_original = self.longitudes


        else:

            if self.verbose:
                print('[INFO] No georeferencing .points file provided. Skipping georeferencing.')

        self.georeferencing_has_run = True

        return None

    def _run_georeferencing_orientation(self) -> None:

        datacube_flipped = check_star_tracker_orientation(adcs_samples=self.adcs['adcssamples'],
                                                         quaternion_s=self.adcs['quaternion_s'],
                                                         quaternion_x=self.adcs['quaternion_x'],
                                                         quaternion_y=self.adcs['quaternion_y'],
                                                         quaternion_z=self.adcs['quaternion_z'],
                                                         velocity_x=self.adcs['velocity_x'],
                                                         velocity_y=self.adcs['velocity_y'],
                                                         velocity_z=self.adcs['velocity_z'])

        if not datacube_flipped:

            if self.l1a_cube is not None:
                self.l1a_cube = self.l1a_cube[:, ::-1, :]

            if self.l1b_cube is not None:  
                self.l1b_cube = self.l1b_cube[:, ::-1, :]
                
            if self.l2a_cubes is not None:
                if isinstance(self.l2a_cubes, dict):
                    for key in self.l2a_cubes.keys():
                        self.l2a_cubes = self.l2a_cubes[key][:, ::-1, :]

        self.datacube_flipped = datacube_flipped

        return None
    
    def _check_georeferencing_has_run(self) -> bool:

        return self.georeferencing_has_run
            

    # Calibration functions
        
    def _run_calibration(self, overwrite: bool = False) -> None:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """


        if self._check_calibration_has_run() and not overwrite:

            if self.verbose:
                    print("[INFO] Calibration has already been run. Skipping.")

            return None

        if self.verbose:
            print('[INFO] Running calibration routines...')

        self._set_calibration_coeff_files()
        self._set_calibration_coeffs()
        self._set_wavelengths()
        self._set_srf()

        self.l1b_cube = self._run_radiometric_calibration()
        self.l1b_cube = self._run_smile_correction(cube=self.l1b_cube)
        self.l1b_cube = self._run_destriping_correction(cube=self.l1b_cube)

        self.calibration_has_run = True

        return None

    def _run_radiometric_calibration(self, cube: np.ndarray = None) -> np.ndarray:

        # Radiometric calibration

        if cube is None:
            cube = self.l1a_cube

        if self.verbose:
            print("[INFO] Running radiometric calibration...")

        cube = self._get_flipped_cube(cube=cube)

        cube = run_radiometric_calibration(cube=cube, 
                                           background_value=self.background_value,
                                           exp=self.exposure,
                                           image_height=self.image_height,
                                           image_width=self.image_width,
                                           frame_count=self.frame_count,
                                           rad_coeffs=self.rad_coeffs)

        cube = self._get_flipped_cube(cube=cube)

        # TODO: The factor by 10 is to fix a bug in which the coeff have a factor of 10
        cube = cube / 10

        return cube

    def _run_smile_correction(self, cube: np.ndarray = None) -> np.ndarray:

        # Smile correction

        if cube is None:
            cube = self.l1a_cube

        if self.verbose:
            print("[INFO] Running smile correction...")

        cube = self._get_flipped_cube(cube=cube)

        cube = run_smile_correction(cube=cube, 
                                    smile_coeffs=self.smile_coeffs)

        cube = self._get_flipped_cube(cube=cube)

        return cube

    def _run_destriping_correction(self, cube: np.ndarray = None) -> np.ndarray:

        # Destriping

        if cube is None:
            cube = self.l1a_cube

        if self.verbose:
            print("[INFO] Running destriping correction...")

        cube = self._get_flipped_cube(cube=cube)

        cube = run_destriping_correction(cube=cube, 
                                         destriping_coeffs=self.destriping_coeffs)

        cube = self._get_flipped_cube(cube=cube)

        return cube

    def _set_calibration_coeffs(self) -> None:

        self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)

        return None

    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        self._set_rad_coeff_file()
        self._set_smile_coeff_file()
        self._set_destriping_coeff_file()
        self._set_spectral_coeff_file()

        return None

    def _set_rad_coeff_file(self, rad_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Get the absolute path for the radiometric coefficients.

        :param rad_coeff_file: Path to radiometric coefficients file (optional)

        :return: None.
        """

        if rad_coeff_file:
            self.rad_coeff_file = rad_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.csv"

            case "nominal":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_radiometric = None

        if csv_file_radiometric:
            rad_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_radiometric}')
        else: 
            rad_coeff_file = None

        self.rad_coeff_file = rad_coeff_file

        return None

    def _set_smile_coeff_file(self, smile_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Get the absolute path for the smile coefficients.

        :param smile_coeff_file: Path to smile coefficients file (optional)

        :return: None.
        """

        if smile_coeff_file:
            self.smile_coeff_file = smile_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.csv"   

            case "nominal":
                csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_smile = None

        if csv_file_smile:
            smile_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_smile}')
        else:
            smile_coeff_file = csv_file_smile

        self.smile_coeff_file = smile_coeff_file

        return None

    def _set_destriping_coeff_file(self, destriping_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Get the absolute path for the destriping coefficients.

        :param destriping_coeff_file: Path to destriping coefficients file (optional)

        :return: None.
        """

        if destriping_coeff_file:
            self.destriping_coeff_file = destriping_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_destriping = None

            case "nominal":
                csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_destriping = None

        if csv_file_destriping:
            destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_destriping}')
        else:
            destriping_coeff_file = csv_file_destriping

        self.destriping_coeff_file = destriping_coeff_file

        return None

    def _set_spectral_coeff_file(self, spectral_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Get the absolute path for the spectral coefficients (wavelengths).

        :param spectral_coeff_file: Path to spectral coefficients file (optional)

        :return: None.
        """

        if spectral_coeff_file:
            self.spectral_coeff_file = spectral_coeff_file
            return None
        
        csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"

        spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_spectral}')

        self.spectral_coeff_file = spectral_coeff_file

        return None

    def _set_wavelengths(self) -> None:

        if self.spectral_coeffs is not None:
            self.wavelengths = self.spectral_coeffs
        else:
            self.wavelengths = None

        return None

    # TODO: move to calibration module?
    def _set_srf(self) -> None:
        """
        Get Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
        used to estimate Sigma for an assumed gaussian distribution of each SRF per band.
        """

        if not any(self.wavelengths):
            self.srf = None

        fwhm_nm = 3.33
        sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

        srf = []
        for band in self.wavelengths:
            center_lambda_nm = band
            start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm), 4)
            soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm), 4)

            srf_wl = [center_lambda_nm]
            lower_wl = []
            upper_wl = []
            for ele in self.wavelengths:
                if start_lambda_nm < ele < center_lambda_nm:
                    lower_wl.append(ele)
                elif center_lambda_nm < ele < soft_end_lambda_nm:
                    upper_wl.append(ele)

            # Make symmetric
            while len(lower_wl) > len(upper_wl):
                lower_wl.pop(0)
            while len(upper_wl) > len(lower_wl):
                upper_wl.pop(-1)

            srf_wl = lower_wl + srf_wl + upper_wl

            good_idx = [(True if ele in srf_wl else False) for ele in self.wavelengths]

            # Delta based on Hypso Sampling (Wavelengths)
            gx = None
            if len(srf_wl) == 1:
                gx = [0]
            else:
                gx = np.linspace(-3 * sigma_nm, 3 * sigma_nm, len(srf_wl))
            gaussian_srf = np.exp(
                -(gx / sigma_nm) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

            # Get final wavelength and SRF
            srf_wl_single = self.wavelengths
            srf_single = np.zeros_like(srf_wl_single)
            srf_single[good_idx] = gaussian_srf

            srf.append([srf_wl_single, srf_single])

        self.srf = srf

        return None

    def _check_calibration_has_run(self) -> bool:

        return self.calibration_has_run
 

    # Geometry computation functions

    def _run_geometry_computation(self, overwrite: bool = False) -> None:

        if self._check_geometry_computation_has_run() and not overwrite:

            if self.verbose:
                    print("[INFO] Geometry computation has already been run. Skipping.")

            return None

        if self.verbose:
            print("[INFO] Running geometry computation...")

        framepose_data = interpolate_at_frame(adcs_pos_df=self.adcs_pos_df,
                                              adcs_quat_df=self.adcs_quat_df,
                                              timestamps_srv=self.timing['timestamps_srv'],
                                              frame_count=self.capture_config['frame_count'],
                                              fps=self.capture_config['fps'],
                                              exposure=self.capture_config['exposure'],
                                              verbose=self.verbose
                                              )


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
                                             verbose=self.verbose
                                             )

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

        self.geometry_computation_has_run = True

        return None

    def _check_geometry_computation_has_run(self) -> bool:

        return self.geometry_computation_has_run


    # Atmospheric correction functions

    def _run_atmospheric_correction(self, product: str, overwrite: bool = False) -> None:

        if self._check_atmospheric_correction_has_run(product=product) and not overwrite:

            if self.verbose:
                print("[INFO] Atmospheric correction has already been run. Skipping.")

            return None

        if self.l2a_cubes is None:
            self.l2a_cubes = {}

        if product is not None:
            product = product.lower()

        match product:
            case "6sv1":
                if self.verbose: 
                    print("[INFO] Running 6SV1 atmospheric correction")
                self.l2a_cubes[product] = self._run_6sv1_atmospheric_correction()
            case "acolite":
                if self.verbose: 
                    print("[INFO] Running ACOLITE atmospheric correction")
                self.l2a_cubes[product] = self._run_acolite_atmospheric_correction()
            case "machi":
                if self.verbose: 
                    print("[INFO] Running MACHI atmospheric correction")
                self.l2a_cubes[product] = self._run_machi_atmospheric_correction()  

            case _:
                print("[ERROR] No such atmospheric correction product supported!")
                return None

        self.atmospheric_correction_has_run = True

        return None

    def _run_6sv1_atmospheric_correction(self) -> np.ndarray:

        # Py6S Atmospheric Correction
        # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
        # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
        # atmos_dict = {
        #     'aot550': 0.01,
        #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
        # }
        # AOT550 parameter gotten from: https://giovanni.gsfc.nasa.gov/giovanni/

        self._run_calibration()
        self._run_geometry_computation()

        # TODO: which values should we use?
        if self.latitudes is None:
            latitudes = self.latitudes_original # fall back on geometry computed values
        else:
            latitudes = self.latitudes

        if self.longitudes is None:
            longitudes = self.longitudes_original # fall back on geometry computed values
        else:
            longitudes = self.longitudes

        atmos_params = {
            'aot550': 0.0580000256
        }

        time_capture = parser.parse(self.iso_time)

        atmos_corrected_cube = run_py6s(wavelengths=self.wavelengths, 
                                        hypercube_L1=self.l1b_cube, 
                                        lat_2d_array=latitudes,
                                        lon_2d_array=longitudes,
                                        solar_azimuth_angles=self.solar_azimuth_angles,
                                        solar_zenith_angles=self.solar_zenith_angles,
                                        sat_azimuth_angles=self.sat_azimuth_angles,
                                        sat_zenith_angles=self.sat_zenith_angles,
                                        iso_time=self.iso_time,
                                        py6s_dict=atmos_params, 
                                        time_capture=time_capture,
                                        srf=self.srf)
        
        return atmos_corrected_cube

    def _run_acolite_atmospheric_correction(self) -> None:

        self._run_calibration()
        self._run_geometry_computation()
        self._check_write_l1b_nc_file_has_run(run_on_false=True)

        # user and password from https://urs.earthdata.nasa.gov/profile
        # optional but good
        atmos_params = {
            'user':'alvarof',
            'password':'nwz7xmu8dak.UDG9kqz'
        }

        if not self.l1b_nc_file.is_file():
            raise Exception("No -l1b.nc file found")
        
        atmos_corrected_cube = run_acolite(output_path=self.tmp_dir, 
                                          atmos_dict=atmos_params, 
                                          nc_file_acoliteready=self.l1b_nc_file)

        return atmos_corrected_cube
    
    # TODO
    def _run_machi_atmospheric_correction(self) -> None:

        print("[WARNING] Minimal Atmospheric Compensation for Hyperspectral Imagers (MACHI) atmospheric correction has not been enabled.")

        return None

        self._run_calibration()
        self._run_geometry_computation()

        return None

        #T, A, objs = run_machi(cube=self.l1b_cube, verbose=self.verbose)
    
    def _check_atmospheric_correction_has_run(self, product: str = None) -> bool:

        if self.atmospheric_correction_has_run:
            if product is None:
                return True
            elif product.lower() in self.l2a_cubes.keys():
                return True
            else:
                return False
        return False


    # Land mask functions

    def _run_land_mask(self, land_mask: str, overwrite: bool = False) -> None:

        if self._check_land_mask_has_run(land_mask=land_mask) and not overwrite:

            if self.verbose:
                print("[INFO] Land mask has already been run. Skipping.")

            return None

        if self.land_masks is None:
            self.land_masks = {}

        key = land_mask

        if key is not None:
            key = key.lower()

        match key:

            case "global":

                if self.verbose:
                    print("[INFO] Running land mask generation...")

                returned_land_mask = self._run_global_land_mask()

                self.land_masks[key] = returned_land_mask
                self._update_active_land_mask(land_mask=key, override=False)

            case "ndwi":

                if self.verbose:
                    print("[INFO] Running NDWI land mask generation...")

                returned_land_mask = self._run_ndwi_land_mask()

                self.land_masks[key] = returned_land_mask
                self._update_active_land_mask(land_mask=key, override=False)

            case "threshold":

                if self.verbose:
                    print("[INFO] Running threshold land mask generation...")

                returned_land_mask = self._run_threshold_land_mask()

                self.land_masks[key] = returned_land_mask
                self._update_active_land_mask(land_mask=key, override=False)

            case _:

                print("[WARNING] No such land mask supported!")
                return None

        self.land_mask_has_run = True

        return None

    def _run_global_land_mask(self) -> np.ndarray:

        self._run_georeferencing()

        land_mask = run_global_land_mask(spatial_dimensions=self.spatial_dimensions,
                                        latitudes=self.latitudes,
                                        longitudes=self.longitudes
                                        )
        
        return land_mask

    def _run_ndwi_land_mask(self) -> np.ndarray:

        self._run_calibration()

        land_mask = run_ndwi_land_mask(cube=self.l1b_cube, 
                                       wavelengths=self.wavelengths,
                                       verbose=self.verbose)

        return land_mask
    
    def _run_threshold_land_mask(self) -> np.ndarray:

        self._run_calibration()

        land_mask = run_threshold_land_mask(cube=self.l1b_cube,
                                            wavelengths=self.wavelengths,
                                            verbose=self.verbose)
    
        return land_mask

    def _update_active_land_mask(self, land_mask: str = None, override: bool = False) -> None:

        if land_mask is None:
            return None

        if land_mask.lower() not in self.land_masks.keys():
            return None

        if self.active_land_mask is None or override:
            self.active_land_mask = self.land_masks[land_mask.lower()]
            self.active_land_mask_name = land_mask.lower()

        self._update_active_mask()

        return None

    def _get_active_land_mask(self) -> tuple[str, np.ndarray]:

        return self.active_land_mask_name, self.active_land_mask

    def _check_land_mask_has_run(self, land_mask: str = None) -> bool:

        if self.land_mask_has_run:
            if land_mask is None:
                return True  
            elif land_mask.lower() in self.land_masks.keys():
                return True  
            else:
                return False
        return False
       

    # Cloud mask functions

    def _run_cloud_mask(self, cloud_mask: str='default', overwrite: bool = False) -> None:

        if self._check_cloud_mask_has_run(cloud_mask=cloud_mask) and not overwrite:

            if self.verbose:
                print("[INFO] Cloud mask has already been run. Skipping.")

            return None

        if self.cloud_masks is None:
            self.cloud_masks = {}

        key = cloud_mask

        if key is not None:
            key = key.lower()

        match key:

            case 'default':

                if self.verbose:
                    print("[INFO] Running cloud mask generation...")
                    print("[WARNING] Cloud mask generation has not been implemented.")

                returned_cloud_mask = run_cloud_mask()

                self.cloud_masks[key] = returned_cloud_mask
                self._update_active_cloud_mask(cloud_mask=key, override=False)

            case _:
                print("[WARNING] No such cloud mask supported!")
                return None

        self.cloud_mask_has_run = True

        return None

    def _update_active_cloud_mask(self, cloud_mask: str = None, override: bool = False) -> None:

        if cloud_mask is None:
            return None

        if cloud_mask.lower() not in self.cloud_masks.keys():
            return None

        if self.active_cloud_mask is None or override:
            self.active_cloud_mask = self.cloud_masks[cloud_mask.lower()]
            self.active_cloud_mask_name = cloud_mask.lower()

        self._update_active_mask()

        return None

    def _get_active_cloud_mask(self) -> tuple[str, np.ndarray]:

        return self.active_cloud_mask_name, self.active_cloud_mask

    def _check_cloud_mask_has_run(self, cloud_mask: str = None) -> bool:

        if self.cloud_mask_has_run:
            if cloud_mask is None:
                return True     
            elif cloud_mask.lower() in self.land_masks.keys():
                return True     
            else:
                return False

        return False


    # Unified mask functions

    def _update_active_mask(self) -> None:

        land_mask_name, land_mask = self._get_active_land_mask()
        cloud_mask_name, cloud_mask = self._get_active_cloud_mask()

        if land_mask is None and cloud_mask is None:
            return None
        
        elif land_mask is None:
            self.active_mask = cloud_mask
        
        elif cloud_mask is None:
            self.active_mask = land_mask
        
        else:
            self.active_mask = land_mask | cloud_mask

        return None

    def _get_active_mask(self) -> np.ndarray:

        return self.active_mask


    # Chlorophyll estimation functions

    def _run_chlorophyll_estimation(self, product: str = None, overwrite: bool = False) -> None:

        if self._check_chlorophyll_estimation_has_run(product=product) and not overwrite:

            if self.verbose:
                print("[INFO] Chlorophyll estimation has already been run. Skipping.")

            return None

        if self.chl is None:
            self.chl = {}

        key = product

        if key is not None:
            key = product.lower()

        match product:

            case "band_ratio":

                if self.verbose:
                    print("[INFO] Running band ratio chlorophyll estimation...")

                self.chl[key] = self._run_band_ratio_chlorophyll_estimation()

            case "6sv1_aqua":

                if self.verbose:
                    print("[INFO] Running 6SV1 AQUA Tuned chlorophyll estimation...")

                self.chl[key] = self._run_6sv1_aqua_tuned_chlorophyll_estimation()

            case "acolite_aqua":

                if self.verbose:
                    print("[INFO] Running ACOLITE AQUA Tuned chlorophyll estimation...")

                self.chl[key] = self._run_acolite_aqua_tuned_chlorophyll_estimation()

            case _:
                print("[ERROR] No such chlorophyll estimation product supported!")
                return None

        self.chlorophyll_estimation_has_run = True

        return None

    def _run_band_ratio_chlorophyll_estimation(self) -> None:

        self._run_calibration()

        chl = run_band_ratio_chlorophyll_estimation(cube=self.l1b_cube,
                                                    mask=self.active_mask, 
                                                    wavelengths=self.wavelengths,
                                                    spatial_dimensions=self.spatial_dimensions
                                                    )

        return chl

    # TODO: complete
    def _run_6sv1_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> None:

        run_tuned_chlorophyll_estimation

        pass

    # TODO: complete
    def _run_acolite_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> None:

        pass

    def _check_chlorophyll_estimation_has_run(self, product: str = None) -> bool:

        if self.chlorophyll_estimation_has_run:
            if product is None:
                return True
            elif product.lower() in self.chl.keys():
                return True
            else:
                return False
        return False


    # L1a file output

    # TODO
    def _write_l1a_nc_file(self, path: str = None, overwrite: bool = False) -> None:

        return None

    # TODO
    def _check_write_l1a_nc_file_has_run(self) -> bool:
        return False


    # L1b file output

    # TODO: refactor
    def _write_l1b_nc_file(self, path: str = None, overwrite: bool = False) -> None:
        """
        Create a l1b.nc file using the radiometrically corrected data. Same structure from the original l1a.nc file
        is used. Required to run ACOLITE as the input is a radiometrically corrected .nc file.

        :return: Nothing.
        """

        #hypso_nc_path = self.l1a_nc_file
        #old_nc = nc.Dataset(hypso_nc_path, 'r', format='NETCDF4')

        #new_path = hypso_nc_path
        #new_path = str(new_path).replace('l1a.nc', 'l1b.nc')

        #if Path(new_path).is_file():
        #    print("L1b.nc file already exists. Not creating it.")
        #    self.l1b_nc_file = Path(new_path)
        #    return

        if self._check_write_l1b_nc_file_has_run() and not overwrite:

            if self.verbose:
                    print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        # Open L1a file
        old_nc = nc.Dataset(self.l1a_nc_file, 'r', format='NETCDF4')

        # Create a new NetCDF file
        with (nc.Dataset(self.l1b_nc_file, 'w', format='NETCDF4') as netfile):
            bands = self.image_width
            lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
            samples = self.image_height  # AKA Cols

            # Set top level attributes -------------------------------------------------
            for md in old_nc.ncattrs():
                set_or_create_attr(netfile,
                                   md,
                                   old_nc.getncattr(md))

            # Manual Replacement
            set_or_create_attr(netfile,
                               attr_name="radiometric_file",
                               attr_value=str(Path(self.rad_coeff_file).name))

            set_or_create_attr(netfile,
                               attr_name="smile_file",
                               attr_value=str(Path(self.smile_coeff_file).name))

            # Destriping Path is the only one which can be None
            if self.destriping_coeff_file is None:
                set_or_create_attr(netfile,
                                   attr_name="destriping",
                                   attr_value="No-File")
            else:
                set_or_create_attr(netfile,
                                   attr_name="destriping",
                                   attr_value=str(Path(self.destriping_coeff_file).name))

            set_or_create_attr(netfile, attr_name="spectral_file", attr_value=str(Path(self.spectral_coeff_file).name))

            set_or_create_attr(netfile, attr_name="processing_level", attr_value="L1B")

            # Create dimensions
            netfile.createDimension('lines', lines)
            netfile.createDimension('samples', samples)
            netfile.createDimension('bands', bands)

            # Create groups
            netfile.createGroup('logfiles')

            netfile.createGroup('products')

            netfile.createGroup('metadata')

            netfile.createGroup('navigation')

            # Adding metadata ---------------------------------------
            meta_capcon = netfile.createGroup('metadata/capture_config')
            for md in old_nc['metadata']["capture_config"].ncattrs():
                set_or_create_attr(meta_capcon,
                                   md,
                                   old_nc['metadata']["capture_config"].getncattr(md))

            # Adding Metatiming --------------------------------------
            meta_timing = netfile.createGroup('metadata/timing')
            for md in old_nc['metadata']["timing"].ncattrs():
                set_or_create_attr(meta_timing,
                                   md,
                                   old_nc['metadata']["timing"].getncattr(md))

            # Meta Temperature -------------------------------------------
            meta_temperature = netfile.createGroup('metadata/temperature')
            for md in old_nc['metadata']["temperature"].ncattrs():
                set_or_create_attr(meta_temperature,
                                   md,
                                   old_nc['metadata']["temperature"].getncattr(md))

            # Meta Corrections -------------------------------------------
            meta_adcs = netfile.createGroup('metadata/adcs')
            for md in old_nc['metadata']["adcs"].ncattrs():
                set_or_create_attr(meta_adcs,
                                   md,
                                   old_nc['metadata']["adcs"].getncattr(md))

            # Meta Corrections -------------------------------------------
            meta_corrections = netfile.createGroup('metadata/corrections')
            for md in old_nc['metadata']["corrections"].ncattrs():
                set_or_create_attr(meta_corrections,
                                   md,
                                   old_nc['metadata']["corrections"].getncattr(md))

            # Meta Database -------------------------------------------
            meta_database = netfile.createGroup('metadata/database')
            for md in old_nc['metadata']["database"].ncattrs():
                set_or_create_attr(meta_database,
                                   md,
                                   old_nc['metadata']["database"].getncattr(md))

            # Set pseudoglobal vars like compression level
            COMP_SCHEME = 'zlib'  # Default: zlib
            COMP_LEVEL = 4  # Default (when scheme != none): 4
            COMP_SHUFFLE = True  # Default (when scheme != none): True

            # Create and populate variables
            Lt = netfile.createVariable(
                'products/Lt', 'uint16',
                ('lines', 'samples', 'bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            Lt.units = "W/m^2/micrometer/sr"
            Lt.long_name = "Top of Atmosphere Measured Radiance"
            Lt.wavelength_units = "nanometers"
            Lt.fwhm = [5.5] * bands
            Lt.wavelengths = np.around(self.spectral_coeffs, 1)
            Lt[:] = self.l1b_cube

            # ADCS Timestamps ----------------------------------------------------
            len_timestamps = old_nc.dimensions["adcssamples"].size
            netfile.createDimension('adcssamples', len_timestamps)

            meta_adcs_timestamps = netfile.createVariable(
                'metadata/adcs/timestamps', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )

            meta_adcs_timestamps[:] = old_nc['metadata']["adcs"]["timestamps"][:]

            # ADCS Position X -----------------------------------------------------
            meta_adcs_position_x = netfile.createVariable(
                'metadata/adcs/position_x', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_position_x[:] = old_nc['metadata']["adcs"]["position_x"][:]

            # ADCS Position Y -----------------------------------------------------
            meta_adcs_position_y = netfile.createVariable(
                'metadata/adcs/position_y', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_position_y[:] = old_nc['metadata']["adcs"]["position_y"][:]

            # ADCS Position Z -----------------------------------------------------
            meta_adcs_position_z = netfile.createVariable(
                'metadata/adcs/position_z', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_position_z[:] = old_nc['metadata']["adcs"]["position_z"][:]

            # ADCS Velocity X -----------------------------------------------------
            meta_adcs_velocity_x = netfile.createVariable(
                'metadata/adcs/velocity_x', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_velocity_x[:] = old_nc['metadata']["adcs"]["velocity_x"][:]

            # ADCS Velocity Y -----------------------------------------------------
            meta_adcs_velocity_y = netfile.createVariable(
                'metadata/adcs/velocity_y', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_velocity_y[:] = old_nc['metadata']["adcs"]["velocity_y"][:]

            # ADCS Velocity Z -----------------------------------------------------
            meta_adcs_velocity_z = netfile.createVariable(
                'metadata/adcs/velocity_z', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_velocity_z[:] = old_nc['metadata']["adcs"]["velocity_z"][:]

            # ADCS Quaternion S -----------------------------------------------------
            meta_adcs_quaternion_s = netfile.createVariable(
                'metadata/adcs/quaternion_s', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_quaternion_s[:] = old_nc['metadata']["adcs"]["quaternion_s"][:]

            # ADCS Quaternion X -----------------------------------------------------
            meta_adcs_quaternion_x = netfile.createVariable(
                'metadata/adcs/quaternion_x', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_quaternion_x[:] = old_nc['metadata']["adcs"]["quaternion_x"][:]

            # ADCS Quaternion Y -----------------------------------------------------
            meta_adcs_quaternion_y = netfile.createVariable(
                'metadata/adcs/quaternion_y', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_quaternion_y[:] = old_nc['metadata']["adcs"]["quaternion_y"][:]

            # ADCS Quaternion Z -----------------------------------------------------
            meta_adcs_quaternion_z = netfile.createVariable(
                'metadata/adcs/quaternion_z', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_quaternion_z[:] = old_nc['metadata']["adcs"]["quaternion_z"][:]

            # ADCS Angular Velocity X -----------------------------------------------------
            meta_adcs_angular_velocity_x = netfile.createVariable(
                'metadata/adcs/angular_velocity_x', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_angular_velocity_x[:] = old_nc['metadata']["adcs"]["angular_velocity_x"][:]

            # ADCS Angular Velocity Y -----------------------------------------------------
            meta_adcs_angular_velocity_y = netfile.createVariable(
                'metadata/adcs/angular_velocity_y', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_angular_velocity_y[:] = old_nc['metadata']["adcs"]["angular_velocity_y"][:]

            # ADCS Angular Velocity Z -----------------------------------------------------
            meta_adcs_angular_velocity_z = netfile.createVariable(
                'metadata/adcs/angular_velocity_z', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_angular_velocity_z[:] = old_nc['metadata']["adcs"]["angular_velocity_z"][:]

            # ADCS ST Quaternion S -----------------------------------------------------
            meta_adcs_st_quaternion_s = netfile.createVariable(
                'metadata/adcs/st_quaternion_s', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_st_quaternion_s[:] = old_nc['metadata']["adcs"]["st_quaternion_s"][:]

            # ADCS ST Quaternion X -----------------------------------------------------
            meta_adcs_st_quaternion_x = netfile.createVariable(
                'metadata/adcs/st_quaternion_x', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_st_quaternion_x[:] = old_nc['metadata']["adcs"]["st_quaternion_x"][:]

            # ADCS ST Quaternion Y -----------------------------------------------------
            meta_adcs_st_quaternion_y = netfile.createVariable(
                'metadata/adcs/st_quaternion_y', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_st_quaternion_y[:] = old_nc['metadata']["adcs"]["st_quaternion_y"][:]

            # ADCS ST Quaternion Z -----------------------------------------------------
            meta_adcs_st_quaternion_z = netfile.createVariable(
                'metadata/adcs/st_quaternion_z', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_st_quaternion_z[:] = old_nc['metadata']["adcs"]["st_quaternion_z"][:]

            # ADCS Control Error -----------------------------------------------------
            meta_adcs_control_error = netfile.createVariable(
                'metadata/adcs/control_error', 'f8',
                ('adcssamples',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE
            )
            meta_adcs_control_error[:] = old_nc['metadata']["adcs"]["control_error"][:]

            # Capcon File -------------------------------------------------------------
            meta_capcon_file = netfile.createVariable(
                'metadata/capture_config/file', 'str')  # str seems necessary for storage of an arbitrarily large scalar
            meta_capcon_file[()] = old_nc['metadata']["capture_config"]["file"][:]  # [()] assignment of scalar to array

            # Metadata: Rad calibration coeff ----------------------------------------------------
            len_radrows = self.rad_coeffs.shape[0]
            len_radcols = self.rad_coeffs.shape[1]

            netfile.createDimension('radrows', len_radrows)
            netfile.createDimension('radcols', len_radcols)
            meta_corrections_rad = netfile.createVariable(
                'metadata/corrections/rad_matrix', 'f4',
                ('radrows', 'radcols'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_rad[:] = self.rad_coeffs

            # Metadata: Spectral coeff ----------------------------------------------------
            len_spectral = self.wavelengths.shape[0]
            netfile.createDimension('specrows', len_spectral)
            meta_corrections_spec = netfile.createVariable(
                'metadata/corrections/spec_coeffs', 'f4',
                ('specrows',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_spec[:] = self.spectral_coeffs

            # Meta Temperature File ---------------------------------------------------------
            meta_temperature_file = netfile.createVariable(
                'metadata/temperature/file', 'str')
            meta_temperature_file[()] = old_nc['metadata']["temperature"]["file"][:]

            # Bin Time ----------------------------------------------------------------------
            bin_time = netfile.createVariable(
                'metadata/timing/bin_time', 'uint16',
                ('lines',))
            bin_time[:] = old_nc['metadata']["timing"]["bin_time"][:]

            # Timestamps -------------------------------------------------------------------
            timestamps = netfile.createVariable(
                'metadata/timing/timestamps', 'uint32',
                ('lines',))
            timestamps[:] = old_nc['metadata']["timing"]["timestamps"][:]

            # Timestamps Service -----------------------------------------------------------
            timestamps_srv = netfile.createVariable(
                'metadata/timing/timestamps_srv', 'f8',
                ('lines',))
            timestamps_srv[:] = old_nc['metadata']["timing"]["timestamps_srv"][:]
            
            # Create Navigation Group --------------------------------------
            try:
                navigation_group = netfile.createGroup('navigation')
                sat_zenith_angle = self.sat_zenith_angles
                sat_azimuth_angle = self.sat_azimuth_angles

                solar_zenith_angle = self.solar_zenith_angles
                solar_azimuth_angle = self.solar_azimuth_angles

                # Unix time -----------------------
                time = netfile.createVariable('navigation/unixtime', 'u8', ('lines',))
                frametime_pose_file = find_file(self.tmp_dir, "frametime-pose", ".csv")
                df = pd.read_csv(frametime_pose_file)
                time[:] = df["timestamp"].values

                # Sensor Zenith --------------------------
                sensor_z = netfile.createVariable(
                    'navigation/sensor_zenith', 'f4', ('lines', 'samples'),
                    # compression=COMP_SCHEME,
                    # complevel=COMP_LEVEL,
                    # shuffle=COMP_SHUFFLE,
                )
                sensor_z[:] = sat_zenith_angle.reshape(self.spatial_dimensions)
                sensor_z.long_name = "Sensor Zenith Angle"
                sensor_z.units = "degrees"
                # sensor_z.valid_range = [-180, 180]
                sensor_z.valid_min = -180
                sensor_z.valid_max = 180

                # Sensor Azimuth ---------------------------
                sensor_a = netfile.createVariable(
                    'navigation/sensor_azimuth', 'f4', ('lines', 'samples'),
                    # compression=COMP_SCHEME,
                    # complevel=COMP_LEVEL,
                    # shuffle=COMP_SHUFFLE,
                )
                sensor_a[:] = sat_azimuth_angle.reshape(self.spatial_dimensions)
                sensor_a.long_name = "Sensor Azimuth Angle"
                sensor_a.units = "degrees"
                # sensor_a.valid_range = [-180, 180]
                sensor_a.valid_min = -180
                sensor_a.valid_max = 180

                # Solar Zenith ----------------------------------------
                solar_z = netfile.createVariable(
                    'navigation/solar_zenith', 'f4', ('lines', 'samples'),
                    # compression=COMP_SCHEME,
                    # complevel=COMP_LEVEL,
                    # shuffle=COMP_SHUFFLE,
                )
                solar_z[:] = solar_zenith_angle.reshape(self.spatial_dimensions)
                solar_z.long_name = "Solar Zenith Angle"
                solar_z.units = "degrees"
                # solar_z.valid_range = [-180, 180]
                solar_z.valid_min = -180
                solar_z.valid_max = 180

                # Solar Azimuth ---------------------------------------
                solar_a = netfile.createVariable(
                'navigation/solar_azimuth', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
                )
                solar_a[:] = solar_azimuth_angle.reshape(self.spatial_dimensions)
                solar_a.long_name = "Solar Azimuth Angle"
                solar_a.units = "degrees"
                # solar_a.valid_range = [-180, 180]
                solar_a.valid_min = -180
                solar_a.valid_max = 180
        
                # Latitude ---------------------------------
                latitude = netfile.createVariable(
                    'navigation/latitude', 'f4', ('lines', 'samples'),
                    # compression=COMP_SCHEME,
                    # complevel=COMP_LEVEL,
                    # shuffle=COMP_SHUFFLE,
                )
                # latitude[:] = lat.reshape(frames, lines)
                latitude[:] = self.latitudes
                latitude.long_name = "Latitude"
                latitude.units = "degrees"
                # latitude.valid_range = [-180, 180]
                latitude.valid_min = -180
                latitude.valid_max = 180

                # Longitude ----------------------------------
                longitude = netfile.createVariable(
                    'navigation/longitude', 'f4', ('lines', 'samples'),
                    # compression=COMP_SCHEME,
                    # complevel=COMP_LEVEL,
                    # shuffle=COMP_SHUFFLE,
                )
                # longitude[:] = lon.reshape(frames, lines)
                longitude[:] = self.longitudes
                longitude.long_name = "Longitude"
                longitude.units = "degrees"
                # longitude.valid_range = [-180, 180]
                longitude.valid_min = -180
                longitude.valid_max = 180

            except Exception as ex:
                print("Navigation Group and Attributes already exist")
                print(ex)


        old_nc.close()

        # Update
        #self.l1b_nc_file = Path(new_path)

        self.write_l1b_nc_file_has_run = True

        return None

    def _check_write_l1b_nc_file_has_run(self) -> bool:

        l1b_nc_file_exists = Path(self.l1b_nc_file).is_file()

        if self.write_l1b_nc_file_has_run and l1b_nc_file_exists:
            return True
        
        return False


    # L2a file output

    # TODO
    def _write_l2a_nc_file(self, path: str = None, product: str = None, overwrite: bool = False) -> None:

        return None

    # TODO
    def _check_write_l2a_nc_file_has_run(self, product: str = None) -> bool:
        return False


    # Other functions

    def _get_flipped_cube(self, cube: np.ndarray) -> np.ndarray:

        if self.datacube_flipped is None:
            return cube
        else:
            if self.datacube_flipped:
                return cube[:, ::-1, :]

            else:
                return cube

        return cube



    # Public methods

    def get_l1a_cube(self) -> np.ndarray:

        return self.l1a_cube

    def generate_l1b_cube(self) -> None:

        self._run_calibration()

        return None

    def get_l1b_cube(self) -> np.ndarray:

        if self._check_calibration_has_run():

            return self.l1b_cube

        if self.verbose:
            print("[ERROR] L1b cube has not yet been generated.")

        return None

    def generate_l2a_cube(self, product: Literal["acolite", "6sv1", "machi"] = "6sv1") -> None:

        self._run_atmospheric_correction(product=product)

        return None

    def get_l2a_cube(self, product: Literal["acolite", "6sv1", "machi"] = "6sv1") -> np.ndarray:

        if product and self._check_atmospheric_correction_has_run(product=product):

            return self.l2a_cubes[product.lower()]
        
        if self.verbose:
            print("[ERROR] " + product.upper() + " L2a cube has not yet been generated.")

        return None

    def get_l2a_cube_dict(self) -> dict:

        return self.l2a_cubes

    def generate_land_mask(self, land_mask: Literal["global", "ndwi", "threshold"] = "global") -> None:

        self._run_land_mask(land_mask=land_mask)

        return None

    def get_land_mask(self, land_mask: Literal["global", "ndwi", "threshold"] = "global") -> np.ndarray:

        if land_mask and self._check_land_mask_has_run(land_mask=land_mask):

            return self.land_masks[land_mask.lower()]
        
        if self.verbose:
            print("[ERROR] " + land_mask + " land mask has not yet been generated.")

        return None

    def get_land_mask_dict(self) -> dict:

        return self.land_masks     
    
    def set_active_land_mask(self, land_mask: Literal["global", "ndwi", "threshold"] = "global"):

        self._update_active_land_mask(land_mask=land_mask, override=True)

    def get_active_land_mask(self) -> tuple[str, np.ndarray]:

        return self._get_active_cloud_mask()

    def generate_cloud_mask(self, cloud_mask: Literal["default"] = "default"):

        self._run_cloud_mask(cloud_mask=cloud_mask)

        return None

    def get_cloud_mask(self, cloud_mask: Literal["default"] = "default") -> np.ndarray:

        if cloud_mask and self._check_cloud_mask_has_run(cloud_mask=cloud_mask):

            return self.cloud_masks[cloud_mask.lower()]
        
        if self.verbose:
            print("[ERROR] " + cloud_mask + " cloud mask has not yet been generated.")
        
        return None
    
    def get_cloud_mask_dict(self) -> dict:

        return self.cloud_masks 

    def set_active_cloud_mask(self, cloud_mask: Literal["default"] = "default"):

        self._update_active_cloud_mask(self, cloud_mask=cloud_mask, override=True)

    def get_active_cloud_mask(self) -> tuple[str, np.ndarray]:

        return self._get_active_land_mask()

    def generate_chlorophyll_estimates(self, 
                                       product: Literal["band_ratio", "6sv1_aqua", "acolite_aqua"]='band_ratio',
                                       model = None):

        self._run_chlorophyll_estimation(self, product=product)

    def get_chlorophyll_estimates(self, 
                                 product: Literal["band_ratio", "6sv1_aqua", "acolite_aqua"]='band_ratio',
                                 ) -> np.ndarray:

        key = product.lower()

        return self.chl[key]

    def get_chlorophyll_estimates_dict(self) -> dict:

        return self.chl

    def generate_product(self):

        pass

    def get_product(self):

        pass

    def get_product_dict(self) -> dict:

        return self.products

    # TODO
    def get_toa_reflectance(self) -> np.ndarray:
        """
        Convert Top Of Atmosphere (TOA) Radiance to TOA Reflectance.

        :return: Array with TOA Reflectance.
        """
        # Get Local variables
        srf = self.srf
        toa_radiance = self.l1b_cube

        scene_date = parser.isoparse(self.iso_time)
        julian_day = scene_date.timetuple().tm_yday
        solar_zenith = self.solar_zenith_angle

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

        return toa_reflectance

    def get_nearest_pixel(self, latitude: float, longitude: float) -> tuple[int, int]:
        """
        Find the nearest pixel in a SwathDefinition given a target latitude and longitude.

        Parameters:
        - swath_def: SwathDefinition object containing latitude and longitude arrays
        - target_lat: Target latitude
        - target_lon: Target longitude

        Returns:
        - (i, j): Indices of the nearest pixel in the swath definition
        """
        # Wrap target coordinates in arrays

        if self.latitudes is None or self.longitudes is None:
            return None

        target_latitudes = np.array([latitude])
        target_longitudes = np.array([longitude])
        
        source_swath_def = geometry.SwathDefinition(lons=self.longitudes, lats=self.latitudes)
        target_swath_def = geometry.SwathDefinition(lons=target_longitudes, lats=target_latitudes)

        # Find nearest neighbor info
        valid_input_index, valid_output_index, index_array, distance_array = get_neighbour_info(
            source_swath_def, target_swath_def, radius_of_influence=np.inf, neighbours=1
        )

        if len(valid_input_index) > 0:
            nearest_index = np.unravel_index(index_array[0], source_swath_def.shape)
            return nearest_index
        else:
            return None

    # TODO check that this works and does image flip need to apply?
    def _haversine(self, lat1, lon1, lat2, lon2):
        """
        WARNING: ChatGPT wrote this... ()

        Calculate the great-circle distance between two points 
        on the Earth using the Haversine formula.
        """
        R = 6371.0  # Radius of the Earth in kilometers
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_nearest_pixel_haversine(self, latitude: float, longitude: float):
        """
        Find the nearest index in 2D latitude and longitude matrices
        to the given target latitude and longitude.

        Parameters:
        - target_lat: Target latitude
        - target_lon: Target longitude

        Returns:
        - (i, j): Tuple of indices in the matrices closest to the target coordinate
        """

        lat_matrix = self.latitudes
        lon_matrix = self.longitudes

        distances = self._haversine(lat_matrix, lon_matrix, latitude, longitude)
        nearest_index = np.unravel_index(np.argmin(distances), distances.shape)
        return nearest_index

    def get_l1a_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:

        if self.l1a_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]

        return spectrum, self.l1a_units

    def get_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:

        if self.l1b_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]

        return spectrum, self.l1b_units

    def get_l2a_spectrum(self, 
                        product: Literal["acolite", "6sv1", "machi"] = "6sv1",
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:


        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        try:
            spectrum = self.l2a_cubes[product][idx[0], idx[1], :]
        except KeyError:
            return None

        return spectrum, self.l2a_units

    def plot_l1a_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> np.ndarray:
        
        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]

        bands = range(0, len(spectrum))

        output_file = Path(self.nc_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(self.l1a_units)
        plt.xlabel("Wavelength (nm)")
        plt.title(f"(lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)
        plt.savefig(output_file)

    def plot_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> np.ndarray:
        
        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]

        output_file = Path(self.nc_dir, self.capture_name + '_l1b_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum)
        plt.ylabel(self.l1b_units)
        plt.xlabel("Wavelength (nm)")
        plt.title(f"(lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)
        plt.savefig(output_file)

    def plot_l2a_spectrum(self, 
                         product: Literal["acolite", "6sv1", "machi"] = "6sv1",
                         latitude=None, 
                         longitude=None,
                         x: int = None,
                         y: int = None
                         ) -> np.ndarray:
        
        if latitude is not None and longitude is not None:
            idx = self.get_nearest_pixel(latitude=latitude, longitude=longitude)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        try:
            spectrum = self.l2a_cubes[product.lower()][idx[0], idx[1], :]
        except KeyError:
            return None

        output_file = Path(self.nc_dir, self.capture_name + '_l2a_' + str(product) + '_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum)
        plt.ylabel(self.l2a_units)
        plt.ylim([0, 1])
        plt.xlabel("Wavelength (nm)")
        plt.title(f"(lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)
        plt.savefig(output_file)

    # TODO
    def write_l1a_nc_file(self, path: str = None) -> None:

        self._write_l1a_nc_file(path=path)

        return None

    # TODO: path override
    def write_l1b_nc_file(self, path: str = None) -> None:

        self._write_l1b_nc_file(path=path)

        return None

    # TODO
    def write_l2a_nc_file(self, path: str = None, product: str = None) -> None:

        self._write_l1b_nc_file(path=path, product=product)

        return None

class Hypso2(Hypso):

    def __init__(self, hypso_path: str, points_path: Union[str, None] = None) -> None:
        
        """
        Initialization of (planned) HYPSO-2 Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(hypso_path=hypso_path, points_path=points_path)

        self.platform = 'hypso2'
        self.sensor = 'hypso2_hsi'

def set_or_create_attr(var, attr_name, attr_value) -> None:
    """
    Set or create an attribute on ".nc" file.

    :param var: Variable on to which assign the attribute
    :param attr_name: Attribute name
    :param attr_value: Attribute value

    :return: No return value
    """

    if attr_name in var.ncattrs():
        var.setncattr(attr_name, attr_value)
        return
    var.UnusedNameAttribute = attr_value
    var.renameAttribute("UnusedNameAttribute", attr_name)
    return




