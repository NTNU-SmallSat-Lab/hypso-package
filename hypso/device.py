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

import time
import scipy.interpolate as si

EXPERIMENTAL_FEATURES = True
SUPPORTED_PRODUCT_LEVELS = ["l1a", "l1b", "l2a"]
SUPPORTED_ATM_CORR_PRODUCTS = ["6sv1", "acolite", "machi"]

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

        # Initialize raw datacube
        self.l1a_cube = None

        # Initialize calibrated datacube
        self.l1b_cube = None

        # Initialize atms corr datacubes dict
        self.l2a_cube = {}

        # Initialize platform and sensor names
        self.platform = None
        self.sensor = None

        # Initialize capture name and target
        self.capture_name = None
        self.capture_region = None

        # Initialize capture date and time
        self.capture_datetime = None

        # Initialize dimensions
        self.dimensions = None
        self.spatial_dimensions = (956, 684)  # 1092 x variable
        self.standard_dimensions = {
            "nominal": 956,  # Along frame_count
            "wide": 1092  # Along image_height (row_count)
        }

        # Initialize projection information
        self.projection_metadata = None

         # Initialize latitude and longitude variables
        self.latitudes = None
        self.longitudes = None
        self.datacube_flipped = False

        # Initialize wavelengths
        self.wavelengths = None
        self.wavelengths_units = r'$nm$'

        # Initialize spectral response function
        self.srf = None

        # Initialize units
        self.units = r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$'

        # Initilize land mask dict
        self.land_mask = {}

        # Initilize cloud mask dict
        self.cloud_mask = {}

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

        #self.rgbGeotiffFilePath = None
        #self.l1cgeotiffFilePath = None
        #self.l2geotiffFilePath = None

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

        # Start processing timer
        #t = time.time()

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

 
        # Load L1a file -----------------------------------------------------
        self._load_l1a_file()

        # Calibration -----------------------------------------------------
        self._run_calibration()

        # Georeferencing -----------------------------------------------------
        self._run_georeferencing()

        # Atmospheric correction -----------------------------------------------------
        # TODO: add flags to make sure run in the correct orders
        #self._run_geometry_computation()
        #self._run_atmospheric_correction(product='6SV1')

        # File output
        #self._write_l1a_file()
        #self._write_l1b_file()
        #self._write_l2a_file()
        #self.write_l1b_nc_file()  # Input for ACOLITE
        #self.l2a_cube = self.find_existing_l2_cube()

        # Land Mask -----------------------------------------------------
        # self._run_land_mask()

        # Cloud Mask -----------------------------------------------------
        # TODO
        #self._run_cloud_mask()

        # Products
        # TODO
        #self.products['chl'] = None
        #self.products['tsm'] = None
        #self.products['pca'] = None
        #self.products['ica'] = None


        # Stop processing timer
        #proc_time = time.time() - t
        #if self.verbose:
        #    print('[INFO] Processing complete. Elapsed time: ' + str(proc_time) + ' seconds.')


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

    def _set_capture_datetime(self) -> None:

        # TODO: implement this function
        self.capture_datetime = None

        return None

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
    
    def _set_rad_coeff_file(self, rad_coeff_file=None) -> None:

        """
        Get the absolute path for the radiometric coefficients.

        :param rad_coeff_file: Path to radiometric coefficients file (optional)

        :return: None.
        """

        if rad_coeff_file:
            self.rad_coeff_file = rad_coeff_file
            return

        match self.info["capture_type"]:
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

    def _set_smile_coeff_file(self, smile_coeff_file=None) -> None:

        """
        Get the absolute path for the smile coefficients.

        :param smile_coeff_file: Path to smile coefficients file (optional)

        :return: None.
        """

        if smile_coeff_file:
            self.smile_coeff_file = smile_coeff_file
            return

        match self.info["capture_type"]:
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

    def _set_destriping_coeff_file(self, destriping_coeff_file=None) -> None:

        """
        Get the absolute path for the destriping coefficients.

        :param destriping_coeff_file: Path to destriping coefficients file (optional)

        :return: None.
        """

        if destriping_coeff_file:
            self.destriping_coeff_file = destriping_coeff_file
            return

        match self.info["capture_type"]:
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

    def _set_spectral_coeff_file(self, spectral_coeff_file=None) -> None:

        """
        Get the absolute path for the spectral coefficients (wavelengths).

        :param spectral_coeff_file: Path to spectral coefficients file (optional)

        :return: None.
        """

        if spectral_coeff_file:
            self.spectral_coeff_file = spectral_coeff_file
            return
        
        csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"

        spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_spectral}')

        self.spectral_coeff_file = spectral_coeff_file

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
    
    def _set_capture_name(self) -> None:

        capture_name = self.hypso_path.stem

        for pl in SUPPORTED_PRODUCT_LEVELS:

            if "-" + pl in capture_name:
                capture_name = capture_name.replace("-" + pl, "")

        #if "-l1a" in capture_name:
        #    capture_name = capture_name.replace("-l1a", "")

        self.capture_name = capture_name

        return None

    def _set_capture_region(self) -> None:

        self._set_capture_name()
        self.capture_region = self.capture_name.split('_')[0].strip('_')

        return None

    def _set_spatial_dimensions(self) -> None:

        self.spatial_dimensions = (self.capture_config["frame_count"], self.info["image_height"])

        if self.verbose:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")

        return None

    def _set_capture_type(self) -> None:

        if self.capture_config["frame_count"] == self.standard_dimensions["nominal"]:
            self.info["capture_type"] = "nominal"

        elif self.info["image_height"] == self.standard_dimensions["wide"]:
            self.info["capture_type"] = "wide"
        else:
            if EXPERIMENTAL_FEATURES:
                print("Number of Rows (AKA frame_count) Is Not Standard")
                self.info["capture_type"] = "custom"
            else:
                raise Exception("Number of Rows (AKA frame_count) Is Not Standard")

        if self.verbose:
            print(f"[INFO] Capture capture type: {self.info['capture_type']}")

        return None

    def _set_info_dict(self) -> None:

        # TODO: move this dictionary into object variables

        info = {}

        info["nc_file"] = self.hypso_path

        info["nc_name"] = self.hypso_path.stem

        file_path = str(self.hypso_path)

        info["l1a_nc_file"] = Path(file_path)
        info["l1b_nc_file"] = Path(file_path.replace("-l1a", "-l1b"))
        info["l2a_nc_file"] = Path(file_path.replace("-l1a", "-l2a"))

        info["tmp_dir"] =  Path(info["nc_file"].parent.absolute(), info["nc_name"].replace("-l1a", "") + "_tmp")
        info["top_folder_name"] = info["tmp_dir"]

        if all([self.target_coords.get('latc'), self.target_coords.get('lonc')]):
            info['target_area'] = self.target_coords['latc'] + ' ' + self.target_coords['lonc']
        else:
            info['target_area'] = None

        info["background_value"] = 8 * self.capture_config["bin_factor"]

        info["x_start"] = self.capture_config["aoi_x"]
        info["x_stop"] = self.capture_config["aoi_x"] + self.capture_config["column_count"]
        info["y_start"] = self.capture_config["aoi_y"]
        info["y_stop"] = self.capture_config["aoi_y"] + self.capture_config["row_count"]
        info["exp"] = self.capture_config["exposure"] / 1000  # in seconds

        info["image_height"] = self.capture_config["row_count"]
        info["image_width"] = int(self.capture_config["column_count"] / self.capture_config["bin_factor"])
        info["im_size"] = info["image_height"] * info["image_width"]


        info["bands"] = info["image_width"]
        info["lines"] = self.capture_config["frame_count"]  # AKA Frames AKA Rows
        info["samples"] = info["image_height"]  # AKA Cols


        # TODO: Verify offset validity. Sivert had 20 here
        UNIX_TIME_OFFSET = 20

        info["start_timestamp_capture"] = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        #    cant compute end timestamp using frame count and frame rate
        #     assuming some default value if fps and exposure not available
        try:
            info["end_timestamp_capture"] = info["start_timestamp_capture"] + self.capture_config["frame_count"] / self.capture_config["fps"] + self.capture_config["exposure"] / 1000.0
        except:
            print("fps or exposure values not found assuming 20.0 for each")
            info["end_timestamp_capture"] = info["start_timestamp_capture"] + self.capture_config["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0
        info["start_timestamp_adcs"] = info["start_timestamp_capture"] - time_margin_start
        info["end_timestamp_adcs"] = info["end_timestamp_capture"] + time_margin_end

        info["unixtime"] = info["start_timestamp_capture"]
        info["iso_time"] = datetime.utcfromtimestamp(info["unixtime"]).isoformat()

        self.info = info

        return None

    def _set_calibration_coeffs(self) -> None:

        self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)
        self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)
        self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)
        self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)

        return None

    def _set_wavelengths(self) -> None:

        if self.spectral_coeffs is not None:
            self.wavelengths = self.spectral_coeffs
        else:
            self.wavelengths = None

        return None

    def _set_adcs_dataframes(self) -> None:

        self._set_adcs_pos_dataframe()
        self._set_adcs_quat_dataframe()

        return None

    def _set_adcs_pos_dataframe(self) -> None:

        # TODO move DataFrame formatting related code to geometry
        position_headers = ["timestamp", "eci x [m]", "eci y [m]", "eci z [m]"]
        
        timestamps = self.adcs["timestamps"]
        pos_x = self.adcs["position_x"]
        pos_y = self.adcs["position_y"]
        pos_z = self.adcs["position_z"]

        pos_array = np.column_stack((timestamps, pos_x, pos_y, pos_z))
        pos_df = pd.DataFrame(pos_array, columns=position_headers)

        self.adcs_pos_df = pos_df

        return None

    def _set_adcs_quat_dataframe(self) -> None:

        # TODO move DataFrame formatting related code to geometry
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


    # Loaders

    def _load_l1a_file(self) -> None:

        self._check_l1a_file_format()

        self._set_capture_name()
        self._set_capture_region()
        self._set_capture_datetime()

        self._load_l1a_cube()
        self._load_l1a_metadata()

        self._set_info_dict()
        self._set_capture_type()
        self._set_spatial_dimensions()
        self._set_adcs_dataframes()

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
            self.target_coords, \
            self.adcs, \
            self.dimensions = load_l1a_nc_metadata(self.hypso_path)
        
        return None

    def _load_l1a_bip_cube(self) -> None:
        
        return None

    def _load_l1a_bip_metadata(self) -> None:
        
        return None


    # Runners

    def _run_calibration(self) -> None:

        self._set_calibration_coeff_files()
        self._set_calibration_coeffs()
        self._set_wavelengths()
        self._set_srf()
        self._generate_l1b_cube()

        self.calibration_has_run = True

        return None

    def _run_georeferencing(self) -> None:

        

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

            # TODO remove lat and lon in info dict
            self.info["lat"] = self.latitudes
            self.info["lon"] = self.longitudes

            self.info["lat_original"] = self.latitudes
            self.info["lon_original"] = self.longitudes


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
                
            if self.l2a_cube is not None:
                if isinstance(self.l2a_cube, dict):
                    for key in self.l2a_cube.keys():
                        self.l2a_cube = self.l2a_cube[key][:, ::-1, :]

        self.datacube_flipped = datacube_flipped

        return None

    def _run_radiometric_calibration(self, cube=None) -> np.ndarray:

        # Radiometric calibration

        if cube is None:
            cube = self.l1a_cube

        if self.verbose:
            print("[INFO] Running radiometric calibration...")

        cube = self._get_flipped_cube(cube=cube)

        cube = run_radiometric_calibration(cube=cube, 
                                           background_value=self.info['background_value'],
                                           exp=self.info['exp'],
                                           image_height=self.info['image_height'],
                                           image_width=self.info['image_width'],
                                           frame_count=self.capture_config['frame_count'],
                                           rad_coeffs=self.rad_coeffs)

        cube = self._get_flipped_cube(cube=cube)

        # TODO: The factor by 10 is to fix a bug in which the coeff have a factor of 10
        #cube_calibrated = run_radiometric_calibration(self.info, self.rawcube, self.calibration_coefficients_dict) / 10
        cube = cube / 10

        return cube

    def _run_smile_correction(self, cube=None) -> np.ndarray:

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

    def _run_destriping_correction(self, cube) -> np.ndarray:

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

    def _run_geometry_computation(self) -> None:

        print("[INFO] Running frame interpolation...")

        framepose_data = interpolate_at_frame(adcs_pos_df=self.adcs_pos_df,
                                              adcs_quat_df=self.adcs_quat_df,
                                              timestamps_srv=self.timing['timestamps_srv'],
                                              frame_count=self.capture_config['frame_count'],
                                              fps=self.capture_config['fps'],
                                              exposure=self.capture_config['exposure'],
                                              verbose=self.verbose
                                              )

        print("[INFO] Running geometry computation...")

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
                                             image_height=self.info['image_height'],
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

    def _run_atmospheric_correction(self, product: str) -> None:

        if self.l2a_cube is None:
            self.l2a_cube = {}

        product = product.lower()

        match product:
            case "6sv1":
                if self.verbose: 
                    print("[INFO] Running 6SV1 atmospheric correction")
                if product not in self.l2a_cube:
                    self.l2a_cube[product] = self._run_6sv1_atmospheric_correction()
            case "acolite":
                if self.verbose: 
                    print("[INFO] Running ACOLITE atmospheric correction")
                if product not in self.l2a_cube:
                    self.l2a_cube[product] = self._run_acolite_atmospheric_correction()
            case "machi":
                if self.verbose: 
                    print("[INFO] Running MACHI atmospheric correction")
                if product not in self.l2a_cube:
                    self.l2a_cube[product] = self._run_machi_atmospheric_correction()  

            case _:
                print("[WARNING] No such atmospheric correction product supported!")

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

        self._check_calibration_has_run()
        self._check_geometry_computation_has_run()


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

        time_capture = parser.parse(self.info['iso_time'])

        atmos_corrected_cube = run_py6s(wavelengths=self.wavelengths, 
                                        hypercube_L1=self.l1b_cube, 
                                        lat_2d_array=latitudes,
                                        lon_2d_array=longitudes,
                                        solar_azimuth_angles=self.solar_azimuth_angles,
                                        solar_zenith_angles=self.solar_zenith_angles,
                                        sat_azimuth_angles=self.sat_azimuth_angles,
                                        sat_zenith_angles=self.sat_zenith_angles,
                                        iso_time=self.info['iso_time'],
                                        py6s_dict=atmos_params, 
                                        time_capture=time_capture,
                                        srf=self.srf)
        
        return atmos_corrected_cube

    def _run_acolite_atmospheric_correction(self) -> None:

        # TODO
        print("[WARNING] ACOLITE atmospheric correction has not been enabled.")

        self._check_calibration_has_run()
        self._check_geometry_computation_has_run()
        self._check_write_l1b_nc_file_has_run()

        # user and password from https://urs.earthdata.nasa.gov/profile
        # optional but good
        atmos_params = {
            'user':'alvarof',
            'password':'nwz7xmu8dak.UDG9kqz'
        }

        if not self.info["l1b_nc_file"].is_file():
            raise Exception("No -l1b.nc file found")
        
        atmos_corrected_cube = run_acolite(output_path=self.info["top_folder_name"], 
                                          atmos_dict=atmos_params, 
                                          nc_file_acoliteready=self.info["l1b_nc_file"])

        return atmos_corrected_cube
    
    def _run_machi_atmospheric_correction(self) -> None:

        # TODO
        print("[WARNING] Minimal Atmospheric Compensation for Hyperspectral Imagers (MACHI) atmospheric correction has not been enabled.")
        
        return None

        #self._check_calibration_has_run()
        #self._check_geometry_computation_has_run()

        #T, A, objs = run_machi(cube=self.l1b_cube, verbose=self.verbose)

    def _run_land_mask(self, product: str) -> None:

        if self.land_mask is None:
            self.land_mask = {}

        product = product.lower()

        match product:

            case "global":

                if self.verbose:
                    print("[INFO] Running land mask generation...")

                self.land_mask[product] = self._run_global_land_mask()

            case "ndwi":

                if self.verbose:
                    print("[INFO] Running NDWI land mask generation...")

                self.land_mask[product] = self._run_ndwi_land_mask()

            case "threshold":

                if self.verbose:
                    print("[INFO] Running threshold land mask generation...")

                self.land_mask[product] = self._run_threshold_land_mask()

            case _:

                print("[WARNING] No such land mask supported!")

    def _run_global_land_mask(self) -> np.ndarray:

        self._check_georeferencing_has_run()

        land_mask = run_global_land_mask(spatial_dimensions=self.spatial_dimensions,
                                        latitudes=self.latitudes,
                                        longitudes=self.longitudes
                                        )
        
        return land_mask

    def _run_ndwi_land_mask(self) -> np.ndarray:

        self._check_calibration_has_run()

        land_mask = run_ndwi_land_mask(cube=self.l1b_cube, 
                                       wavelengths=self.wavelengths,
                                       verbose=self.verbose)

        return land_mask
    
    def _run_threshold_land_mask(self) -> np.ndarray:

        self._check_calibration_has_run()

        land_mask = run_threshold_land_mask(cube=self.l1b_cube,
                                            wavelengths=self.wavelengths,
                                            verbose=self.verbose)
    
        return land_mask

    def _run_cloud_mask(self, product: str) -> None:

        if self.cloud_mask is None:
            self.cloud_mask = {}

        if self.verbose:
            print("[INFO] Running cloud mask generation...")
            print("[WARNING] Cloud mask generation has not been implemented.")
        
        product = "general"

        cloud_mask = run_cloud_mask(spatial_dimensions=self.spatial_dimensions,
                                        latitudes=self.latitudes,
                                        longitudes=self.longitudes
                                        )
        
        self.cloud_mask = cloud_mask[product]

        return None





    def _run_chlorophyll_estimation(self, product: str) -> None:

        if self.products is None:
            self.products = {}

        product = product.lower()

        match product:

            case "band_ratio":

                if self.verbose:
                    print("[INFO] Running band ratio chlorophyll estimation...")

                key = "chl_" + product

                self.products[key] = self._run_band_ratio_chlorophyll_estimation()

            case "6sv1_aqua":

                if self.verbose:
                    print("[INFO] Running 6SV1 AQUA Tuned chlorophyll estimation...")

                key = "chl_" + product

                self.products[key] = self._run_6sv1_aqua_chlorophyll_estimation()

            case "acolite_aqua":

                if self.verbose:
                    print("[INFO] Running ACOLITE AQUA Tuned chlorophyll estimation...")

                key = "chl_" + product

                self.products[key] = self._run_acolite_aqua_chlorophyll_estimation()

            case _:

                print("[WARNING] No such chlorophyll estimation product supported!")

    def _run_band_ratio_chlorophyll_estimation(self) -> None:

        pass

    def _run_6sv1_aqua_chlorophyll_estimation(self, model) -> None:

        pass

    def _run_acolite_aqua_chlorophyll_estimation(self, model) -> None:

        pass








    # Other

    def _get_flipped_cube(self, cube: np.ndarray) -> np.ndarray:

        if self.datacube_flipped is None:
            return cube
        else:
            if self.datacube_flipped:
                return cube[:, ::-1, :]

            else:
                return cube

        return cube

    def _check_georeferencing_has_run(self, run=True) -> bool:
        if run:
            if not self.georeferencing_has_run:

                if self.verbose:
                    print("[INFO] Georeferencing has not been run yet. Running now...")

                self._run_georeferencing()
                return True

        if self.georeferencing_has_run:
            return True

        return False
    
    def _check_calibration_has_run(self, run=True) -> bool:
        if run:
            if not self.calibration_has_run:

                if self.verbose:
                    print("[INFO] Calibration has not been run yet. Running now...")

                self._run_calibration()
                return True

        if self.calibration_has_run:
            return True

        return False
    
    def _check_geometry_computation_has_run(self, run=True) -> bool:
        if run:
            if not self.geometry_computation_has_run:

                if self.verbose:
                    print("[INFO] Geometry computation has not been run yet. Running now...")

                self._run_geometry_computation()
                return True

        if self.geometry_computation_has_run:
            return True

        return False
    
    def _check_write_l1b_nc_file_has_run(self, run=True) -> bool:
        if run:
            if not self.geometry_computation_has_run:

                if self.verbose:
                    print("[INFO] L1b .nc file has not been created yet. Creating it now...")
    
                self._write_l1b_nc_file()
                return True

        if self.geometry_computation_has_run:
            return True

        return False     

    def _check_atmospheric_correction_has_run(self, product=None, run=True) -> bool:
        if run:
            if not self.atmospheric_correction_has_run:
                self._run_atmospheric_correction(product=product)
                return True

        if self.atmospheric_correction_has_run:
            return True

        return False
    
    def _check_l1a_file_format(self) -> None:

        # Check that hypso_path file is a NetCDF file:
        #if not self.hypso_path.suffix == '.nc':
        #    raise Exception("Incorrect HYPSO Path. Only .nc files supported")

        match self.hypso_path.suffix:
            case '.nc':
                pass
            case '.bip':
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
            case _:
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
    
        return None

    def _generate_l1b_cube(self) -> None:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        self.l1b_cube = self._run_radiometric_calibration()
        self.l1b_cube = self._run_smile_correction(cube=self.l1b_cube)
        self.l1b_cube = self._run_destriping_correction(cube=self.l1b_cube)

        return None


    # Public methods

    def display_geometry_information(self) -> None:

        # Notes:
        # - along_track, frame_count, lines, rows, y, latitude: 956, 598
        # - cross_track, row_count, image_height, samples, cols, x, longitude: 684, 1092
        # - spectral, image_width, bands, z: 120
        print('Spatial dimensions: ' + str(self.spatial_dimensions))
        print('Standard dimensions: ' + str(self.standard_dimensions))
        print('Row count: ' + str(self.capture_config['row_count']))
        print('Image height: ' + str(self.info['image_height']))
        print('Image width: ' + str(self.info['image_width']))
        print('Frame count: ' + str(self.capture_config['frame_count']))
        #print('Frame height: ' + str(frame_height))
        print('Lines: ' + str(self.dimensions['lines']))
        print('Samples: ' + str(self.dimensions['samples']))
        print('Bands: ' + str(self.dimensions['bands']))

        return None

    # TODO
    def get_spectra(self, position_dict: dict, product: Literal["L1C", "L2-6SV1", "L2-ACOLITE"] = "L1C",
                    filename: Union[str, None] = None, plot: bool = True) -> Union[pd.DataFrame, None]:
        """
        Get spectra values from the indicated coordinated or pixel.

        :param position_dict: Dictionary with the inputs required.\n
            *** If Coordinates are Input *** \n
            ``{"lat":59.5,"lon":10}``\n
            *** If pixel location passed *** \n
            ``{"X":200,"Y":83}``
        :param product: Product of which to retrieve the spectral signal. Can either be "L1C", "L2-ACOLITE" or "L2-6SV1"
        :param filename: Path on which to save the spectra as a .csv file. Filename should contain the ".csv" extension.
            Example: "/Documents/export_signal.csv". If None, the spectral signal won´t be saved as a file.
        :param plot: If True, the spectral signal will be plotted.

        :return: None if the coordinates/pixel location are not withing the captured image. Otherwise, a pandas dataframe
            containing the spectral signal is returned.
        """

        position_keys = list(position_dict.keys())
        if "lat" in position_keys or "lon" in position_keys:
            postype = "coord"
        elif "X" in position_keys or "Y" in position_keys:
            postype = "pix"
        else:
            raise Exception("Keys of ")

        # To Store data
        spectra_data = []
        multiplier = 1  # Multiplier for the signal. In case signal is compressed differently.
        posX = None
        posY = None
        lat = None
        lon = None
        transformed_lon = None
        transformed_lat = None
        # Open the raster

        # Find Geotiffs
        self.find_geotiffs()

        # Check if full (120 band) tiff exists
        if self.l1cgeotiffFilePath is None and self.l2geotiffFilePath is None:
            raise Exception("No Full-Band GeoTiff, Force Restart")

        path_to_read = None
        cols = []
        if product == "L1C":
            if self.l1cgeotiffFilePath is None:
                raise Exception("L1C product does not exist.")
            elif self.l1cgeotiffFilePath is not None:
                path_to_read = self.l1cgeotiffFilePath
                cols = ["wl", "radiance"]
        elif "L2" in product:
            l2_engine = product.split("-")[1]
            if self.l2geotiffFilePath is None:
                raise Exception("L2 product does not exist.")
            elif self.l2geotiffFilePath is not None:
                try:
                    path_to_read = self.l2geotiffFilePath[l2_engine.upper()]
                except:
                    raise Exception(f"There is no L2 Geotiff for {l2_engine.upper()}")

                cols = ["wl", "rrs"]

        else:
            raise Exception("Wrong product type.")

        with rasterio.open(str(path_to_read)) as dataset:
            dataset_crs = dataset.crs
            print("Dataset CRS: ", dataset_crs)

            # Create Projection with Dataset CRS
            dataset_proj = prj.Proj(dataset_crs)  # your data crs

            # Find Corners of Image (For Development)
            boundbox = dataset.bounds
            left_bottom = dataset_proj(
                boundbox[0], boundbox[1], inverse=True)
            right_top = dataset_proj(
                boundbox[2], boundbox[3], inverse=True)

            if postype == 'coord':
                # Get list to two variables
                lat = position_dict["lat"]
                lon = position_dict["lon"]
                # lat, lon = position
                # Transform Coordinates to Image CRS
                transformed_lon, transformed_lat = dataset_proj(
                    lon, lat, inverse=False)
                # Get pixel coordinates from map coordinates
                posY, posX = dataset.index(
                    transformed_lon, transformed_lat)

            elif postype == 'pix':
                posX = int(position_dict["X"])
                posY = int(position_dict["Y"])

                # posX = int(position[0])
                # posY = int(position[1])

                transformed_lon = dataset.xy(posX, posY)[0]
                transformed_lat = dataset.xy(posX, posY)[1]

                # Transform from the GeoTiff CRS
                lon, lat = dataset_proj(
                    transformed_lon, transformed_lat, inverse=True)

            # Window size is 1 for a Single Pixel or 3 for a 3x3 windowd
            N = 3
            # Build an NxN window
            window = rasterio.windows.Window(
                posX - (N // 2), posY - (N // 2), N, N)

            # Read the data in the window
            # clip is a nbands * N * N numpy array
            clip = dataset.read(window=window)
            if N != 1:
                clip = np.mean(clip, axis=(1, 2))

            clip = np.squeeze(clip)

            # Append data to Array
            # Multiplier for Values like Sentinel 2 which need 1/10000
            spectra_data = clip * multiplier

        # Return None if outside of boundaries or alpha channel is 0
        if posX < 0 or posY < 0 or self.projection_metadata["rgba_data"][3, posY, posX] == 0:
            print("Location not covered by image --------------------------\n")
            return None

        # Print Coordinate and Pixel Matching
        print("(lat, lon) -→ (X, Y) : (%s, %s) -→ (%s, %s)" %
              (lat, lon, posX, posY))

        df_band = pd.DataFrame(np.column_stack((self.wavelengths, spectra_data)), columns=cols)
        df_band["lat"] = lat
        df_band["lon"] = lon
        df_band["X"] = posX
        df_band["Y"] = posY

        if filename is not None:
            df_band.to_csv(filename, index=False)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.wavelengths, spectra_data)
            if product == "L1C":
                plt.ylabel(self.units)
            elif "L2" in product:
                plt.ylabel("Rrs [0,1]")
                plt.ylim([0, 1])
            plt.xlabel("Wavelength (nm)")
            plt.title(f"(lat, lon) -→ (X, Y) : ({lat}, {lon}) -→ ({posX}, {posY})")
            plt.grid(True)
            plt.show()

        return df_band

    def get_l1a_cube(self) -> np.ndarray:

        return self.l1a_cube

    def get_l1b_cube(self) -> np.ndarray:

        self._check_calibration_has_run()

        return self.l1b_cube

    def get_l2a_cube(self, product: Literal["acolite", "6sv1", "machi"]="6sv1") -> dict:

        #self._check_geometry_computation_has_run()

        if not self._check_atmospheric_correction_has_run(run=False):
            self._run_atmospheric_correction(product=product)
            
        product = product.lower()

        return self.l2a_cube[product]
            



    def get_land_mask(self, product: Literal["global", "ndwi", "threshold"]) -> np.ndarray:

        self._run_land_mask(product=product)

        return self.land_mask[product]   

    def get_land_masks(self) -> dict:


        return self.land_mask     
    
    def get_cloud_mask(self, product: str) -> np.ndarray:

        self._run_cloud_mask(product=product)

        return self.cloud_mask[product]
    
    def get_cloud_masks(self) -> dict:

        return self.cloud_mask 

    def get_chlorophyll_estimate(self, 
                                 product:  Literal["band_ratio", "6sv1_aqua", "acolite_aqua"],
                                model=None
                                ) -> np.ndarray:

        self._run_chlorophyll_estimation(product=product)

        key = 'chl_' + product

        return self.products[key]

    def get_products(self) -> dict:

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

        scene_date = parser.isoparse(self.info['iso_time'])
        julian_day = scene_date.timetuple().tm_yday
        solar_zenith = self.info['solar_zenith_angle']

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

    # TODO
    def get_projection_metadata(self) -> dict:
        """
        Returns the projection metadata dictionary. This is either extracted from the RGBA, L1 or L1C GeoTiff

        :return: Dictionary containing the projection information for the capture.
        """

        top_folder_name = self.info["top_folder_name"]
        current_project = {}

        # Find Geotiffs
        self.find_geotiffs()

        # -----------------------------------------------------------------
        # Get geotiff data for rgba first    ------------------------------
        # -----------------------------------------------------------------
        if self.rgbGeotiffFilePath is not None:
            print("RGBA Tif File: ", self.rgbGeotiffFilePath.name)
            # Load GeoTiff Metadata with gdal
            ds = gdal.Open(str(self.rgbGeotiffFilePath))
            data = ds.ReadAsArray()
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            inproj = osr.SpatialReference()
            inproj.ImportFromWkt(proj)

            boundbox = None
            crs = None
            with rasterio.open(self.rgbGeotiffFilePath) as dataset:
                crs = dataset.crs
                boundbox = dataset.bounds

            current_project = {
                "rgba_data": data,
                "gt": gt,
                "proj": proj,
                "inproj": inproj,
                "boundbox": boundbox,
                "crs": str(crs).lower()
            }

        # -----------------------------------------------------------------
        # Get geotiff data for full second   ------------------------------
        # -----------------------------------------------------------------
        full_path = None
        if self.l2geotiffFilePath is not None:
            # Load GeoTiff Metadata with gdal
            first_key = list(self.l2geotiffFilePath)[0]
            path_found = self.l2geotiffFilePath[first_key]
            ds = gdal.Open(str(path_found))
            data = ds.ReadAsArray()
            current_project["data"] = data
            print("Full L2 Tif File: ", path_found.name)
        if self.l1cgeotiffFilePath is not None:
            # Load GeoTiff Metadata with gdal
            ds = gdal.Open(str(self.l1cgeotiffFilePath))
            data = ds.ReadAsArray()
            current_project["data"] = data
            print("Full L1C Tif File: ", self.l1cgeotiffFilePath.name)

        return current_project

    # TODO

    def write_l1b_nc_file(self) -> None:

        self._write_l1b_nc_file()

        return None

    def _write_l1b_nc_file(self) -> None:
        """
        Create a l1b.nc file using the radiometrically corrected data. Same structure from the original l1a.nc file
        is used. Required to run ACOLITE as the input is a radiometrically corrected .nc file.

        :return: Nothing.
        """

        # TODO: can this be implemented with satpy NetCDF reader?

        hypso_nc_path = self.info["l1a_nc_file"]
        old_nc = nc.Dataset(hypso_nc_path, 'r', format='NETCDF4')

        new_path = hypso_nc_path
        new_path = str(new_path).replace('l1a.nc', 'l1b.nc')

        if Path(new_path).is_file():
            print("L1b.nc file already exists. Not creating it.")
            self.info["l1b_nc_file"] = Path(new_path)
            return

        # Create a new NetCDF file
        with (nc.Dataset(new_path, 'w', format='NETCDF4') as netfile):
            bands = self.info["image_width"]
            lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
            samples = self.info["image_height"]  # AKA Cols

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
                frametime_pose_file = find_file(self.info["top_folder_name"], "frametime-pose", ".csv")
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
        self.info["l1b_nc_file"] = Path(new_path)

        self.write_l1b_nc_file_has_run = True

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




