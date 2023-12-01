from typing import Union
from osgeo import gdal, osr
import numpy as np
import pandas as pd
from importlib.resources import files
import rasterio
from pathlib import Path
from dateutil import parser

import pyproj as prj
from .calibration import crop_and_bin_matrix, calibrate_cube, get_coefficients_from_dict, get_coefficients_from_file, \
    smile_correct_cube, destriping_correct_cube
from .georeference import start_coordinate_correction, generate_full_geotiff, generate_rgb_geotiff
from .reading import load_nc, load_directory
from hypso.utils import find_dir, find_file, find_all_files
from .atmospheric import run_py6s

EXPERIMENTAL_FEATURES = True


class Satellite:
    def __init__(self, hypso_path, points_path=None) -> None:
        self.DEBUG = False
        self.spatialDim = (956, 684)  # 1092 x variable
        self.standardDimensions = {
            "nominal": 956,  # Along frame_count
            "wide": 1092  # Along image_height (row_count)
        }
        self.units = r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$'
        self.rgbGeotiffFilePath = None
        self.l1cgeotiffFilePath = None
        self.l2geotiffFilePath = None
        if points_path is not None:
            self.pointsPath = Path(points_path)
        else:
            self.pointsPath = points_path

        # Make absolute path
        hypso_path = Path(hypso_path).absolute()

        # Check if file or directory passed
        if hypso_path.is_dir():
            self.info, self.rawcube, self.spatialDim = load_directory(hypso_path, self.standardDimensions)
            self.info["top_folder_name"] = hypso_path
        elif hypso_path.suffix == '.nc':
            # Obtain metadata from files
            self.info, self.rawcube, self.spatialDim = load_nc(hypso_path, self.standardDimensions)
            self.info["top_folder_name"] = self.info["tmp_dir"]
        else:
            raise Exception("Incorrect HYPSO Path")

        # Correction Coefficients ----------------------------------------
        self.calibration_coeffs_file_dict = self.get_calibration_coefficients_path()
        self.calibration_coefficients_dict = get_coefficients_from_dict(
            self.calibration_coeffs_file_dict)

        # Wavelengths -----------------------------------------------------
        self.spectral_coeff_file = self.get_spectral_coefficients_path()
        self.spectral_coefficients = get_coefficients_from_file(
            self.spectral_coeff_file)
        self.wavelengths = self.spectral_coefficients

        # Calibrate and Correct Cube Variables and load existing L2 Cube  ----------------------------------------
        self.l1b_cube = self.get_calibrated_and_corrected_cube()
        self.l2a_cube = self.find_existing_l2_cube()

        # Generated afterwards
        self.waterMask = None

        # Get Projection Metadata from created geotiff
        self.projection_metadata = self.get_projection_metadata(self.info["top_folder_name"])

    def get_geotiff(self, product="L2", force_reload=False, atmos_dict=None):

        if product != "L2" and product != "L1C":
            raise Exception("Wrong product")

        if product == "L2" and atmos_dict is None:
            raise Exception("Atmospheric Dictionary is needed")

        if force_reload:
            # Delete geotiff dir and generate a new rgba one
            self.delete_geotiff_dir(self.info["top_folder_name"])

        # Generate RGB/RGBA Geotiff with Projection metadata and L1B
        generate_rgb_geotiff(self)

        # Get Projection Metadata from created geotiff
        self.projection_metadata = self.get_projection_metadata(self.info["top_folder_name"])

        # Before Generating new Geotiff we check if .points file exists and update 2D coord
        self.info["lat"], self.info["lon"] = start_coordinate_correction(
            self.pointsPath, self.info, self.projection_metadata)

        atmos_corrected_cube=None
        atmos_model = None

        # TODO: Add Acolite
        if product == "L2":
            atmos_model = atmos_dict["model"].upper()
            try:
                atmos_corrected_cube = self.l2a_cube[atmos_model]
            except:
                if atmos_model == "6SV1":
                    # Py6S Atmospheric Correction
                    # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
                    # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
                    # atmos_dict = {
                    #     'model': '6sv1',
                    #     'aot550': 0.01,
                    #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
                    # }
                     atmos_corrected_cube = run_py6s(self.wavelengths, self.l1b_cube, self.info, self.info["lat"], self.info["lon"],
                                             atmos_dict, time_capture=parser.parse(self.info['iso_time']))
                elif atmos_model == "ACOLITE":
                    pass

            # Store the l2a_cube just generated
            if self.l2a_cube is None:
                self.l2a_cube = {}

            self.l2a_cube[atmos_model] = atmos_corrected_cube

            with open(Path(self.info["top_folder_name"], "geotiff",f'L2_{atmos_model}.npy'), 'wb') as f:
                np.save(f, atmos_corrected_cube)

        if force_reload:
            # Delete geotiff dir and generate a new rgba one
            self.delete_geotiff_dir(self.info["top_folder_name"])

        # Generate RGBA/RGBA and Full Geotiff with corrected metadata and L2A if exists (if not L1B)
        generate_rgb_geotiff(self)
        generate_full_geotiff(self, product=product, L2_key=atmos_model)

        # Get Projection Metadata from created geotiff
        self.projection_metadata = self.get_projection_metadata(self.info["top_folder_name"])

    def delete_geotiff_dir(self, top_folder_name: Path):
        tiff_name = "geotiff"
        geotiff_dir = find_dir(top_folder_name, tiff_name)

        self.rgbGeotiffFilePath = None
        self.l1cgeotiffFilePath = None
        self.l2geotiffFilePath = None

        if geotiff_dir is not None:
            print("Forcing Reload: Deleting geotiff Directory...")
            import shutil
            shutil.rmtree(geotiff_dir, ignore_errors=True)

    def find_geotiffs(self, top_folder_name: Path):
        self.rgbGeotiffFilePath = find_file(top_folder_name, "rgba_8bit", ".tif")
        self.l1cgeotiffFilePath = find_file(top_folder_name, "-full_L1C", ".tif")
        self.l2geotiffFilePath = find_file(top_folder_name, "-full_L2", ".tif")

    def find_existing_l2_cube(self) -> Union[dict, None]:
        found_l2_npy = find_all_files(path= Path(self.info["top_folder_name"],"geotiff"),
                  str_in_file="L2",
                  suffix=".npy")

        if found_l2_npy is None:
            return None

        dict_L2 = None
        # Save Generated Cube as "npy" (for faster loading
        for l2_file in found_l2_npy:
            correction_model = str(l2_file.stem).split("_")[1]
            correction_model = correction_model.upper()
            l2_cube=None
            with open(l2_file, 'rb') as f:
                print(f"Found {l2_file.name}")
                l2_cube = np.load(f)

            if dict_L2 is None:
                dict_L2 = {}
            dict_L2[correction_model] = l2_cube

        return dict_L2

    def get_projection_metadata(self, top_folder_name: Path) -> dict:

        current_project = {}

        # Find Geotiffs
        self.find_geotiffs(top_folder_name)

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
            ds = gdal.Open(str(self.l2geotiffFilePath))
            data = ds.ReadAsArray()
            current_project["data"] = data
            print("Full L2 Tif File: ", self.l2geotiffFilePath.name)
        if self.l1cgeotiffFilePath is not None:
            # Load GeoTiff Metadata with gdal
            ds = gdal.Open(str(self.l1cgeotiffFilePath))
            data = ds.ReadAsArray()
            current_project["data"] = data
            print("Full L1C Tif File: ", self.l1cgeotiffFilePath.name)



        return current_project

    def get_calibration_coefficients_path(self) -> dict:
        csv_file_radiometric = None
        csv_file_smile = None
        csv_file_destriping = None

        if self.info["capture_type"] == "nominal":
            csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv"
            csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"
            csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"
        elif self.info["capture_type"] == "wide":
            csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.csv"
            csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"
            csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"

        elif self.info["capture_type"] == "custom":
            bin_x = self.info["bin_factor"]

            # Radiometric ---------------------------------
            full_coeff = get_coefficients_from_file(files('hypso.calibration').joinpath(
                f'data/{"radiometric_calibration_matrix_HYPSO-1_full_v1.csv"}'),
            )

            csv_file_radiometric = crop_and_bin_matrix(
                full_coeff,
                self.info["x_start"],
                self.info["x_stop"],
                self.info["y_start"],
                self.info["y_stop"],
                bin_x)

            # Smile ---------------------------------
            full_coeff = get_coefficients_from_file(files('hypso.calibration').joinpath(
                f'data/{"spectral_calibration_matrix_HYPSO-1_full_v1.csv"}'),
            )
            csv_file_smile = crop_and_bin_matrix(
                full_coeff,
                self.info["x_start"],
                self.info["x_stop"],
                self.info["y_start"],
                self.info["y_stop"],
                bin_x)

            # Destriping ---------------------------------
            rows_img = self.info["frame_count"]  # Due to way image is captured
            cols_img = self.info["image_height"]

            if (rows_img < self.standardDimensions["nominal"]):
                self.info["capture_type"] = "nominal"

            elif (cols_img == self.standardDimensions["wide"]):
                self.info["capture_type"] = "wide"
            csv_file_destriping = None

        rad_coeff_file = csv_file_radiometric if not isinstance(csv_file_radiometric, str) else files(
            'hypso.calibration').joinpath(f'data/{csv_file_radiometric}')

        smile_coeff_file = csv_file_smile if not isinstance(csv_file_smile, str) else files(
            'hypso.calibration').joinpath(f'data/{csv_file_smile}')
        destriping_coeff_file = csv_file_destriping if not isinstance(csv_file_destriping, str) else files(
            'hypso.calibration').joinpath(f'data/{csv_file_destriping}')

        coeff_dict = {"radiometric": rad_coeff_file,
                      "smile": smile_coeff_file,
                      "destriping": destriping_coeff_file}

        return coeff_dict

    def get_spectral_coefficients_path(self) -> str:
        csv_file = "spectral_bands_HYPSO-1_v1.csv"
        wl_file = files(
            'hypso.calibration').joinpath(f'data/{csv_file}')
        return wl_file

    def get_spectra(self, position_dict: dict, product: str = "L2", postype: str = 'coord', multiplier=1, filename=None,
                    plot=True):
        '''
        files_path: Works with a directorie with GeoTiff files. Uses the metadata, and integrated CRS
        position:
            [lat, lon] if postype=='coord'
            [X, Y| if postype == 'pix'
        postye:
            'coord' assumes latitude and longitude are passed.
            'pix' receives X and Y values
        '''
        # To Store Data
        spectra_data = []

        posX = None
        posY = None
        lat = None
        lon = None
        transformed_lon = None
        transformed_lat = None
        # Open the raster

        # Find Geotiffs
        self.find_geotiffs(self.info["top_folder_name"])

        # Check if full (120 band) tiff exists
        if self.l1cgeotiffFilePath is None and self.l2geotiffFilePath is None:
            raise Exception("No Full-Band GeoTiff, Force Restart")

        path_to_read = None
        cols = []
        if product == "L2":
            if self.l2geotiffFilePath is None:
                raise Exception("L2 product does not exist.")
            elif self.l2geotiffFilePath is not None:
                path_to_read = self.l2geotiffFilePath
                cols = ["wl", "rrs"]
        elif product == "L1C":
            if self.l1cgeotiffFilePath is None:
                raise Exception("L1C product does not exist.")
            elif self.l1cgeotiffFilePath is not None:
                path_to_read = self.l1cgeotiffFilePath
                cols = ["wl", "radiance"]
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

        if filename != None:
            df_band.to_csv(filename, index=False)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.wavelengths, spectra_data)
            if product == "L1C":
                plt.ylabel(self.units)
            elif product == "L2":
                plt.ylabel("Rrs [0,1]")
                plt.ylim([0, 1])
            plt.xlabel("Wavelength (nm)")
            plt.title(f"(lat, lon) -→ (X, Y) : ({lat}, {lon}) -→ ({posX}, {posY})")
            plt.grid(True)
            plt.show()

        return df_band

    def get_calibrated_and_corrected_cube(self):
        ''' Calibrate cube.

        Includes:
        - Radiometric calibration
        - Smile correction
        - Destriping

        Assumes all coefficients has been adjusted to the frame size (cropped and
        binned), and that the data cube contains 12-bit values.
        '''

        # Radiometric calibration
        # TODO: The factor by 10 is to fix a bug in which the coeff have a factor of 10
        cube_calibrated = calibrate_cube(
            self.info, self.rawcube, self.calibration_coefficients_dict) / 10

        # Smile correction
        cube_smile_corrected = smile_correct_cube(
            cube_calibrated, self.calibration_coefficients_dict)

        # Destriping
        cube_destriped = destriping_correct_cube(
            cube_smile_corrected, self.calibration_coefficients_dict)

        return cube_destriped


