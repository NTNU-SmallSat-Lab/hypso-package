import glob
import os
from os import listdir
from os.path import isfile, join
from osgeo import gdal, osr
import numpy as np
import pandas as pd
from datetime import datetime
from importlib.resources import files
import netCDF4 as nc
import rasterio
import cartopy.crs as ccrs
import pyproj as prj
import pathlib
from .calibration import crop_and_bin_matrix, calibrate_cube, get_coefficients_from_dict, get_coefficients_from_file, smile_correct_cube, destriping_correct_cube
from .georeference import start_coordinate_correction, generate_geotiff

EXPERIMENTAL_FEATURES=True

class Satellite:
    def __init__(self, top_folder_name) -> None:
        self.DEBUG = False
        self.spatialDim = (956, 684)  # 1092 x variable
        self.standardDimensions = {
            "nominal": 956,  # Along frame_count
            "wide": 1092  # Along image_height (row_count)
        }
        self.info = self.get_metainfo(top_folder_name)
        self.units = r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$'
        self.rawcube = self.get_raw_cube(top_folder_name)

        # Correction Coefficients ----------------------------------------
        self.calibration_coeffs_file_dict = self.get_calibration_coefficients_path()
        self.calibration_coefficients_dict = get_coefficients_from_dict(
            self.calibration_coeffs_file_dict)

        # Wavelengths -----------------------------------------------------
        self.spectral_coeff_file = self.get_spectral_coefficients_path()
        self.spectral_coefficients = get_coefficients_from_file(
            self.spectral_coeff_file)
        self.wavelengths = self.spectral_coefficients

        # Calibrate and Correct Cube ----------------------------------------
        self.l1b_cube = self.get_calibrated_and_corrected_cube()

        # Get projection from RGBA GeoTiff----------------------------------------
        self.projection_metadata = self.get_projection_metadata(
            top_folder_name)

        # Before Generating new Geotiff we check if .points file exists
        self.info["lat"], self.info["lon"] = start_coordinate_correction(
            top_folder_name, self.info, self.projection_metadata)

        # Create Geotiff (L1C) and Correct Coordinate if .points file exists to get cube corrected
        # WARNING: Old RGB and RGBa should not be used for Lat and Lon as they are wrong
        self.projection_metadata = self.generate_full_geotiff(
            top_folder_name)

        self.l2a_cube = None

        # Generated afterwards
        self.waterMask = None

    def get_raw_cube(self, top_folder_name) -> np.ndarray:
        # find file ending in .bip
        path_to_bip = None
        for file in os.listdir(top_folder_name):
            if file.endswith(".bip"):
                path_to_bip = os.path.join(top_folder_name, file)
                break

        cube = np.fromfile(path_to_bip, dtype="uint16")
        if self.DEBUG:
            print(path_to_bip)
            print(cube.shape)
        cube = cube.reshape(
            (-1, self.info["image_height"], self.info["image_width"]))

        # reverse the order of the third dimension
        cube = cube[:, :, ::-1]

        return cube

    def get_metainfo(self, top_folder_name: str) -> dict:
        """Get the metadata from the top folder of the data.

        Args:
            top_folder_name (str): The name of the top folder of the data.

        Returns:
            dict: The metadata.
        """
        info = {}
        info["top_folder_name"] = top_folder_name
        info["folder_name"] = top_folder_name.split("/")[-1]

        # find folder with substring "hsi0" or throw error
        for folder in os.listdir(top_folder_name):
            if "hsi0" in folder:
                raw_folder = folder
                break
        else:
            raise ValueError("No folder with metadata found.")

        # combine top_folder_name and raw_folder to get the path to the raw
        # data
        config_file_path = os.path.join(
            top_folder_name, raw_folder, "capture_config.ini"
        )

        def is_integer_num(n) -> bool:
            if isinstance(n, int):
                return True
            if isinstance(n, float):
                return n.is_integer()
            return False

        # read all lines in the config file
        with open(config_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # split the line at the equal sign
                line = line.split("=")
                # if the line has two elements, add the key and value to the
                # info dict
                if len(line) == 2:
                    key = line[0].strip()
                    value = line[1].strip()
                    try:
                        if is_integer_num(float(value)):
                            info[key] = int(value)
                        else:
                            info[key] = float(value)
                    except BaseException:
                        info[key] = value

        timetamp_file = os.path.join(
            top_folder_name, raw_folder, "timestamps.txt")

        try:
            with open(timetamp_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "." not in line:
                        continue
                    s_part = line.split(".")[0]
                    if s_part.strip().isnumeric():
                        info["unixtime"] = int(s_part) + 20
                        info["iso_time"] = datetime.utcfromtimestamp(
                            info["unixtime"]
                        ).isoformat()
                        break
        except:
            print("No timestamps.txt file. Necessary for atmospheric correction.")
            info["unixtime"] = None
            info["iso_time"]=None

        # find local_angle_csv file with substring "local-angles.csv" or throw error
        for file in os.listdir(top_folder_name):
            if "local-angles.csv" in file:
                local_angle_csv = file
                break
        else:
            raise ValueError("No local-angles.csv file found.")

        local_angle_df = pd.read_csv(top_folder_name + "/" + local_angle_csv)

        solar_za = local_angle_df["Solar Zenith Angle [degrees]"].tolist()
        solar_aa = local_angle_df["Solar Azimuth Angle [degrees]"].tolist()
        sat_za = local_angle_df["Satellite Zenith Angle [degrees]"].tolist()
        sat_aa = local_angle_df["Satellite Azimuth Angle [degrees]"].tolist()

        # Calculates the average solar/sat azimuth/zenith angle.
        average_solar_za = np.round(np.average(solar_za), 5)
        average_solar_aa = np.round(np.average(solar_aa), 5)
        average_sat_za = np.round((np.average(sat_za)), 5)
        average_sat_aa = np.round(np.average(sat_aa), 5)

        info["solar_zenith_angle"] = average_solar_za
        info["solar_azimuth_angle"] = average_solar_aa
        info["sat_zenith_angle"] = average_sat_za
        info["sat_azimuth_angle"] = average_sat_aa

        info["background_value"] = 8 * info["bin_factor"]

        info["x_start"] = info["aoi_x"]
        info["x_stop"] = info["aoi_x"] + info["column_count"]
        info["y_start"] = info["aoi_y"]
        info["y_stop"] = info["aoi_y"] + info["row_count"]
        info["exp"] = info["exposure"] / 1000  # in seconds

        info["image_height"] = info["row_count"]
        info["image_width"] = int(info["column_count"] / info["bin_factor"])
        info["im_size"] = info["image_height"] * info["image_width"]

        # Update Spatial Dim if not standard
        rows_img = info["frame_count"]  # Due to way image is captured
        cols_img = info["image_height"]

        if (rows_img == self.standardDimensions["nominal"]):
            info["capture_type"] = "nominal"

        elif (cols_img == self.standardDimensions["wide"]):
            info["capture_type"] = "wide"
        else:
            if EXPERIMENTAL_FEATURES:
                print("Number of Rows (AKA frame_count) Is Not Standard")
                info["capture_type"] = "custom"
            else:
                raise Exception("Number of Rows (AKA frame_count) Is Not Standard")

        self.spatialDim = (rows_img, cols_img)

        print(
            f"Processing *{info['capture_type']}* Image with Dimensions: {self.spatialDim}")

        # Find Coordinates of the Center of the Image
        # TODO get center lat from .dat lat lon files , average it
        pos_file = ""
        foldername = info["top_folder_name"]
        for file in os.listdir(foldername):
            if file.endswith("geometric-meta-info.txt"):
                pos_file = os.path.join(foldername, file)
                break

        if pos_file == "":
            raise ValueError(f"Could not find position file in {foldername}")

        found_pos = False
        with open(pos_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "lat lon" in line:
                    info["latc"] = float(line.split(
                        "lat lon")[1].split(" ")[1])
                    info["lonc"] = float(line.split(
                        "lat lon")[1].split(" ")[2])
                    found_pos = True
                    break

        if not found_pos:
            raise ValueError(f"Could not find position in {pos_file}")

        # Find 2D Coordinate
        dat_files = glob.glob(top_folder_name + "/*.dat")
        longitude_dataPath = [f for f in dat_files if "longitudes" in f][0]
        latitude_dataPath = [f for f in dat_files if "latitudes" in f][0]

        # Load Latitude
        info["lat"] = np.fromfile(latitude_dataPath, dtype="float32")

        info["lat"] = info["lat"].reshape(
            self.spatialDim)
        # Load Longitude
        info["lon"] = np.fromfile(longitude_dataPath, dtype="float32")
        info["lon"] = info["lon"].reshape(
            self.spatialDim)

        # info["lon"], info["lat"] = np.meshgrid(info["lon"], info["lat"], sparse=True)
        if self.DEBUG:
            print(info)

        return info

    def generate_full_geotiff(self, top_folder_name: str):
        tiff_name = "geotiff-full"

        geotiff_dir = [
            f.path
            for f in os.scandir(top_folder_name)
            if (f.is_dir() and (tiff_name in os.path.basename(os.path.normpath(f))))
        ]

        if len(geotiff_dir) != 0:
            geotiff_dir = geotiff_dir[0]
        else:

            # Nowe we generate the geotiff with corrected lon and lat
            generate_geotiff(self)
            geotiff_dir = [
                f.path
                for f in os.scandir(top_folder_name)
                if (f.is_dir() and (tiff_name in os.path.basename(os.path.normpath(f))))
            ]
            if len(geotiff_dir) != 0:
                geotiff_dir = geotiff_dir[0]
            else:
                raise Exception(
                    "Could not create Full L1C GeoTiff Directory Found")

        self.geotiffFilePath = [
            join(geotiff_dir, f)
            for f in listdir(geotiff_dir)
            if (isfile(join(geotiff_dir, f)) and ("-full" in f)and f.endswith('.tif'))
        ][0]

        # Load GeoTiff Metadata with gdal
        ds = gdal.Open(self.geotiffFilePath)
        # Not hyperspectral, fewer bands
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)

        boundbox = None
        with rasterio.open(self.geotiffFilePath) as dataset:
            crs = dataset.crs
            boundbox = dataset.bounds

        modified_project_metadata = self.projection_metadata.copy()
        modified_project_metadata["data"] = data
        modified_project_metadata["gt"] = gt
        modified_project_metadata["proj"] = proj
        modified_project_metadata["inproj"] = inproj
        return modified_project_metadata

    def get_projection_metadata(self, top_folder_name: str) -> dict:

        tiff_name = "geotiff"
        geotiff_dir = [
            f.path
            for f in os.scandir(top_folder_name)
            if (f.is_dir() and (tiff_name in os.path.basename(os.path.normpath(f))) and ("geotiff-full" not in os.path.basename(os.path.normpath(f))))
        ]

        if len(geotiff_dir) != 0:
            geotiff_dir = geotiff_dir[0]
        else:
            raise Exception("No RGBA Tiff Directory Found")

        self.rgbGeotiffFilePath = [
            join(geotiff_dir, f)
            for f in listdir(geotiff_dir)
            if (isfile(join(geotiff_dir, f)) and ("8bit" in f))
        ][0]

        # Load GeoTiff Metadata with gdal
        ds = gdal.Open(self.rgbGeotiffFilePath)
        # Not hyperspectral, fewer bands
        rgba_data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        inproj = osr.SpatialReference()
        inproj.ImportFromWkt(proj)

        boundbox = None
        crs = None
        with rasterio.open(self.rgbGeotiffFilePath) as dataset:
            crs = dataset.crs
            boundbox = dataset.bounds

        return {
            "rgba_data": rgba_data,
            "gt": gt,
            "proj": proj,
            "inproj": inproj,
            "boundbox": boundbox,
            "crs": str(crs).lower()
        }

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
            full_coeff=get_coefficients_from_file(files('hypso.calibration').joinpath(
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
            full_coeff=get_coefficients_from_file(files('hypso.calibration').joinpath(
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

    def get_spectra(self, position: list, postype: str = 'coord', multiplier=1, filename=None, plot=True):
        '''
        files_path: Works with a directorie with GeoTiff files. Uses the metadata, and integrated CRS
        position:
            [lat, lon] if postype=='coord'
            [X, Y| if postype == 'pix'
        postye:
            'coord' assumes latitude and longitude are passed.
            'pix' receives X and Y values
        '''
        # Read All GeoTiff files in Directory
        # onlyfiles = natsorted([f for f in listdir(files_path)
        #    if isfile(join(files_path, f))])

        # To Store Data
        spectra_data = []

        # Get Columns Name
        cols = ['lat', 'lon', 'X', 'Y']

        for wl in self.wavelengths:
            cols.append("wl"+str(np.round(wl, 2)).replace(".", "_"))

        posX = None
        posY = None
        lat = None
        lon = None
        transformed_lon=None
        transformed_lat=None
        # Open the raster
        with rasterio.open(self.geotiffFilePath) as dataset:
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
                lat, lon = position
                # Transform Coordinates to Image CRS
                transformed_lon, transformed_lat = dataset_proj(
                    lon, lat, inverse=False)
                # Get pixel coordinates from map coordinates
                posY, posX = dataset.index(
                    transformed_lon, transformed_lat)

            elif postype == 'pix':
                posX = int(position[0])
                posY = int(position[1])

                transformed_lon = dataset.xy(posX, posY)[0]
                transformed_lat = dataset.xy(posX, posY)[1]

                # Transform from the GeoTiff CRS
                lon, lat = dataset_proj(
                    transformed_lon, transformed_lat, inverse=True)

            # Window size is 1 for a Single Pixel
            N = 1
            # Build an NxN window
            window = rasterio.windows.Window(
                posX - (N // 2), posY - (N // 2), N, N)

            # Read the data in the window
            # clip is a nbands * N * N numpy array
            clip = dataset.read(window=window)
            clip = np.squeeze(clip)

            # Append data to Array
            # Multiplier for Values like Sentinel 2 which need 1/10000
            spectra_data = clip * multiplier

        # print(spectra_data)
        # # Add Lat, Lon, PosX and PosY to spectra data List
        # spectra_data = [transformed_lat,
        #                 transformed_lon, posX, posY] + list(spectra_data)

        if posX<0 or posY<0 or self.projection_metadata["rgba_data"][3,posY,posX]==0:
            print("Location not covered by image.")
            return None
        
        # Print Coordinate and Pixel Matching
        print("(lat, lon) -→ (X, Y) : (%s, %s) -→ (%s, %s)" %
                (lat, lon, posX, posY))
            
        expanded_spectra_data = list([lat,
                        lon, posX, posY] + list(spectra_data))

        # Get Dataframe to Store
        df_band = pd.DataFrame([expanded_spectra_data], columns=cols)

        if filename != None:
            df_band.to_csv(filename, index=False)

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,5))
            plt.plot(self.wavelengths, spectra_data)
            plt.ylabel(self.units)
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
        cube_calibrated = calibrate_cube(
            self.info, self.rawcube, self.calibration_coefficients_dict)

        # Smile correction
        cube_smile_corrected = smile_correct_cube(
            cube_calibrated, self.calibration_coefficients_dict)

        # Destriping
        cube_destriped = destriping_correct_cube(
            cube_smile_corrected, self.calibration_coefficients_dict)

        return cube_destriped
