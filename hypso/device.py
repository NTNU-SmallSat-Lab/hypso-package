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
from .calibration import crop_and_bin_matrix, calibrate_cube, get_coefficients_from_dict, get_coefficients_from_file, \
    smile_correct_cube, destriping_correct_cube
from .georeference import start_coordinate_correction, generate_full_geotiff, generate_rgb_geotiff
from .reading import load_nc
from hypso.utils import find_dir, find_file, find_all_files
from .atmospheric import run_py6s, run_acolite

EXPERIMENTAL_FEATURES = True


def set_or_create_attr(var, attr_name, attr_value):
    if attr_name in var.ncattrs():
        var.setncattr(attr_name, attr_value)
        return
    var.UnusedNameAttribute = attr_value
    var.renameAttribute("UnusedNameAttribute", attr_name)
    return


class Hypso:
    def __init__(self, hypso_path, points_path=None) -> None:
        self.projection_metadata = None
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
        if hypso_path.suffix == '.nc':
            # Obtain metadata from files
            self.info, self.rawcube, self.spatialDim = load_nc(hypso_path, self.standardDimensions)
            self.info["top_folder_name"] = self.info["tmp_dir"]
            self.info["nc_file"] = hypso_path
        else:
            raise Exception("Incorrect HYPSO Path. Only .nc files supported")

        # Correction Coefficients ----------------------------------------
        self.calibration_coeffs_file_dict = self.get_calibration_coefficients_path()
        self.calibration_coefficients_dict = get_coefficients_from_dict(
            self.calibration_coeffs_file_dict, self)

        # Wavelengths -----------------------------------------------------
        self.spectral_coeff_file = self.get_spectral_coefficients_path()
        self.spectral_coefficients = get_coefficients_from_file(
            self.spectral_coeff_file)
        self.wavelengths = self.spectral_coefficients

        # Calibrate and Correct Cube Variables and load existing L2 Cube  ----------------------------------------
        self.l1b_cube = self.get_calibrated_and_corrected_cube()

        # Create L1B .nc File
        self.create_l1b_nc_file(hypso_path)  # Input for ACOLITE

        self.l2a_cube = self.find_existing_l2_cube()

        # Generated afterwards
        self.waterMask = None

        # Get SRF
        self.srf = self.get_srf()

        # Generate RGB/RGBA Geotiff with Projection metadata and L1B
        # If points file used, run twice to use correct coordinates
        message_list = ["Getting Projection Data without lat/lon correction =========================================",
                        "Getting Projection Data with coordinate correction ========================================="]
        range_correction = 1
        if points_path is not None:
            range_correction = 2
        for i in range(range_correction):
            print(message_list[i])
            generate_rgb_geotiff(self, overwrite=True)
            # Get Projection Metadata from created geotiff
            self.projection_metadata = self.get_projection_metadata()
            # Before Generating new Geotiff we check if .points file exists and update 2D coord
            if i == 0:
                self.info["lat"], self.info["lon"] = start_coordinate_correction(
                    self.pointsPath, self.info, self.projection_metadata)

    def get_srf(self):
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

        return srf

    # TODO: Replace L1A with the same but only the rawcube instead of L1B
    def create_l1b_nc_file(self, hypso_nc_path):
        old_nc = nc.Dataset(hypso_nc_path, 'r', format='NETCDF4')
        new_path = hypso_nc_path
        new_path = str(new_path).replace('l1a.nc', 'l1b.nc')

        if Path(new_path).is_file():
            print("L1b.nc file already exists. Not creating it.")
            self.info["nc_file"] = Path(new_path)
            return

        # Create a new NetCDF file
        with (nc.Dataset(new_path, 'w', format='NETCDF4') as netfile):
            bands = self.info["image_width"]
            lines = self.info["frame_count"]  # AKA Frames AKA Rows
            samples = self.info["image_height"]  # AKA Cols

            # Set top level attributes -------------------------------------------------
            for md in old_nc.ncattrs():
                set_or_create_attr(netfile,
                                   md,
                                   old_nc.getncattr(md))

            # Manual Replacement
            set_or_create_attr(netfile,
                               attr_name="radiometric_file",
                               attr_value=str(Path(self.calibration_coeffs_file_dict["radiometric"]).name))

            set_or_create_attr(netfile,
                               attr_name="smile_file",
                               attr_value=str(Path(self.calibration_coeffs_file_dict["smile"]).name))

            # Destriping Path is the only one which can be None
            if self.calibration_coeffs_file_dict["destriping"] is None:
                set_or_create_attr(netfile,
                                   attr_name="destriping",
                                   attr_value="No-File")
            else:
                set_or_create_attr(netfile,
                                   attr_name="destriping",
                                   attr_value=str(Path(self.calibration_coeffs_file_dict["destriping"]).name))

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
            Lt.wavelengths = np.around(self.spectral_coefficients, 1)
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
            len_radrows = self.calibration_coefficients_dict["radiometric"].shape[0]
            len_radcols = self.calibration_coefficients_dict["radiometric"].shape[1]
            netfile.createDimension('radrows', len_radrows)
            netfile.createDimension('radcols', len_radcols)
            meta_corrections_rad = netfile.createVariable(
                'metadata/corrections/rad_matrix', 'f4',
                ('radrows', 'radcols'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_rad[:] = self.calibration_coefficients_dict["radiometric"]

            # Metadata: Spectral coeff ----------------------------------------------------
            len_spectral = self.wavelengths.shape[0]
            netfile.createDimension('specrows', len_spectral)
            meta_corrections_spec = netfile.createVariable(
                'metadata/corrections/spec_coeffs', 'f4',
                ('specrows',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_spec[:] = self.spectral_coefficients

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
                sat_zenith_angle = self.info["sat_zenith_angle"]
                sat_azimuth_angle = self.info["sat_azimuth_angle"]

                solar_zenith_angle = self.info["solar_zenith_angle"]
                solar_azimuth_angle = self.info["solar_azimuth_angle"]

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
                sensor_z[:] = sat_zenith_angle.reshape(self.spatialDim)
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
                sensor_a[:] = sat_azimuth_angle.reshape(self.spatialDim)
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
                solar_z[:] = solar_zenith_angle.reshape(self.spatialDim)
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
                solar_a[:] = solar_azimuth_angle.reshape(self.spatialDim)
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
                latitude[:] = self.info["lat"]
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
                longitude[:] = self.info["lon"]
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
        self.info["nc_file"] = Path(new_path)

    def create_geotiff(self, product="L2", force_reload=False, atmos_dict=None):

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
            self.projection_metadata = self.get_projection_metadata()

        atmos_corrected_cube = None
        atmos_model = None

        if product == "L2":
            atmos_model = atmos_dict["model"].upper()
            try:
                atmos_corrected_cube = self.l2a_cube[atmos_model]
            except Exception as err:
                if atmos_model == "6SV1":
                    # Py6S Atmospheric Correction
                    # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
                    # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
                    # atmos_dict = {
                    #     'model': '6sv1',
                    #     'aot550': 0.01,
                    #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
                    # }
                    atmos_corrected_cube = run_py6s(self.wavelengths, self.l1b_cube, self.info, self.info["lat"],
                                                    self.info["lon"],
                                                    atmos_dict, time_capture=parser.parse(self.info['iso_time']),
                                                    srf=self.srf)
                elif atmos_model == "ACOLITE":
                    print("Getting ACOLITE L2")
                    if not self.info["nc_file"].is_file():
                        raise Exception("No -l1b.nc file found")
                    file_name_l1b = self.info["nc_file"].name
                    print(f"Found {file_name_l1b}")
                    atmos_corrected_cube = run_acolite(self.info, atmos_dict, self.info["nc_file"])

            # Store the l2a_cube just generated
            if self.l2a_cube is None:
                self.l2a_cube = {}

            self.l2a_cube[atmos_model] = atmos_corrected_cube

            with open(Path(self.info["top_folder_name"], "geotiff", f'L2_{atmos_model}.npy'), 'wb') as f:
                np.save(f, atmos_corrected_cube)

        # Generate RGBA/RGBA and Full Geotiff with corrected metadata and L2A if exists (if not L1B)
        generate_full_geotiff(self, product=product, L2_key=atmos_model)

    def delete_geotiff_dir(self, top_folder_name: Path):
        tiff_name = "geotiff"
        geotiff_dir = find_dir(top_folder_name, tiff_name)

        self.rgbGeotiffFilePath = None
        self.l1cgeotiffFilePath = None
        self.l2geotiffFilePath = None

        if geotiff_dir is not None:
            print("Deleting geotiff Directory...")
            import shutil
            shutil.rmtree(geotiff_dir, ignore_errors=True)

    def find_geotiffs(self):
        top_folder_name = self.info["top_folder_name"]
        self.rgbGeotiffFilePath = find_file(top_folder_name, "rgba_8bit", ".tif")
        self.l1cgeotiffFilePath = find_file(top_folder_name, "-full_L1C", ".tif")

        L2_dict = {}
        L2_files = find_all_files(top_folder_name, "-full_L2", ".tif")
        if len(L2_files) > 0:
            for f in find_all_files(top_folder_name, "-full_L2", ".tif"):
                key = str(f.stem).split("_")[-1]
                L2_dict[key] = f
            self.l2geotiffFilePath = L2_dict

    def find_existing_l2_cube(self) -> Union[dict, None]:
        found_l2_npy = find_all_files(path=Path(self.info["top_folder_name"], "geotiff"),
                                      str_in_file="L2",
                                      suffix=".npy")

        if found_l2_npy is None:
            return None

        dict_L2 = None
        # Save Generated Cube as "npy" (for faster loading
        for l2_file in found_l2_npy:
            correction_model = str(l2_file.stem).split("_")[1]
            correction_model = correction_model.upper()
            l2_cube = None
            with open(l2_file, 'rb') as f:
                print(f"Found {l2_file.name}")
                l2_cube = np.load(f)

            if dict_L2 is None:
                dict_L2 = {}
            dict_L2[correction_model] = l2_cube

        return dict_L2

    def get_projection_metadata(self) -> dict:
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

    def get_calibration_coefficients_path(self) -> dict:
        csv_file_radiometric = None
        csv_file_smile = None
        csv_file_destriping = None

        if self.info["capture_type"] == "custom":

            # Radiometric ---------------------------------
            full_rad_coeff_file = files('hypso.calibration').joinpath(f'data/{"radiometric_calibration_matrix_HYPSO-1_full_v1.csv"}')

            # Smile ---------------------------------
            full_smile_coeff_file = files('hypso.calibration').joinpath(f'data/{"spectral_calibration_matrix_HYPSO-1_full_v1.csv"}')

            # Destriping (not available for custom)
            full_destripig_coeff_file = None

            return {"radiometric": full_smile_coeff_file,
                    "smile": full_smile_coeff_file,
                    "destriping": full_destripig_coeff_file}

        elif self.info["capture_type"] == "nominal":
            csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv"
            csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"
            csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"
        elif self.info["capture_type"] == "wide":
            csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.csv"
            csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"
            csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"



        rad_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_radiometric}')

        smile_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_smile}')
        destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_destriping}')

        coeff_dict = {"radiometric": rad_coeff_file,
                      "smile": smile_coeff_file,
                      "destriping": destriping_coeff_file}

        return coeff_dict

    def get_spectral_coefficients_path(self) -> str:
        csv_file = "spectral_bands_HYPSO-1_v1.csv"
        wl_file = files(
            'hypso.calibration').joinpath(f'data/{csv_file}')
        return str(wl_file)

    def get_spectra(self, position_dict: dict, product: str = "L2", postype: str = 'coord', multiplier=1, filename=None,
                    plot=True, L2_engine="6SV1"):
        """

        :param position_dict:
            [lat, lon] if postype=='coord'
            [X, Y| if postype == 'pix'

        :param product:
        :param postype:
            'coord' assumes latitude and longitude are passed.
            'pix' receives X and Y values

        :param multiplier:
        :param filename:
        :param plot:
        :param L2_engine:

        :return:
        """

        # To Store data
        spectra_data = []

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
        if product == "L2":
            if self.l2geotiffFilePath is None:
                raise Exception("L2 product does not exist.")
            elif self.l2geotiffFilePath is not None:
                try:
                    path_to_read = self.l2geotiffFilePath[L2_engine.upper()]
                except:
                    raise Exception(f"There is no L2 Geotiff for {L2_engine.upper()}")

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

        if filename is not None:
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

    def get_calibrated_and_corrected_cube(self) -> np.ndarray:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: Numpy Array with corrected cub
        """

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

    def toa_reflectance_from_toa_radiance(self):
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
