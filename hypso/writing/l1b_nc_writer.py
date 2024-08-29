from hypso.writing import set_or_create_attr
from hypso import Hypso
from pathlib import Path
import netCDF4 as nc
import numpy as np


def l1b_nc_writer(satobj: Hypso, dst_l1b_nc_file: str, src_l1a_nc_file: str) -> None:
    """
    Create a l1b.nc file using the radiometrically corrected data. Same structure from the original l1a.nc file
    is used. Required to run ACOLITE as the input is a radiometrically corrected .nc file.

    :return: Nothing.
    """

    # Open L1a file
    old_nc = nc.Dataset(src_l1a_nc_file, 'r', format='NETCDF4')

    # Create a new NetCDF file
    with (nc.Dataset(dst_l1b_nc_file, 'w', format='NETCDF4') as netfile):
        bands = satobj.image_width
        lines = satobj.capture_config["frame_count"]  # AKA Frames AKA Rows
        samples = satobj.image_height  # AKA Cols

        # Set top level attributes -------------------------------------------------
        for md in old_nc.ncattrs():
            set_or_create_attr(netfile,
                                md,
                                old_nc.getncattr(md))

        # Manual Replacement
        set_or_create_attr(netfile,
                            attr_name="radiometric_file",
                            attr_value=str(Path(satobj.rad_coeff_file).name))

        set_or_create_attr(netfile,
                            attr_name="smile_file",
                            attr_value=str(Path(satobj.smile_coeff_file).name))

        # Destriping Path is the only one which can be None
        if satobj.destriping_coeff_file is None:
            set_or_create_attr(netfile,
                                attr_name="destriping",
                                attr_value="No-File")
        else:
            set_or_create_attr(netfile,
                                attr_name="destriping",
                                attr_value=str(Path(satobj.destriping_coeff_file).name))

        set_or_create_attr(netfile, attr_name="spectral_file", attr_value=str(Path(satobj.spectral_coeff_file).name))

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
        Lt.wavelengths = np.around(satobj.spectral_coeffs, 1)
        Lt[:] = satobj.l1b_cube.to_numpy()

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
        len_radrows = satobj.rad_coeffs.shape[0]
        len_radcols = satobj.rad_coeffs.shape[1]

        netfile.createDimension('radrows', len_radrows)
        netfile.createDimension('radcols', len_radcols)
        meta_corrections_rad = netfile.createVariable(
            'metadata/corrections/rad_matrix', 'f4',
            ('radrows', 'radcols'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE)
        meta_corrections_rad[:] = satobj.rad_coeffs

        # Metadata: Spectral coeff ----------------------------------------------------
        len_spectral = satobj.wavelengths.shape[0]
        netfile.createDimension('specrows', len_spectral)
        meta_corrections_spec = netfile.createVariable(
            'metadata/corrections/spec_coeffs', 'f4',
            ('specrows',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE)
        meta_corrections_spec[:] = satobj.spectral_coeffs

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
        #navigation_group = netfile.createGroup('navigation')

        try:
            # Latitude ---------------------------------
            latitude = netfile.createVariable(
                'navigation/latitude', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            # latitude[:] = lat.reshape(frames, lines)
            latitude[:] = satobj.latitudes
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
            longitude[:] = satobj.longitudes
            longitude.long_name = "Longitude"
            longitude.units = "degrees"
            # longitude.valid_range = [-180, 180]
            longitude.valid_min = -180
            longitude.valid_max = 180

        except Exception as ex:
            print("[WARNING] Unable to write latitude and longitude information to NetCDF file. L1b may be incomplete. Please run georeferencing.")
            print("[WARNING] Encountered exception: " + str(ex))


        try:
            sat_zenith_angle = satobj.sat_zenith_angles
            sat_azimuth_angle = satobj.sat_azimuth_angles

            solar_zenith_angle = satobj.solar_zenith_angles
            solar_azimuth_angle = satobj.solar_azimuth_angles

            # Unix time -----------------------
            time = netfile.createVariable('navigation/unixtime', 'u8', ('lines',))

            df = satobj.framepose_df

            time[:] = df["timestamp"].values

            # Sensor Zenith --------------------------
            sensor_z = netfile.createVariable(
                'navigation/sensor_zenith', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_z[:] = sat_zenith_angle.reshape(satobj.spatial_dimensions)
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
            sensor_a[:] = sat_azimuth_angle.reshape(satobj.spatial_dimensions)
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
            solar_z[:] = solar_zenith_angle.reshape(satobj.spatial_dimensions)
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
            solar_a[:] = solar_azimuth_angle.reshape(satobj.spatial_dimensions)
            solar_a.long_name = "Solar Azimuth Angle"
            solar_a.units = "degrees"
            # solar_a.valid_range = [-180, 180]
            solar_a.valid_min = -180
            solar_a.valid_max = 180
    
        except Exception as ex:
            print("[WARNING] Unable to write navigation angles to NetCDF file. L1b file may be incomplete. Please run geometry computations.")
            print("[WARNING] Encountered exception: " + str(ex))


    old_nc.close()

    return None
