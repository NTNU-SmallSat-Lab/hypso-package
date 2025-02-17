from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np
from .navigation_group_writer import navigation_group_writer

def l1c_nc_writer(satobj, dst_nc: str, datacube: str = False) -> None:
    """
    Create a l1c.nc file using the radiometrically corrected data and navigation data.

    :return: Nothing.
    """

    # Open L1a file
    #old_nc = nc.Dataset(src_l1a_nc_file, 'r', format='NETCDF4')

    # Create a new NetCDF file
    with (nc.Dataset(dst_nc, 'w', format='NETCDF4') as netfile):
        bands = satobj.image_width
        lines = satobj.capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        samples = satobj.image_height  # AKA Cols

        # Set top level attributes -------------------------------------------------
        for md in satobj.ncattrs:
            set_or_create_attr(netfile,
                                md,
                                satobj.ncattrs[md])

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

        # Adding metadata ---------------------------------------
        meta_capcon = netfile.createGroup('metadata/capture_config')
        for md in getattr(satobj, 'capture_config_attrs'):
            set_or_create_attr(meta_capcon,
                                md,
                                getattr(satobj, 'capture_config_attrs')[md])

        # Adding timing --------------------------------------
        meta_timing = netfile.createGroup('metadata/timing')
        for md in getattr(satobj, 'timing_attrs'):
            set_or_create_attr(meta_timing,
                                md,
                                getattr(satobj, 'timing_attrs')[md])

        # Adding Temperature -------------------------------------------
        meta_temperature = netfile.createGroup('metadata/temperature')
        for md in getattr(satobj, 'temperature_attrs'):
            set_or_create_attr(meta_temperature,
                                md,
                                getattr(satobj, 'temperature_attrs')[md])


        # Adding ADCS -------------------------------------------
        meta_adcs = netfile.createGroup('metadata/adcs')
        for md in getattr(satobj, 'adcs_attrs'):
            set_or_create_attr(meta_adcs,
                                md,
                                getattr(satobj, 'adcs_attrs')[md])

        # Adding Corrections -------------------------------------------
        meta_corrections = netfile.createGroup('metadata/corrections')
        for md in getattr(satobj, 'corrections_attrs'):
            set_or_create_attr(meta_corrections,
                                md,
                                getattr(satobj, 'corrections_attrs')[md])


        # Adding Database -------------------------------------------
        meta_database = netfile.createGroup('metadata/database')
        for md in getattr(satobj, 'database_attrs'):
            set_or_create_attr(meta_database,
                                md,
                                getattr(satobj, 'database_attrs')[md])




        # Set pseudoglobal vars like compression level
        COMP_SCHEME = 'zlib'  # Default: zlib
        COMP_LEVEL = 4  # Default (when scheme != none): 4
        COMP_SHUFFLE = True  # Default (when scheme != none): True


        # Create and populate variables

        if datacube:

            # Store as datacube
            Lt = netfile.createVariable(
                'products/Lt', 'f8',
                ('lines', 'samples', 'bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            Lt.units = "W/m^2/micrometer/sr"
            Lt.long_name = "Top-of-Atmosphere Radiance"
            Lt.wavelength_units = "nanometers"
            Lt.fwhm = [5.5] * bands
            Lt.wavelengths = np.around(satobj.spectral_coeffs, 1)
            Lt[:] = satobj.l1b_cube.to_numpy()

        else:

            # Store as bands
            Lt_cube = satobj.l1b_cube.to_numpy()
            for band in range(0, Lt_cube.shape[-1]):

                wave = np.around(satobj.spectral_coeffs, 1)[band]
                wave_name = str(int(wave))
                name = 'Lt_' + wave_name

                Lt = netfile.createVariable(
                    'products/' + name, 'f8',
                    ('lines', 'samples'),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                Lt.units = "W/m^2/micrometer/sr"
                Lt.long_name = "Top-of-Atmosphere Radiance Band " + str(band) + " (" + wave_name + " nm)"
                Lt.wavelength_units = "nanometers"
                Lt.fwhm = 5.5
                Lt.wavelength = wave

                #Lt.f0 = None
                #Lt.width = 5.5
                Lt.wave = wave
                Lt.parameter = name
                Lt.wave_name = wave_name
                Lt.band = band

                Lt[:] = Lt_cube[:,:,band]


        # ADCS Timestamps ----------------------------------------------------
        len_timestamps = getattr(satobj, 'dimensions')["adcssamples"] #.size
        netfile.createDimension('adcssamples', len_timestamps)

        meta_adcs_timestamps = netfile.createVariable(
            'metadata/adcs/timestamps', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )

        meta_adcs_timestamps[:] = getattr(satobj, 'adcs_vars')["timestamps"][:]

        # ADCS Position X -----------------------------------------------------
        meta_adcs_position_x = netfile.createVariable(
            'metadata/adcs/position_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_x[:] = getattr(satobj, 'adcs_vars')["position_x"][:]

        # ADCS Position Y -----------------------------------------------------
        meta_adcs_position_y = netfile.createVariable(
            'metadata/adcs/position_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_y[:] = getattr(satobj, 'adcs_vars')["position_y"][:]

        # ADCS Position Z -----------------------------------------------------
        meta_adcs_position_z = netfile.createVariable(
            'metadata/adcs/position_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_z[:] = getattr(satobj, 'adcs_vars')["position_z"][:]

        # ADCS Velocity X -----------------------------------------------------
        meta_adcs_velocity_x = netfile.createVariable(
            'metadata/adcs/velocity_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_x[:] = getattr(satobj, 'adcs_vars')["velocity_x"][:]

        # ADCS Velocity Y -----------------------------------------------------
        meta_adcs_velocity_y = netfile.createVariable(
            'metadata/adcs/velocity_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_y[:] = getattr(satobj, 'adcs_vars')["velocity_y"][:]

        # ADCS Velocity Z -----------------------------------------------------
        meta_adcs_velocity_z = netfile.createVariable(
            'metadata/adcs/velocity_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_z[:] = getattr(satobj, 'adcs_vars')["velocity_z"][:]

        # ADCS Quaternion S -----------------------------------------------------
        meta_adcs_quaternion_s = netfile.createVariable(
            'metadata/adcs/quaternion_s', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_s[:] = getattr(satobj, 'adcs_vars')["quaternion_s"][:]

        # ADCS Quaternion X -----------------------------------------------------
        meta_adcs_quaternion_x = netfile.createVariable(
            'metadata/adcs/quaternion_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_x[:] = getattr(satobj, 'adcs_vars')["quaternion_x"][:]

        # ADCS Quaternion Y -----------------------------------------------------
        meta_adcs_quaternion_y = netfile.createVariable(
            'metadata/adcs/quaternion_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_y[:] = getattr(satobj, 'adcs_vars')["quaternion_y"][:]

        # ADCS Quaternion Z -----------------------------------------------------
        meta_adcs_quaternion_z = netfile.createVariable(
            'metadata/adcs/quaternion_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_z[:] = getattr(satobj, 'adcs_vars')["quaternion_z"][:]

        # ADCS Angular Velocity X -----------------------------------------------------
        meta_adcs_angular_velocity_x = netfile.createVariable(
            'metadata/adcs/angular_velocity_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_x[:] = getattr(satobj, 'adcs_vars')["angular_velocity_x"][:]

        # ADCS Angular Velocity Y -----------------------------------------------------
        meta_adcs_angular_velocity_y = netfile.createVariable(
            'metadata/adcs/angular_velocity_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_y[:] = getattr(satobj, 'adcs_vars')["angular_velocity_y"][:]

        # ADCS Angular Velocity Z -----------------------------------------------------
        meta_adcs_angular_velocity_z = netfile.createVariable(
            'metadata/adcs/angular_velocity_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_z[:] = getattr(satobj, 'adcs_vars')["angular_velocity_z"][:]

        # ADCS ST Quaternion S -----------------------------------------------------
        meta_adcs_st_quaternion_s = netfile.createVariable(
            'metadata/adcs/st_quaternion_s', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_st_quaternion_s[:] = getattr(satobj, 'adcs_vars')["st_quaternion_s"][:]

        # ADCS ST Quaternion X -----------------------------------------------------
        meta_adcs_st_quaternion_x = netfile.createVariable(
            'metadata/adcs/st_quaternion_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_st_quaternion_x[:] = getattr(satobj, 'adcs_vars')["st_quaternion_x"][:]

        # ADCS ST Quaternion Y -----------------------------------------------------
        meta_adcs_st_quaternion_y = netfile.createVariable(
            'metadata/adcs/st_quaternion_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_st_quaternion_y[:] = getattr(satobj, 'adcs_vars')["st_quaternion_y"][:]

        # ADCS ST Quaternion Z -----------------------------------------------------
        meta_adcs_st_quaternion_z = netfile.createVariable(
            'metadata/adcs/st_quaternion_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_st_quaternion_z[:] = getattr(satobj, 'adcs_vars')["st_quaternion_z"][:]

        # ADCS Control Error -----------------------------------------------------
        meta_adcs_control_error = netfile.createVariable(
            'metadata/adcs/control_error', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_control_error[:] = getattr(satobj, 'adcs_vars')["control_error"][:]


        # Capcon File -------------------------------------------------------------
        meta_capcon_file = netfile.createVariable(
            'metadata/capture_config/file', 'str')  # str seems necessary for storage of an arbitrarily large scalar
        meta_capcon_file[()] = getattr(satobj, 'capture_config_vars')["file"][:]  # [()] assignment of scalar to array


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
        meta_temperature_file[()] = getattr(satobj, 'temperature_vars')["file"][:]

        # Bin Time ----------------------------------------------------------------------
        bin_time = netfile.createVariable(
            'metadata/timing/bin_time', 'uint16',
            ('lines',))
        bin_time[:] = getattr(satobj, 'timing_vars')["bin_time"][:]

        # Timestamps -------------------------------------------------------------------
        timestamps = netfile.createVariable(
            'metadata/timing/timestamps', 'uint32',
            ('lines',))
        timestamps[:] = getattr(satobj, 'timing_vars')["timestamps"][:]

        # Timestamps Service -----------------------------------------------------------
        timestamps_srv = netfile.createVariable(
            'metadata/timing/timestamps_srv', 'f8',
            ('lines',))
        timestamps_srv[:] = getattr(satobj, 'timing_vars')["timestamps_srv"][:]

        navigation_group_writer(satobj=satobj, netfile=netfile, product_level="L1c")
    
    return None