from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np
from .navigation_group_writer import navigation_group_writer
from .calibration_filenames_writer import calibration_filenames_writer
from pathlib import Path

def write_l2_nc_file(satobj, correction: str = None, overwrite: bool = False, **kwargs) -> None:
    
    l2_nc_file_list = []

    parent_dir = satobj.parent_dir
    capture_name = satobj.capture_name

    if correction is not None:
        l2_nc_file_list.append(correction)

    else:
        for correction in satobj.l2_cubes.keys():
            l2_nc_file_list.append(correction)


    for correction in l2_nc_file_list:

        l2_nc_file = Path(parent_dir, str(capture_name) + "-l2-" + str(correction) + ".nc")

        print(l2_nc_file)

        if Path(l2_nc_file).is_file() and not overwrite:

            if satobj.VERBOSE:
                print("[INFO] L2 NetCDF file has already been generated. Skipping.")

            return None

        l2_nc_writer(satobj=satobj,
                    correction = correction, 
                    dst_nc=l2_nc_file, 
                    **kwargs
                    )


    return None


def l2_nc_writer(satobj, correction: str, dst_nc: str, datacube: str = True) -> None:
    """
    Create a l2.nc file using the bottom-of-atmosphere data.

    :return: Nothing.
    """

    # Create a new NetCDF file
    with (nc.Dataset(dst_nc, 'w', format='NETCDF4') as netfile):
        bands = satobj.image_width
        lines = satobj.nc_capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        samples = satobj.image_height  # AKA Cols

        # Set top level attributes -------------------------------------------------
        for md in satobj.nc_attrs:
            set_or_create_attr(netfile,
                                md,
                                satobj.nc_attrs[md])

        set_or_create_attr(netfile, attr_name="processing_level", attr_value="L2")

        # Add calibration file names
        calibration_filenames_writer(satobj=satobj, netfile=netfile)

        # Create dimensions
        netfile.createDimension('lines', lines)
        netfile.createDimension('samples', samples)
        netfile.createDimension('bands', bands)

        # Create groups
        #netfile.createGroup('logfiles')

        netfile.createGroup('products')

        netfile.createGroup('metadata')

        # Adding metadata ---------------------------------------
        meta_capcon = netfile.createGroup('metadata/capture_config')
        for md in getattr(satobj, 'nc_capture_config_attrs'):
            set_or_create_attr(meta_capcon,
                                md,
                                getattr(satobj, 'nc_capture_config_attrs')[md])

        # Adding timing --------------------------------------
        meta_timing = netfile.createGroup('metadata/timing')
        for md in getattr(satobj, 'nc_timing_attrs'):
            set_or_create_attr(meta_timing,
                                md,
                                getattr(satobj, 'nc_timing_attrs')[md])

        # # Adding Temperature -------------------------------------------
        # meta_temperature = netfile.createGroup('metadata/temperature')
        # for md in getattr(satobj, 'nc_temperature_attrs'):
        #     set_or_create_attr(meta_temperature,
        #                         md,
        #                         getattr(satobj, 'nc_temperature_attrs')[md])


        # Adding ADCS -------------------------------------------
        meta_adcs = netfile.createGroup('metadata/adcs')
        for md in getattr(satobj, 'nc_adcs_attrs'):
            set_or_create_attr(meta_adcs,
                                md,
                                getattr(satobj, 'nc_adcs_attrs')[md])

        # Adding Corrections -------------------------------------------
        meta_corrections = netfile.createGroup('metadata/corrections')
        for md in getattr(satobj, 'nc_corrections_attrs'):
            set_or_create_attr(meta_corrections,
                                md,
                                getattr(satobj, 'nc_corrections_attrs')[md])


        # # Adding Database -------------------------------------------
        # meta_database = netfile.createGroup('metadata/database')
        # for md in getattr(satobj, 'nc_database_attrs'):
        #     set_or_create_attr(meta_database,
        #                         md,
        #                         getattr(satobj, 'nc_database_attrs')[md])




        # Set pseudoglobal vars like compression level
        COMP_SCHEME = 'zlib'  # Default: zlib
        COMP_LEVEL = 4  # Default (when scheme != none): 4
        COMP_SHUFFLE = True  # Default (when scheme != none): True


        # Create and populate variables
        if datacube:

            # Store as datacube
            rrs = netfile.createVariable(
                'products/rrs', 'f4',
                ('lines', 'samples', 'bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            rrs.units = ""
            rrs.long_name = "Bottom-of-Atmosphere Reflectance"
            rrs.wavelength_units = "nanometers"
            rrs.fwhm = satobj.fwhm
            rrs.wavelengths = np.around(satobj.wavelengths, 1)
            rrs[:] = satobj.l2_cubes[correction].to_numpy()

        else:

            # Store as bands
            rrs_cube = satobj.l2_cubes[correction].to_numpy()
            for band in range(0, rrs_cube.shape[-1]):

                wave = np.around(satobj.wavelengths, 1)[band]
                wave_name = str(int(wave))
                name = 'rrs_' + wave_name

                rrs = netfile.createVariable(
                    'products/' + name, 'f4',
                    ('lines', 'samples'),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                rrs.units = ""
                rrs.long_name = "Bottom-of-Atmosphere Reflectance Band " + str(band) + " (" + wave_name + " nm)"
                rrs.wavelength_units = "nanometers"
                rrs.fwhm = satobj.fwhm[band]
                rrs.wavelength = wave

                rrs.radiation_wavelength = float(satobj.wavelengths[band]),
                rrs.radiation_wavelength_unit = "nm"

                #rrs.f0 = None
                #rrs.width = satobj.fwhm[band]
                rrs.wave = wave
                rrs.parameter = name
                rrs.wave_name = wave_name
                rrs.band = band

                rrs[:] = rrs_cube[:,:,band]


        # ADCS Timestamps ----------------------------------------------------
        len_timestamps = getattr(satobj, 'nc_dimensions')["adcssamples"] #.size
        netfile.createDimension('adcssamples', len_timestamps)

        meta_adcs_timestamps = netfile.createVariable(
            'metadata/adcs/timestamps', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )

        meta_adcs_timestamps[:] = getattr(satobj, 'nc_adcs_vars')["timestamps"][:]

        # ADCS Position X -----------------------------------------------------
        meta_adcs_position_x = netfile.createVariable(
            'metadata/adcs/position_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_x[:] = getattr(satobj, 'nc_adcs_vars')["position_x"][:]

        # ADCS Position Y -----------------------------------------------------
        meta_adcs_position_y = netfile.createVariable(
            'metadata/adcs/position_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_y[:] = getattr(satobj, 'nc_adcs_vars')["position_y"][:]

        # ADCS Position Z -----------------------------------------------------
        meta_adcs_position_z = netfile.createVariable(
            'metadata/adcs/position_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_position_z[:] = getattr(satobj, 'nc_adcs_vars')["position_z"][:]

        # ADCS Velocity X -----------------------------------------------------
        meta_adcs_velocity_x = netfile.createVariable(
            'metadata/adcs/velocity_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_x[:] = getattr(satobj, 'nc_adcs_vars')["velocity_x"][:]

        # ADCS Velocity Y -----------------------------------------------------
        meta_adcs_velocity_y = netfile.createVariable(
            'metadata/adcs/velocity_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_y[:] = getattr(satobj, 'nc_adcs_vars')["velocity_y"][:]

        # ADCS Velocity Z -----------------------------------------------------
        meta_adcs_velocity_z = netfile.createVariable(
            'metadata/adcs/velocity_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_velocity_z[:] = getattr(satobj, 'nc_adcs_vars')["velocity_z"][:]

        # ADCS Quaternion S -----------------------------------------------------
        meta_adcs_quaternion_s = netfile.createVariable(
            'metadata/adcs/quaternion_s', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_s[:] = getattr(satobj, 'nc_adcs_vars')["quaternion_s"][:]

        # ADCS Quaternion X -----------------------------------------------------
        meta_adcs_quaternion_x = netfile.createVariable(
            'metadata/adcs/quaternion_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_x[:] = getattr(satobj, 'nc_adcs_vars')["quaternion_x"][:]

        # ADCS Quaternion Y -----------------------------------------------------
        meta_adcs_quaternion_y = netfile.createVariable(
            'metadata/adcs/quaternion_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_y[:] = getattr(satobj, 'nc_adcs_vars')["quaternion_y"][:]

        # ADCS Quaternion Z -----------------------------------------------------
        meta_adcs_quaternion_z = netfile.createVariable(
            'metadata/adcs/quaternion_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_quaternion_z[:] = getattr(satobj, 'nc_adcs_vars')["quaternion_z"][:]

        # ADCS Angular Velocity X -----------------------------------------------------
        meta_adcs_angular_velocity_x = netfile.createVariable(
            'metadata/adcs/angular_velocity_x', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_x[:] = getattr(satobj, 'nc_adcs_vars')["angular_velocity_x"][:]

        # ADCS Angular Velocity Y -----------------------------------------------------
        meta_adcs_angular_velocity_y = netfile.createVariable(
            'metadata/adcs/angular_velocity_y', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_y[:] = getattr(satobj, 'nc_adcs_vars')["angular_velocity_y"][:]

        # ADCS Angular Velocity Z -----------------------------------------------------
        meta_adcs_angular_velocity_z = netfile.createVariable(
            'metadata/adcs/angular_velocity_z', 'f8',
            ('adcssamples',),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE
        )
        meta_adcs_angular_velocity_z[:] = getattr(satobj, 'nc_adcs_vars')["angular_velocity_z"][:]

        # # ADCS ST Quaternion S -----------------------------------------------------
        # meta_adcs_st_quaternion_s = netfile.createVariable(
        #     'metadata/adcs/st_quaternion_s', 'f8',
        #     ('adcssamples',),
        #     compression=COMP_SCHEME,
        #     complevel=COMP_LEVEL,
        #     shuffle=COMP_SHUFFLE
        # )
        # meta_adcs_st_quaternion_s[:] = getattr(satobj, 'nc_adcs_vars')["st_quaternion_s"][:]

        # # ADCS ST Quaternion X -----------------------------------------------------
        # meta_adcs_st_quaternion_x = netfile.createVariable(
        #     'metadata/adcs/st_quaternion_x', 'f8',
        #     ('adcssamples',),
        #     compression=COMP_SCHEME,
        #     complevel=COMP_LEVEL,
        #     shuffle=COMP_SHUFFLE
        # )
        # meta_adcs_st_quaternion_x[:] = getattr(satobj, 'nc_adcs_vars')["st_quaternion_x"][:]

        # # ADCS ST Quaternion Y -----------------------------------------------------
        # meta_adcs_st_quaternion_y = netfile.createVariable(
        #     'metadata/adcs/st_quaternion_y', 'f8',
        #     ('adcssamples',),
        #     compression=COMP_SCHEME,
        #     complevel=COMP_LEVEL,
        #     shuffle=COMP_SHUFFLE
        # )
        # meta_adcs_st_quaternion_y[:] = getattr(satobj, 'nc_adcs_vars')["st_quaternion_y"][:]

        # # ADCS ST Quaternion Z -----------------------------------------------------
        # meta_adcs_st_quaternion_z = netfile.createVariable(
        #     'metadata/adcs/st_quaternion_z', 'f8',
        #     ('adcssamples',),
        #     compression=COMP_SCHEME,
        #     complevel=COMP_LEVEL,
        #     shuffle=COMP_SHUFFLE
        # )
        # meta_adcs_st_quaternion_z[:] = getattr(satobj, 'nc_adcs_vars')["st_quaternion_z"][:]

        # # ADCS Control Error -----------------------------------------------------
        # meta_adcs_control_error = netfile.createVariable(
        #     'metadata/adcs/control_error', 'f8',
        #     ('adcssamples',),
        #     compression=COMP_SCHEME,
        #     complevel=COMP_LEVEL,
        #     shuffle=COMP_SHUFFLE
        # )
        # meta_adcs_control_error[:] = getattr(satobj, 'nc_adcs_vars')["control_error"][:]


        # # Capcon File -------------------------------------------------------------
        # meta_capcon_file = netfile.createVariable(
        #     'metadata/capture_config/file', 'str')  # str seems necessary for storage of an arbitrarily large scalar
        # meta_capcon_file[()] = getattr(satobj, 'nc_capture_config_vars')["file"][:]  # [()] assignment of scalar to array


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

        # Metadata: Wavelengths or Spectral coeffs ----------------------------------------------------
        try:
            wavelengths = satobj.spectral_coeffs
            len_spectral = satobj.wavelengths.shape[0]
            netfile.createDimension('bands', len_spectral)
            meta_corrections_wl = netfile.createVariable(
                'metadata/corrections/wavelengths', 'f4',
                ('bands',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_wl[:] = wavelengths
        except:
            pass
        try:
            spectral_coeffs = satobj.spectral_coeffs
            len_spectral = satobj.wavelengths.shape[0]
            netfile.createDimension('specrows', len_spectral)
            meta_corrections_spec = netfile.createVariable(
                'metadata/corrections/spec_coeffs', 'f4',
                ('specrows',),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            meta_corrections_spec[:] = spectral_coeffs
        except:
            pass


        # # Meta Temperature File ---------------------------------------------------------
        # meta_temperature_file = netfile.createVariable(
        #     'metadata/temperature/file', 'str')
        # meta_temperature_file[()] = getattr(satobj, 'nc_temperature_vars')["file"][:]

        # # Bin Time ----------------------------------------------------------------------
        # bin_time = netfile.createVariable(
        #     'metadata/timing/bin_time', 'uint16',
        #     ('lines',))
        # bin_time[:] = getattr(satobj, 'nc_timing_vars')["bin_time"][:]

        # Timestamps -------------------------------------------------------------------
        timestamps = netfile.createVariable(
            'metadata/timing/timestamps', 'f8',
            ('lines',))
        timestamps[:] = getattr(satobj, 'nc_timing_vars')["timestamps"][:]

        # # Timestamps Service -----------------------------------------------------------
        # timestamps_srv = netfile.createVariable(
        #     'metadata/timing/timestamps_srv', 'f8',
        #     ('lines',))
        # timestamps_srv[:] = getattr(satobj, 'nc_timing_vars')["timestamps_srv"][:]
    
        navigation_group_writer(satobj=satobj, netfile=netfile)

    return None