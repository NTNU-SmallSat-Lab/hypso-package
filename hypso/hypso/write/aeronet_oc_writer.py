import numpy as np
from pathlib import Path
import netCDF4 as nc
from datetime import datetime

from .utils import set_or_create_attr



def write_aeronet_oc_matchup_nc_file(satobj, matchup_data, atmospheric_correction: str = None, dst_nc: str = None, datacube: str = True):
    """
    Write AERONET-OC matchup extraction results to a NetCDF file.
    
    Parameters:
    -----------
    satobj : object
        Satellite object containing metadata
    matchup_data : dict
        The dictionary returned by aeronet_oc_extract_matchup_area
    dst_nc : str or Path, optional
        Directory to save the file. If None, uses satobj.parent_dir
    atmospheric_correction : str
        Atmospheric correction algorithm used (default "polymer")
    
    Returns:
    --------
    Path : Path to the created NetCDF file
    """
    
    if matchup_data is None:
        print("Error: No extraction result provided")
        return None
    
    # Determine output directory
    if dst_nc is None:
        dst_nc = Path(satobj.parent_dir)
    else:
        dst_nc = Path(dst_nc)
    
    dst_nc.mkdir(parents=True, exist_ok=True)
    

    # Create filename
    capture_name = satobj.capture_name
    hypso_name = matchup_data.get("hypso_name", "unknown")
    aeronet_name = matchup_data.get("aeronet_name", "unknown")

    product_symbol = getattr(satobj, "product_symbol", None)
    product_level = getattr(satobj, "product_level", None)
    
    label = getattr(satobj, "label", None)
    if label:
        label = "_" + str(label)
    else:
        label = ""


    filename_parts = [
        capture_name,
        "AERONET-OC",
        hypso_name,
        aeronet_name,
        product_level,
        atmospheric_correction,
        label
    ]

    # Keep only non-None and non-empty values
    parts = [p for p in filename_parts if p]

    filename = "_".join(parts) + ".nc"

    output_path = dst_nc / filename
    
    # Compression settings
    COMP_SCHEME = 'zlib'
    COMP_LEVEL = 4
    COMP_SHUFFLE = True
    
    # Get extracted data
    extracted_cube = matchup_data["extracted_cube"]  # Shape: (n_size, n_size, n_bands)
    n_size = matchup_data["requested_size"]
    n_bands = extracted_cube.shape[2]
    
    # Get wavelengths if available
    #wavelengths = getattr(satobj, 'wavelengths', None)
    #fwhm = getattr(satobj, 'fwhm', None)

    # Get AERONET-OC wavelengths
    aeronet_wavelengths = matchup_data.get("aeronet_wavelengths")["values"]
    n_aeronet_bands = len(aeronet_wavelengths)

    
    # Create NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as netfile:
        
        # Set global attributes
        set_or_create_attr(netfile, "title", f"AERONET-OC Matchup for {capture_name}")
        set_or_create_attr(netfile, "hypso_capture_name", capture_name)
        set_or_create_attr(netfile, "hypso_target_name", hypso_name)
        set_or_create_attr(netfile, "aeronet_oc_site_name", aeronet_name)
        set_or_create_attr(netfile, "atmospheric_correction_algorithm", atmospheric_correction)
        set_or_create_attr(netfile, "processing_level", "L2_AERONET_OC_MATCHUP")
        set_or_create_attr(netfile, "creation_date", datetime.now().isoformat())
        
        # AERONET site info
        set_or_create_attr(netfile, "aeronet_latitude", matchup_data.get("aeronet_latitude"))
        set_or_create_attr(netfile, "aeronet_longitude", matchup_data.get("aeronet_longitude"))
        
        # AERONET info

        aeronet_time = matchup_data.get("aeronet_time")["values"]
        aeronet_date = matchup_data.get("aeronet_date")["values"]
        if aeronet_time and aeronet_date:
            combined = datetime.combine(aeronet_date, aeronet_time)
            set_or_create_attr(netfile, "aeronet_datetime", combined.isoformat())


        # HYPSO matchup info
        set_or_create_attr(netfile, "hypso_pixel_x", matchup_data.get("center_x"))
        set_or_create_attr(netfile, "hypso_pixel_y", matchup_data.get("center_y"))
        
        # Window info
        set_or_create_attr(netfile, "window_size", n_size)
        set_or_create_attr(netfile, "is_edge_case", matchup_data.get("is_edge_case", False))
        set_or_create_attr(netfile, "valid_pixel_count", matchup_data.get("valid_pixel_count", 0))
        set_or_create_attr(netfile, "valid_pixel_percentage", matchup_data.get("valid_pixel_percentage", 0))
        
        # Edge case details if applicable
        if matchup_data.get("is_edge_case", False):
            set_or_create_attr(netfile, "truncated_top", matchup_data.get("truncated_top", 0))
            set_or_create_attr(netfile, "truncated_bottom", matchup_data.get("truncated_bottom", 0))
            set_or_create_attr(netfile, "truncated_left", matchup_data.get("truncated_left", 0))
            set_or_create_attr(netfile, "truncated_right", matchup_data.get("truncated_right", 0))
        




        # Create dimensions
        #netfile.createDimension('window_size', n_size)
        #netfile.createDimension('bands', n_bands)

        netfile.createDimension('lines', n_size)
        netfile.createDimension('samples', n_size)
        netfile.createDimension('bands', n_bands)

        netfile.createDimension('aeronet_lines', 1)
        netfile.createDimension('aeronet_samples', 1)
        netfile.createDimension('aeronet_bands', n_aeronet_bands)


        # Create groups
        netfile.createGroup('products')
        netfile.createGroup('products/hypso')
        netfile.createGroup('products/aeronet')
        netfile.createGroup('geometry')

        # Set pseudoglobal vars like compression level
        COMP_SCHEME = 'zlib'  # Default: zlib
        COMP_LEVEL = 4  # Default (when scheme != none): 4
        COMP_SHUFFLE = True  # Default (when scheme != none): True


        # Latitude ---------------------------------
        latitude = netfile.createVariable(
            'geometry/latitude', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        latitude[:] = matchup_data.get("latitudes")
        latitude.long_name = "Latitude"
        latitude.units = "degrees"
        # latitude.valid_range = [-180, 180]
        latitude.valid_min = -180
        latitude.valid_max = 180

        # Longitude ----------------------------------
        longitude = netfile.createVariable(
            'geometry/longitude', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        longitude[:] = matchup_data.get("longitudes")
        longitude.long_name = "Longitude"
        longitude.units = "degrees"
        # longitude.valid_range = [-180, 180]
        longitude.valid_min = -180
        longitude.valid_max = 180




        # Latitude (AERONET) ---------------------------------
        latitude = netfile.createVariable(
            'geometry/aeronet_latitude', 'f4', ('aeronet_lines', 'aeronet_samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        latitude[:] = matchup_data.get("aeronet_latitude")
        latitude.long_name = "Latitude (AERONET)"
        latitude.units = "degrees"
        # latitude.valid_range = [-180, 180]
        latitude.valid_min = -180
        latitude.valid_max = 180

        # Longitude (AERONET) ----------------------------------
        longitude = netfile.createVariable(
            'geometry/aeronet_longitude', 'f4', ('aeronet_lines', 'aeronet_samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        longitude[:] = matchup_data.get("aeronet_longitude")
        longitude.long_name = "Longitude (AERONET)"
        longitude.units = "degrees"
        # longitude.valid_range = [-180, 180]
        longitude.valid_min = -180
        longitude.valid_max = 180


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


        # Create and populate variables
        if datacube:

            # Store as datacube
            product_data = netfile.createVariable(
                'products/hypso/' + product_symbol, 'f4',
                ('lines', 'samples', 'bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            product_data.units = ""
            product_data.long_name = "HYPSO product_data"
            product_data.wavelength_units = "nanometers"
            #product_data.fwhm = satobj.fwhm
            product_data.wavelengths = np.around(satobj.wavelengths, 1)
            product_data[:] = extracted_cube

        else:

            # Store as bands
            product_data_cube = extracted_cube
            for band in range(0, n_bands):

                wave = np.around(satobj.wavelengths, 1)[band]
                wave_name = str(int(wave))
                name = product_symbol + '_' + wave_name

                product_data = netfile.createVariable(
                    'products/hypso/' + name, 'f4',
                    ('lines', 'samples'),
                    compression=COMP_SCHEME,
                    complevel=COMP_LEVEL,
                    shuffle=COMP_SHUFFLE)
                
                product_data.units = ""
                product_data.long_name = "HYPSO product_data"
                product_data.wavelength_units = "nanometers"
                #product_data.fwhm = satobj.fwhm[band]
                product_data.wavelength = wave

                product_data.radiation_wavelength = float(satobj.wavelengths[band]),
                product_data.radiation_wavelength_unit = "nm"

                #product_data.f0 = None
                #product_data.width = satobj.fwhm[band]
                product_data.wave = wave
                product_data.parameter = name
                product_data.wave_name = wave_name
                product_data.band = band

                product_data.coordinates = '/geometry/longitude /geometry/latitude'
                product_data.grid_mapping = '/geometry/crs_wgs84'

                product_data[:] = product_data_cube[:,:,band]


        # Write AERONET-OC datasets
        for aeronet_var_name in ['Rrs', 'Lwn', 'Lwn_fQ', 'Rho', 'Solar_Zenith_Angle']:

            aeronet_var_values = matchup_data[aeronet_var_name]["values"]
            aeronet_var_units = matchup_data[aeronet_var_name]["metadata"]["units"]

            # Store as datacube
            aeronet_var = netfile.createVariable(
                'products/aeronet/' + aeronet_var_name, 'f4',
                ('aeronet_lines', 'aeronet_samples', 'aeronet_bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            aeronet_var.units = aeronet_var_units
            aeronet_var.long_name = "AERONET-OC " + aeronet_var_name
            aeronet_var.wavelength_units = "nanometers"
            aeronet_var.wavelengths = aeronet_wavelengths
            aeronet_var[:] = aeronet_var_values


    print(f"Successfully wrote AERONET-OC matchup to: {output_path}")
    print(f"  Window size: {n_size}x{n_size}")
    print(f"  Valid pixels: {matchup_data.get('valid_pixel_percentage', 0):.1f}%")
    
    return output_path


