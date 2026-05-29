import numpy as np
from pathlib import Path
import netCDF4 as nc
from datetime import datetime

from .utils import set_or_create_attr



def write_aeronet_oc_matchup_nc_file(satobj, extraction_result, correction: str, dst_nc: str = None, datacube: str = True):
    """
    Write AERONET-OC matchup extraction results to a NetCDF file.
    
    Parameters:
    -----------
    satobj : object
        Satellite object containing metadata
    extraction_result : dict
        The dictionary returned by aeronet_oc_extract_matchup_area
    dst_nc : str or Path, optional
        Directory to save the file. If None, uses satobj.parent_dir
    correction : str
        Atmospheric correction algorithm used (default "polymer")
    
    Returns:
    --------
    Path : Path to the created NetCDF file
    """
    
    if extraction_result is None:
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
    hypso_name = extraction_result.get("hypso_name", "unknown")
    aeronet_name = extraction_result.get("aeronet_name", "unknown")
    
    filename = f"{capture_name}_AERONET-OC_{hypso_name}_{aeronet_name}_{correction}.nc"
    output_path = dst_nc / filename
    
    # Compression settings
    COMP_SCHEME = 'zlib'
    COMP_LEVEL = 4
    COMP_SHUFFLE = True
    
    # Get extracted data
    extracted_cube = extraction_result["extracted_cube"]  # Shape: (n_size, n_size, n_bands)
    n_size = extraction_result["requested_size"]
    n_bands = extracted_cube.shape[2]
    
    # Get wavelengths if available
    wavelengths = getattr(satobj, 'wavelengths', None)
    fwhm = getattr(satobj, 'fwhm', None)
    
    # Create NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as netfile:
        
        # Set global attributes
        set_or_create_attr(netfile, "title", f"AERONET-OC Matchup for {capture_name}")
        set_or_create_attr(netfile, "hypso_capture_name", capture_name)
        set_or_create_attr(netfile, "hypso_target_name", hypso_name)
        set_or_create_attr(netfile, "aeronet_oc_site_name", aeronet_name)
        set_or_create_attr(netfile, "atmospheric_correction_algorithm", correction)
        set_or_create_attr(netfile, "processing_level", "L2_AERONET_OC_MATCHUP")
        set_or_create_attr(netfile, "creation_date", datetime.now().isoformat())
        
        # AERONET site info
        set_or_create_attr(netfile, "aeronet_latitude", extraction_result.get("aeronet_latitude"))
        set_or_create_attr(netfile, "aeronet_longitude", extraction_result.get("aeronet_longitude"))
        
        # HYPSO matchup info
        set_or_create_attr(netfile, "hypso_pixel_x", extraction_result.get("center_x"))
        set_or_create_attr(netfile, "hypso_pixel_y", extraction_result.get("center_y"))
        
        # Window info
        set_or_create_attr(netfile, "window_size", n_size)
        set_or_create_attr(netfile, "is_edge_case", extraction_result.get("is_edge_case", False))
        set_or_create_attr(netfile, "valid_pixel_count", extraction_result.get("valid_pixel_count", 0))
        set_or_create_attr(netfile, "valid_pixel_percentage", extraction_result.get("valid_pixel_percentage", 0))
        
        # Edge case details if applicable
        if extraction_result.get("is_edge_case", False):
            set_or_create_attr(netfile, "truncated_top", extraction_result.get("truncated_top", 0))
            set_or_create_attr(netfile, "truncated_bottom", extraction_result.get("truncated_bottom", 0))
            set_or_create_attr(netfile, "truncated_left", extraction_result.get("truncated_left", 0))
            set_or_create_attr(netfile, "truncated_right", extraction_result.get("truncated_right", 0))
        

        # Create dimensions
        #netfile.createDimension('window_size', n_size)
        #netfile.createDimension('bands', n_bands)

        netfile.createDimension('lines', n_size)
        netfile.createDimension('samples', n_size)
        netfile.createDimension('bands', n_bands)
        
        # Create groups
        netfile.createGroup('products')
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
        latitude[:] = extraction_result.get("latitudes")
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
        longitude[:] = extraction_result.get("longitudes")
        longitude.long_name = "Longitude"
        longitude.units = "degrees"
        # longitude.valid_range = [-180, 180]
        longitude.valid_min = -180
        longitude.valid_max = 180


        try:
            l2a_variable_name = satobj.l2a_cubes[correction].attrs['l2_variable_name']
        except Exception as ex:
            print["[WARNING] No 'l2_variable_name' attrribute found. Defaulting to 'rrs'"]
            print(ex)
            l2a_variable_name = "rrs"

        extracted_cube

        # Create and populate variables
        if datacube:

            # Store as datacube
            rrs = netfile.createVariable(
                'products/' + l2a_variable_name.lower(), 'f4',
                ('lines', 'samples', 'bands'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE)
            rrs.units = ""
            rrs.long_name = "Bottom-of-Atmosphere Reflectance"
            rrs.wavelength_units = "nanometers"
            rrs.fwhm = satobj.fwhm
            rrs.wavelengths = np.around(satobj.wavelengths, 1)
            rrs[:] = extracted_cube

        else:

            # Store as bands
            rrs_cube = extracted_cube
            for band in range(0, n_bands):

                wave = np.around(satobj.wavelengths, 1)[band]
                wave_name = str(int(wave))
                name = l2a_variable_name.lower() + '_' + wave_name

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

                rrs.coordinates = '/geometry/longitude /geometry/latitude'
                rrs.grid_mapping = '/geometry/crs_wgs84'

                rrs[:] = rrs_cube[:,:,band]

















    '''
        # Write reflectance cube
        reflectance_var = netfile.createVariable(
            'reflectance_cube', 'f4',
            ('window_size', 'window_size', 'bands'),
            compression=COMP_SCHEME,
            complevel=COMP_LEVEL,
            shuffle=COMP_SHUFFLE,
            fill_value=np.nan
        )
        reflectance_var.units = "sr^-1"
        reflectance_var.long_name = "Remote sensing reflectance (Rrs) extracted area"
        reflectance_var.coordinates = "latitude longitude"
        
        if wavelengths is not None:
            reflectance_var.wavelengths = wavelengths[:n_bands]
            reflectance_var.wavelength_units = "nanometers"
        if fwhm is not None:
            reflectance_var.fwhm = fwhm[:n_bands]
        
        reflectance_var[:] = extracted_cube
        
        # Write latitudes if available
        if extraction_result.get("latitudes") is not None:
            lat_var = netfile.createVariable(
                'latitude', 'f4',
                ('window_size', 'window_size'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE,
                fill_value=np.nan
            )
            lat_var.units = "degrees_north"
            lat_var.long_name = "Latitude of each pixel"
            lat_var[:] = extraction_result["latitudes"]
        
        # Write longitudes if available
        if extraction_result.get("longitudes") is not None:
            lon_var = netfile.createVariable(
                'longitude', 'f4',
                ('window_size', 'window_size'),
                compression=COMP_SCHEME,
                complevel=COMP_LEVEL,
                shuffle=COMP_SHUFFLE,
                fill_value=np.nan
            )
            lon_var.units = "degrees_east"
            lon_var.long_name = "Longitude of each pixel"
            lon_var[:] = extraction_result["longitudes"]
        
        # Compute and write per-band statistics
        statistics_group = netfile.createGroup('statistics')
        
        valid_mask = ~np.isnan(extracted_cube[:, :, 0])
        valid_pixels = np.sum(valid_mask)
        
        for band_idx in range(n_bands):
            band_data = extracted_cube[:, :, band_idx]
            band_valid = band_data[~np.isnan(band_data)]
            
            if len(band_valid) > 0:
                band_stats = statistics_group.createVariable(f'band_{band_idx:03d}', 'f4')
                band_stats.mean = np.mean(band_valid)
                band_stats.std = np.std(band_valid)
                band_stats.median = np.median(band_valid)
                band_stats.min = np.min(band_valid)
                band_stats.max = np.max(band_valid)
                band_stats.cv = np.std(band_valid) / np.mean(band_valid) if np.mean(band_valid) != 0 else np.nan
                band_stats.n_valid_pixels = len(band_valid)
                
                if wavelengths is not None and band_idx < len(wavelengths):
                    band_stats.wavelength = wavelengths[band_idx]
        
        # Write overall statistics
        overall_stats = statistics_group.createVariable('overall', 'f4')
        overall_stats.window_size = n_size
        overall_stats.total_pixels = n_size * n_size
        overall_stats.valid_pixels = valid_pixels
        overall_stats.valid_percentage = extraction_result.get("valid_pixel_percentage", 0)
        overall_stats.is_edge_case = 1 if extraction_result.get("is_edge_case", False) else 0
        
        # Copy relevant satellite metadata if available
        if hasattr(satobj, 'nc_attrs'):
            metadata_group = netfile.createGroup('satellite_metadata')
            for key, value in satobj.nc_attrs.items():
                if key not in ['title', 'processing_level']:
                    try:
                        set_or_create_attr(metadata_group, key, value)
                    except:
                        pass
    '''

    print(f"Successfully wrote AERONET-OC matchup to: {output_path}")
    print(f"  Window size: {n_size}x{n_size}")
    print(f"  Valid pixels: {extraction_result.get('valid_pixel_percentage', 0):.1f}%")
    
    return output_path


