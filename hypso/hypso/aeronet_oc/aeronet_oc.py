import numpy as np

import os
import sys
import numpy as np
from pathlib import Path

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import re
from importlib.resources import files

#from .utils import download_aeronet_oc_data, read_aeronet_oc_data, parse_aeronet_oc_products






def download_aeronet_oc_data(site_name, year, month, day, data_level=1.0, output_dir='aeronet_data'):
    """
    Download Lwn data for a single site and date.
    Skips download if file already exists.
    
    Parameters:
    -----------
    site_name : str
        AERONET site name
    year, month, day : int
        Date for download
    data_level : float
        1.0 or 1.5
    output_dir : str
        Base directory for storing files
    
    Returns:
    --------
    str or None
        Path to the downloaded file if successful, None if failed
    """
    
    # Create subdirectory for this site
    site_dir = Path(output_dir) / site_name
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    data_type = 'LWN10' if data_level == 1.0 else 'LWN15'
    filename = f"{site_name}_{data_type}_{year}{month:02d}{day:02d}.csv"
    filepath = site_dir / filename
    
    # Check if file already exists
    if filepath.exists():
        print(f"File already exists: {filepath}")
        return str(filepath)
    
    # Build URL
    url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
           f"?site={site_name}&year={year}&month={month}&day={day}"
           f"&year2={year}&month2={month}&day2={day}"
           f"&{data_type}=1&AVG=10&if_no_html=1")
    
    print(url)

    print(f"Downloading: {site_name} for {year}-{month:02d}-{day:02d}")
    
    # Download
    response = requests.get(url, verify=False)
    
    if response.status_code == 200:
        # Save raw response text to file
        with open(filepath, 'w') as f:
            f.write(response.text)
        print(f"Saved to: {filepath}")
        return str(filepath)
    else:
        print(f"Error for {site_name}: HTTP {response.status_code}")
        return None
    


def aeronet_oc_read_data(filepath):
    """
    Read AERONET CSV file, skipping the first 5 metadata lines.
    Line 6 (index 5) contains the column headers.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip first 5 metadata lines, keep from line 6 onwards
    data_lines = lines[5:]  # lines[5] is the 6th line (column headers)
    data_text = ''.join(data_lines)
    
    # Read into dataframe
    df = pd.read_csv(StringIO(data_text), delimiter=',')
    
    return df




def aeronet_oc_detect_matchups(satobj, aeronet_oc_sites_csv_path, atmospheric_correction="polymer"):

    # Load the CSV file
    df = pd.read_csv(aeronet_oc_sites_csv_path)  # Replace with your actual file path
    print(df.columns.tolist())

    capture_target = str(satobj.capture_target).lower()

    print("[INFO] Searching AERONET-OC site matchup for " + capture_target + " HYPSO target...")

    matching_rows = df[df['HYPSO_NAME'] == capture_target]

    if not matching_rows.empty:

        matchups = []

        print("[INFO] Matching rows:")
        print(matching_rows)

        for idx in range(len(matching_rows)):
            try:
                print("idx")

                # Get the first matching row
                #row = matching_rows.iloc[0]
                #row = matching_row
                #print(row)

                print(idx)
                row = matching_rows.iloc[idx]
                print(row)

                hypso_name = row.HYPSO_NAME  # Note: spaces become underscores
                aeronet_name = row.AERONETOC_NAME
                aeronet_latitude = row.LATITUDE
                aeronet_longitude = row.LONGITUDE
                elevation = row.ELEVATION
                
                matchup = {
                    "hypso_name": row.HYPSO_NAME,
                    "aeronet_name": row.AERONETOC_NAME,
                    "aeronet_latitude": row.LATITUDE,
                    "aeronet_longitude": row.LONGITUDE,
                    "elevation": row.ELEVATION
                }

                print("Detected AERONET-OC site match:")
                print(f"{hypso_name} - {aeronet_name}: ({aeronet_latitude}, {aeronet_longitude})")

                hypso_latitudes = satobj.latitudes
                hypso_longitudes = satobj.longitudes

                capture_shape = satobj.l2a_cube[atmospheric_correction].shape[0:2]

                min_error = np.inf
                for i in range(capture_shape[0]):
                    for j in range(capture_shape[1]):
                        error = np.abs(hypso_latitudes[i, j] - aeronet_latitude) + np.abs(hypso_longitudes[i, j] - aeronet_longitude)
                        if error < min_error:
                            min_error = error
                            y_point = i
                            x_point = j

                print(f"HYPSO Closest Geographic Coordinates (lat,lon): ({hypso_latitudes[y_point, x_point]}, {hypso_longitudes[y_point, x_point]})")
                print(f"HYPSO Coordinates (y,x): ({y_point}, {x_point})")

                matchup["hypso_x_point"] = x_point
                matchup["hypso_y_point"] = y_point
                matchup["hypso_latitude"] = hypso_latitudes[y_point, x_point]
                matchup["hypso_y_longitudes"] = hypso_longitudes[y_point, x_point]
            
                matchups.append(matchup)

            except Exception as ex:
                print(ex)
                print("[INFO] Error parsing AERONET-OC site information:")
                print(df)
                continue


        return matchups

    else:
        print("No AERONET-OC site matchup detected for " + capture_target)
        return None







def aeronet_oc_download_data(satobj, matchup, aeronet_oc_data_dir, data_level=1.0):

    if matchup is None:
        print("[WARNING] No matchup detected!")
        return None


    dt = satobj.capture_datetime

    output_file = f"aeronet_{dt.year}{dt.month:02d}{dt.day:02d}.csv"

    site_name = matchup["aeronet_name"]


    aeronet_oc_data_file = download_aeronet_oc_data(
                            site_name=site_name,  
                            year=dt.year,
                            month=dt.month,
                            day=dt.day,
                            data_level=data_level,
                            output_dir=aeronet_oc_data_dir
                    )

    return aeronet_oc_data_file





def aeronet_oc_get_closest_matchup_data(satobj, aeronet_oc_data_file, df_time_column="Time(hh:mm:ss)", df_date_column="Date(dd-mm-yyyy)"):

    try:
        df = aeronet_oc_read_data(aeronet_oc_data_file)
    except pd.errors.EmptyDataError as ex:
        print(ex)
        print(f"[WARNING] Matchup data in {aeronet_oc_data_file} is empty.")
        return None

    dt = satobj.capture_datetime

    # Extract time from datetime
    target_time = dt.time()
    
    # Parse AERONET time strings to datetime.time objects
    df['parsed_time'] = pd.to_datetime(df[df_time_column], format='%H:%M:%S').dt.time
    df['parsed_date'] = pd.to_datetime(df[df_date_column], format='%d:%m:%Y')
    
    
    # Calculate time difference (convert to minutes for easier comparison)
    def time_diff(time_obj):
        # Convert time to minutes since midnight
        target_minutes = target_time.hour * 60 + target_time.minute + target_time.second / 60
        row_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
        
        # Circular time difference (handles wrap around midnight)
        diff = abs(row_minutes - target_minutes)
        diff = min(diff, 1440 - diff)  # 1440 minutes in a day
        return diff
    
    df['time_diff_minutes'] = df['parsed_time'].apply(time_diff)
    
    # Find row with minimum time difference
    closest_idx = df['time_diff_minutes'].idxmin()
    closest_row = df.loc[closest_idx]
    
    print(f"Target time: {target_time}")
    print(f"Closest AERONET time: {closest_row[df_time_column]}")
    print(f"Difference: {closest_row['time_diff_minutes']:.2f} minutes")

    df_series = closest_row

    return df_series

'''
def aeronet_oc_matchup_aeronet_data(satobj, matchup_aeronet_data):
    """
    Extract and organize all AERONET-OC products from a pandas Series (row).
    
    Parameters:
    -----------
    closest_series : pandas Series
        A single row from AERONET-OC DataFrame (result of .iloc[] or .loc[])
    
    Returns:
    --------
    dict : Dictionary containing categorized products by type and wavelength
    """
    
    products = {
        'Lw': {},
        'Lt': {},
        'Lwn': {},
        'Lwn_fQ': {},
        'Rho': {},
        'aeronet_solar_zenith_angle': {},
        'aeronet_wavelengths': {},
        'aeronet_time': {},
        'aeronet_date': {},
    }


    products['aeronet_time'] = matchup_aeronet_data['parsed_time']
    products['aeronet_date'] = matchup_aeronet_data['parsed_date']

    # Iterate through all columns in the Series
    for col_name in matchup_aeronet_data.index:
        
        # Parse Lw[412], Lw[443], etc.
        if col_name.startswith('Lwn[') and col_name.endswith(']'):

            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Lwn'][wavelength] = value
        
    
        # Parse Lw_f/Q[412], Lw_f/Q[443], etc.
        if col_name.startswith('Lw_f/Q[') and col_name.endswith(']'):

            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Lwn_fQ'][wavelength] = value

        # Parse Rho[412], Rho[443], etc.
        if col_name.startswith('Rho[') and col_name.endswith(']'):

            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Rho'][wavelength] = value

        # Parse Solar_Zenith_Angle[412], Solar_Zenith_Angle[443], etc.
        if col_name.startswith('Solar_Zenith_Angle[') and col_name.endswith(']'):

            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['aeronet_solar_zenith_angle'][wavelength] = value


        # Parse Exact_Wavelengths(um)_412, etc.
        if col_name.startswith('Exact_Wavelengths(um)_'):

            value = matchup_aeronet_data[col_name]
            try:
                wavelength = int(col_name.split('_')[-1])
            except:
                break

            products['aeronet_wavelengths'][wavelength] = value


    return products
'''





def aeronet_oc_matchup_aeronet_data(satobj, matchup_aeronet_data):
    """
    Extract and organize all AERONET-OC products from a pandas Series (row).
    
    Parameters:
    -----------
    satobj : object
        Satellite object containing matching information
    matchup_aeronet_data : pandas Series
        A single row from AERONET-OC DataFrame (result of .iloc[] or .loc[])
    
    Returns:
    --------
    dict : Dictionary containing categorized products by type and wavelength,
           with metadata including units and descriptions
    """
    
    # Define metadata for each product type
    metadata = {
        'Lw': {
            'description': 'Water-leaving radiance',
            'units': 'mW cm^-2 μm^-1 sr^-1',
            'long_name': 'Water-leaving radiance'
        },
        'Lt': {
            'description': 'Total radiance',
            'units': 'mW cm^-2 μm^-1 sr^-1',
            'long_name': 'Total radiance'
        },
        'Lwn': {
            'description': 'Normalized water-leaving radiance',
            'units': 'mW cm^-2 μm^-1 sr^-1',
            'long_name': 'Normalized water-leaving radiance'
        },
        'Lwn_fQ': {
            'description': 'Normalized water-leaving radiance with f/Q correction',
            'units': 'mW cm^-2 μm^-1 sr^-1',
            'long_name': 'Normalized water-leaving radiance (f/Q corrected)'
        },
        'Rho': {
            'description': 'Remote sensing reflectance',
            'units': 'sr^-1',
            'long_name': 'Remote sensing reflectance'
        },
        'Solar_Zenith_Angle': {
            'description': 'Solar zenith angle at time of AERONET measurement',
            'units': 'degrees',
            'long_name': 'Solar zenith angle'
        },
        'aeronet_wavelengths': {
            'description': 'Exact measurement wavelengths',
            'units': 'nanometers (nm)',
            'long_name': 'AERONET measured wavelengths'
        },
        'aeronet_time': {
            'description': 'Time of AERONET measurement',
            'units': 'UTC',
            'long_name': 'Measurement time'
        },
        'aeronet_date': {
            'description': 'Date of AERONET measurement',
            'units': 'YYYY-MM-DD',
            'long_name': 'Measurement date'
        },
        'Rrs': {
            'description': 'Remote sensing reflectance',
            'units': 'sr^-1',
            'long_name': 'Remote sensing reflectance'
        },
    }
    
    products = {
        'Lw': {'values': {}, 'metadata': metadata['Lw']},
        'Lt': {'values': {}, 'metadata': metadata['Lt']},
        'Lwn': {'values': {}, 'metadata': metadata['Lwn']},
        'Lwn_fQ': {'values': {}, 'metadata': metadata['Lwn_fQ']},
        'Rho': {'values': {}, 'metadata': metadata['Rho']},
        'Solar_Zenith_Angle': {'values': {}, 'metadata': metadata['Solar_Zenith_Angle']},
        'aeronet_wavelengths': {'values': {}, 'metadata': metadata['aeronet_wavelengths']},
        'aeronet_time': {'values': None, 'metadata': metadata['aeronet_time']},
        'aeronet_date': {'values': None, 'metadata': metadata['aeronet_date']},
        'Rrs': {'values': None, 'metadata': metadata['Rrs']},
    }

    def convert_from_dict(product):
        """Convert wavelength dictionary to sorted arrays"""
        values = product.get("values", {})
        if not values:
            return np.array([]), []
        
        wavelengths = sorted(values.keys())
        values_list = [values[wl] for wl in wavelengths]
        values_array = np.array(values_list)
        
        return values_array, wavelengths

    products['aeronet_time']['values'] = matchup_aeronet_data['parsed_time']
    products['aeronet_date']['values'] = matchup_aeronet_data['parsed_date']

    # Iterate through all columns in the Series
    for col_name in matchup_aeronet_data.index:
        
        # Parse Lwn[412], Lwn[443], etc.
        if col_name.startswith('Lwn[') and col_name.endswith(']'):
            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))
                products['Lwn']['values'][wavelength] = value
        
        # Parse Lwn_f/Q[412], Lwn_f/Q[443], etc.
        if col_name.startswith('Lwn_f/Q[') and col_name.endswith(']'):
            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))
                products['Lwn_fQ']['values'][wavelength] = value

        # Parse Rho[412], Rho[443], etc.
        if col_name.startswith('Rho[') and col_name.endswith(']'):
            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))
                products['Rho']['values'][wavelength] = value

        # Parse Solar_Zenith_Angle[412], Solar_Zenith_Angle[443], etc.
        if col_name.startswith('Solar_Zenith_Angle[') and col_name.endswith(']'):
            value = matchup_aeronet_data[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))
                products['Solar_Zenith_Angle']['values'][wavelength] = value

        # Parse Exact_Wavelengths(um)_412, etc.
        if col_name.startswith('Exact_Wavelengths(um)_'):
            value = matchup_aeronet_data[col_name]
            try:
                wavelength = int(col_name.split('_')[-1])
                # Convert from micrometers to nanometers and store
                if value != -999:
                    value = value * 1000
                products['aeronet_wavelengths']['values'][wavelength] = value
            except (ValueError, IndexError):
                continue

    # Convert dictionaries to arrays for easier handling
    for product_key in ['Lwn', 'Lwn_fQ', 'Rho', 'Solar_Zenith_Angle', 'aeronet_wavelengths']:
        values_array, wavelengths = convert_from_dict(products[product_key])
        products[product_key]['values'] = values_array
        products[product_key]['wavelengths'] = wavelengths

    # Calculate Rrs from Lwn if needed (requires solar zenith angle)

    Lwn_wavelengths = products['aeronet_wavelengths']['values']
    Lwn = products['Lwn']['values']
    Rrs = aeronet_oc_calculate_rrs(Lwn, Lwn_wavelengths)

    products['Rrs']['values'] = Rrs

    return products


def aeronet_oc_calculate_rrs(Lwn, wavelengths):

    # Load SSI
    f0 = aeronet_oc_load_ssi()

    # Filter out indices where Lwn is -999
    valid_indices = [i for i, l in enumerate(Lwn) if l != -999]
    invalid_indicies = [i for i, l in enumerate(Lwn) if l == -999]

    wavelengths = np.array(wavelengths)
    Lwn = np.array(Lwn)

    #wavelengths = wavelengths*1000

    F0_values = []
    for wl in wavelengths:
        F0_value = np.interp(wl, f0['wave'], f0['data'])
        F0_values.append(F0_value)
    F0 = np.array(F0_values)

    # We apply a factor of 10 here since TSIS-1 F0 units are mW/(m^2 nm) and AERONET-OC Lwn units are mW/(cm^2 sr um) 
    aeronet_Rrs = 10 * Lwn/(F0)

    aeronet_Rrs[invalid_indicies] = -999

    print(aeronet_Rrs)

    return aeronet_Rrs
    






def aeronet_oc_calculate_rrs_csiro(Lwn, wavelengths):


    # Approach taken in "Evaluation of the ACOLITE atmospheric correction algorithm at a tropical coastal site" (2025)

    # Load SSI
    f0 = aeronet_oc_load_ssi()

    # Filter out indices where Lwn is -999
    valid_indices = [i for i, l in enumerate(Lwn) if l != -999]
    invalid_indicies = [i for i, l in enumerate(Lwn) if l == -999]

    wavelengths = np.array(wavelengths)
    Lwn = np.array(Lwn)

    #wavelengths = wavelengths*1000

    F0_values = []
    for wl in wavelengths:
        F0_value = np.interp(wl, f0['wave'], f0['data'])
        F0_values.append(F0_value)
    F0 = np.array(F0_values)

    # We apply a factor of 10 here since TSIS-1 F0 units are mW/(m^2 nm) and AERONET-OC Lwn units are mW/(cm^2 sr um) 
    aeronet_Rrs = 10 * Lwn/(F0)

    aeronet_Rrs[invalid_indicies] = -999

    print(aeronet_Rrs)

    return aeronet_Rrs













def aeronet_oc_generate_matchup(satobj,
                                matchup,
                                AERONET_OC_DATA_DIR,
                                #atmospheric_correction = "polymer",
                                #n_size = 5
                                ):

    aeronet_oc_data_file = aeronet_oc_download_data(satobj, 
                                                    matchup, 
                                                    AERONET_OC_DATA_DIR)


    matchup_aeronet_data = aeronet_oc_get_closest_matchup_data(satobj, aeronet_oc_data_file)

    if matchup_aeronet_data is None:
        return None

    matchup_aeronet_data = aeronet_oc_matchup_aeronet_data(satobj, matchup_aeronet_data)



    return matchup_aeronet_data




def aeronet_oc_load_ssi():

    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"))
    ds = np.load(solar_data_path)

    solar_wavelengths = ds["solar_x"] 
    ssi = ds["solar_y"] * 1000 # convert to milliwatts

    #return ssi, solar_wavelengths

    f0 = {"wave":np.asarray(solar_wavelengths), "data":np.asarray(ssi)}

    return f0



def aeronet_oc_matchup_load_hypso_data(satobj, matchup, atmospheric_correction="polymer", n_size=5):
    """
    Extract an NxN area from the datacube centered at the matchup point.
    Out-of-bounds indices are filled with NaN to always return exactly NxN.
    
    Parameters:
    -----------
    matchup : dict
        The matchup dictionary returned by aeronet_oc_detect_matchups
    satobj : object
        Satellite object containing l2a_cube
    atmospheric_correction : str
        Atmospheric correction algorithm to use (default "polymer")
    n_size : int
        Size of the square window to extract (default 5, meaning 5x5 area)
        Must be an odd number to have a true center.
        
    Returns:
    --------
    dict : Dictionary containing extracted area and metadata, or None if invalid
    """
    
    if matchup is None:
        print("No matchup provided. Cannot extract area.")
        return None
    
    # Validate n_size
    if n_size < 1:
        print(f"[ERROR] n_size must be >= 1, got {n_size}")
        return None
    
    if n_size % 2 == 0:
        print(f"Warning: n_size={n_size} is even. Using n_size={n_size+1} for proper centering.")
        n_size = n_size + 1
    
    # Get center coordinates
    center_y = matchup.get("hypso_y_point")
    center_x = matchup.get("hypso_x_point")
    
    if center_x is None or center_y is None:
        print("[ERROR] Missing center coordinates in matchup dictionary")
        return None
    

    cube_name = satobj.cube_name

    # Get the datacube
    if not hasattr(satobj, cube_name):
        print(f"[ERROR] satobj has no {cube_name} attribute!")
        return None


    hypso_product_level = satobj.product_level
    hypso_product_symbol = satobj.product_symbol

    print(f"[INFO] Detected HYPSO product level: {hypso_product_level}")

    if hypso_product_level == "l2a":

        datacube = satobj.l2a_cube.get(atmospheric_correction)
        if datacube is None:
            print(f"[ERROR] {atmospheric_correction} not found in l2a_cube")
            return None
        
    else:

        datacube = getattr(satobj, cube_name, None)
        if datacube is None:
            print(f"[ERROR] {cube_name} not found!")
            return None


    
    if len(datacube.shape) == 3:
        # (height, width, bands)
        height, width, n_bands = datacube.shape
        print(f"Detected cube shape: (height={height}, width={width}, bands={n_bands})")
    else:
        print(f"[ERROR] Unexpected datacube shape: {datacube.shape}")
        return None
    
    # Validate center coordinates are within bounds
    if center_y < 0 or center_y >= height or center_x < 0 or center_x >= width:
        print(f"[ERROR] Center coordinates (x={center_x}, y={center_y}) out of bounds for cube of size (width={width}, height={height})")
        return None
    
    # Calculate window boundaries
    half_size = n_size // 2
    y_start = center_y - half_size
    y_end = center_y + half_size + 1
    x_start = center_x - half_size
    x_end = center_x + half_size + 1
    
    # Cube is (height, width, bands) - output (n_size, n_size, bands)
    extracted_area = np.full((n_size, n_size, n_bands), np.nan, dtype=datacube.dtype)
    
    # Calculate overlap between requested window and actual cube
    src_y_start = max(0, y_start)
    src_y_end = min(height, y_end)
    src_x_start = max(0, x_start)
    src_x_end = min(width, x_end)
    
    # Destination (output) coordinates
    dst_y_start = max(0, -y_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, -x_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    
    # Check if there's any overlap
    if src_y_start < src_y_end and src_x_start < src_x_end:
        # Extract valid data from cube and place into output array
        # (height, width, bands)
        valid_data = datacube[src_y_start:src_y_end, src_x_start:src_x_end, :]
        
        # Place into output array
        extracted_area[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = valid_data
    
    # Extract corresponding latitudes and longitudes if available
    latitudes_area = None
    longitudes_area = None
    
    if hasattr(satobj, 'latitudes') and satobj.latitudes is not None:
        if satobj.latitudes.shape == (height, width):
            latitudes_area = np.full((n_size, n_size), np.nan, dtype=satobj.latitudes.dtype)
            valid_lats = satobj.latitudes[src_y_start:src_y_end, src_x_start:src_x_end]
            latitudes_area[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = valid_lats
        else:
            print(f"Warning: latitudes shape {satobj.latitudes.shape} doesn't match cube shape ({height}, {width})")
    
    if hasattr(satobj, 'longitudes') and satobj.longitudes is not None:
        if satobj.longitudes.shape == (height, width):
            longitudes_area = np.full((n_size, n_size), np.nan, dtype=satobj.longitudes.dtype)
            valid_lons = satobj.longitudes[src_y_start:src_y_end, src_x_start:src_x_end]
            longitudes_area[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = valid_lons
        else:
            print(f"Warning: longitudes shape {satobj.longitudes.shape} doesn't match cube shape ({height}, {width})")
    
    # Calculate truncation amounts (for reporting)
    truncate_top = max(0, -y_start)
    truncate_bottom = max(0, y_end - height)
    truncate_left = max(0, -x_start)
    truncate_right = max(0, x_end - width)
    
    is_edge_case = (truncate_top > 0 or truncate_bottom > 0 or truncate_left > 0 or truncate_right > 0)
    
    # Calculate valid pixel statistics across all bands
    # Method 1: Pixel is valid if ANY band has data (useful for edge cases)
    valid_mask_any = ~np.all(np.isnan(extracted_area), axis=2)  # Shape: (n_size, n_size)
    valid_pixel_count_any = np.sum(valid_mask_any)
    valid_pixel_percentage_any = 100 * valid_pixel_count_any / (n_size * n_size)
    
    # Method 2: Pixel is valid only if ALL bands have data (stricter, for complete spectra)
    valid_mask_all = ~np.any(np.isnan(extracted_area), axis=2)  # Shape: (n_size, n_size)
    valid_pixel_count_all = np.sum(valid_mask_all)
    valid_pixel_percentage_all = 100 * valid_pixel_count_all / (n_size * n_size)
    
    # Per-band valid pixel counts
    valid_pixels_per_band = np.sum(~np.isnan(extracted_area), axis=(0, 1))  # Shape: (n_bands,)
    
    # Print edge case information
    if is_edge_case:
        print(f"Edge case detected: Window extends beyond cube boundaries")
        print(f"  Requested: {n_size}x{n_size} centered at (x={center_x}, y={center_y})")
        print(f"  Out-of-bounds pixels filled with NaN")
        if truncate_top > 0:
            print(f"  {truncate_top} row(s) truncated from top (filled with NaN)")
        if truncate_bottom > 0:
            print(f"  {truncate_bottom} row(s) truncated from bottom (filled with NaN)")
        if truncate_left > 0:
            print(f"  {truncate_left} column(s) truncated from left (filled with NaN)")
        if truncate_right > 0:
            print(f"  {truncate_right} column(s) truncated from right (filled with NaN)")
        
        print(f"  Valid pixels (any band): {valid_pixel_percentage_any:.1f}% of window")
        print(f"  Valid pixels (all bands): {valid_pixel_percentage_all:.1f}% of window")
    
    # Create result dictionary
    result = {
        "extracted_cube": extracted_area,  # Shape: (n_size, n_size, n_bands)
        "center_x": center_x,
        "center_y": center_y,
        "requested_size": n_size,
        "actual_size": n_size,
        "y_start": y_start,
        "y_end": y_end,
        "x_start": x_start,
        "x_end": x_end,
        "truncated_top": truncate_top,
        "truncated_bottom": truncate_bottom,
        "truncated_left": truncate_left,
        "truncated_right": truncate_right,
        "is_edge_case": is_edge_case,
        # Valid pixel statistics (any band)
        "valid_pixel_count": valid_pixel_count_any,
        "valid_pixel_percentage": valid_pixel_percentage_any,
        # Valid pixel statistics (all bands)
        "valid_pixel_count_all_bands": valid_pixel_count_all,
        "valid_pixel_percentage_all_bands": valid_pixel_percentage_all,
        # Per-band valid pixel counts
        "valid_pixels_per_band": valid_pixels_per_band,
        "latitudes": latitudes_area,
        "longitudes": longitudes_area,
        "atmospheric_correction": atmospheric_correction,
        "hypso_name": matchup.get("hypso_name"),
        "aeronet_name": matchup.get("aeronet_name"),
        "aeronet_latitude": matchup.get("aeronet_latitude"),
        "aeronet_longitude": matchup.get("aeronet_longitude"),
        "hypso_latitude": matchup.get("hypso_latitude"),
        "hypso_longitude": matchup.get("hypso_y_longitudes"),
        "hypso_product_level": hypso_product_level,
        "hypso_product_symbol": hypso_product_symbol
    }
    
    print(f"Successfully extracted {n_size}x{n_size} area (shape: {extracted_area.shape})")
    print(f"  Valid pixels (any band): {valid_pixel_count_any}/{n_size*n_size} ({valid_pixel_percentage_any:.1f}%)")
    print(f"  Valid pixels (all bands): {valid_pixel_count_all}/{n_size*n_size} ({valid_pixel_percentage_all:.1f}%)")
    
    return result