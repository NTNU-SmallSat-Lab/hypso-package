import os
import sys
import numpy as np
from pathlib import Path

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import re




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
    


def read_aeronet_oc_data(filepath):
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


def parse_aeronet_oc_products(closest_series):
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
        'Solar_Zenith_Angle': {},
        'Exact_Wavelengths': {}
    }
    
    # Iterate through all columns in the Series
    for col_name in closest_series.index:
        
        # Parse Lw[412], Lw[443], etc.
        if col_name.startswith('Lwn[') and col_name.endswith(']'):

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Lwn'][wavelength] = value
        
    
        # Parse Lw_f/Q[412], Lw_f/Q[443], etc.
        if col_name.startswith('Lw_f/Q[') and col_name.endswith(']'):

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Lwn_fQ'][wavelength] = value

        # Parse Rho[412], Rho[443], etc.
        if col_name.startswith('Rho[') and col_name.endswith(']'):

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Rho'][wavelength] = value

        # Parse Solar_Zenith_Angle[412], Solar_Zenith_Angle[443], etc.
        if col_name.startswith('Solar_Zenith_Angle[') and col_name.endswith(']'):

            value = closest_series[col_name]
            match = re.search(r'\[(\d+)nm\]', col_name)
            if match:
                wavelength = int(match.group(1))

            products['Solar_Zenith_Angle'][wavelength] = value


        # Parse Exact_Wavelengths(um)_412, etc.
        if col_name.startswith('Exact_Wavelengths(um)_'):

            value = closest_series[col_name]
            try:
                wavelength = int(col_name.split('_')[-1])
            except:
                break

            products['Exact_Wavelengths'][wavelength] = value


    return products