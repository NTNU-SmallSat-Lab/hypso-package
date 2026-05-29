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

from .utils import download_aeronet_oc_data, read_aeronet_oc_data, parse_aeronet_oc_products


def aeronet_oc_detect_matchup(satobj, aeronet_oc_sites_csv_path, ac_algorithm="polymer"):

    # Load the CSV file
    df = pd.read_csv(aeronet_oc_sites_csv_path)  # Replace with your actual file path
    print(df.columns.tolist())

    capture_target = str(satobj.capture_target).lower()

    print("Searching AERONET-OC site matchup for " + capture_target + " HYPSO target...")

    matching_rows = df[df['HYPSO_NAME'] == capture_target]

    if not matching_rows.empty:
        # Get the first matching row
        row = matching_rows.iloc[0]

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

        capture_shape = satobj.l2a_cube[ac_algorithm].shape[0:2]

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
        
        return matchup

    else:
        print("No AERONET-OC site matchup detected for " + capture_target)
        return None







def aeronet_oc_download_data(satobj, matchup, aeronet_oc_data_dir, data_level=1.0):

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


def aeronet_oc_read_data(aeronet_oc_file):

    return read_aeronet_oc_data(aeronet_oc_file)


def aeronet_oc_read_matchup_data(satobj, aeronet_oc_data_file, df_time_column="Time(hh:mm:ss)"):

    df = aeronet_oc_read_data(aeronet_oc_data_file)

    dt = satobj.capture_datetime

    # Extract time from datetime
    target_time = dt.time()
    
    # Parse AERONET time strings to datetime.time objects
    df['parsed_time'] = pd.to_datetime(df[df_time_column], format='%H:%M:%S').dt.time
    
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


def aeronet_oc_parse_products(satobj, df_series):

    products = parse_aeronet_oc_products(df_series)

    return products



def aeronet_oc_load_matchup_products(satobj, aeronet_oc_data_file):
    df_series = aeronet_oc_read_matchup_data(satobj, aeronet_oc_data_file)
    products = aeronet_oc_parse_products(satobj, df_series)

    return products

#def aeronet_oc_get_matchup_data(satobj, aeronet_oc_site)




def aeronet_oc_load_ssi():

    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"))
    ds = np.load(solar_data_path)

    solar_wavelengths = ds["solar_x"] 
    ssi = ds["solar_y"] * 1000 # convert to milliwatts

    #return ssi, solar_wavelengths

    f0 = {"wave":np.asarray(solar_wavelengths), "data":np.asarray(ssi)}

    return f0



def aeronet_oc_extract_matchup_area(matchup, satobj, ac_algorithm="polymer", n_size=5):
    """
    Extract an NxN area from the datacube centered at the matchup point.
    
    Parameters:
    -----------
    matchup : dict
        The matchup dictionary returned by aeronet_oc_detect_matchup
    satobj : object
        Satellite object containing l2a_cube
    ac_algorithm : str
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
        print(f"Error: n_size must be >= 1, got {n_size}")
        return None
    
    if n_size % 2 == 0:
        print(f"Warning: n_size={n_size} is even. Using n_size={n_size+1} for proper centering.")
        n_size = n_size + 1
    
    # Get center coordinates
    center_x = matchup.get("hypso_x_point")
    center_y = matchup.get("hypso_y_point")
    
    if center_x is None or center_y is None:
        print("Error: Missing center coordinates in matchup dictionary")
        return None
    
    # Get the datacube
    if not hasattr(satobj, 'l2a_cube'):
        print("Error: satobj has no l2a_cube attribute")
        return None
    
    datacube = satobj.l2a_cube.get(ac_algorithm)
    if datacube is None:
        print(f"Error: {ac_algorithm} not found in l2a_cube")
        return None
    
    # Get cube shape (bands, height, width)
    if len(datacube.shape) == 3:
        n_bands, height, width = datacube.shape
    else:
        print(f"Error: Unexpected datacube shape: {datacube.shape}")
        return None
    
    # Validate center coordinates are within bounds
    if center_y < 0 or center_y >= height or center_x < 0 or center_x >= width:
        print(f"Error: Center coordinates ({center_y}, {center_x}) out of bounds for cube of size ({height}, {width})")
        return None
    
    # Calculate window boundaries
    half_size = n_size // 2
    y_start = center_y - half_size
    y_end = center_y + half_size + 1
    x_start = center_x - half_size
    x_end = center_x + half_size + 1
    
    # Track edge adjustments
    y_start_original = y_start
    y_end_original = y_end
    x_start_original = x_start
    x_end_original = x_end
    
    # Handle edge cases (clip to valid ranges)
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    
    # Check if window is completely outside the cube
    if y_start >= height or y_end <= 0 or x_start >= width or x_end <= 0:
        print(f"Error: Window completely outside cube boundaries")
        return None
    
    # Calculate actual window size
    actual_size_y = y_end - y_start
    actual_size_x = x_end - x_start
    
    # Calculate how much we truncated
    truncate_top = max(0, -y_start_original)
    truncate_bottom = max(0, y_end_original - height)
    truncate_left = max(0, -x_start_original)
    truncate_right = max(0, x_end_original - width)
    
    # Check if we have a valid window (at least 1x1)
    if actual_size_y < 1 or actual_size_x < 1:
        print(f"Error: Invalid window size: {actual_size_y}x{actual_size_x}")
        return None
    
    # Extract the area
    extracted_area = datacube[:, y_start:y_end, x_start:x_end]
    
    # Extract corresponding latitudes and longitudes if available
    latitudes_area = None
    longitudes_area = None
    
    if hasattr(satobj, 'latitudes') and satobj.latitudes is not None:
        if satobj.latitudes.shape == (height, width):
            latitudes_area = satobj.latitudes[y_start:y_end, x_start:x_end]
        else:
            print(f"Warning: latitudes shape {satobj.latitudes.shape} doesn't match cube shape ({height}, {width})")
    
    if hasattr(satobj, 'longitudes') and satobj.longitudes is not None:
        if satobj.longitudes.shape == (height, width):
            longitudes_area = satobj.longitudes[y_start:y_end, x_start:x_end]
        else:
            print(f"Warning: longitudes shape {satobj.longitudes.shape} doesn't match cube shape ({height}, {width})")
    
    # Check if extraction was successful (not empty)
    if extracted_area.size == 0:
        print("Error: Extracted area is empty")
        return None
    
    # Print edge case information
    if actual_size_y != n_size or actual_size_x != n_size:
        print(f"Edge case detected: Window truncated")
        print(f"  Requested: {n_size}x{n_size} centered at ({center_y}, {center_x})")
        print(f"  Actual: {actual_size_y}x{actual_size_x}")
        if truncate_top > 0:
            print(f"  Truncated {truncate_top} row(s) from top")
        if truncate_bottom > 0:
            print(f"  Truncated {truncate_bottom} row(s) from bottom")
        if truncate_left > 0:
            print(f"  Truncated {truncate_left} column(s) from left")
        if truncate_right > 0:
            print(f"  Truncated {truncate_right} column(s) from right")
    
    # Create result dictionary
    result = {
        "extracted_cube": extracted_area,
        "center_x": center_x,
        "center_y": center_y,
        "requested_size": n_size,
        "actual_size_y": actual_size_y,
        "actual_size_x": actual_size_x,
        "y_start": y_start,
        "y_end": y_end,
        "x_start": x_start,
        "x_end": x_end,
        "truncated_top": truncate_top,
        "truncated_bottom": truncate_bottom,
        "truncated_left": truncate_left,
        "truncated_right": truncate_right,
        "is_edge_case": (actual_size_y != n_size or actual_size_x != n_size),
        "latitudes": latitudes_area,
        "longitudes": longitudes_area,
        "ac_algorithm": ac_algorithm,
        "hypso_name": matchup.get("hypso_name"),
        "aeronet_name": matchup.get("aeronet_name"),
        "aeronet_latitude": matchup.get("aeronet_latitude"),
        "aeronet_longitude": matchup.get("aeronet_longitude")
    }
    
    print(f"Successfully extracted {actual_size_y}x{actual_size_x} area")
    print(f"Extracted cube shape: {extracted_area.shape}")
    
    return result