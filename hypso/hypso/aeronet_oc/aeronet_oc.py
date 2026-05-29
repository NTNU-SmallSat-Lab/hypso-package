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


def aeronet_oc_load_ssi():

    solar_data_path = str(files('hypso.reflectance').joinpath("hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.npz"))
    ds = np.load(solar_data_path)

    solar_wavelengths = ds["solar_x"] 
    ssi = ds["solar_y"] * 1000 # convert to milliwatts

    #return ssi, solar_wavelengths

    f0 = {"wave":np.asarray(solar_wavelengths), "data":np.asarray(ssi)}

    return f0



def aeronet_oc_extract_matchup_area(satobj, matchup, ac_algorithm="polymer", n_size=5):
    """
    Extract an NxN area from the datacube centered at the matchup point.
    Out-of-bounds indices are filled with NaN to always return exactly NxN.
    
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
    center_y = matchup.get("hypso_y_point")
    center_x = matchup.get("hypso_x_point")
    
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
    
    if len(datacube.shape) == 3:
        # (height, width, bands)
        height, width, n_bands = datacube.shape
        print(f"Detected cube shape: (height={height}, width={width}, bands={n_bands})")
    else:
        print(f"Error: Unexpected datacube shape: {datacube.shape}")
        return None
    
    # Validate center coordinates are within bounds
    if center_y < 0 or center_y >= height or center_x < 0 or center_x >= width:
        print(f"Error: Center coordinates (x={center_x}, y={center_y}) out of bounds for cube of size (width={width}, height={height})")
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
        "ac_algorithm": ac_algorithm,
        "hypso_name": matchup.get("hypso_name"),
        "aeronet_name": matchup.get("aeronet_name"),
        "aeronet_latitude": matchup.get("aeronet_latitude"),
        "aeronet_longitude": matchup.get("aeronet_longitude"),
        "hypso_latitude": matchup.get("hypso_latitude"),
        "hypso_longitude": matchup.get("hypso_y_longitudes")
    }
    
    print(f"Successfully extracted {n_size}x{n_size} area (shape: {extracted_area.shape})")
    print(f"  Valid pixels (any band): {valid_pixel_count_any}/{n_size*n_size} ({valid_pixel_percentage_any:.1f}%)")
    print(f"  Valid pixels (all bands): {valid_pixel_count_all}/{n_size*n_size} ({valid_pixel_percentage_all:.1f}%)")
    
    return result