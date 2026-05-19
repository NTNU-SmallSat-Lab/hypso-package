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