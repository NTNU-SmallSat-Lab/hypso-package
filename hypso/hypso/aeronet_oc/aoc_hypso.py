"""
Helper functions for PACE Hackweek Validation Tutorial.

Authors:
    James Allen and Anna Windle
"""
import datetime
from pathlib import Path
import re

import earthaccess
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pvlib.solarposition as sunpos
from scipy import stats, odr
import seaborn as sns
import xarray as xr
import os



HYPSO_TO_AOC_LIST = [
  {
    "HYPSO_NAME": "aeronetgalata",
    "AOC_NAME": "Galata_Platform",
    "LATITUDE": 43.04462,
    "LONGITUDE": 28.19319,
    "ELEVATION": 31
  },
  {
    "HYPSO_NAME": "aeronetgloria",
    "AOC_NAME": "Section-7_Platform",
    "LATITUDE": 44.54580,
    "LONGITUDE": 29.44660,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "aeronetvenice",
    "AOC_NAME": "AAOT",
    "LATITUDE": 45.31390,
    "LONGITUDE": 12.50830,
    "ELEVATION": 10
  },
  {
    "HYPSO_NAME": "annapolis",
    "AOC_NAME": "Chesapeake_Bay",
    "LATITUDE": 39.12351,
    "LONGITUDE": -76.34890,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "gustav",
    "AOC_NAME": "Gustav_Dalen_Tower",
    "LATITUDE": 58.59417,
    "LONGITUDE": 17.46683,
    "ELEVATION": 25
  },
  {
    "HYPSO_NAME": "chesapeake",
    "AOC_NAME": "CBBT",
    "LATITUDE": 37.03667,
    "LONGITUDE": -76.07660,
    "ELEVATION": 20
  },
  {
    "HYPSO_NAME": "kemigawa",
    "AOC_NAME": "Kemigawa_Offshore",
    "LATITUDE": 35.61083,
    "LONGITUDE": 140.02333,
    "ELEVATION": 8.2
  },
  {
    "HYPSO_NAME": "lucinda",
    "AOC_NAME": "Lucinda",
    "LATITUDE": -18.51980,
    "LONGITUDE": 146.38610,
    "ELEVATION": 8
  },
  {
    "HYPSO_NAME": "moby",
    "AOC_NAME": "MOBY",
    "LATITUDE": 20.8266,
    "LONGITUDE": -157.2015,
    "ELEVATION": 1.2
  },
  {
    "HYPSO_NAME": "mvco",
    "AOC_NAME": "MVCO",
    "LATITUDE": 41.32500,
    "LONGITUDE": -70.56670,
    "ELEVATION": 10
  },
  {
    "HYPSO_NAME": "ngomeni",
    "AOC_NAME": "San_Marco_Platform",
    "LATITUDE": -2.94167,
    "LONGITUDE": 40.21472,
    "ELEVATION": 20
  },
  {
    "HYPSO_NAME": "palgrunden",
    "AOC_NAME": "Palgrunden",
    "LATITUDE": 58.75533,
    "LONGITUDE": 13.15150,
    "ELEVATION": 49
  },
  {
    "HYPSO_NAME": "socheongcho",
    "AOC_NAME": "Socheongcho",
    "LATITUDE": 37.42313,
    "LONGITUDE": 124.73804,
    "ELEVATION": 28
  },
  {
    "HYPSO_NAME": "section7platform",
    "AOC_NAME": "Section-7_Platform",
    "LATITUDE": 44.54580,
    "LONGITUDE": 29.44660,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "section7platform",
    "AOC_NAME": "Gloria",
    "LATITUDE": 44.59997,
    "LONGITUDE": 29.35967,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "gloria",
    "AOC_NAME": "Section-7_Platform",
    "LATITUDE": 44.54580,
    "LONGITUDE": 29.44660,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "gloria",
    "AOC_NAME": "Gloria",
    "LATITUDE": 44.59997,
    "LONGITUDE": 29.35967,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "plocan",
    "AOC_NAME": "PLOCAN_Tower",
    "LATITUDE": 28.04112,
    "LONGITUDE": -15.38511,
    "ELEVATION": 12
  },
  {
    "HYPSO_NAME": "zeebrugge",
    "AOC_NAME": "Thornton_C-power",
    "LATITUDE": 51.53250,
    "LONGITUDE": 2.95528,
    "ELEVATION": 30
  },
  {
    "HYPSO_NAME": "laplata",
    "AOC_NAME": "RdP-EsNM",
    "LATITUDE": -34.81800,
    "LONGITUDE": -57.89590,
    "ELEVATION": 9
  },
  {
    "HYPSO_NAME": "blanca",
    "AOC_NAME": "Bahia_Blanca",
    "LATITUDE": -39.14833,
    "LONGITUDE": -61.72167,
    "ELEVATION": 15
  },
  {
    "HYPSO_NAME": "cocobeach",
    "AOC_NAME": "Banana_River",
    "LATITUDE": 28.36699,
    "LONGITUDE": -80.63328,
    "ELEVATION": 15
  },
  {
    "HYPSO_NAME": "wilmington",
    "AOC_NAME": "Frying_Pan_Tower",
    "LATITUDE": 33.48530,
    "LONGITUDE": -77.59010,
    "ELEVATION": 41
  },
  {
    "HYPSO_NAME": "longisland",
    "AOC_NAME": "LISCO",
    "LATITUDE": 40.95452,
    "LONGITUDE": -73.34177,
    "ELEVATION": 12
  },
  {
    "HYPSO_NAME": "grizzlybay",
    "AOC_NAME": "Grizzly_Bay",
    "LATITUDE": 38.10817,
    "LONGITUDE": -122.05621,
    "ELEVATION": 4
  },
  {
    "HYPSO_NAME": "ariake",
    "AOC_NAME": "ARIAKE_TOWER",
    "LATITUDE": 33.10362,
    "LONGITUDE": 130.27195,
    "ELEVATION": 15
  },
  {
    "HYPSO_NAME": "ariake",
    "AOC_NAME": "ARIAKE_TOWER_2",
    "LATITUDE": 33.11400,
    "LONGITUDE": 130.29000,
    "ELEVATION": 5
  }
]




def get_aoc_names(satobj):
    """
    Convert a HYPSO_NAME to all corresponding AOC_NAME values.
    
    Args:
        hypso_name (str): The HYPSO_NAME to look up
        
    Returns:
        list: A list of matching AOC_NAME values (empty list if none found)
    """

    hypso_name = str(satobj.capture_target).lower()

    matches = [entry["AOC_NAME"] for entry in HYPSO_TO_AOC_LIST 
               if entry["HYPSO_NAME"] == hypso_name]
    return matches


def format_capture_date(satobj):
    """
    Extract and format the capture_datetime from satobj as YYYY-MM-DD.
    
    Args:
        satobj: Object with capture_datetime attribute
        
    Returns:
        str: Date in format YYYY-MM-DD (e.g., "2024-06-01")
    """
    dt = satobj.capture_datetime
    return dt.strftime("%Y-%m-%d")



def build_aeronet_queries(satobj, data_level=15):
    """
    Generate arguments dictionary for process_aeronet function.
    
    Args:
        satobj: Object with capture_target and capture_datetime attributes
        
    Returns:
        dict: Arguments for process_aeronet including aoc_site, start_date, 
              end_date, and data_level
    """
    # Get AOC names (take first one if multiple, or None if none)
    aoc_names = get_aoc_names(satobj)

    # Get formatted date
    date_str = format_capture_date(satobj)

    aoc_queries = []

    for aoc_name in aoc_names:
        aoc_site = aoc_name
    
        # Create args dict
        aoc_query = {
        "aoc_site": aoc_site,
        "start_date": date_str,
        "end_date": date_str,
        "data_level": data_level
        }

        aoc_queries.append(aoc_query)
    
    return aoc_queries


def plot_aoi_rgb(datacube, satobj, center_y, center_x, box_size=5, rgb_bands=None):
    """
    Plot RGB image of the AOI from HYPSO datacube.
    
    Parameters
    ----------
    datacube : xarray.DataArray
        The hyperspectral datacube
    satobj : HYPSO object
        HYPSO captures containing wavelengths
    center_y, center_x : int
        Center coordinates of the AOI
    box_size : int
        Size of the box around center (default 5 for 5x5)
    rgb_bands : dict or None
        Dictionary with 'red', 'green', 'blue' wavelengths (nm)
        If None, uses default bands: red=645, green=555, blue=465
    """
    
    # Default RGB bands (typical true color)
    if rgb_bands is None:
        rgb_bands = {'red': 645, 'green': 555, 'blue': 465}
    
    # Get wavelengths
    wavelengths = satobj.wavelengths
    
    # Find closest band indices for RGB
    red_idx = np.argmin(np.abs(wavelengths - rgb_bands['red']))
    green_idx = np.argmin(np.abs(wavelengths - rgb_bands['green']))
    blue_idx = np.argmin(np.abs(wavelengths - rgb_bands['blue']))
    
    print(f"[INFO] RGB bands used:")
    print(f"  Red: {wavelengths[red_idx]:.1f}nm (target: {rgb_bands['red']}nm)")
    print(f"  Green: {wavelengths[green_idx]:.1f}nm (target: {rgb_bands['green']}nm)")
    print(f"  Blue: {wavelengths[blue_idx]:.1f}nm (target: {rgb_bands['blue']}nm)")
    
    # Extract RGB bands from the full datacube or AOI
    if hasattr(datacube, 'isel'):
        # For xarray DataArray
        red_band = datacube.isel(band=red_idx).values
        green_band = datacube.isel(band=green_idx).values
        blue_band = datacube.isel(band=blue_idx).values
    else:
        # For numpy array (height, width, bands)
        red_band = datacube[:, :, red_idx]
        green_band = datacube[:, :, green_idx]
        blue_band = datacube[:, :, blue_idx]

    def normalize_band(band):
        vmin, vmax = np.percentile(band[~np.isnan(band)], (2, 98))
        normalized = (band - vmin) / (vmax - vmin)
        return np.clip(normalized, 0, 1)
        
    red_norm = normalize_band(red_band)
    green_norm = normalize_band(green_band)
    blue_norm = normalize_band(blue_band)
    
    # Stack into RGB image
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    # Calculate box boundaries
    half_box = box_size // 2
    y_start = max(center_y - half_box, 0)
    y_end = min(center_y + half_box + 1, red_band.shape[0])
    x_start = max(center_x - half_box, 0)
    x_end = min(center_x + half_box + 1, red_band.shape[1])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Full scene with AOI box
    ax1.imshow(rgb_image)
    
    # Draw rectangle around AOI
    rect = plt.Rectangle((x_start - 0.5, y_start - 0.5), 
                         x_end - x_start, y_end - y_start,
                         linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    
    # Mark center pixel
    ax1.plot(center_x, center_y, 'r+', markersize=10, linewidth=2, label='Center pixel')
    
    ax1.set_title(f'Full Scene with {box_size}x{box_size} AOI Box\nCenter: ({center_y}, {center_x})')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    ax1.legend()
    
    # Plot 2: Zoomed AOI
    aoi_rgb = rgb_image[y_start:y_end, x_start:x_end, :]
    ax2.imshow(aoi_rgb)
    
    # Mark center pixel in zoomed view
    center_in_aoi_y = center_y - y_start
    center_in_aoi_x = center_x - x_start
    ax2.plot(center_in_aoi_x, center_in_aoi_y, 'r+', markersize=10, linewidth=2)
    
    ax2.set_title(f'Area of Interest (AOI) - {box_size}x{box_size} pixels')
    ax2.set_xlabel('Pixel X (local)')
    ax2.set_ylabel('Pixel Y (local)')
    
    # Add grid to AOI
    ax2.set_xticks(range(aoi_rgb.shape[1]))
    ax2.set_yticks(range(aoi_rgb.shape[0]))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()


    save = True
    # Save the plot if requested
    if save and hasattr(satobj, 'capture_dir') and satobj.capture_dir:
        # Create filename with timestamp and AOI info
        filename = f"aoi_rgb_{box_size}x{box_size}_center_{center_y}_{center_x}.png"
        save_path = os.path.join(satobj.capture_dir, filename)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] RGB AOI plot saved to: {save_path}")
    


    plt.show()
    
    return rgb_image, aoi_rgb





def process_hypso(satobj, latitude, longitude, atmospheric_correction="polymer"):
    """
    Download and process HYPSO data for matchups.


    Workflow:
        1. Load HYPSO capture
        2. Find closest pixel to station, extract 5x5 pixel box
            2a. Exclude pixels based on l2_flags (future)
        3. Filtered mean to get single spectra
        4. Compute statistics and save data row
        5. Organize output pandas dataframe

    Parameters
    ----------
    satobj : HYPSO object
        HYPSO captures.
    latitude : float
        In decimal degrees for Aeronet-OC site for matchups
    longitude : float
        In decimal degrees (negative West) for Aeronet-OC site for matchups

    Returns
    -------
    pandas DataFrame object
        Flattened table of HYPSO capture matchup.

    """



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
        print(f"[INFO] Detected cube shape: (height={height}, width={width}, bands={n_bands})")
    else:
        print(f"[ERROR] Unexpected datacube shape: {datacube.shape}")
        return None



    hypso_datetime = satobj.capture_datetime
    print(f"Running Capture: {hypso_datetime}")

    if hasattr(hypso_datetime, 'tzinfo') and hypso_datetime.tzinfo is not None:
        hypso_datetime = hypso_datetime.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    else:
        # Assume it's already UTC if naive
        hypso_datetime = hypso_datetime

    """
    hypso_latitudes = satobj.latitudes
    hypso_longitudes = satobj.longitudes    

    # Find x and y coord of AERONET-OC site based on provided lat/lon
    capture_shape = datacube.shape[0:2]

    min_error = np.inf
    for i in range(capture_shape[0]):
        for j in range(capture_shape[1]):
            error = np.abs(hypso_latitudes[i, j] - latitude) + np.abs(hypso_longitudes[i, j] - longitude)
            if error < min_error:
                min_error = error
                y_point = i
                x_point = j

    print(f"[INFO] AERONET-OC Geographic Coordinates (lat,lon): ({latitude}, {longitude})")
    print(f"[INFO] HYPSO Closest Geographic Coordinates (lat,lon): ({hypso_latitudes[y_point, x_point]}, {hypso_longitudes[y_point, x_point]})")
    print(f"[INFO] HYPSO Coordinates (y,x): ({y_point}, {x_point})")

    hypso_x_point = x_point
    hypso_y_point = y_point
    hypso_latitude = hypso_latitudes[y_point, x_point]
    hypso_y_longitudes = hypso_longitudes[y_point, x_point]
    """

    hypso_wavelengths = satobj.wavelengths



    sat_lat = satobj.latitudes
    sat_lon = satobj.longitudes

    # Calculate the Euclidean distance for 2D lat/lon arrays
    distances = np.sqrt((sat_lat - latitude)**2 + (sat_lon - longitude)**2)

    # Find the index of the minimum distance
    # Dimensions are (lines, pixels)
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    center_y, center_x = min_dist_idx

    # Get indices for a 5x5 box around the center pixel
    y_start = max(center_y - 2, 0)
    y_end = min(center_y + 2 + 1, sat_lat.shape[0])
    x_start = max(center_x - 2, 0)
    x_end = min(center_x + 2 + 1, sat_lat.shape[1])

    print(f"[INFO] AERONET-OC Geographic Coordinates (lat,lon): ({latitude}, {longitude})")
    print(f"[INFO] HYPSO Closest Geographic Coordinates (lat,lon): ({sat_lat[center_y, center_x]}, {sat_lon[center_y, center_x]})")
    print(f"[INFO] HYPSO Coordinates (y,x): ({center_y}, {center_x})")


    data = datacube.isel(
        y=slice(y_start, y_end),
        x=slice(x_start, x_end)
    )


    data_values = data.values


    # Get stats - calculate mean and std across spatial dimensions (y, x)
    data_mean = np.mean(data_values, axis=(0, 1))  # Shape: (n_bands,)
    data_std = np.std(data_values, axis=(0, 1))    # Shape: (n_bands,)

    # Matchup criteria uses cv as median of 405-570nm
    data_cv = data_std / data_mean
    cv_mask = (hypso_wavelengths >= 405) & (hypso_wavelengths <= 570)
    data_cv = np.median(data_cv[cv_mask])

    # Put in dictionary of the row
    row = {
        "hypso_datetime": hypso_datetime,
        "hypso_cv": data_cv,
        "hypso_latitude": float(sat_lat[center_y, center_x]),
        "hypso_longitude": float(sat_lon[center_y, center_x]),
        "hypso_pixel_valid": data_values.shape[0] * data_values.shape[1],  # 5x5 = 25
        "hypso_box_size": f"{data_values.shape[0]}x{data_values.shape[1]}"
    }

    # Add mean spectra to the row dictionary
    for wavelength, mean_value in zip(hypso_wavelengths, data_mean):
        key = f'hypso_{hypso_product_symbol.lower()}{int(wavelength)}'
        row[key] = float(mean_value)


    #for v in data_mean:
    #    print(v)

    print(f"[INFO] Processed box: {data_values.shape[0]}x{data_values.shape[1]}")
    print(f"[INFO] CV median (405-570nm): {data_cv:.4f}")


    plot_aoi_rgb(datacube, satobj, center_y, center_x, box_size=5, rgb_bands=None)


    return pd.DataFrame(row, index=[0])

















def match_hypso_data(df_hypso, df_aoc, cv_max=0.5, senz_max=60.0,
               min_percent_valid=55.0, max_time_diff=180, std_max=1.5):
    """Create matchup dataframe based on selection criteria.

    Parameters
    ----------
    df_hypso : pandas dataframe
        HYPSO data from flat validation file.
    df_aoc : pandas dataframe
        Field data from flat validation file.
    cv_max : float, default 0.15
        Maximum coefficient of variation (stdev/mean) for sat data.
    senz_max : float, default 60.0
        Maximum sensor zenith for sat data.
    min_percent_valid : float, default 55.0
        Minimum percentage of valid satellite pixels.
    max_time_diff : int, default 180
        Maximum time difference (minutes) between sat and field matchup.
    std_max : float, default 1.5
        If multiple valid field matchups, select within std_max stdevs of mean.

    Returns
    -------
    pandas dataframe of matchups for product
    """
    # Setup
    time_window = pd.Timedelta(minutes=max_time_diff)
    df_match_list = []



    # Make copies to avoid modifying originals
    #df_aoc = df_aoc.copy()
    #df_hypso = df_hypso.copy()
    

    # Filter Field data based on Solar Zenith
    df_aoc_filtered = df_aoc[df_aoc['aoc_solar_zenith'] <= senz_max]

    # Filter satellite data based on cv threshold
    df_hypso_filtered = df_hypso[df_hypso['hypso_cv'] <= cv_max]

    # Filter satellite data based on percent good pixels
    df_hypso_filtered = df_hypso_filtered[
        df_hypso_filtered['hypso_pixel_valid'] >= min_percent_valid * 25 / 100]

    for _, hypso_row in df_hypso_filtered.iterrows():
        # Filter field data based on time difference and coordinates
        hypso_datetime_aware = pd.to_datetime(hypso_row['hypso_datetime']).tz_localize('UTC')
        time_diff = abs(df_aoc_filtered['aoc_datetime']-hypso_datetime_aware)
        within_time = time_diff <= time_window
        within_lat = 0.2 >= abs(
            df_aoc_filtered['aoc_latitude'] - hypso_row['hypso_latitude'])
        within_lon = 0.2 >= abs(
            df_aoc_filtered['aoc_longitude'] - hypso_row['hypso_longitude'])
        field_matches = df_aoc_filtered[within_time & within_lat & within_lon]

        if field_matches.shape[0] > 5:
            # Filter by Standard Deviation for rrs columns
            rrs_cols = [col for col in field_matches.columns
                        if col.startswith('aoc_rrs')]
            if rrs_cols:
                mean_spectra = field_matches[rrs_cols].mean(axis=0)
                std_spectra = field_matches[rrs_cols].std(axis=0)
                within_std = (abs(field_matches[rrs_cols] - mean_spectra)
                              <= std_max * std_spectra)
                field_matches = field_matches[within_std.all(axis=1)]

        if not field_matches.empty:
            # Select the best match based on time delta
            time_diff = abs(
                field_matches['aoc_datetime']-hypso_datetime_aware)
            best_match = field_matches.loc[time_diff.idxmin()]
            df_match_list.append({**best_match.to_dict(), **hypso_row.to_dict()})

    df_match = pd.DataFrame(df_match_list)
    return df_match