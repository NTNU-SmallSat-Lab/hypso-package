import numpy as np
from datetime import datetime, timedelta, timezone
from shapely.geometry import Polygon
import earthaccess
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import xarray as xr

def extract_footprint_and_date(satobj):

    try:
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
    except:
        latitudes = satobj.latitudes_direct
        longitudes = satobj.longitudes_direct


    # Extract edge pixels to make a precise footprint polygon
    edge_lons = np.concatenate([
        longitudes[0, :],        # top
        longitudes[:, -1],       # right
        longitudes[-1, ::-1],    # bottom reversed
        longitudes[::-1, 0]      # left reversed
    ])
    edge_lats = np.concatenate([
        latitudes[0, :],
        latitudes[:, -1],
        latitudes[-1, ::-1],
        latitudes[::-1, 0]
    ])

    footprint_precise = Polygon(zip(edge_lons, edge_lats))

    # Simple bounding rectangle (CCW)
    min_lon, min_lat, max_lon, max_lat = footprint_precise.bounds
    simple_polygon_ccw = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ]

    # Extract date from file name using regex
    file_date = datetime.fromtimestamp(satobj.unixtime, tz=timezone.utc)
    temporal_range = (file_date - timedelta(hours=12), file_date + timedelta(hours=12))


    return footprint_precise, simple_polygon_ccw, temporal_range



def download_viirs_aot(footprint_polygon, temporal_range, local_path='data_aerosol'):
    """
    Search and download VIIRS AOT 550 nm granules for given footprint and temporal range.

    Parameters:
        footprint_polygon : shapely.geometry.Polygon
            The area of interest.
        temporal_range : tuple(date, date)
            Start and end date for search.
        local_path : str
            Folder to save downloaded granules.

    Returns:
        files : list of str
            Paths to downloaded granules.
    """
    # Log in (only once per session)
    earthaccess.login()

    print(earthaccess.status())
    
    # Satellites to query
    short_names = ['AERDB_L2_VIIRS_SNPP', 'AERDB_L2_VIIRS_NOAA20']
    
    # Extract bounding box from footprint
    min_lon, min_lat, max_lon, max_lat = footprint_polygon.bounds
    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    
    all_results = []
    for sn in short_names:
        results = earthaccess.search_data(
            short_name=sn,
            temporal=temporal_range,
            bounding_box=bounding_box
        )
        all_results.extend(results)
        print(f"Found {len(results)} granules for {sn}")
    
    print(f"\nTotal granules found: {len(all_results)}")
    
    # Download the files
    try:
        files = earthaccess.download(all_results, local_path=local_path)
        print(f"Downloaded {len(files)} files to {local_path}")
    except ValueError as ex:
        print("[WARNING] No VIIRS files downloaded.")
        files = None
    
    return files



def get_aot_in_swath(files_f, footprint_precise, latitudes, longitudes, aot_var="Aerosol_Optical_Thickness_550_Land_Ocean", name="default", local_path='data_aerosol'):
    """
    Plot VIIRS AOT 550 nm data from a list of NetCDF files and filter by a swath footprint polygon.
    
    Parameters:
    - files_f: list of file paths (NetCDF files)
    - footprint_precise: shapely Polygon defining the footprint
    - bounding_box: optional [lon_min, lat_min, lon_max, lat_max] for zooming the plot
    - aot_var: string, name of the AOT variable in NetCDF
    """
    all_aot, all_lat, all_lon = [], [], []

    # Read all files
    for f in files_f:
        #print(f)
        ds = xr.open_dataset(f, decode_timedelta=False)
        if aot_var in ds:
            print("Found AOT data.")
            aot = ds[aot_var]
            lat = ds["Latitude"]
            lon = ds["Longitude"]

            mask = aot > 0
            all_aot.append(aot.where(mask).values.flatten())
            all_lat.append(lat.where(mask).values.flatten())
            all_lon.append(lon.where(mask).values.flatten())
        ds.close()

    
    # Concatenate
    all_aot = np.concatenate(all_aot)
    all_lat = np.concatenate(all_lat)
    all_lon = np.concatenate(all_lon)

    lon_min = np.min(longitudes)
    lat_min = np.min(latitudes)

    lon_max = np.max(longitudes)
    lat_max = np.max(latitudes)


    savefig_path = Path(local_path)
    savefig_path = savefig_path.joinpath(str(name) + ".png")

    '''
    # Scatter plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': cartopy.crs.PlateCarree()})
    pcm = ax.scatter(all_lon, all_lat, c=all_aot, cmap='viridis', s=1, alpha=0.7)

    # Coastlines, borders, grid
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Polygon boundary
    lon_boundary, lat_boundary = footprint_precise.boundary.xy

    ax.plot(lon_boundary, lat_boundary, color='red', linewidth=2, label='Swath Footprint')
    #ax.set_xlim(bounding_box[0] - 5, bounding_box[2] + 5)  # lon_min, lon_max
    #ax.set_ylim(bounding_box[1] - 5, bounding_box[3] + 5)  # lat_min, lat_max
    ax.set_xlim(lon_min - 5, lon_max + 5)  # lon_min, lon_max
    ax.set_ylim(lat_min - 5, lat_max + 5)  # lat_min, lat_max

    # Colorbar
    fig.colorbar(pcm, ax=ax, label='AOT 550 nm')
    ax.set_title('VIIRS Deep Blue Aerosol Optical Thickness (550 nm) - All Granules')


    ax.legend()
    #plt.show()
    #plt.show(block=True) 
    plt.savefig(savefig_path)
    plt.close()
    '''


    poly_coords = np.array(footprint_precise.exterior.coords)
    poly_path = matplotlib.path.Path(poly_coords)
    points = np.vstack((all_lon, all_lat)).T

    # Vectorized check: True if inside footprint
    inside_mask = poly_path.contains_points(points)

    # Apply mask to AOT data
    aot_inside = all_aot[inside_mask]

    return all_aot, all_lat, all_lon, aot_inside
