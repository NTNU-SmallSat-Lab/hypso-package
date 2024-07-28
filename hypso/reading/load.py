import numpy as np
from datetime import datetime
import netCDF4 as nc
from hypso import georeference
from pathlib import Path
from hypso.utils import is_integer_num
from typing import Tuple

EXPERIMENTAL_FEATURES = True

def load_l1a_nc_cube(nc_file_path: Path) -> np.ndarray:
    """
    Load l1a.nc Hypso Capture file metadata

    :param nc_file_path: Absolute path to the l1a.nc file

    :return: raw cube numpy array (digital counts)
    """

    nc_cube = load_cube_from_nc_file(nc_file_path)

    return nc_cube

def load_l1a_nc_metadata(nc_file_path: Path) -> Tuple[dict, dict, dict, dict, dict]:
    """
    Load l1a.nc Hypso Capture file metadata

    :param nc_file_path: Absolute path to the l1a.nc file

    :return: "capture_config" dictionary with Hypso capture information, "timing" dictionary with Hypso timing information, "target_coords" dictionary with Hypso target coordinate information, "adcs" dictionary with Hypso ADCS information, and "dimensions" dictionary with Hypso capture spatial dimensions information
    """
    # Get metadata
    #quaternion_csv_path = Path(nc_info["tmp_dir"], "quaternion.csv")
    #position_csv_path = Path(nc_info["tmp_dir"], "position.csv")
    #timestamps_srv_path = Path(nc_info["tmp_dir"], "timestamps_services.txt")
    #frametime_pose_csv_path = Path(nc_info["tmp_dir"], "frametime-pose.csv")
    # TODO: Check if local-angles.csv is to be used or sun-azimuth.dat and etc
    #local_angles_csv_path = Path(nc_info["tmp_dir"], "local-angles.csv")
    #sat_azimuth_path = Path(nc_info["tmp_dir"], "sat-azimuth.dat")
    #sat_zenith_path = Path(nc_info["tmp_dir"], "sat-zenith.dat")
    #solar_azimuth_path = Path(nc_info["tmp_dir"], "sun-azimuth.dat")
    #solar_zenith_path = Path(nc_info["tmp_dir"], "sun-zenith.dat")
    #latitude_dataPath = Path(nc_info["tmp_dir"], "latitudes.dat")
    #longitude_dataPath = Path(nc_info["tmp_dir"], "longitudes.dat")

    nc_capture_config = load_capture_config_from_nc_file(nc_file_path)
    nc_timing = load_timing_from_nc_file(nc_file_path)
    nc_target_coords = load_target_coords_from_nc_file(nc_file_path)
    nc_adcs = load_adcs_from_nc_file(nc_file_path)
    nc_dimensions = load_dimensions_from_nc_file(nc_file_path)

    return nc_capture_config, nc_timing, nc_target_coords, nc_adcs, nc_dimensions

def load_adcs_from_nc_file(nc_file_path: Path) -> Tuple[dict, tuple]:
    """
    Get the metadata from the top folder of the data.

    :param nc_file_path: Path to the name of the tmp folder
    :param standard_dimensions: Dictionary with Hypso standard dimensions

    :return: "adcs" dictionary with ADCS metadata and a tuple with spatial dimensions
    """

    # ------------------------------------------------------------------------
    # ADCS info -----------------------------------------------------------
    # ------------------------------------------------------------------------

    adcs = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["adcs"]
        
        for key in group.variables.keys():
            
            value = group.variables[key][:]

            adcs[key] = value

        adcs['adcssamples'] = len(f.dimensions['adcssamples'])

    return adcs

# TODO
def create_tmp_dir(nc_file_path: Path) -> str:
    """
    Create empty directory for generated files.

    :param nc_file_path: Path to the name of the NetCDF file

    :return: temp_dir path
    """

    nc_name = nc_file_path.stem
    temp_dir = Path(nc_file_path.parent.absolute(), nc_name.replace("-l1a", "") + "_tmp")

    #info["tmp_dir"] = temp_dir
    #info["top_folder_name"] = info["tmp_dir"]

    temp_dir.mkdir(parents=True, exist_ok=True)

    return str(temp_dir)

def load_capture_config_from_nc_file(nc_file_path: Path) -> dict:

    # ------------------------------------------------------------------------
    # Capture info -----------------------------------------------------------
    # ------------------------------------------------------------------------

    capture_config = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["capture_config"]
        for attrname in group.ncattrs():

            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    capture_config[attrname] = int(value)
                else:
                    capture_config[attrname] = float(value)
            except BaseException:
                capture_config[attrname] = value

    return capture_config

def load_timing_from_nc_file(nc_file_path: Path) -> dict:

    timing = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["timing"]
        
        for key in group.variables.keys():
            value = group.variables[key][:]
            timing[key] = value
        timing['lines'] = len(f.dimensions['lines'])


        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    timing[attrname] = int(value)
                else:
                    timing[attrname] = float(value)
            except BaseException:
                timing[attrname] = value

    return timing

def load_target_coords_from_nc_file(nc_file_path: Path) -> dict:

    # ------------------------------------------------------------------------
    # Target Lat Lon ---------------------------------------------------------
    # ------------------------------------------------------------------------

    target_coords = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f
        try:
            # returns string 'False' if doesnt exist
            target_coords_attr = getattr(group, "target_coords")
            if target_coords_attr == 'False':
                target_lat = None
                target_lon = None
            else:
                target_coords_attr = target_coords_attr.split(" ")
                target_lat = target_coords_attr[0]
                target_lon = target_coords_attr[1]
        except AttributeError:
            target_lat = None
            target_lon = None

        target_coords["latc"] = target_lat
        target_coords["lonc"] = target_lon

    return target_coords

def load_cube_from_nc_file(nc_file_path: Path) -> np.ndarray:
    """
    Get Raw Cube from Hypso l1a.nc File

    :param nc_file_path: Absolute path to l1a.nc file

    :return: Numpy array with raw Cube (digital counts) extracted from nc file
    """
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["products"]
        # 16-bit according to Original data Capture
        cube = np.array(group.variables["Lt"][:], dtype='uint16')

        return cube

def load_dimensions_from_nc_file(nc_file_path: Path) -> dict:

    dimensions = {}
    
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.dimensions
        
        for key in group.keys():
            value = group[key]
            dimensions[key] = len(value)


                
    return dimensions