import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple

from .utils import load_capture_config_from_nc_file, \
                    load_timing_from_nc_file, \
                    load_target_coords_from_nc_file, \
                    load_adcs_from_nc_file, \
                    load_dimensions_from_nc_file, \
                    load_navigation_from_nc_file, \
                    load_database_from_nc_file, \
                    load_corrections_from_nc_file, \
                    load_logfiles_from_nc_file, \
                    load_temperature_from_nc_file, \
                    load_ncattrs_from_nc_file

def load_l1a_nc(nc_file_path: Path) -> Tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict, dict, np.ndarray]:

    nc_capture_config, \
    nc_timing, \
    nc_target_coords, \
    nc_adcs, \
    nc_dimensions, \
    nc_navigation, \
    nc_database, \
    nc_corrections, \
    nc_temperature, \
    nc_attrs = load_l1a_nc_metadata(nc_file_path=nc_file_path)

    nc_cube = load_l1a_nc_cube(nc_file_path=nc_file_path)

    return nc_capture_config, \
            nc_timing, \
            nc_target_coords, \
            nc_adcs, \
            nc_dimensions, \
            nc_navigation, \
            nc_database, \
            nc_corrections, \
            nc_temperature, \
            nc_attrs, \
            nc_cube


def load_l1a_nc_cube(nc_file_path: Path) -> np.ndarray:
    """
    Get Raw Cube from Hypso l1a.nc File

    :param nc_file_path: Absolute path to l1a.nc file

    :return: Numpy array with raw data cube extracted from nc file
    """
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["products"]
        # 16-bit according to Original data Capture
        cube = np.array(group.variables["Lt"][:], dtype='double')

        return cube

def load_l1a_nc_metadata(nc_file_path: Path) -> Tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict, dict, dict]:
    """
    Load l1a.nc Hypso Capture file metadata

    :param nc_file_path: Absolute path to the l1a.nc file

    :return: "capture_config" dictionary with Hypso capture information, "timing" dictionary with Hypso timing information, "target_coords" dictionary with Hypso target coordinate information, "adcs" dictionary with Hypso ADCS information, and "dimensions" dictionary with Hypso capture spatial dimensions information
    """

    nc_capture_config = load_capture_config_from_nc_file(nc_file_path)
    nc_timing = load_timing_from_nc_file(nc_file_path)
    nc_target_coords = load_target_coords_from_nc_file(nc_file_path)
    nc_adcs = load_adcs_from_nc_file(nc_file_path)
    nc_dimensions = load_dimensions_from_nc_file(nc_file_path)
    nc_navigation = load_navigation_from_nc_file(nc_file_path)
    nc_database = load_database_from_nc_file(nc_file_path)
    nc_corrections = load_corrections_from_nc_file(nc_file_path)
    #nc_logfiles = load_logfiles_from_nc_file(nc_file_path)
    nc_temperature = load_temperature_from_nc_file(nc_file_path)
    nc_attrs = load_ncattrs_from_nc_file(nc_file_path)

    return nc_capture_config, \
            nc_timing, \
            nc_target_coords, \
            nc_adcs, \
            nc_dimensions,\
            nc_navigation, \
            nc_database, \
            nc_corrections, \
            nc_temperature, \
            nc_attrs
