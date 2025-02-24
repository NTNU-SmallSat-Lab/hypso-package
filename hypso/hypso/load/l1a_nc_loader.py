import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple

from .utils import load_capture_config_from_nc_file, \
                    load_timing_from_nc_file, \
                    load_adcs_from_nc_file, \
                    load_dimensions_from_nc_file, \
                    load_database_from_nc_file, \
                    load_corrections_from_nc_file, \
                    load_logfiles_from_nc_file, \
                    load_temperature_from_nc_file, \
                    load_ncattrs_from_nc_file

def load_l1a_nc(nc_file_path: Path) -> Tuple[dict, dict, dict, dict, dict, dict, np.ndarray]:

    nc_metadata_vars, nc_metadata_attrs = load_l1a_nc_metadata(nc_file_path=nc_file_path)

    nc_naivigation_vars = {}
    nc_navigation_attrs = {}

    nc_cube = load_l1a_nc_cube(nc_file_path=nc_file_path)

    nc_cube_attrs = {}

    nc_global_metadata = load_l1a_global_nc_metadata(nc_file_path=nc_file_path)

    return nc_metadata_vars, \
            nc_metadata_attrs, \
            nc_naivigation_vars, \
            nc_navigation_attrs, \
            nc_global_metadata, \
            nc_cube_attrs, \
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

        try:
            cube = np.array(group.variables["dn"][:], dtype='double')
        except KeyError:
            cube = np.array(group.variables["Lt"][:], dtype='double')

        return cube


def load_l1a_global_nc_metadata(nc_file_path: Path):

    global_metadata = {}

    global_metadata['dimensions'] = load_dimensions_from_nc_file(nc_file_path)
    global_metadata['ncattrs'] = load_ncattrs_from_nc_file(nc_file_path)

    return global_metadata


def load_l1a_nc_metadata(nc_file_path: Path) -> Tuple[dict, dict]:
    """
    Load l1a.nc Hypso Capture file metadata

    :param nc_file_path: Absolute path to the l1a.nc file

    :return: "metadata_vars" dictionary with metadata variables, "metadata_attrs" dictionary with metadata attributes, "metadata_global" dictionary with global metadata attributes and dimensions, 
    """

    metadata_vars = {}

    metadata_vars['capture_config'] = load_capture_config_from_nc_file(nc_file_path)[0]
    metadata_vars['timing'] = load_timing_from_nc_file(nc_file_path)[0]
    metadata_vars['adcs'] = load_adcs_from_nc_file(nc_file_path)[0]
    metadata_vars['database'] = load_database_from_nc_file(nc_file_path)[0]
    metadata_vars['corrections'] = load_corrections_from_nc_file(nc_file_path)[0]
    metadata_vars['logfiles'] = load_logfiles_from_nc_file(nc_file_path)[0]
    metadata_vars['temperature'] = load_temperature_from_nc_file(nc_file_path)[0]

    metadata_attrs = {}

    metadata_attrs['capture_config'] = load_capture_config_from_nc_file(nc_file_path)[1]
    metadata_attrs['timing'] = load_timing_from_nc_file(nc_file_path)[1]
    metadata_attrs['adcs'] = load_adcs_from_nc_file(nc_file_path)[1]
    metadata_attrs['database'] = load_database_from_nc_file(nc_file_path)[1]
    metadata_attrs['corrections'] = load_corrections_from_nc_file(nc_file_path)[1]
    metadata_attrs['logfiles'] = load_logfiles_from_nc_file(nc_file_path)[1]
    metadata_attrs['temperature'] = load_temperature_from_nc_file(nc_file_path)[1]

    return metadata_vars, metadata_attrs
