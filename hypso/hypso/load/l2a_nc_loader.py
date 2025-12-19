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
                    load_ncattrs_from_nc_file, \
                    load_geometry_from_nc_file, \
                    load_gcp_from_nc_file

def load_l2a_nc(nc_file_path: Path) -> Tuple[dict, dict, dict, dict, dict, dict, np.ndarray]:

    nc_metadata_vars, nc_metadata_attrs = load_l2a_nc_metadata(nc_file_path=nc_file_path)

    nc_geometry_vars, nc_geometry_attrs = load_l2a_nc_geometry(nc_file_path=nc_file_path)

    nc_cube = load_l2a_nc_cube(nc_file_path=nc_file_path)

    nc_cube_attrs = load_l2a_nc_cube_attrs(nc_file_path=nc_file_path)

    nc_global_metadata = load_l2a_global_nc_metadata(nc_file_path=nc_file_path)

    nc_gcp_vars, nc_gcp_attrs = load_l2a_nc_gcp(nc_file_path=nc_file_path)

    return nc_metadata_vars, \
            nc_metadata_attrs, \
            nc_geometry_vars, \
            nc_geometry_attrs, \
            nc_gcp_vars, \
            nc_gcp_attrs, \
            nc_global_metadata, \
            nc_cube_attrs, \
            nc_cube


def load_l2a_nc_cube(nc_file_path: Path) -> np.ndarray:
    """
    Get Raw Cube from Hypso l2a.nc File

    :param nc_file_path: Absolute path to l2a.nc file

    :return: Numpy array with raw data cube extracted from nc file
    """
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["products"]
        
        try:
            # 16-bit according to Original data Capture
            cube = np.array(group.variables["rrs"][:], dtype='double')
        except Exception as ex:
            print("[INFO] Loading rrs cube from separate bands...")

            height, width = np.array(group.variables[list(group.variables)[0]][:], dtype='double').shape
            depth = len(list(group.variables))

            cube = np.empty((height,width,depth))

            for idx, rrs_band in enumerate(list(group.variables)):

                print("[INFO] Loading band " + str(idx) + "...")

                band = np.array(group.variables[rrs_band][:], dtype='double')

                cube[:,:,idx] = band

            print("[INFO] Loading bands complete.")

        return cube


def load_l2a_nc_cube_attrs(nc_file_path: Path) -> np.ndarray:
    """
    Get Raw Cube from Hypso l2a.nc File

    :param nc_file_path: Absolute path to l2a.nc file

    :return: Numpy array with raw data cube extracted from nc file
    """
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        
        try:
            group = f.groups["products"]["rrs"]

            nc_cube_attrs = {}
            for attrname in group.ncattrs():
                value = getattr(group, attrname)
                nc_cube_attrs[attrname] = value
        except Exception as ex:

            nc_cube_attrs = {}

            nc_cube_attrs["units"] = ""
            nc_cube_attrs["long_name"] = "Remote sensing reflectance"
            nc_cube_attrs["wavelength_units"] = "nanometers"

            wavelength_list = []
            fwhm_list = []

            group = f.groups["products"]

            for idx, rrs_band in enumerate(list(group.variables)):
                
                subgroup = f.groups["products"][rrs_band]

                wavelength = getattr(subgroup, "wavelength")
                fwhm = getattr(subgroup, "fwhm")

                wavelength_list.append(wavelength)
                fwhm_list.append(fwhm)


            wavelengths = np.array(wavelength_list)
            fwhm = np.array(fwhm_list)

            nc_cube_attrs["wavelengths"] = wavelengths
            nc_cube_attrs["fwhm"] = fwhm

        return nc_cube_attrs


def load_l2a_global_nc_metadata(nc_file_path: Path):

    global_metadata = {}

    global_metadata['dimensions'] = load_dimensions_from_nc_file(nc_file_path)
    global_metadata['ncattrs'] = load_ncattrs_from_nc_file(nc_file_path)

    return global_metadata


def load_l2a_nc_geometry(nc_file_path: Path):
    
    geometry_vars, geometry_attrs = load_geometry_from_nc_file(nc_file_path)

    return geometry_vars, geometry_attrs


def load_l2a_nc_metadata(nc_file_path: Path) -> Tuple[dict, dict]:
    """
    Load l2a.nc Hypso Capture file metadata

    :param nc_file_path: Absolute path to the l2a.nc file

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


def load_l2a_nc_gcp(nc_file_path: Path) -> Tuple[dict, dict]:
    """
    Load l2a.nc Hypso Capture file GCPs

    :param nc_file_path: Absolute path to the l2a.nc file

    :return: "gcp_vars" dictionary with gcp variables, "gcp_attrs" dictionary with gcp attributes 
    """

    gcp_vars = {}
    gcp_attrs = {}

    gcp_vars, gcp_attrs = load_gcp_from_nc_file(nc_file_path)

    return gcp_vars, gcp_attrs