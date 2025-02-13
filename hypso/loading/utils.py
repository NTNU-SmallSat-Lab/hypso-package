#import numpy as np
#from datetime import datetime
import netCDF4 as nc
#from hypso import georeferencing
from pathlib import Path
from hypso.utils import is_integer_num
from typing import Tuple


def load_adcs_from_nc_file(nc_file_path: Path) -> Tuple[dict, tuple]:
    """
    Get the metadata from the top folder of the data.

    :param nc_file_path: Path to the name of the tmp folder
    :param standard_dimensions: Dictionary with Hypso standard dimensions

    :return: "adcs" dictionary with ADCS metadata and a tuple with spatial dimensions
    """

    adcs = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["adcs"]
        
        for key in group.variables.keys():
            
            value = group.variables[key][:]

            adcs[key] = value

        adcs['adcssamples'] = len(f.dimensions['adcssamples'])

    return adcs

 
def load_capture_config_from_nc_file(nc_file_path: Path) -> dict:

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


def load_dimensions_from_nc_file(nc_file_path: Path) -> dict:

    dimensions = {}
    
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.dimensions
        
        for key in group.keys():
            value = group[key]
            dimensions[key] = len(value)

    return dimensions


def load_navigation_from_nc_file(nc_file_path: Path) -> dict:

    navigation = {}
    
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        try:
            group = f.groups["navigation"]

            for key in group.variables.keys():

                try:
                    value = group.variables[key][:]
                    navigation[key] = value
                except:
                    pass
        except:
            pass

    return navigation


