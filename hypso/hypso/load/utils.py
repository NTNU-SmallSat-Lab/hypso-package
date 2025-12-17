#import numpy as np
#from datetime import datetime
import netCDF4 as nc
#from hypso import georeferencing
from pathlib import Path
from hypso.utils import is_integer_num
from typing import Tuple


def load_adcs_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:
    """
    Get the metadata from the top folder of the data.

    :param nc_file_path: Path to the name of the tmp folder
    :param standard_dimensions: Dictionary with Hypso standard dimensions

    :return: "adcs" dictionary with ADCS metadata and a tuple with spatial dimensions
    """

    adcs_attrs = {}
    adcs_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["adcs"]
        
        for key in group.variables.keys():
            value = group.variables[key][:]
            adcs_vars[key] = value

        adcs_vars['adcssamples'] = len(f.dimensions['adcssamples'])

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    adcs_attrs[attrname] = int(value)
                else:
                    adcs_attrs[attrname] = float(value)
            except BaseException:
                adcs_attrs[attrname] = value

    return (adcs_vars, adcs_attrs)

 
def load_capture_config_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    capture_config_attrs = {}
    capture_config_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["capture_config"]

        for key in group.variables.keys():
            value = group.variables[key][:]
            capture_config_vars[key] = value

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    capture_config_attrs[attrname] = int(value)
                else:
                    capture_config_attrs[attrname] = float(value)
            except BaseException:
                capture_config_attrs[attrname] = value

    return (capture_config_vars, capture_config_attrs)


def load_timing_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    timing_attrs = {}
    timing_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["timing"]
        
        for key in group.variables.keys():
            value = group.variables[key][:]
            timing_vars[key] = value
        timing_vars['lines'] = len(f.dimensions['lines'])

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    timing_attrs[attrname] = int(value)
                else:
                    timing_attrs[attrname] = float(value)
            except BaseException:
                timing_attrs[attrname] = value

    return (timing_vars, timing_attrs)




def load_dimensions_from_nc_file(nc_file_path: Path) -> dict:

    dimensions = {}
    
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.dimensions
        
        for key in group.keys():
            value = group[key]
            dimensions[key] = len(value)

    return dimensions




def load_database_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    database_attrs = {}
    database_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        try:
            group = f.groups["metadata"]["database"]

            for attrname in group.ncattrs():
                value = getattr(group, attrname)
                try:
                    if is_integer_num(float(value)):
                        database_attrs[attrname] = int(value)
                    else:
                        database_attrs[attrname] = float(value)
                except BaseException:
                    database_attrs[attrname] = value

        except Exception as ex:
            #print(ex)
            pass

    return (database_vars, database_attrs)



def load_corrections_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    corrections_attrs = {}
    corrections_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["corrections"]
        
        for key in group.variables.keys():
            value = group.variables[key][:]
            corrections_vars[key] = value
        

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    corrections_attrs[attrname] = int(value)
                else:
                    corrections_attrs[attrname] = float(value)
            except BaseException:
                corrections_attrs[attrname] = value

    return (corrections_vars, corrections_attrs)


def load_logfiles_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    logfiles_attrs = {}
    logfiles_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        try:
            group = f.groups["logfiles"]

            for key in group.variables.keys():
                value = group.variables[key][:]
                logfiles_vars[key] = value

            for attrname in group.ncattrs():
                value = getattr(group, attrname)
                try:
                    if is_integer_num(float(value)):
                        logfiles_attrs[attrname] = int(value)
                    else:
                        logfiles_attrs[attrname] = float(value)
                except BaseException:
                    logfiles_attrs[attrname] = value
        except Exception as ex:
            #print(ex)
            pass

    return (logfiles_vars, logfiles_attrs)



def load_temperature_from_nc_file(nc_file_path: Path) -> Tuple[dict, dict]:

    temperature_attrs = {}
    temperature_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        try:
            group = f.groups["metadata"]["temperature"]
            
            for key in group.variables.keys():
                value = group.variables[key][:]
                temperature_vars[key] = value

            for attrname in group.ncattrs():
                value = getattr(group, attrname)
                try:
                    if is_integer_num(float(value)):
                        temperature_attrs[attrname] = int(value)
                    else:
                        temperature_attrs[attrname] = float(value)
                except BaseException:
                    temperature_attrs[attrname] = value
        
        except Exception as ex:
            #print(ex)
            pass

    return (temperature_vars, temperature_attrs)



def load_ncattrs_from_nc_file(nc_file_path: Path) -> dict:

    attrs = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    attrs[attrname] = int(value)
                else:
                    attrs[attrname] = float(value)
            except BaseException:
                attrs[attrname] = value


    return attrs




def load_navigation_from_nc_file(nc_file_path: Path) -> dict:

    navigation_attrs = {}
    navigation_vars = {}

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["navigation"]
        
        for key in group.variables.keys():
            value = group.variables[key][:]
            navigation_vars[key] = value

        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    navigation_attrs[attrname] = int(value)
                else:
                    navigation_attrs[attrname] = float(value)
            except BaseException:
                navigation_attrs[attrname] = value

    return (navigation_vars, navigation_attrs)


    '''
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
    '''


