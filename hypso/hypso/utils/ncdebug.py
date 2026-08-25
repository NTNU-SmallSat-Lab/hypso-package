"""NetCDF inspection/debugging helpers (print_nc/compare_netcdf_files and
their support functions) - the surviving, coherent slice of the old
utils/utils_file.py grab-bag after its unused functions (file finders,
MyProgressBar, a duplicate haversine, find_closest_water_lat_lon_match) were
removed with zero callers confirmed. Developer conveniences for a
NetCDF-centric package: print a file's full group/variable/attribute tree, or
diff two files' structures."""
from numbers import Number
from typing import Union

import netCDF4 as nc
import numpy as np
import pandas as pd


def flatten_dict(nested_dict: dict) -> dict:
    """
    Flatten a nested dictionary into a single level dictionary

    :param nested_dict: Nested dictionary to flatten

    :return: Single level dictionary with keys corresponding to the previous nested levels
    """
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict: dict) -> pd.DataFrame:
    """
    Convert nested dictionary to DataFrame.

    :param values_dict: Nested dictionary

    :return: Dataframe where the nested level are displayed as a column.
    """

    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    # df.index = pd.MultiIndex.from_tuples(df.index)
    # df = df.unstack(level=-1)
    # df.columns = df.columns.map("{0[1]}".format)
    return df


def navigate_recursive_nc(nc_file: nc.Dataset, path: str = '', depth: int = 0) -> dict:
    """
    Navigate recursively the structure of a .nc file and return the dictionary structure. The function will call
    itself to navigate recursively.

    :param nc_file: Open .nc Dataset file
    :param path: Relative path to append to recursivness
    :param depth: Depth at which we are navigating

    :return: Dictionary of dictionaries containing the structure of the .nc file
    """
    label = path + nc_file.name
    tree_structure = {
        label: {}
    }
    # Dimensions -----------------------------------
    tree_structure[label]["dimensions"] = {}
    group_dims = list(nc_file.dimensions.keys())
    for gd in group_dims:
        tree_structure[label]["dimensions"][gd] = nc_file.dimensions[gd].size

    # Group Attributes --------------------------------------------------------
    tree_structure[label]["group_attributes"] = {}
    group_attrs = nc_file.ncattrs()
    for ga in group_attrs:
        tree_structure[label]["group_attributes"][ga] = nc_file.getncattr(ga)

    # Variables -------------------------------------------------------------
    group_variables = nc_file.variables
    tree_structure[label]["variables"] = {}
    for gv in group_variables:
        tree_structure[label]["variables"][gv] = {}
        tree_structure[label]["variables"][gv]["dimensions"] = group_variables[gv].dimensions
        tree_structure[label]["variables"][gv]["value"] = group_variables[gv][:]

    # Variable Attributes -----------------------------------------------
    tree_structure[label]["variables_attributes"] = {}
    variables = nc_file.variables.keys()
    variables_attributes = []
    for v in variables:
        try:
            attrs = nc_file[v].ncattrs()
            variables_attributes.append(attrs)
        except AttributeError:
            pass
    for v, attr_list in zip(variables, variables_attributes):
        for a in attr_list:
            attr_tmp = nc_file[v].getncattr(a)  # Get attribute
            tree_structure[label]["variables_attributes"][v + "-" + a] = attr_tmp

    # Sub groups -----------------------------------------------------------
    tree_structure[label]["subgroups"] = list(nc_file.groups.keys())

    for g in nc_file.groups.keys():
        if nc_file.name == '/':
            newname = path + nc_file.name
        else:
            newname = path + nc_file.name + '/'
        recursive_dict = navigate_recursive_nc(nc_file.groups[g], path=newname, depth=depth + 1)
        recursive_keys = list(recursive_dict.keys())
        for k in recursive_keys:
            tree_structure[k] = recursive_dict[k]

    return tree_structure


def list_array_1d_to_string(arr: Union[np.ndarray, list, tuple]) -> Union[tuple, str, Number]:
    """
    Converts 1D numpy array to string

    :param arr: 1D numpy array of numbers or strings to convert to string

    :return: String of combined values or the same value if not a list
    """
    var_str = ''
    end_var_str = ''
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        var_str = '['
        end_var_str = ']'
    elif isinstance(arr, tuple):
        var_str = '('
        end_var_str = ')'
    else:  # if int or float or not a list
        return arr

    for ss in arr:
        var_str += str(ss).replace("'", '')
        var_str += ', '
    var_str = ''.join(var_str.rsplit(', ', 1))
    var_str += end_var_str

    return var_str


def recursive_print_nc(nc_file: nc.Dataset, path: str = '', depth: int = 0) -> None:
    """
    Navigate recursively the structure of a .nc file and print the structure. The function will call
    itself to navigate recursively.

    :param nc_file: Open .nc Dataset file
    :param path: Relative path to append to recursivness
    :param depth: Depth at which we are navigating

    :return: No return.
    """

    indent = ''
    for i in range(depth):
        indent += '  '

    print(indent, '--- GROUP: "', path + nc_file.name, '" ---', sep='')

    print(indent, 'DIMENSIONS: ', sep='', end='')
    for d in nc_file.dimensions.keys():
        print(f"{d} ({nc_file.dimensions[d].size})", end=', ')
    print('')

    print(indent, 'GROUP ATTRIBUTES: ', sep='', end='')
    for a in nc_file.ncattrs():
        print(a, end=', ')
    print('')

    print(indent, 'VARIABLES: ', sep='', end='')
    for v in nc_file.variables.keys():
        var_str_tmp = nc_file[v].dimensions
        var_str = list_array_1d_to_string(var_str_tmp)

        print(v, f"{var_str}", end=', ')
    print('')

    # Variable Attributes ------------------------------------------------
    var_str = nc_file.variables.keys()
    curr_var_atrr = []
    for v in var_str:
        try:
            attrs = nc_file[v].ncattrs()
            curr_var_atrr.append(attrs)
        except AttributeError:
            pass
    print(indent, 'VAR ATTRIBUTES: ', sep='')
    if len(curr_var_atrr) > 0:

        for v, attr_list in zip(var_str, curr_var_atrr):
            if len(attr_list) > 0:
                print('')
                print(indent, indent, v)
                for a in attr_list:
                    attr_tmp = nc_file[v].getncattr(a)
                    attr_string = list_array_1d_to_string(attr_tmp)
                    if isinstance(attr_tmp, np.ndarray):
                        print(indent, indent, f"---{a.strip()} {attr_tmp.shape}: {attr_string}")
                    elif isinstance(attr_tmp, list) or isinstance(attr_tmp, tuple):
                        print(indent, indent, f"---{a.strip()} {len(attr_tmp)}: {attr_string}")
                    else:
                        attr_string = str(attr_tmp)
                        print(indent, indent, f"---{a.strip()}: {attr_string}")

        print('')

    # Sub Groups ---------------------------------------------------------
    print(indent, 'SUB-GROUPS: ', sep='', end='')
    for g in nc_file.groups.keys():
        print(g, end=', ')
    print('')
    print('')

    for g in nc_file.groups.keys():
        if nc_file.name == '/':
            newname = path + nc_file.name
        else:
            newname = path + nc_file.name + '/'
        recursive_print_nc(nc_file.groups[g], path=newname, depth=depth + 1)


def print_nc(nc_file: str) -> None:
    """
    Print the contents of a .nc file
    :param nc_file_path: Absolute path to a .nc file

    :return: No return
    """
    recursive_print_nc(nc.Dataset(nc_file, format="NETCDF4"))


def compare_netcdf_files(file1: str, file2: str) -> pd.DataFrame:
    """
    Compare two .nc files and returns a pandas DataFrame where each attribute, variable and values are compared side
    by side.

    :param file1: Absolute path to file 1
    :param file2: Absolute path to file 2

    :return: Dataframe with the compared characteristics
    """
    file1 = Path(file1).absolute()
    file2 = Path(file2).absolute()
    file1_structure = navigate_recursive_nc(nc.Dataset(str(file1), format="NETCDF4"))
    file2_structure = navigate_recursive_nc(nc.Dataset(str(file2), format="NETCDF4"))

    df1 = nested_dict_to_df(file1_structure)
    df1.rename(columns={0: file1.name}, inplace=True)
    df1['label'] = df1.index
    df1.reset_index(inplace=True, drop=True)

    df2 = nested_dict_to_df(file2_structure)
    df2.rename(columns={0: file2.name}, inplace=True)
    df2['label'] = df2.index
    df2.reset_index(inplace=True, drop=True)

    # Merged on Column and Indicate which label is in both dataframes
    d = {"left_only": f"Only present in {file1.name}",
         "right_only": f"Only present in {file2.name}",
         "both": "Present in Both"}

    merged = pd.merge(df1, df2, on="label", how="outer", indicator=True)
    merged['_merge'] = merged['_merge'].map(d)

    merged.rename(columns={'_merge': "presence"}, inplace=True)

    # Validate if values are equal or differente
    merged["validator"] = "N/A"

    validator_res = []
    for idx, row in merged.iterrows():
        try:
            if np.all(row[file1.name] == row[file2.name]):
                validator_res.append("equal")
            else:
                validator_res.append("different")
        except Exception as e:
            validator_res.append("different")

    merged['validator'] = pd.Series(validator_res)

    # Change Column Order
    merged = merged[['label', 'presence', 'validator', file1.name, file2.name]]

    return merged
