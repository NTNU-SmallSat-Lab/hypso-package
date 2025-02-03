from pathlib import Path
import numpy as np
import netCDF4 as nc
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
from importlib.resources import files
from math import asin, cos, radians, sin, sqrt
import progressbar
import pandas as pd
from typing import Literal, Union, Tuple
from numbers import Number


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


def HSI2RGB(wY, HSI, d: Literal[50, 55, 65, 75] = 65, threshold=0.002) -> np.ndarray:
    """
    Calculate a colore accurate RGB image with HSI values. Original function used for the paper
    "Creating RGB Images from Hyperspectral Images Using a Color Matching Function" by
    M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson, H. Deborah and J. R. Sveinsson,

    If you use this method, please cite the following paper:
    @INPROCEEDINGS{hsi2rgb,
    author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson}
    and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}},
    booktitle={IEEE International Geoscience and Remote Sensing Symposium},
    title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function},
    year={2020}, volume={}, number={}, pages={}}

    Paper is available at
    https://www.researchgate.net/profile/Jakob_Sigurdsson

    :param wY: Wavelengths in nm
    :param HSI: Hyperspectral Image as a (#pixels x #bands) matrix,
    :param d: Determines the illuminant used, if in doubt use d65. Values of  50, 55, 65, 75
    :param threshold: True if RGB thesholding should be done to increase contrast

    :return: RGB Image array
    """

    (ydim, xdim, zdim) = HSI.shape
    HSI = np.reshape(HSI, [-1, zdim]) / np.nanmax(HSI)

    # Load reference illuminant
    illuminant_path = files('hypso.utils').joinpath('data/D_illuminants.mat')
    D = spio.loadmat(str(illuminant_path))
    w = D['wxyz'][:, 0]
    x = D['wxyz'][:, 1]
    y = D['wxyz'][:, 2]
    z = D['wxyz'][:, 3]
    D = D['D']

    i = {50: 2,
         55: 3,
         65: 1,
         75: 4}
    wI = D[:, 0]
    I = D[:, i[d]]

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(wY)  # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w, x, extrapolate=True)(wY)  # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w, y, extrapolate=True)(wY)  # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w, z, extrapolate=True)(wY)  # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i = bisect(wY, 780)
    HSI = HSI[:, 0:i] / HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute k
    k = 1 / np.trapz(y * I, wY)

    # Compute X,Y & Z for image
    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)

    XYZ = np.array([X, Y, Z])

    # Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    sRGB = M @ XYZ

    # Gamma correction
    """Convert sRGB values to physically linear ones. The transformation is
           uniform in RGB, so *srgb* can be of any shape.

           *srgb* values should range between 0 and 1, inclusively.

        """
    gamma = ((sRGB + 0.055) / 1.055) ** 2.4
    scale = sRGB / 12.92
    sRGB = np.where(sRGB > 0.04045, gamma, scale)

    # gamma_map = sRGB > 0.0031308
    # sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
    # sRGB[np.invert(gamma_map)] = 12.92 * sRGB[np.invert(gamma_map)]

    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB[sRGB > 1] = 1
    sRGB[sRGB < 0] = 0

    if threshold:
        for idx in range(3):
            y = sRGB[idx, :]
            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            th = b[0]
            i = a < threshold
            if i.any():
                th = b[i][-1]
            y = y - th
            y[y < 0] = 0

            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b) / 2
            a = np.cumsum(a) / np.sum(a)
            i = a > 1 - threshold
            th = b[i][0]
            y[y > th] = th
            y = y / th
            sRGB[idx, :] = y

    R = np.reshape(sRGB[0, :], [ydim, xdim])
    G = np.reshape(sRGB[1, :], [ydim, xdim])
    B = np.reshape(sRGB[2, :], [ydim, xdim])

    return np.dstack((R, G, B))


class MyProgressBar:
    """
    Class used to display a progress bar on the screen for bulk downloading.
    """

    def __init__(self, text_prefix):
        self.pbar = None
        self.text_prefix = text_prefix

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size,
                                                widgets=[
                                                    progressbar.Bar('=', f'Downloading: {self.text_prefix} [', ']', ),
                                                    ' ', progressbar.Percentage(), ], )
            # self.pbar = progressbar.Percentage()
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def find_all_files(path: Path, str_in_file: str, suffix: Union[str, None] = None,
                   type: Literal["partial", "exact"] = "partial") -> list:
    """
    Find recursively all files in a specified directory.

    :param path: Directory path on which to look for files.
    :param str_in_file: Check if this string is in the file name
    :param suffix: Check if file extension matches the suffix provided
    :param type: Type of comparisson of "str_in_file". If "partial", then "str_in_file" will be checked to be in the filename,
        otherwise "exact" will check if the filename matches exactly the string provided.

    :return: List of file paths
    """
    all_files = []
    for subpath in path.rglob("*"):
        if subpath.is_file():
            if type == "partial":
                if suffix is not None:
                    if str_in_file in subpath.name and subpath.suffix == suffix:
                        all_files.append(subpath.absolute())
                elif suffix is None:
                    if str_in_file in subpath.name:
                        all_files.append(subpath.absolute())

            elif type == "exact":
                if suffix is not None:
                    if str_in_file == subpath.name and subpath.suffix == suffix:
                        all_files.append(subpath.absolute())
                elif suffix is None:
                    if str_in_file == subpath.name:
                        all_files.append(subpath.absolute())

    return all_files


def find_file(path: Path, str_in_file: str, suffix: Union[str, None] = None,
              type: Literal["partial", "exact"] = "partial") -> Union[Path, None]:
    """
    Find recursively a single file (or first occurance) in a specified directory.

    :param path: Directory path on which to look for files.
    :param str_in_file: Check if this string is in the file name
    :param suffix: Check if file extension matches the suffix provided
    :param type: Type of comparisson of "str_in_file". If "partial", then "str_in_file" will be checked to be in the filename,
        otherwise "exact" will check if the filename matches exactly the string provided.

    :return: Path of the file or None if no file found
    """
    for subpath in path.rglob("*"):
        if subpath.is_file():
            if type == "partial":
                if suffix is not None:
                    if str_in_file in subpath.name and subpath.suffix == suffix:
                        return subpath.absolute()
                elif suffix is None:
                    if str_in_file in subpath.name:
                        return subpath.absolute()

            elif type == "exact":
                if suffix is not None:
                    if str_in_file == subpath.name and subpath.suffix == suffix:
                        return subpath.absolute()
                elif suffix is None:
                    if str_in_file == subpath.name:
                        return subpath.absolute()

    return None

def find_dir(path: Path, str_in_dir: str, type: Literal["partial", "exact"] = "partial") -> Union[Path, None]:
    """
    Find recursively a directory (or first occurance) in a specified directory.

    :param path: Directory path on which to look for files.
    :param str_in_dir: Check if this string is in the directory name
    :param type: Type of comparisson of "str_in_file". If "partial", then "str_in_file" will be checked to be in the filename,
        otherwise "exact" will check if the filename matches exactly the string provided.

    :return: Path of the file or None if no file found
    """

    for subpath in path.rglob("*"):
        if subpath.is_dir():
            if type == "partial":
                if str_in_dir in subpath.name:
                    return subpath.absolute()
            elif type == "exact":
                if str_in_dir == subpath.name:
                    return subpath.absolute()

    return None


def is_integer_num(n) -> bool:
    """
    Check if a number is an integer

    :param n: Number to check
    :return: Boolean value indicating whether or not the number is an integer
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

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

def print_nc(nc_file: str) -> None:
    """
    Print the contents of a .nc file
    :param nc_file_path: Absolute path to a .nc file

    :return: No return
    """
    recursive_print_nc(nc.Dataset(nc_file, format="NETCDF4"))


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

'''
def find_closest_water_lat_lon_match(sat_obj, target_lat: float, target_lon: float, water_pixel_filter:bool=False) -> Tuple[int, int, float]:
    """
    Find the closest Image lat and lon to the targat specificed that is also a water pixel.

    :param water_pixel_filter: If true, the closest pixel should also be classified as water pixel.
    :param sat_obj: Hypso satellite object
    :param target_lat: Target latitude
    :param target_lon: Target longitude

    :return: Row, col and dist_to_target in km from the closes water pixel
    """

    lat_2d = sat_obj.latitudes
    lon_2d = sat_obj.longitudes
    waterMask = sat_obj.waterMask
    row = 0
    col = 0
    dist_to_target_from_sat = 0.0

    if np.nanmax(lat_2d) > target_lat > np.nanmin(lat_2d) and np.nanmax(lon_2d) > target_lon > np.nanmin(lon_2d):

        if water_pixel_filter:
            water_pix_mask = waterMask.flatten()
            coordinates = [c for c in zip(lat_2d.flatten()[water_pix_mask], lon_2d.flatten()[water_pix_mask])]
        else:
            coordinates = [c for c in zip(lat_2d.flatten(), lon_2d.flatten())]

        xy = (target_lat, target_lon)

        # Closest distance with Pithagoras --------------
        dist = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
        closest_existing_coord = min(coordinates, key=lambda co: dist(co, xy))
        lat_found = closest_existing_coord[0]
        lon_found = closest_existing_coord[1]

        found_idx = np.argwhere(np.logical_and(
            lat_2d == lat_found,
            lon_2d == lon_found,
        ))[0]

        try:
            row = found_idx[0]
            col = found_idx[1]

            dist_to_target_from_sat = haversine(lon_found, lat_found, target_lon, target_lat)

            print(
                f"Satellite (Lat:Lon)  : {closest_existing_coord[1]} for  \n desired Lat: {target_lat} - Lon {target_lon} found at Row {row}:Col {col}")
            print(f"Distance (Km) from Satellite coordinate to Target:"
                  f"{dist_to_target_from_sat}"
                  )
        except:
            raise Exception("Lat and Lon not found")

        # e.g.
        # l1b_cube[found[0],found[1],:]

        return row, col, dist_to_target_from_sat

    else:
        raise Exception("Lat and Lon not within Capture 2D Array Lat/Lon values")
'''

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    Source: https://stackoverflow.com/a/4913653

    :param lon1: Longitude of first point
    :param lat1: Latitude of first point
    :param lon2: Longitude of second point
    :param lat2: Latitude of second point

    :return: Distance in km between two points on earth
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km
