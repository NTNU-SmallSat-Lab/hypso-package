from pathlib import Path
import numpy as np
import netCDF4 as nc
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
from importlib.resources import files
from math import asin, cos, radians, sin, sqrt
import progressbar

def HSI2RGB(wY, HSI, d=65, threshold=0.002):
    # wY: wavelengths in nm
    # Y : HSI as a (#pixels x #bands) matrix,
    # dims: x & y dimension of image
    # d: 50, 55, 65, 75, determines the illuminant used, if in doubt use d65
    # thresholdRGB : True if thesholding should be done to increase contrast
    #
    #
    # If you use this method, please cite the following paper:
    #  M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson,
    #  H. Deborah and J. R. Sveinsson,
    #  "Creating RGB Images from Hyperspectral Images Using a Color Matching Function",
    #  IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020
    #
    #  @INPROCEEDINGS{hsi2rgb,
    #  author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson}
    #  and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}},
    #  booktitle={IEEE International Geoscience and Remote Sensing Symposium},
    #  title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function},
    #  year={2020}, volume={}, number={}, pages={}}
    #
    # Paper is available at
    # https://www.researchgate.net/profile/Jakob_Sigurdsson
    #
    #

    (ydim, xdim, zdim) = HSI.shape
    HSI = np.reshape(HSI, [-1, zdim]) / np.nanmax(HSI)

    # Load reference illuminant
    illuminant_path = files('hypso.utils').joinpath('data/D_illuminants.mat')
    D = spio.loadmat(illuminant_path)
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
    sRGB =  np.where(sRGB > 0.04045, gamma, scale)

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

class MyProgressBar():
    def __init__(self,text_prefix):
        self.pbar = None
        self.text_prefix=text_prefix


    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size,
                                              widgets=[progressbar.Bar('=', f'Downloading: {self.text_prefix} [', ']',), ' ', progressbar.Percentage(),],)
            #self.pbar = progressbar.Percentage()
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def find_all_files(path: Path, str_in_file: str, suffix=None, type="partial"):
    all_files=[]
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

def find_file(path:Path,str_in_file:str, suffix=None, type="partial"):
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

def find_dir(path:Path,str_in_dir:str,type="partial"):
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
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def print_nc(nc_file_path):
    recursive_print_nc(nc.Dataset(nc_file_path, format="NETCDF4"))

def list_array_1d_to_string(arr):
    var_str = ''
    end_var_str = ''
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        var_str = '['
        end_var_str = ']'
    elif isinstance(arr, tuple):
        var_str = '('
        end_var_str = ')'
    else: # if int or float or not a list
        return arr

    for ss in arr:
        var_str += str(ss).replace("'", '')
        var_str += ', '
    var_str = ''.join(var_str.rsplit(', ', 1))
    var_str += end_var_str

    return var_str

def recursive_print_nc(nc_file, path='', depth=0):

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

        print(v,f"{var_str}", end=', ')
    print('')

    # Variable Attributes ------------------------------------------------
    var_str = nc_file.variables.keys()
    curr_var_atrr=[]
    for v in var_str:
        try:
            attrs = nc_file[v].ncattrs()
            curr_var_atrr.append(attrs)
        except AttributeError:
            pass
    print(indent, 'VAR ATTRIBUTES: ', sep='')
    if len(curr_var_atrr) > 0:

        for v, attr_list in zip(var_str,curr_var_atrr):
            if len(attr_list) > 0:
                print('')
                print(indent,indent, v)
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



def pseudo_convolution(cube, watermask, center=[], size=5):
    total_spectra = None

    start_row = center[0] - size // 2
    start_col = center[1] - size // 2
    for i in range(size):
        current_row = start_row + (i + 1)
        for j in range(size):
            current_col = start_col + (j + 1)
            if watermask[current_row, current_col] == True:
                if total_spectra is None:
                    total_spectra = cube[current_row, current_col, :]
                else:
                    total_spectra = np.column_stack((total_spectra, cube[current_row, current_col, :]))

    if len(total_spectra.shape) == 1:
        return total_spectra

    else:
        return np.mean(total_spectra, axis=1)


def find_closest_water_lat_lon_match(lat_2d: np.array, lon_2d: np.array, waterMask: np.array, target_lat, target_lon):
    if np.nanmax(lat_2d) > target_lat > np.nanmin(lat_2d) and np.nanmax(lon_2d) > target_lon > np.nanmin(lon_2d):

        water_pix = waterMask.flatten()

        coordinates = [c for c in zip(lat_2d.flatten()[water_pix], lon_2d.flatten()[water_pix])]

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


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
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