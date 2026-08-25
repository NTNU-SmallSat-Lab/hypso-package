# Trimmed from the old utils_file.py grab-bag (see ncdebug.py's docstring):
# the unused file finders/MyProgressBar/haversine-duplicate/
# find_closest_water_lat_lon_match were removed (zero callers confirmed across
# this repo, hypso-processing-pipeline, and the demo), HSI2RGB moved to
# hypso.spectral_analysis (it renders RGB from spectra - not a file utility).
from .misc import is_integer_num
from .ncdebug import print_nc, recursive_print_nc, navigate_recursive_nc, \
                      compare_netcdf_files, nested_dict_to_df, flatten_dict, \
                      list_array_1d_to_string
