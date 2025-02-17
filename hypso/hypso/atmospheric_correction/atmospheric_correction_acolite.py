import numpy as np
from importlib.resources import files
from urllib.parse import urlparse, unquote

from pathlib import Path
import netCDF4 as nc

import sys

def run_acolite(acolite_path: str, output_path: str, nc_file_path: str) -> np.ndarray:
    """
    Run the ACOLITE correction model. Adjustments of the original files are made to ensure they work for HYPSO

    :param hypso_info: Dictionary containing the hypso capture information
    :param atmos_dict: Dictionary containing the information required for the atmospheric correction
    :param nc_file_acoliteready: Absolute path for the .nc file for ACOLITE (L1B)

    :return: Returns surface reflectanc corrected spectral image
    """
    # https://odnature.naturalsciences.be/remsem/acolite-forum/viewtopic.php?t=238
    
    # add acolite clone to Python path and import acolite


    print(sys.path)
    sys.path.append(acolite_path)
    print(sys.path)

    import acolite as ac

    # optional file with processing settings
    # if set to None defaults will be used
    settings_file = None

    # import settings
    settings = ac.acolite.settings.load(settings_file)

    # set settings provided above
    settings['inputfile'] = nc_file_path
    settings['output'] = output_path

    settings['polygon'] = None
    settings['l2w_parameters'] = None
    settings['rgb_rhot'] = True
    settings['rgb_rhos'] = True
    settings['map_l2w'] = False

    # user and password from https://urs.earthdata.nasa.gov/profile
    # optional but good
    settings['EARTHDATA_u'] = "alvarof"
    settings['EARTHDATA_p'] = "nwz7xmu8dak.UDG9kqz"

    # other settings can also be provided here, e.g.
    # settings['s2_target_res'] = 60
    # settings['dsf_path_reflectance'] = 'fixed'
    # settings['l2w_parameters'] = ['t_nechad', 't_dogliotti']

    # process the current bundle
    processed = ac.acolite.acolite_run(settings=settings)

    acolite_l2_file = processed[0]['l2r'][0]

    # Maintainer comment:
    # Source: https://odnature.naturalsciences.be/remsem/acolite-forum/viewtopic.php?t=311
    # - L1R, containing the top-of-atmosphere reflectance (rhot_*) as converted to the ACOLITE format from the sensor specific L1 files
    # - L2R, containing the top-of-atmosphere reflectance (rhot_*) and the surface-level reflectance after atmospheric correction (rhos_*)
    # - L2W, containing user requested parameters, e.g. water-leaving radiance reflectance (rhow_*), Remote sensing reflectance (Rrs_*), or outputs from any of the included parameter retrieval algorithms

    # Read .nc
    final_acolite_l2 = None

    with nc.Dataset(acolite_l2_file, format="NETCDF4") as f:
        group = f
        keys = [i for i in f.variables.keys()]

        toa_keys = [k for k in keys if 'rhos' not in k]
        surface_keys = [kk for kk in keys if 'rhot' not in kk]

        # Add Cube

        for i, k in enumerate(surface_keys):
            current_channel = np.array(group.variables[k][:])
            if final_acolite_l2 is None:
                final_acolite_l2 = np.empty(
                    (current_channel.shape[0], current_channel.shape[1], len(surface_keys)))

            final_acolite_l2[:, :, i] = current_channel

        # TODO: Confirm if zeros should be appended at the beginning or end
        # ACOLITE returns 118 bands
        # If number of bands less that 120, append zeros to the end
        delta = int(120 - final_acolite_l2.shape[2])
        if delta > 0:
            for _ in range(delta):
                zeros_arr = np.zeros((final_acolite_l2.shape[0], final_acolite_l2.shape[1]), dtype=float)
                final_acolite_l2 = np.dstack((final_acolite_l2, zeros_arr))

    return final_acolite_l2