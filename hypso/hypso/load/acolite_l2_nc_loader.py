import logging
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple
import xarray as xr
import re

logger = logging.getLogger(__name__)



def load_acolite_l2r_nc(nc_file_path: Path) -> dict:

    logger.info("Opening ACOLITE L2R NetCDF file %s", nc_file_path)

    datasets = {}

    dim_names_3d = ["y", "x", "band"]
    dim_names_2d = ["y", "x"]


    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        varnames = list(f.variables)

        rhot_vars = [v for v in varnames if v.startswith("rhot_")]
        rhos_vars = [v for v in varnames if v.startswith("rhos_")]
        other_vars = [v for v in varnames if not v.startswith(("rhos_", "rhot_"))]

        refl_vars = sorted(rhos_vars)

        logger.info("Accessing rhos reflectances (%d bands)", len(refl_vars))

        data = _load_reflectances_cube(f, refl_vars, dim_names_3d)

        if data is not None:
            datasets['rhos'] = data

    return datasets








def load_acolite_l2w_nc(nc_file_path: Path) -> dict:

    logger.info("Opening ACOLITE L2W NetCDF file %s", nc_file_path)

    datasets = {}

    dim_names_3d = ["y", "x", "band"]
    dim_names_2d = ["y", "x"]


    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        varnames = list(f.variables)

        rrs_vars = [v for v in varnames if v.startswith("Rrs")]
        other_vars = [v for v in varnames if not v.startswith("Rrs_")]

        refl_vars = sorted(rrs_vars)

        logger.info("Accessing rhot reflectances (%d bands)", len(refl_vars))

        data = _load_reflectances_cube(f, refl_vars, dim_names_3d)

        if data is not None:
            datasets['Rrs'] = data


        for other_var in other_vars:

            logger.info("Loading %s", other_var)

            try:
                data = f[other_var][:]

                attrs = {a: getattr(f[other_var], a) for a in f[other_var].ncattrs()}

                data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)

                datasets[other_var] = data

            except Exception:
                logger.warning("Unable to load %s", other_var)


    return datasets





def _load_reflectances_cube(f, refl_vars, dim_names_3d):


    if len(refl_vars) == 0:
        logger.error("No datasets found!")
        return None

    height, width = np.array(f[refl_vars[0]][:], dtype='double').shape
    depth = len(refl_vars)

    data = np.empty((height,width,depth))

    wavelengths = []
    per_band_attrs = {}

    for idx, refl_var in enumerate(refl_vars):

        try:
            band = np.array(f[refl_var][:], dtype='double')
            
            attrs = {a: getattr(f[refl_var], a) for a in f[refl_var].ncattrs()}

        except Exception:
            logger.warning("Unable to load %s", refl_var)
            break

        data[:,:,idx] = band

        per_band_attrs[idx] = attrs

        #print(f[refl_var].ncattrs())
        wavelengths.append(float(f[refl_var].wavelength))

    data = xr.DataArray(data, dims=dim_names_3d, coords={"band": wavelengths})
    data.attrs['per_band'] = per_band_attrs

    return data
