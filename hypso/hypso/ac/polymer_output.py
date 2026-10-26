import logging

import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple
import xarray as xr
import re

logger = logging.getLogger(__name__)


# For NetCDF output from Feb 2026 version of Polymer
def load_polymer_l2_v2_nc(nc_file_path: Path) -> dict:

    polymer_datasets = [
        "latitude",
        "longitude",
        "rho_w",
        "logchl",
        "logfb",
        "Rgli",
        "Rnir",
        "flags",
        "bands"
    ]


    datasets = {}

    dim_names_3d = ["y", "x", "band"]
    dim_names_2d = ["y", "x"]


    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        logger.info("Opening Polymer L2 NetCDF output file %s", nc_file_path)


        varnames = list(f.variables)

        #print(varnames)



        for polymer_dataset in polymer_datasets:

            logger.info("Loading %s", polymer_dataset)

            try:
                data = f[polymer_dataset][:]

                attrs = {a: getattr(f[polymer_dataset], a) for a in f[polymer_dataset].ncattrs()}


                if len(data.shape) == 2:
                    data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)
                if len(data.shape) == 3:
                    data = xr.DataArray(data, dims=dim_names_3d, attrs=attrs)

                datasets[polymer_dataset] = data


            except Exception:
                logger.warning("Unable to load %s", polymer_dataset)



        #rhot_vars = [v for v in varnames if v.startswith("rhot_")]
        #rhos_vars = [v for v in varnames if v.startswith("rhos_")]
        #other_vars = [v for v in varnames if not v.startswith(("rhos_", "rhot_"))]

        #refl_vars = sorted(rhos_vars)

        #print("[INFO] Accessing rhos reflectances (" + str(len(refl_vars)) + " bands)")

        #data = _load_reflectances_cube(f, refl_vars, dim_names_3d)

        #if data is not None:
        #    datasets['rhos'] = data

    return datasets




# For NetCDF output from Oct 2025 version of Polymer
def load_polymer_l2_v1_nc(nc_file_path: Path) -> dict:

    polymer_datasets = [
        "bands",
        "chla",
        "fb",
        "flags",
        "latitude",
        "longitude",
        "Rgli",
        "Rnir",
        "SPM",
    ]


    datasets = {}

    dim_names_3d = ["y", "x", "band"]
    dim_names_2d = ["y", "x"]


    with nc.Dataset(nc_file_path, format="NETCDF4") as f:

        logger.info("Opening Polymer L2 NetCDF output file %s", nc_file_path)


        varnames = list(f.variables)

        rho_w_varnames = sorted([var for var in varnames if var.startswith('rho_w_')])

        for polymer_dataset in polymer_datasets:

            logger.info("Loading %s", polymer_dataset)

            try:
                data = f[polymer_dataset][:]

                attrs = {a: getattr(f[polymer_dataset], a) for a in f[polymer_dataset].ncattrs()}


                if len(data.shape) == 2:
                    data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)
                if len(data.shape) == 3:
                    data = xr.DataArray(data, dims=dim_names_3d, attrs=attrs)

                datasets[polymer_dataset] = data


            except Exception:
                logger.warning("Unable to load %s", polymer_dataset)


        logger.info("Accessing rho_w reflectances (%d bands)", len(rho_w_varnames))

        data = _load_reflectances_cube_v1(f, rho_w_varnames, datasets['bands'], dim_names_3d)

        if data is not None:
            datasets['rho_w'] = data

    return datasets






def _load_reflectances_cube_v2(f, refl_vars, dim_names_3d):


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





def _load_reflectances_cube_v1(f, refl_vars, wavelengths, dim_names_3d):


    if len(refl_vars) == 0:
        logger.error("No datasets found!")
        return None

    height, width = np.array(f[refl_vars[0]][:], dtype='double').shape
    depth = len(refl_vars)

    data = np.empty((height,width,depth))

    #wavelengths = []
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
        #wavelengths.append(float(f[refl_var].wavelength))

    data = xr.DataArray(data, dims=dim_names_3d, coords={"band": wavelengths})
    data.attrs['per_band'] = per_band_attrs

    return data


def read_polymer_reflectance(nc_file_path) -> tuple:
    """(rho_w as a (y, x, bands) float array, band wavelengths in nm) from a
    Polymer output file, whichever way the build stores it: one 3D `rho_w`
    variable, or one `rho_w_<nm>` variable per band (what the v2 build
    writes for HYPSO - load_polymer_l2_v2_nc misses these). Bands are ordered
    by wavelength; wavelengths come from the file's `bands` variable when it
    matches the per-band names, else from the names."""
    with nc.Dataset(nc_file_path) as f:
        bands_var = np.asarray(f["bands"][:], dtype=float) if "bands" in f.variables else None
        if "rho_w" in f.variables and f["rho_w"].ndim == 3:
            cube = np.ma.filled(np.ma.asarray(f["rho_w"][:], dtype=float), np.nan)
            if f["rho_w"].dimensions[0] == "bands" or (bands_var is not None and cube.shape[0] == bands_var.size
                                                         and cube.shape[-1] != bands_var.size):
                cube = np.moveaxis(cube, 0, -1)
            return cube, bands_var
        names = [n for n in f.variables if re.fullmatch(r"rho_w_\d+", n)]
        if not names:
            raise ValueError(f"{nc_file_path}: no rho_w variable(s) in Polymer output")
        names.sort(key=lambda n: int(n.rsplit("_", 1)[1]))
        suffixes = np.array([int(n.rsplit("_", 1)[1]) for n in names], dtype=float)
        if bands_var is not None and bands_var.size == len(names) and np.array_equal(np.rint(np.sort(bands_var)), suffixes):
            wavelengths = np.sort(bands_var)
        else:
            wavelengths = suffixes
        cube = np.stack([np.ma.filled(np.ma.asarray(f[n][:], dtype=float), np.nan) for n in names], axis=-1)
        return cube, wavelengths
