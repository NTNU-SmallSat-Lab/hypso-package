import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple
import xarray as xr
import re


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

        print("[INFO] Opening Polymer L2 NetCDF output file " + str(nc_file_path))


        varnames = list(f.variables) 

        #print(varnames)



        for polymer_dataset in polymer_datasets:

            print("[INFO] Loading " + str(polymer_dataset))

            try:
                data = f[polymer_dataset][:]

                attrs = {a: getattr(f[polymer_dataset], a) for a in f[polymer_dataset].ncattrs()}


                if len(data.shape) == 2:
                    data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)
                if len(data.shape) == 3:
                    data = xr.DataArray(data, dims=dim_names_3d, attrs=attrs)

                datasets[polymer_dataset] = data


            except:
                print("[WARNING] Unable to load " + str(polymer_dataset))



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

        print("[INFO] Opening Polymer L2 NetCDF output file " + str(nc_file_path))


        varnames = list(f.variables) 

        rho_w_varnames = sorted([var for var in varnames if var.startswith('rho_w_')])

        for polymer_dataset in polymer_datasets:

            print("[INFO] Loading " + str(polymer_dataset))

            try:
                data = f[polymer_dataset][:]

                attrs = {a: getattr(f[polymer_dataset], a) for a in f[polymer_dataset].ncattrs()}


                if len(data.shape) == 2:
                    data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)
                if len(data.shape) == 3:
                    data = xr.DataArray(data, dims=dim_names_3d, attrs=attrs)

                datasets[polymer_dataset] = data


            except:
                print("[WARNING] Unable to load " + str(polymer_dataset))


        print("[INFO] Accessing rhos reflectances (" + str(len(rho_w_varnames)) + " bands)")

        data = _load_reflectances_cube_v1(f, rho_w_varnames, datasets['bands'], dim_names_3d)

        if data is not None:
            datasets['rho_w'] = data

    return datasets







def _load_reflectances_cube_v2(f, refl_vars, dim_names_3d):


    if len(refl_vars) == 0:
        print("[ERROR] No datasets found!")
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

        except:
            print("[WARNING] Unable to load " + str(refl_var))
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
        print("[ERROR] No datasets found!")
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

        except:
            print("[WARNING] Unable to load " + str(refl_var))
            break

        data[:,:,idx] = band

        per_band_attrs[idx] = attrs

        #print(f[refl_var].ncattrs())
        #wavelengths.append(float(f[refl_var].wavelength))

    data = xr.DataArray(data, dims=dim_names_3d, coords={"band": wavelengths})
    data.attrs['per_band'] = per_band_attrs

    return data
