import numpy as np
from pathlib import Path
import netCDF4 as nc
import xarray as xr
import re



def load_ocsmart_h5(h5_file_path: Path) -> dict:

    #Wavelengths: 12, -2

    ocsmart_datasets = [
        "L2_flags",
        "Latitude",
        "Longitude",
        "Relative_azimuth",
        "Sensor_zenith",
        "Solar_zenith",
        "chlor_a(oci)",
        "chlor_a(yoc)",
        "tsm(yoc)"
    ]


    ocsmart_subgroups = [
        "Lr",
        "Lrc",
        "Lt",
        "Rrs"
    ]

    datasets = {}

    dim_names_3d = ["y", "x", "band"]
    dim_names_2d = ["y", "x"]

    # netCDF4 rather than h5py (2026-08-31) - OC-SMART's .h5 output is a
    # plain HDF5 file with no netCDF-specific features, and netCDF4 reads
    # it fine (top-level datasets as variables, subgroups as groups,
    # attrs via .ncattrs()) - verified against a real OC-SMART output.
    # This process already imports netCDF4 everywhere else (L1x/L2A
    # writing); h5py was the only thing pulling in a second, differently
    # versioned copy of the HDF5 C library into the same process, a known
    # cause of the native segfaults/heap corruption observed 2026-08-30/31
    # (moby_2025-01-08, tristandacunha) - see the AC pipeline's own
    # ac_runners_hypso.py comment on why this loader exists at all.
    with nc.Dataset(h5_file_path, "r") as f:

        print("[INFO] Opening OC-SMART HDF5 file " + str(h5_file_path))

        for ocsmart_dataset in ocsmart_datasets:

            print("[INFO] Loading " + str(ocsmart_dataset))

            try:
                var = f.variables[ocsmart_dataset]
                data = var[:]
                attrs = {attr: getattr(var, attr) for attr in var.ncattrs()}

                data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)

                datasets[ocsmart_dataset] = data


            except:
                print("[WARNING] Unable to load " + str(ocsmart_dataset))




        for ocsmart_subgroup in ocsmart_subgroups:

            ocsmart_subgroup_datasets = list(f.groups[ocsmart_subgroup].variables.keys())


            print("[INFO] Accessing subgroup " + str(ocsmart_subgroup) + " (" + str(len(ocsmart_subgroup_datasets)) + " bands)")

            height, width = np.array(f.groups[ocsmart_subgroup].variables[ocsmart_subgroup_datasets[0]][:], dtype='double').shape
            depth = len(ocsmart_subgroup_datasets)

            data = np.empty((height,width,depth))

            wavelengths = []

            for idx, ocsmart_subgroup_dataset in enumerate(ocsmart_subgroup_datasets):

                #print("[INFO] Loading " + str(ocsmart_subgroup_dataset))
                #print("[INFO] Loading band " + str(idx) + "...")

                try:
                    band = np.array(f.groups[ocsmart_subgroup].variables[ocsmart_subgroup_dataset][:], dtype='double')

                except:
                    print("[WARNING] Unable to load " + str(ocsmart_dataset))
                    break

                data[:,:,idx] = band

                wavelength = int(re.search(r"(\d+)", ocsmart_subgroup_dataset).group(1))
                wavelengths.append(wavelength)



            data = xr.DataArray(data, dims=dim_names_3d, coords={"band": wavelengths})
            #data.assign_coords(band=wavelengths)

            datasets[ocsmart_subgroup] = data


    return datasets

