import logging
import numpy as np
from pathlib import Path
import netCDF4 as nc
import xarray as xr
import re

logger = logging.getLogger(__name__)



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


    # netCDF4 rather than h5py (production 89ac5e46) - OC-SMART's .h5 output
    # is a plain HDF5 file with no netCDF-specific features, and netCDF4
    # reads it fine (top-level datasets as variables, subgroups as groups,
    # attrs via .ncattrs()). This process already imports netCDF4
    # everywhere else; h5py was the only thing pulling in a second,
    # differently versioned copy of the HDF5 C library into the same
    # process, a known cause of the native segfaults/heap corruption
    # observed 2026-08-30/31 (moby_2025-01-08, tristandacunha).
    with nc.Dataset(h5_file_path, "r") as f:

        logger.info("Opening OC-SMART HDF5 file %s", h5_file_path)

        for ocsmart_dataset in ocsmart_datasets:

            logger.info("Loading %s", ocsmart_dataset)

            try:
                var = f.variables[ocsmart_dataset]
                data = var[:]
                attrs = {attr: getattr(var, attr) for attr in var.ncattrs()}

                data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)

                datasets[ocsmart_dataset] = data


            except Exception:
                logger.warning("Unable to load %s", ocsmart_dataset)




        for ocsmart_subgroup in ocsmart_subgroups:

            ocsmart_subgroup_datasets = list(f.groups[ocsmart_subgroup].variables.keys())


            logger.info("Accessing subgroup %s (%d bands)", ocsmart_subgroup, len(ocsmart_subgroup_datasets))

            height, width = np.array(f.groups[ocsmart_subgroup].variables[ocsmart_subgroup_datasets[0]][:], dtype='double').shape
            depth = len(ocsmart_subgroup_datasets)

            data = np.empty((height,width,depth))

            wavelengths = []

            for idx, ocsmart_subgroup_dataset in enumerate(ocsmart_subgroup_datasets):

                #print("[INFO] Loading " + str(ocsmart_subgroup_dataset))
                #print("[INFO] Loading band " + str(idx) + "...")

                try:
                    band = np.array(f.groups[ocsmart_subgroup].variables[ocsmart_subgroup_dataset][:], dtype='double')

                except Exception:
                    # was str(ocsmart_dataset) - a leftover outer-loop variable
                    # name, not the dataset actually being loaded here
                    logger.warning("Unable to load %s", ocsmart_subgroup_dataset)
                    break

                data[:,:,idx] = band

                wavelength = int(re.search(r"(\d+)", ocsmart_subgroup_dataset).group(1))
                wavelengths.append(wavelength)



            data = xr.DataArray(data, dims=dim_names_3d, coords={"band": wavelengths})
            #data.assign_coords(band=wavelengths)

            datasets[ocsmart_subgroup] = data


    return datasets

