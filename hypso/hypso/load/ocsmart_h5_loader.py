import logging
import numpy as np
from pathlib import Path
import h5py
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


    with h5py.File(h5_file_path, "r") as f:
        
        logger.info("Opening OC-SMART HDF5 file %s", h5_file_path)

        if False:
            for name, obj in f.items():
                if isinstance(obj, h5py.Group):
                    logger.debug("%s -> subgroup", name)
                elif isinstance(obj, h5py.Dataset):
                    logger.debug("%s -> variable (dataset)", name)


        for ocsmart_dataset in ocsmart_datasets:

            logger.info("Loading %s", ocsmart_dataset)

            try:
                data = f[ocsmart_dataset][:]

                # f[ocsmart_dataset].ncattrs() doesn't exist on h5py objects
                # (that's a netCDF4 API method) - this file is opened with
                # h5py.File, whose attribute dict is .attrs. The old
                # .ncattrs() call raised AttributeError unconditionally here,
                # silently discarding every one of these datasets (caught by
                # the bare except below) even though they're actually
                # present in the file.
                attrs = dict(f[ocsmart_dataset].attrs)

                data = xr.DataArray(data, dims=dim_names_2d, attrs=attrs)

                datasets[ocsmart_dataset] = data


            except Exception:
                logger.warning("Unable to load %s", ocsmart_dataset)




        for ocsmart_subgroup in ocsmart_subgroups:

            ocsmart_subgroup_datasets = list(f[ocsmart_subgroup].keys())


            logger.info("Accessing subgroup %s (%d bands)", ocsmart_subgroup, len(ocsmart_subgroup_datasets))

            height, width = np.array(f[ocsmart_subgroup][ocsmart_subgroup_datasets[0]][:], dtype='double').shape
            depth = len(list(f[ocsmart_subgroup].keys()))

            data = np.empty((height,width,depth))

            wavelengths = []

            for idx, ocsmart_subgroup_dataset in enumerate(ocsmart_subgroup_datasets):

                #print("[INFO] Loading " + str(ocsmart_subgroup_dataset))
                #print("[INFO] Loading band " + str(idx) + "...")

                try:
                    band = np.array(f[ocsmart_subgroup][ocsmart_subgroup_dataset][:], dtype='double')

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

