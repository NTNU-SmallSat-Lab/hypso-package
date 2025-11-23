import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Tuple
import h5py
import xarray as xr
import re



def load_ocsmart_h5(h5_file_path: Path) -> Tuple[np.ndarray, dict]:

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
        
        for name, obj in f.items():
            if isinstance(obj, h5py.Group):
                print(name, "→ subgroup")
            elif isinstance(obj, h5py.Dataset):
                print(name, "→ variable (dataset)")


        for ocsmart_dataset in ocsmart_datasets:

            print("[INFO] Loading " + str(ocsmart_dataset))

            try:
                data = f[ocsmart_dataset][:]

                data = xr.DataArray(data, dims=dim_names_2d)

                datasets[ocsmart_dataset] = data

            except:
                print("[WARNING] Unable to load " + str(ocsmart_dataset))




        for ocsmart_subgroup in ocsmart_subgroups:

            ocsmart_subgroup_datasets = list(f[ocsmart_subgroup].keys())


            print("[INFO] Accessing subgroup " + str(ocsmart_subgroup) + " (" + str(len(ocsmart_subgroup_datasets)) + " bands)")

            height, width = np.array(f[ocsmart_subgroup][ocsmart_subgroup_datasets[0]][:], dtype='double').shape
            depth = len(list(f[ocsmart_subgroup].keys()))

            data = np.empty((height,width,depth))

            wavelengths = []

            for idx, ocsmart_subgroup_dataset in enumerate(ocsmart_subgroup_datasets):

                #print("[INFO] Loading " + str(ocsmart_subgroup_dataset))
                #print("[INFO] Loading band " + str(idx) + "...")

                try:
                    band = np.array(f[ocsmart_subgroup][ocsmart_subgroup_dataset][:], dtype='double')
                    
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


#h5_file_path = "/home/cameron/Dokumenter/aeronetvenice_2025-06-22T10-46-15Z/HYPSO2_HSI_aeronetvenice_2025-06-22T10-46-15Z-l1d_L2_OCSMART.h5"
#datasets = load_ocsmart_h5(h5_file_path)
#print(datasets["Rrs"])
