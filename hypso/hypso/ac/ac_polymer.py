import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path


def ac_polymer_srf_getter(srf_nc_path: Path, ):

    ds = xr.open_dataset(srf_nc_path)

    print("Reached ac_polymer_srf_getter")
    print(ds)
    # TODO: load NetCDF

    return ds