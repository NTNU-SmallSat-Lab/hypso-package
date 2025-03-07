from pathlib import Path
from typing import Union

import netCDF4 as nc

from .hypso1 import Hypso1
from .hypso2 import Hypso2


def Hypso(path: Union[str, Path], verbose=False):

    try:
        with nc.Dataset(path, format="NETCDF4") as f:

            sat_id = getattr(f, "sat_id")

        if sat_id == "HYPSO-1":

            return Hypso1(path=path, verbose=verbose)


        elif sat_id == "HYPSO-2":

            return Hypso2(path=path, verbose=verbose)

        else:
            print("[ERROR] Unrecognized file.")

    except Exception as ex:
        print(ex)

    return None

    
