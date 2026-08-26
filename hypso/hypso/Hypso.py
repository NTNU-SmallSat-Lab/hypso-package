import logging
from pathlib import Path
from typing import Union

import netCDF4 as nc

from .hypso1 import Hypso1
from .hypso2 import Hypso2

logger = logging.getLogger(__name__)


def Hypso(path: Union[str, Path], label: str = None, load_cube: bool = True, verbose=False):

    try:
        with nc.Dataset(path, format="NETCDF4") as f:

            sat_id = getattr(f, "sat_id")

        if sat_id == "HYPSO-1":

            return Hypso1(path=path, label=label, verbose=verbose, load_cube=load_cube)


        elif sat_id == "HYPSO-2":

            return Hypso2(path=path, label=label, verbose=verbose, load_cube=load_cube)

        else:
            logger.error("Unrecognized file.")

    except Exception:
        logger.exception("Failed to open capture file.")

    return None

    
