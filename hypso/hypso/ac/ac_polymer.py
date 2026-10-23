import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path


def ac_polymer_srf_getter(srf_nc_path: Path):

    ds = xr.open_dataset(srf_nc_path)

    return ds


def ac_polymer_srf_getter_from_l1(l1_nc_path: Path):
    """
    Polymer srf_getter reading the per-band SRFs (Band_<i> with their own
    wav_Band_<i> coordinate) from the L1 file's own metadata/srf group.

    Returns only the Band_* variables, ordered by band index: Polymer's
    get_SRF(rename_method='bands') asserts the dataset has exactly one
    variable per band, so the group's esun/effective_fwhm/csiro_* variables
    must not be passed on.

    Files written before the group held SRFs fall back to the single
    <sensor>_srf.nc next to the L1 file (ac_polymer_srf_getter's source).
    """
    l1_nc_path = Path(l1_nc_path)

    with xr.open_dataset(l1_nc_path, group="metadata/srf") as group:
        bands = [v for v in group.data_vars if v.startswith("Band_")]
        if bands:
            bands = sorted(bands, key=lambda v: int(group[v].attrs["index"]))
            srf = group[bands].load()
            srf.attrs = {"desc": f"Spectral response functions from {l1_nc_path.name} (metadata/srf)"}
            return srf

    siblings = sorted(l1_nc_path.parent.glob("*_srf.nc"))
    if len(siblings) != 1:
        raise FileNotFoundError(f"No Band_* SRFs in {l1_nc_path} metadata/srf and "
                                f"{len(siblings)} *_srf.nc files next to it (expected 1)")
    print(f"[WARNING] No SRFs in {l1_nc_path.name} metadata/srf; using {siblings[0].name}")
    return ac_polymer_srf_getter(siblings[0])
