import logging
import numpy as np
from pathlib import Path
import netCDF4 as nc
import xarray as xr
import re

logger = logging.getLogger(__name__)



def load_ocsmart_h5(h5_file_path: Path) -> dict:
    """Read an OC-SMART L2 HDF5 file: everything it holds.

    Returns
    -------
    dict
        ``{variable name: xarray.DataArray}``: the 2D root variables
        (``L2_flags``, ``chlor_a(oci)``, geometry, ...) as ``(y, x)``, and
        each per-band group (``Rrs``, ``Lt``, ``Lr``, ``Lrc``, ``Lwp``,
        ``tg_sol``, ``tg_sen``, whatever ``l2_prod`` asked for) as a float32
        ``(y, x, band)`` cube on its band wavelengths (``band``: nm, from the
        ``<name>_<nm>nm`` dataset names).
    """
    datasets = {}
    # netCDF4 rather than h5py (production 89ac5e46): h5py pulled a second,
    # differently versioned HDF5 C library into the same process, a known cause
    # of native segfaults (2026-08-30/31).
    with nc.Dataset(h5_file_path, "r") as f:
        logger.info("Opening OC-SMART HDF5 file %s", h5_file_path)
        for name, var in f.variables.items():
            datasets[name] = xr.DataArray(var[:], dims=["y", "x"][:var.ndim],
                                          attrs={attr: getattr(var, attr) for attr in var.ncattrs()})
        for group_name, group in f.groups.items():
            names = [n for n in group.variables if re.search(r"_(\d+)nm$", n)]
            if not names:
                logger.warning("OC-SMART group %s has no <name>_<nm>nm datasets; skipped", group_name)
                continue
            names.sort(key=lambda n: int(re.search(r"_(\d+)nm$", n).group(1)))
            cube = np.stack([np.ma.filled(np.ma.asarray(group.variables[n][:], dtype=np.float32), np.nan)
                             for n in names], axis=-1)
            wavelengths = [int(re.search(r"_(\d+)nm$", n).group(1)) for n in names]
            datasets[group_name] = xr.DataArray(cube, dims=["y", "x", "band"], coords={"band": wavelengths})
    return datasets
