import logging
import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)


def ac_polymer_srf_getter(srf_nc_path: Path):
    """Polymer ``srf_getter`` reading HYPSO's spectral response functions
    from a standalone SRF NetCDF file. See also
    :func:`ac_polymer_srf_getter_from_l1`, which reads them from the L1
    product's ``metadata/srf``."""

    ds = xr.open_dataset(srf_nc_path)

    return ds


# The dotted path Polymer's eotools.srf.get_SRF resolves at runtime via
# importlib.import_module + getattr (a string-based plugin hook, not a normal
# Python import - see PolymerAdapter/_polymer_driver.py's docstrings for the
# full mechanism). Derived from the function object itself, not hand-typed,
# so this string can never silently drift out of sync with
# ac_polymer_srf_getter's actual module/name if either ever changes -
# every caller (the driver, tests) should import and use this constant
# instead of retyping the string.
SRF_GETTER_PATH = f"{ac_polymer_srf_getter.__module__}.{ac_polymer_srf_getter.__qualname__}"


def ac_polymer_srf_getter_from_l1(l1_nc_path: Path):
    """
    Polymer srf_getter reading the per-band SRFs (Band_<i> with their own
    wav_Band_<i> coordinate) from the L1 file's own metadata/srf group.

    Returns only the Band_* variables, ordered by band index: Polymer's
    get_SRF(rename_method='bands') asserts the dataset has exactly one
    variable per band, so the group's F0/effective_fwhm variables must
    not be passed on.

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
    logger.warning(f"No SRFs in {l1_nc_path.name} metadata/srf; using {siblings[0].name}")
    return ac_polymer_srf_getter(siblings[0])


# Same derivation as SRF_GETTER_PATH, for the L1-file getter the Polymer
# driver actually passes (2026-10-23).
SRF_GETTER_FROM_L1_PATH = f"{ac_polymer_srf_getter_from_l1.__module__}.{ac_polymer_srf_getter_from_l1.__qualname__}"
