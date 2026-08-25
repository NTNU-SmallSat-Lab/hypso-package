import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path


def ac_polymer_srf_getter(srf_nc_path: Path):

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