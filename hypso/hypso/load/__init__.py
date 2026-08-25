# load_l1a_nc stays on its own loader (raw ground-segment input, not written by
# this package - see hypso.io.reader's module docstring for why). load_l1b_nc/
# l1c_nc/l1d_nc/l2a_nc come from hypso.io.reader (the schema-driven reader that
# replaced this package's old per-level loader files, since deleted).
# Names/signatures unchanged.
from .l1a_nc_loader import load_l1a_nc, load_l1a_nc_cube, load_l1a_nc_metadata
from .ocsmart_h5_loader import load_ocsmart_h5
from .acolite_l2_nc_loader import load_acolite_l2r_nc, load_acolite_l2w_nc
from .polymer_l2_nc_loader import load_polymer_l2_v1_nc, load_polymer_l2_v2_nc

# Resolved on first attribute access rather than at import time (PEP 562) because
# hypso.io.reader imports hypso.load.utils, so an eager `from hypso.io.reader
# import ...` here makes the two packages mutually import-dependent: whichever of
# hypso.load / hypso.io is imported second fails with a partially-initialized
# module. Deferring it makes the import order irrelevant. Names/signatures
# unchanged for callers - `from hypso.load import load_l1b_nc` still works.
_IO_READER_NAMES = frozenset({"load_l1b_nc", "load_l1c_nc", "load_l1d_nc", "load_l2a_nc"})


def __getattr__(name):
    if name in _IO_READER_NAMES:
        from hypso.io import reader
        return getattr(reader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")