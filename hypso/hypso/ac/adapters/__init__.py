"""Atmospheric-correction adapter registry (self.ac composition - part of the
HypsoCapture breakup called for in the approved refactor plan, see
REFACTOR_PROGRESS.md).

One adapter class per external AC tool (Polymer/ACOLITE/OC-SMART), behind the
shared ACAdapter run_correction/open_output interface (base.py), registered the
same way sensors are (hypso.sensors). This pass is organizational, not a
rewrite: every method body is the corresponding HypsoCapture ac_* method body
relocated verbatim, and HypsoCapture keeps every public ac_* name as a thin
delegating wrapper (e.g. ac_polymer_run_correction ->
self.ac.polymer.run_correction(self, ...)), so nothing external changes.
The seam this buys: a future rewrite of one tool's internals now has an
isolated target that touches neither the other tools nor HypsoCapture.

Dark-pixel subtraction is not an ACAdapter: it has no external tool to run or
output file to open (it computes in-memory from the L1D cube directly), so it
doesn't implement run_correction/open_output and isn't in _REGISTRY below -
see hypso/ac/ac_dark_pixel_subtraction.py. It's still exposed on the same
self.ac namespace as a bare function (self.ac.dark_pixel_subtraction(satobj,
...)) purely for call-site naming consistency with self.ac.polymer/.acolite/
.ocsmart - HypsoCapture.ac_dark_pixel_subtraction is a thin delegating
wrapper over it, matching the ac_polymer_run_correction -> self.ac.polymer.
run_correction(self, ...) pattern used for the real adapters.

Adding a future AC tool means adding one adapter module here and registering
its instance below.
"""
from types import SimpleNamespace

from .base import ACAdapter, ACRunError, get_inferred_wavelength_band_map, run_subprocess_driver
from .polymer import PolymerAdapter
from .acolite import ACOLITEAdapter
from .ocsmart import OCSMARTAdapter
from ..ac_dark_pixel_subtraction import ac_dark_pixel_subtraction

# Stateless singletons (all per-capture state lives on the satobj passed to
# every call), shared safely between captures - see ACAdapter's docstring.
POLYMER_ADAPTER = PolymerAdapter()
ACOLITE_ADAPTER = ACOLITEAdapter()
OCSMART_ADAPTER = OCSMARTAdapter()

_REGISTRY: dict[str, ACAdapter] = {
    adapter.key: adapter
    for adapter in (POLYMER_ADAPTER, ACOLITE_ADAPTER, OCSMART_ADAPTER)
}

# What HypsoCapture exposes as self.ac: attribute access per tool
# (satobj.ac.polymer.run_correction(satobj, ...)), plus dark_pixel_subtraction
# as a bare function (see module docstring for why it's not in _REGISTRY).
AC_ADAPTERS = SimpleNamespace(dark_pixel_subtraction=ac_dark_pixel_subtraction, **_REGISTRY)


def get_ac_adapter(key: str) -> ACAdapter:
    """Look up a registered adapter by key ("polymer"/"acolite"/"ocsmart").
    Raises KeyError with the list of known tools if key isn't registered."""
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"No AC adapter registered for {key!r}. Known tools: {sorted(_REGISTRY)}. "
            f"See hypso/ac/adapters/polymer.py etc. for how to add a new one."
        ) from None


def registered_ac_adapters() -> list[ACAdapter]:
    return list(_REGISTRY.values())
