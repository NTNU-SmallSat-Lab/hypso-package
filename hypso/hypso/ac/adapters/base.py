"""Shared interface for atmospheric-correction tool adapters (see this
subpackage's __init__.py docstring). Also holds get_inferred_wavelength_band_map,
the wavelength-matching helper every adapter's open_output path uses to map a
tool's output bands back onto HYPSO band indices - previously
HypsoBase._get_inferred_wavelength_band_map (zero external callers, confirmed by
grep, so it moved here outright with no wrapper kept on HypsoBase)."""
import numpy as np


class ACAdapter:
    """One adapter per external atmospheric-correction tool, behind a shared
    run_correction/open_output interface. This pass is *organizational* (the
    approved plan's "prepare the AC functions to be refactored"): every method
    body is today's HypsoBase method body relocated verbatim - same subprocess/
    sys.path/external-tool-parsing logic, not rewritten - so a future rewrite of
    one tool's internals has a clean, isolated target that doesn't touch the
    other tools or HypsoBase.

    Adapters are stateless (all per-capture state lives on the satobj passed to
    every call), so the module-level instances in this subpackage are shared
    safely between captures.
    """

    #: Registry key, e.g. "polymer" - also the l2a_cube correction key convention.
    key: str = None

    def run_correction(self, satobj, **kwargs):
        """Run the external AC tool on this capture's L1 product."""
        raise NotImplementedError

    def open_output(self, satobj, **kwargs):
        """Read the tool's output file(s) into satobj.l2a_cube."""
        raise NotImplementedError


def get_inferred_wavelength_band_map(satobj, inferred_wavelengths):

    # Map inferred wavelengths to HYPSO wavelengths
    A = np.array(inferred_wavelengths, dtype=float)
    B = np.array(satobj.wavelengths, dtype=float)

    index_map = {}
    indices_unique = []

    for a in A:
        ix = np.argmin(np.abs(B - a))
        if ix not in index_map: # ensure uniqueness
            index_map[ix] = a
            indices_unique.append(ix)
        else:
            print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

    wl_band_map = np.array(indices_unique, dtype=int)


    return wl_band_map
