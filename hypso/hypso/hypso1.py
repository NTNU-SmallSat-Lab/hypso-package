from pathlib import Path
from typing import Union

from .HypsoBase import HypsoBase
from .sensors.hypso1 import HYPSO1_PROFILE, get_hypso1_wavelengths  # noqa: F401 (re-exported)


class Hypso1(HypsoBase):
    """HYPSO-1 capture. Thin subclass over HypsoBase + HYPSO1_PROFILE (see
    hypso.sensors) - kept for named-class/isinstance() compatibility and as
    the documented entry point for HYPSO-1 captures; all the sensor-specific
    data (fwhm, fwhm_lookup_wl/fwhm_lookup_fwhm, calibration-file resolver) that used to live
    here now lives in the profile instead."""

    def __init__(self, path: Union[str, Path], label: str = None, load_cube: bool = True, verbose: bool = False) -> None:
        super().__init__(path=path, sensor_profile=HYPSO1_PROFILE, label=label, load_cube=load_cube, verbose=verbose)
