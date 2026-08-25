from pathlib import Path
from typing import Union

from .HypsoBase import HypsoBase
from .sensors.hypso2 import HYPSO2_PROFILE, get_hypso2_wavelengths  # noqa: F401 (re-exported)


class Hypso2(HypsoBase):
    """HYPSO-2 capture. Thin subclass over HypsoBase + HYPSO2_PROFILE (see
    hypso.sensors) - kept for named-class/isinstance() compatibility and as
    the documented entry point for HYPSO-2 captures; all the sensor-specific
    data (fwhm, srf_wl/srf_fwhm, calibration-file resolver) that used to live
    here now lives in the profile instead."""

    def __init__(self, path: Union[str, Path], label: str = None, load_cube: bool = True, verbose: bool = False) -> None:
        super().__init__(path=path, sensor_profile=HYPSO2_PROFILE, label=label, load_cube=load_cube, verbose=verbose)
