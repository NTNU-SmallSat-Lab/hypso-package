"""Sensor-profile registry.

Replaces the old pattern of one hand-written HypsoCapture subclass per sensor
(hypso1.py/hypso2.py each duplicating the same ~50-line __init__ just to set
different instrument constants) with a single generic descriptor,
SensorProfile, plus a registry keyed by the file's own sat_id attribute.

Adding a future sensor means adding one profile module (see hypso1.py/
hypso2.py in this package for the pattern) and registering it here - no new
HypsoCapture subclass is required, though one may still be written for
isinstance()-style identity if that's useful to callers (see hypso.hypso1/
hypso.hypso2, which now do exactly that: a ~10-line subclass that just
forwards the right profile to HypsoCapture).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SensorProfile:
    """Everything HypsoCapture needs to know about one sensor, previously
    hardcoded per-subclass. See REFACTOR_PROGRESS.md / ARCHITECTURE_PROPOSAL.md
    for how this list was derived (every self.<attr> HypsoCapture reads that a
    subclass, not HypsoCapture itself, used to be responsible for setting).

    calibration_files is the pluggable replacement for the old
    Hypso1/Hypso2._set_calibration_coeff_files override: given
    (capture_type, coeff_type), it must return a dict with keys
    "radiometric", "smile", "destriping", "spectral" - each an absolute path
    to a calibration coefficient file. Today this is always
    get_hypsoN_calibration_files from the matching hypsoN_calibration sibling
    package, wrapped in a small adapter (see hypso1.py/hypso2.py in this
    package) so its (capture_type, coeff_type=...) signature matches this
    Callable's shape exactly.
    """
    key: str
    sat_id: str
    sensor: str
    platform: str
    fwhm: np.ndarray
    fwhm_lookup_wl: np.ndarray
    fwhm_lookup_fwhm: np.ndarray
    calibration_files: Callable[[str, str], dict]
    #: Ordered (capture_type, satobj_attr, expected_value) rules used to
    #: classify a loaded capture - the first rule whose getattr(satobj,
    #: satobj_attr) == expected_value wins; no match falls back to "custom".
    #: Drives what calibration_files(capture_type, ...) above receives (see
    #: io/dispatch.py's check_capture_type). Previously one hardcoded,
    #: sensor-agnostic if/elif chain shared by every sensor regardless of
    #: which SensorProfile was active - moved here so a future sensor with
    #: different frame/image dimensions declares its own thresholds instead
    #: of silently matching (or failing to match) another sensor's.
    capture_type_thresholds: tuple[tuple[str, str, int], ...]


_REGISTRY: dict[str, SensorProfile] = {}


def register_sensor(profile: SensorProfile) -> None:
    """Register a SensorProfile under both its sat_id (e.g. "HYPSO-2", the
    key files are actually looked up by - see hypso.Hypso's factory) and its
    short key (e.g. "hypso2"), so callers can look it up either way."""
    _REGISTRY[profile.sat_id] = profile
    _REGISTRY[profile.key] = profile


def get_sensor_profile(sat_id: str) -> SensorProfile:
    """Look up a registered SensorProfile by sat_id (as read from a
    capture's NetCDF attrs, e.g. "HYPSO-2") or by its short key
    (e.g. "hypso2"). Raises KeyError with the list of known sensors if
    sat_id isn't registered - this is the error a future sensor's
    capture file would hit before its profile module is written and
    imported."""
    try:
        return _REGISTRY[sat_id]
    except KeyError:
        known = sorted({p.sat_id for p in _REGISTRY.values()})
        raise KeyError(
            f"No SensorProfile registered for {sat_id!r}. Known sensors: {known}. "
            f"See hypso/sensors/hypso1.py or hypso2.py for how to add a new one."
        ) from None


def registered_sensors() -> list[SensorProfile]:
    """Every distinct registered profile (deduplicated - each is registered
    under two keys, see register_sensor)."""
    seen = {}
    for profile in _REGISTRY.values():
        seen[profile.sat_id] = profile
    return list(seen.values())


# Import side effects register HYPSO-1/HYPSO-2 on package import.
from . import hypso1 as _hypso1  # noqa: E402,F401
from . import hypso2 as _hypso2  # noqa: E402,F401
