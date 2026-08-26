"""Registry for custom calibration coefficient sets, keyed by sensor + name.

Lets a user plug in their own radiometric/smile/destriping/spectral coefficient
files without editing the bundled hypso1_calibration/hypso2_calibration packages
(see hypso.sensors.hypso1._calibration_files / hypso.sensors.hypso2._calibration_files,
which wrap those bundled resolvers). Register once, then pass coeff_type=<name>
anywhere a built-in coeff_type ("moved"/"adjusted"/"original") is accepted today
- HypsoCapture._set_calibration_coeff_files checks this registry before falling
back to the sensor's built-in presets. For a true one-off set (no reuse), pass
coeff_files=... directly to generate_l1b_cube/_set_calibration_coeff_files
instead of registering anything.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

COEFF_FILE_KEYS = ("radiometric", "smile", "destriping", "spectral", "spectral_full_frame")

_CUSTOM_COEFFS: dict = {}


def register_calibration_coeffs(sat_id: str, name: str, files: dict) -> None:
    """Register a named, reusable set of calibration coefficient files for a sensor.

    :param sat_id: sensor identifier this set applies to, e.g. "HYPSO-1" (matches
        satobj.sat_id / SensorProfile.sat_id).
    :param name: the coeff_type name this set will be selected under, e.g.
        "my_lab_calibration". Must not collide with a sensor's own built-in
        coeff_type names ("moved"/"adjusted"/"original") - those are resolved
        first and would shadow a custom set registered under the same name.
    :param files: dict supplying a path (str/Path) for at least one of
        "radiometric", "smile", "destriping", "spectral", "spectral_full_frame".
        Missing keys default to None (that correction stage is then skipped,
        same as the built-in resolvers already do for a capture_type that
        doesn't need e.g. destriping).
    """
    unknown = set(files) - set(COEFF_FILE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown calibration file key(s) {sorted(unknown)}; expected a subset of {COEFF_FILE_KEYS}"
        )
    resolved = {key: files.get(key) for key in COEFF_FILE_KEYS}
    logger.info("Registered custom calibration coefficient set %r for sensor %r", name, sat_id)
    _CUSTOM_COEFFS[(sat_id, name)] = resolved


def get_custom_calibration_coeffs(sat_id: str, name: str) -> Optional[dict]:
    """Look up a previously-registered custom coefficient set, or None if
    `name` isn't a registered custom set for this sensor (caller should then
    fall back to the sensor's built-in coeff_type presets)."""
    return _CUSTOM_COEFFS.get((sat_id, name))


def registered_calibration_coeffs() -> list:
    """List (sat_id, name) pairs for every currently-registered custom set."""
    return list(_CUSTOM_COEFFS.keys())
