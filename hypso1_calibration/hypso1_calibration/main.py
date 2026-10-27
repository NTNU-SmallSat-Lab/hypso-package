from . import _catalog

# Which coefficient file each correction uses, per version and capture type,
# and the versions each release selects - data/coefficients.yaml (data-only
# edits for new coefficients; see that file's header). Selection logic:
# _catalog.py, shared with hypso2_calibration.
CATALOG = _catalog.load("hypso1_calibration")


def resolve_coefficients(coeff_type: str = None, version=None) -> dict:
    """The coefficient selection for a coeff_type and version, without file
    paths (see _catalog.resolve). coeff_type: "moved" (default), "abs" or
    "lab"; the deprecated names "adjusted" and "original" map to (abs, V2)
    and (lab, V1). version: "V2" (default), "V1", or {correction: version}."""
    return _catalog.resolve(CATALOG, coeff_type, version)


def get_hypso1_calibration_files(capture_type="custom", coeff_type=None, version=None) -> dict:
    """
    Get the absolute paths of the calibration coefficients included in the
    package (radiometric, smile, destriping, spectral) for a capture type
    ("nominal", "wide", "custom", "moon"). See resolve_coefficients for
    coeff_type and version; the defaults are moved / V2.

    :return: {"radiometric", "smile", "destriping", "spectral",
        "spectral_full_frame": path or None, "crop_modes", "selection"}.
    """
    return _catalog.calibration_files("hypso1_calibration", CATALOG, capture_type, coeff_type, version)


def optional_model(kind: str, version: str = None):
    """An optional model outside the releases (e.g. "keystone"): (path, entry)
    or (None, None) - see _catalog.optional_model."""
    return _catalog.optional_model("hypso1_calibration", CATALOG, kind, version)
