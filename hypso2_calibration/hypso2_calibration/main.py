from . import _catalog

# Which coefficient file each correction uses, per version, and the versions
# each release selects - data/coefficients.yaml (data-only edits for new
# coefficients; see that file's header). Selection logic: _catalog.py.
CATALOG = _catalog.load("hypso2_calibration")

CORRECTIONS = ("radiometric", "smile", "spectral", "spectral_full_frame")


def resolve_coefficients(coeff_type: str = None, version=None) -> dict:
    """The coefficient selection for a coeff_type and version, without file
    paths: {"coeff_type", "release", "radiometric": "V3/abs", "smile": "V3",
    "spectral": "V2", "spectral_full_frame": "V2"}.

    :param coeff_type: radiometric variant ("abs", "moved", "lab");
        default CATALOG["default_coeff_type"]. The deprecated names
        "adjusted" and "original" map to (abs, V2) and (lab, V1).
    :param version: a release name ("V3", "V2", "V1"; default
        CATALOG["default_release"]), or {correction: version} to mix, e.g.
        {"radiometric": "V3", "spectral": "V3"}. Corrections missing from
        the dict come from the default release.
    """
    return _catalog.resolve(CATALOG, coeff_type, version)


def get_hypso2_calibration_files(capture_type=None, coeff_type=None, version=None) -> dict:
    """
    Get the absolute paths of the calibration coefficients included in the
    package (radiometric, smile, destriping, spectral). See
    resolve_coefficients for coeff_type and version; the defaults are
    abs / V3. capture_type selects a capture-type-specific file where the
    catalog has one (e.g. the wide matrix for V3 abs).

    :return: {"radiometric", "smile", "destriping", "spectral",
        "spectral_full_frame": path or None, "crop_modes": {kind: "as_is" |
        "crop_and_bin"} for files that declare one, "selection":
        resolve_coefficients(...) (plus "radiometric_capture_type" when a
        capture-type file was chosen)}.
    """
    return _catalog.calibration_files("hypso2_calibration", CATALOG, capture_type, coeff_type, version)


def optional_model(kind: str, version: str = None):
    """An optional model outside the releases (e.g. "keystone"): (path, entry)
    or (None, None) - see _catalog.optional_model."""
    return _catalog.optional_model("hypso2_calibration", CATALOG, kind, version)
