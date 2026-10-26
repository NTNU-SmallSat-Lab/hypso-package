"""HYPSO-2 versioned calibration coefficients (hypso2_calibration
data/coefficients.yaml, REFACTOR_PROGRESS.md ×48)."""
from pathlib import Path

import pytest

from hypso.calibration.manifest import sha256_file
from hypso2_calibration import get_hypso2_calibration_files
from hypso2_calibration.main import CATALOG, resolve_coefficients


def _names(files):
    return {k: (Path(str(v)).name if v is not None else None) for k, v in files.items()
            if k not in ("selection", "crop_modes")}


def test_default_is_abs_v3_with_v2_spectral():
    files = get_hypso2_calibration_files()
    assert _names(files) == {
        "radiometric": "h2_radiometric_calibration_matrix_full_v3_abs.npz",
        "smile": "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz",
        "destriping": None,
        "spectral": "spectral_array_calibrated_poly_full_v2.npz",   # V2: OC-SMART compatibility
        "spectral_full_frame": "spectral_array_calibrated_poly_full_v2.npz",
    }
    assert files["selection"] == {"coeff_type": "abs", "release": "V3", "radiometric": "V3/abs",
                                  "smile": "V3", "spectral": "V2", "spectral_full_frame": "V2"}


@pytest.mark.parametrize("coeff_type, version, radiometric, smile, spectral", [
    ("moved", "V3", "h2_radiometric_calibration_matrix_full_moved_v3.npz",
     "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz", "spectral_array_calibrated_poly_full_v2.npz"),
    ("moved", "V2", "h2_radiometric_calibration_matrix_full_moved_v2.npz",
     "smile_correction_matrix_HYPSO-2_full_v2.npz", "spectral_array_calibrated_poly_full_v2.npz"),
    ("abs", "v2", "h2_radiometric_calibration_matrix_full_v2_abs.npz",
     "smile_correction_matrix_HYPSO-2_full_v2.npz", "spectral_array_calibrated_poly_full_v2.npz"),
    ("lab", "V1", "h2_radiometric_calibration_matrix_full_v1_lab.npz",
     "smile_correction_matrix_HYPSO-2_full_v2.npz", "h2_spectral_calibration_wavelengths_center_row_v1.npz"),
])
def test_releases(coeff_type, version, radiometric, smile, spectral):
    names = _names(get_hypso2_calibration_files(coeff_type=coeff_type, version=version))
    assert (names["radiometric"], names["smile"], names["spectral"]) == (radiometric, smile, spectral)


def test_mixing_versions():
    with pytest.warns(UserWarning, match="renames integer-wavelength bands"):
        files = get_hypso2_calibration_files(coeff_type="moved", version={"spectral": "V3"})
    names = _names(files)
    assert names["radiometric"] == "h2_radiometric_calibration_matrix_full_moved_v3.npz"
    assert names["spectral"] == "spectral_array_calibrated_poly_full_v3.npz"
    assert files["selection"]["release"] == "custom"
    # V3 radiometric with every other correction from V2
    files = get_hypso2_calibration_files(version={"radiometric": "V3", "smile": "V2"})
    assert _names(files)["smile"] == "smile_correction_matrix_HYPSO-2_full_v2.npz"
    assert files["selection"]["radiometric"] == "V3/abs"


@pytest.mark.parametrize("alias, coeff_type, release", [("adjusted", "abs", "V2"), ("original", "lab", "V1")])
def test_deprecated_names(alias, coeff_type, release):
    with pytest.warns(DeprecationWarning):
        selection = resolve_coefficients(alias)
    assert (selection["coeff_type"], selection["release"]) == (coeff_type, release)


@pytest.mark.parametrize("kwargs", [
    dict(coeff_type="lab", version="V3"),        # lab exists only in V1
    dict(coeff_type="moved", version="V4"),
    dict(version={"destripe": "V3"}),            # unknown correction
])
def test_invalid_selections(kwargs):
    with pytest.raises(ValueError):
        resolve_coefficients(**kwargs)


def test_every_catalog_file_exists():
    from importlib.resources import files
    data = files("hypso2_calibration").joinpath("data")
    names = [v["file"] for versions in CATALOG["radiometric"].values() for v in versions.values()]
    names += [o["file"] for versions in CATALOG["radiometric"].values() for v in versions.values()
              for o in (v.get("capture_types") or {}).values()]
    names += [v["file"] for kind in ("smile", "spectral") for v in CATALOG[kind].values()]
    missing = [n for n in names if not data.joinpath(n).is_file()]
    assert not missing


# The V2/V1 files are the pre-V3 files renamed, byte for byte: sha256 values
# from the coefficient manifest before the renames.
_PRE_V3_SHA256 = {
    ("radiometric", "V2", "moved"): "ceec3db9e1a164c060da2a273e8287c8d70cc1703f044c07068e427bd14d9513",
    ("radiometric", "V2", "abs"): "e022ac744aac9be56301e3d6880606264ecb541e2515f435bc33e2fdf3308a14",
    ("radiometric", "V1", "lab"): "933d83251050d9587f0e71d452511ff63b6a5212e52ecf0a7c90b92063783f29",
    ("smile", "V2", None): "3537f5d2bbc7c1d87e183b88959321802707b2a14ef8c89a38bfe9862d4dd4cd",
    ("spectral", "V2", None): "40627dd626d612c682085545734816e5c4324acb0a17ab3f2848a1ee3306c027",
    ("spectral", "V1", None): "b5eb0724927319dee8b050adba0221ab3e040916b2e4f1df436fdf869ea666c7",
}


@pytest.mark.parametrize("key, sha", list(_PRE_V3_SHA256.items()))
def test_pre_v3_files_unchanged(key, sha):
    from importlib.resources import files
    kind, version, variant = key
    entry = CATALOG[kind][version][variant] if variant else CATALOG[kind][version]
    assert sha256_file(files("hypso2_calibration").joinpath(f"data/{entry['file']}")) == sha


def test_wide_captures_use_the_wide_v3_abs_matrix():
    # Production 73eb6b8a: V3 abs uses the wide matrix for wide captures,
    # as is (no crop/bin).
    wide = get_hypso2_calibration_files(capture_type="wide")
    assert _names(wide)["radiometric"] == "h2_radiometric_calibration_matrix_wide_v3_abs.npz"
    assert wide["crop_modes"] == {"radiometric": "as_is"}
    assert wide["selection"]["radiometric_capture_type"] == "wide"
    for capture_type, coeff_type, version in (("nominal", "abs", "V3"), ("wide", "moved", "V3"),
                                              ("wide", "abs", "V2")):
        files = get_hypso2_calibration_files(capture_type=capture_type, coeff_type=coeff_type, version=version)
        assert "wide" not in _names(files)["radiometric"] and files["crop_modes"] == {}
