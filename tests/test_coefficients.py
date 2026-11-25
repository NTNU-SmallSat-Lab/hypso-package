"""HYPSO-2 versioned calibration coefficients (hypso2_calibration
data/coefficients.yaml, REFACTOR_PROGRESS.md ×48)."""
from pathlib import Path

import numpy as np
import pytest

from hypso.calibration.manifest import sha256_file
from hypso2_calibration import get_hypso2_calibration_files
from hypso2_calibration.main import CATALOG, resolve_coefficients


def _names(files):
    return {k: (Path(str(v)).name if v is not None else None) for k, v in files.items()
            if k not in ("selection", "crop_modes")}


def test_default_is_abs_v31():
    # V3.1, production's default (polymer_v3 c390ca27, ×92): V3 spectral and the in-orbit FWHM table
    files = get_hypso2_calibration_files()
    assert _names(files) == {
        "radiometric": "h2_radiometric_calibration_matrix_full_v31_abs.npz",
        "smile": "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz",
        "destriping": None,
        "spectral": "spectral_array_calibrated_poly_full_v3.npz",
        "spectral_full_frame": "spectral_array_calibrated_poly_full_v3.npz",
        "fwhm": "h2_srf_fwhm_v31.npz",
    }
    assert files["selection"] == {"coeff_type": "abs", "release": "V3.1", "radiometric": "V3.1/abs",
                                  "smile": "V3", "spectral": "V3", "spectral_full_frame": "V3", "fwhm": "V3.1"}


def test_v31v2spec_and_release_names():
    # for OC-SMART: V3.1 with the pre-V3 band names; production's names work; V3 maps to V3.1
    files = get_hypso2_calibration_files(version="V3.1v2spec")
    assert _names(files)["spectral"] == "spectral_array_calibrated_poly_full_v2.npz"
    assert _names(files)["radiometric"] == "h2_radiometric_calibration_matrix_full_v31_abs.npz"
    assert files["selection"]["fwhm"] == "V3.1"
    assert get_hypso2_calibration_files(version="v31v2spec")["selection"]["release"] == "V3.1v2spec"
    assert get_hypso2_calibration_files(version="v31")["selection"]["release"] == "V3.1"
    with pytest.warns(DeprecationWarning, match="superseded by V3.1"):
        assert get_hypso2_calibration_files(version="V3")["selection"]["release"] == "V3.1"
    assert get_hypso2_calibration_files(version="V2")["selection"]["fwhm"] is None     # sensor profile table


@pytest.mark.parametrize("coeff_type, version, radiometric, smile, spectral", [
    ("moved", "V3.1", "h2_radiometric_calibration_matrix_full_moved_v3.npz",
     "smile_correction_matrix_HYPSO-2_full_sobel_v3.npz", "spectral_array_calibrated_poly_full_v3.npz"),
    ("moved", "V2", "h2_radiometric_calibration_matrix_full_moved_v2.npz",
     "smile_correction_matrix_HYPSO-2_full_v2.npz", "spectral_array_calibrated_poly_full_v2.npz"),
    ("abs", "v2", "h2_radiometric_calibration_matrix_full_v2_abs.npz",
     "smile_correction_matrix_HYPSO-2_full_v2.npz", "spectral_array_calibrated_poly_full_v2.npz"),
    # V1 = the complete lab set (×94): lab gains, lab smile, band centres from the lab smile's row 608
    ("lab", "V1", "h2_radiometric_calibration_matrix_full_v1_lab.npz",
     "smile_correction_matrix_HYPSO-2_full_lab.npz", "h2_spectral_array_lab_smile_row608.npz"),
    ("lab", "V1v2spec", "h2_radiometric_calibration_matrix_full_v1_lab.npz",
     "smile_correction_matrix_HYPSO-2_full_lab.npz", "spectral_array_calibrated_poly_full_v2.npz"),
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
    # V3.1 radiometric with V2 smile, everything else from the default release
    files = get_hypso2_calibration_files(version={"radiometric": "V3.1", "smile": "V2"})
    assert _names(files)["smile"] == "smile_correction_matrix_HYPSO-2_full_v2.npz"
    assert files["selection"]["radiometric"] == "V3.1/abs"


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
# from the coefficient manifest before the renames. Except V1's smile and
# spectral (×94): the lab delivery's smile (= production polymer_v1 ad8866c7)
# and its row 608 (polymer_v1 2b560445), replacing the V2 smile and the old
# lab centre row (b5eb0724..., now in reference/).
_PRE_V3_SHA256 = {
    ("radiometric", "V2", "moved"): "ceec3db9e1a164c060da2a273e8287c8d70cc1703f044c07068e427bd14d9513",
    ("radiometric", "V2", "abs"): "e022ac744aac9be56301e3d6880606264ecb541e2515f435bc33e2fdf3308a14",
    ("radiometric", "V1", "lab"): "933d83251050d9587f0e71d452511ff63b6a5212e52ecf0a7c90b92063783f29",
    ("smile", "V2", None): "3537f5d2bbc7c1d87e183b88959321802707b2a14ef8c89a38bfe9862d4dd4cd",
    ("spectral", "V2", None): "40627dd626d612c682085545734816e5c4324acb0a17ab3f2848a1ee3306c027",
    ("smile", "V1", None): "46e9aedd54e85c2aa3c08fb6f2fe1bab4f482c7e3d37af9558dcfa138664f587",
    ("spectral", "V1", None): "b372ffdc15ad74774b900218a0c1e0428d816f9ba7815f701d1ceae0f46b70f3",
}


@pytest.mark.parametrize("key, sha", list(_PRE_V3_SHA256.items()))
def test_pre_v3_files_unchanged(key, sha):
    from importlib.resources import files
    kind, version, variant = key
    entry = CATALOG[kind][version][variant] if variant else CATALOG[kind][version]
    assert sha256_file(files("hypso2_calibration").joinpath(f"data/{entry['file']}")) == sha


def test_wide_captures_use_the_wide_v31_abs_matrix():
    # V3.1 abs uses the wide matrix for wide captures, as is (no crop/bin), like V3 did (production 73eb6b8a)
    wide = get_hypso2_calibration_files(capture_type="wide")
    assert _names(wide)["radiometric"] == "h2_radiometric_calibration_matrix_wide_v31_abs.npz"
    assert wide["crop_modes"] == {"radiometric": "as_is"}
    assert wide["selection"]["radiometric_capture_type"] == "wide"
    for capture_type, coeff_type, version in (("nominal", "abs", "V3.1"), ("wide", "moved", "V3.1"),
                                              ("wide", "abs", "V2")):
        files = get_hypso2_calibration_files(capture_type=capture_type, coeff_type=coeff_type, version=version)
        assert "wide" not in _names(files)["radiometric"] and files["crop_modes"] == {}


def test_catalog_module_copies_identical():
    # Both calibration packages carry the same _catalog.py (they cannot share code).
    import hypso1_calibration._catalog as one
    import hypso2_calibration._catalog as two
    assert Path(one.__file__).read_text() == Path(two.__file__).read_text()


def test_hypso1_versions_and_names():
    # HYPSO-1 on HYPSO-2's scheme (REFACTOR_PROGRESS.md ×69): V3.1 (default, ×92) / V2 / V1;
    # a variant without a version takes the newest release that has it (moved -> V2, lab -> V1).
    from hypso1_calibration import get_hypso1_calibration_files, resolve_coefficients as resolve1
    assert resolve1() == {"coeff_type": "abs", "release": "V3.1", "radiometric": "V3.1/abs", "smile": "V3",
                          "destriping": "V1", "spectral": "V3", "spectral_full_frame": "V2", "fwhm": "V3.1"}
    assert resolve1("moved") == {"coeff_type": "moved", "release": "V2", "radiometric": "V2/moved", "smile": "V1",
                                 "destriping": "V1", "spectral": "V2", "spectral_full_frame": "V2", "fwhm": None}
    names = lambda f: {k: (Path(str(v)).name if v else None) for k, v in f.items() if k not in ("crop_modes", "selection")}
    v31 = get_hypso1_calibration_files("wide")
    assert names(v31)["radiometric"] == "radiometric_calibration_matrix_HYPSO-1_wide_v31.npz"
    assert names(v31)["smile"] == "smile_correction_matrix_HYPSO-1_full_sobel_v3.npz"
    assert v31["crop_modes"] == {"radiometric": "as_is", "smile": "crop_and_bin"}
    assert names(get_hypso1_calibration_files("wide", version="v31v2spec"))["spectral"] == \
        "spectral_array_calibrated_poly_full.npz"
    wide = names(get_hypso1_calibration_files("wide", coeff_type="abs", version="V2"))
    assert wide["radiometric"] == "radiometric_calibration_matrix_HYPSO-1_full_v1_adjusted_v11.npz"
    assert wide["smile"] == "smile_correction_matrix_HYPSO-1_wide_v1.npz"
    assert wide["destriping"] == "destriping_matrix_HYPSO-1_wide_v1.npz"
    lab = names(get_hypso1_calibration_files("custom", coeff_type="lab", version="V1"))
    assert lab["radiometric"] == "radiometric_calibration_matrix_HYPSO-1_full_v1.npz" and lab["destriping"] is None
    assert lab["spectral"] == "spectral_calibration_wavelengths_center_row_HYPSO-1.npz"
    moon = names(get_hypso1_calibration_files("moon"))
    assert moon["radiometric"] == "radiometric_calibration_matrix_HYPSO-1_moon.npz"
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert names(get_hypso1_calibration_files("wide", coeff_type="adjusted")) == wide



@pytest.mark.parametrize("package", ["hypso1_calibration", "hypso2_calibration"])
def test_data_folder_is_organized(package):
    # ×93: every file in data/ is in its correction's folder and either selected by the catalog or listed
    # under reference_files; every path the catalog names exists
    import yaml
    from importlib.resources import files
    data = Path(str(files(package).joinpath("data")))
    catalog = yaml.safe_load((data / "coefficients.yaml").read_text())

    def paths(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "file" and isinstance(value, str):
                    yield value
                else:
                    yield from paths(value)
    selected = {p for section in ("radiometric", "smile", "destriping", "spectral", "fwhm", "optional")
                for p in paths(catalog.get(section, {}))}
    reference = set(catalog.get("reference_files", {}))
    on_disk = {str(p.relative_to(data)) for p in data.rglob("*.npz")}
    assert selected | reference == on_disk
    assert not selected & reference
    assert all(p.split("/")[0] == "reference" for p in reference)
    assert all(p.split("/")[0] in ("radiometric", "smile", "destriping", "spectral", "fwhm", "optional")
               for p in selected)



def test_v1_spectral_is_the_lab_smile_row_608():
    # ×94 (user decision 2026-10-28): V1's band centres are row 608 of the lab smile, the centre row the
    # smile correction resamples onto for the AOC window (aoi_y 62, 1092 rows)
    files = get_hypso2_calibration_files(coeff_type="lab", version="V1")
    with np.load(files["smile"]) as smile, np.load(files["spectral"]) as spectral:
        np.testing.assert_array_equal(spectral[spectral.files[0]], smile[smile.files[0]][62 + 1092 // 2])
