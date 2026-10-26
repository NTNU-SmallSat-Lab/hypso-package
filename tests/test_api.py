"""hypso.open / hypso.Product - the public API (REFACTOR_PROGRESS.md ×50)."""
import json

import numpy as np
import pytest

import hypso
from conftest import BASELINE_PATH, COEFF_TYPE, COEFF_VERSION, L1A_PATH, LATS_PATH, LONS_PATH, requires_real_capture
from test_regression_real_capture import assert_matches_signature

pytestmark = requires_real_capture


@pytest.fixture(scope="module")
def l1a():
    return hypso.open(L1A_PATH)


@pytest.fixture(scope="module")
def l1d(l1a):
    return (l1a.calibrate(COEFF_TYPE, version=COEFF_VERSION)
               .georeference(latitudes=LATS_PATH, longitudes=LONS_PATH, direct=False)
               .toa_reflectance())


def test_full_chain_matches_golden_baseline(l1a, l1d):
    # The baseline was recorded from the pre-refactor package (tests/baseline).
    baseline = json.loads(BASELINE_PATH.read_text())
    l1b = l1a.calibrate(COEFF_TYPE, version=COEFF_VERSION)
    assert_matches_signature("l1a_cube", l1a.cube.values, baseline["l1a_cube"])
    assert_matches_signature("l1b_cube", l1b.cube.values, baseline["l1b_cube"])
    assert_matches_signature("l1d_cube", l1d.cube.values, baseline["l1d_cube"])
    assert_matches_signature("latitude", l1d.geometry["latitude"].values, baseline["latitude"])
    assert np.allclose(l1d.wavelengths, baseline["wavelengths"])


def test_product_is_immutable(l1a):
    l1b = l1a.calibrate()
    assert l1a.level == "L1A" and l1b.level == "L1B"
    with pytest.raises(AttributeError):
        l1a.level = "L1B"
    tree = l1a.tree
    tree.attrs["processing_level"] = "X"
    assert l1a.level == "L1A"


def test_contents(l1d):
    assert l1d.level == "L1D" and l1d.sat_id == "HYPSO-2" and l1d.target == "aeronetvenice"
    assert l1d.cube.name == "rhot" and l1d.cube.dims == ("lines", "samples", "bands")
    assert {"latitude", "longitude", "solar_zenith"} <= set(l1d.geometry.data_vars)
    assert l1d.flags == {}
    assert [s["step"] for s in l1d.provenance] == ["l1b", "l1c", "l1d"]
    assert l1d.provenance[0]["coefficient_release"] == COEFF_VERSION
    assert "rhot" in repr(l1d)


def test_wrong_level_explains(l1a):
    with pytest.raises(ValueError, match=r'needs an L1C product, this one is L1A or \.to\("L1D"\)'):
        l1a.toa_reflectance()


def test_to_runs_missing_steps(l1a, l1d):
    via_to = l1a.to("L1D", calibrate={"coeff_type": COEFF_TYPE, "version": COEFF_VERSION},
                    georeference={"latitudes": LATS_PATH, "longitudes": LONS_PATH, "direct": False})
    np.testing.assert_array_equal(via_to.cube.values, l1d.cube.values)
    with pytest.raises(ValueError):
        l1d.to("L1B")


def test_write_names_and_round_trips(l1d, tmp_path):
    path = l1d.write(out_dir=tmp_path)
    assert path.name == "HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc"
    reopened = hypso.open(path)
    assert reopened.level == "L1D"
    np.testing.assert_array_equal(reopened.cube.values, l1d.cube.values.astype(np.float32))
    assert [s["step"] for s in reopened.provenance] == ["l1b", "l1c", "l1d"]
    assert path.with_name(l1d.filename(label="test")).name.endswith(".L1D.test.nc")


def test_dps_matches_old_function(l1d):
    from hypso.ac.ac_dark_pixel_subtraction import dark_pixel_subtraction_per_band
    l2a = l1d.correct_atmosphere("dps")
    expected, dark = dark_pixel_subtraction_per_band(l1d.cube.values)
    np.testing.assert_array_equal(l2a.cube.values, expected)
    assert l2a.level == "L2A" and l2a.cube.name == "Rrs" and l2a.atmospheric_correction == "dps"
    assert l2a.filename() == "HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.dps.nc"
    assert [s["step"] for s in l2a.provenance][-1] == "l2a_dps"
    assert "latitude" in l2a.geometry


def test_dps_masked_ignores_flagged_pixels(l1d):
    import xarray as xr
    from hypso.masks.flags import FLAGS_NAME, encode_flags
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:100] = True
    tree = l1d.tree
    root = tree.to_dataset(inherit=False)
    root[FLAGS_NAME] = encode_flags({"land": land})
    tree.dataset = root
    flagged = hypso.Product(tree)
    l2a = flagged.correct_atmosphere("dps", masked=True)
    assert np.isnan(l2a.cube.values[:100]).all()
    assert not np.isnan(l2a.cube.values[100:]).any()
    dark = np.nanmin(np.where(land[..., None], np.nan, l1d.cube.values), axis=(0, 1))
    np.testing.assert_allclose(l2a.cube.values[100:], np.clip(l1d.cube.values[100:] - dark, 0, None))
    with pytest.raises(ValueError, match="pixel_flags"):
        l1d.correct_atmosphere("dps", masked=True)


def test_correct_atmosphere_level_and_method_errors(l1a, l1d):
    with pytest.raises(ValueError, match="needs an L1D"):
        l1a.correct_atmosphere("dps")
    with pytest.raises(ValueError, match="unknown method"):
        l1d.correct_atmosphere("magic")
