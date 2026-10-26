"""hypso.processing: processing steps as tree -> tree functions, on the real
reference capture. End-to-end values are checked against the pre-refactor
baseline in test_regression_real_capture.py; each step was also checked
identical to the removed capture-object path before it was deleted
(REFACTOR_PROGRESS.md ×47-×52)."""
import numpy as np
import pytest

from conftest import COEFF_TYPE, COEFF_VERSION, LATS_PATH, LONS_PATH, requires_real_capture
from hypso.processing import capture_params, l1a_to_l1b, l1b_to_l1c, l1c_to_l1d

pytestmark = requires_real_capture


def test_capture_params(l1a):
    # Values the capture-object loader derived for this capture.
    params = capture_params(l1a.tree)
    assert (params.capture_type, params.sat_id) == ("wide", "HYPSO-2")
    assert params.spatial_dimensions == (598, 1092)
    assert (params.bin_factor, params.background_value, params.image_width) == (9, 72, 120)
    assert np.isclose(params.exposure, 0.04)
    assert params.iso_time == "2025-03-04T10:38:05.079200+00:00"


def test_l1a_to_l1b_leaves_input_unchanged(l1a):
    tree = l1a.tree
    before = tree.dataset["dn"].values.copy()
    corrections_attrs, root_attrs = dict(tree["metadata/corrections"].attrs), dict(tree.attrs)
    out = l1a_to_l1b(tree, "moved", version="V3")
    assert out["metadata/corrections"].attrs["radiometric_coefficients_version"] == "moved"
    assert dict(tree["metadata/corrections"].attrs) == corrections_attrs
    assert dict(tree.attrs) == root_attrs and tree.attrs["processing_level"] == "L1A"
    np.testing.assert_array_equal(tree.dataset["dn"].values, before)


def test_steps_reject_wrong_levels(l1a, l1b, l1c):
    with pytest.raises(ValueError, match="L1A"):
        l1a_to_l1b(l1b.tree)
    with pytest.raises(ValueError, match="L1B"):
        l1b_to_l1c(l1c.tree)
    with pytest.raises(ValueError, match="L1C"):
        l1c_to_l1d(l1a.tree)


def test_l1b_provenance(l1b):
    step = l1b.tree["metadata/processing/l1b"].attrs
    assert step["coeff_type"] == COEFF_TYPE and step["coefficient_release"] == COEFF_VERSION
    assert "radiometric" in step["corrections_applied"] and len(step["radiometric_sha256"]) == 64


def test_l1b_to_l1c_direct_and_fallback(l1b):
    with_direct = l1b_to_l1c(l1b.tree, latitudes=LATS_PATH, longitudes=LONS_PATH, direct=True).dataset
    assert {"latitude_direct", "solar_zenith_direct"} <= set(with_direct.data_vars)
    # Without coordinates the direct georeferencing becomes the main geometry.
    fallback = l1b_to_l1c(l1b.tree).dataset
    np.testing.assert_array_equal(fallback["latitude"].values, with_direct["latitude_direct"].values)
    np.testing.assert_array_equal(fallback["solar_zenith"].values, with_direct["solar_zenith_direct"].values)


def test_l1b_to_l1c_argument_errors(l1b):
    with pytest.raises(ValueError, match="both"):
        l1b_to_l1c(l1b.tree, latitudes=LATS_PATH)
    with pytest.raises(ValueError, match="shape"):
        l1b_to_l1c(l1b.tree, latitudes=np.zeros((2, 2)), longitudes=np.zeros((2, 2)))


def test_l1d_provenance_and_srfs(l1d):
    step = l1d.tree["metadata/processing/l1d"].attrs
    assert step["ssi_source"] == "tsis" and step["solar_zenith"] == "solar_zenith"
    srf = l1d.tree["metadata/srf"]
    assert np.isclose(step["earth_sun_distance_au"], srf.attrs["earth_sun_distance_au"])
    assert len([n for n in srf.dataset.data_vars if n.startswith("Band_")]) == 120


def test_wide_capture_abs_v3_uses_wide_matrix(l1a):
    # The fixture is a wide capture: V3 abs uses the wide matrix as is
    # (production 73eb6b8a; identical L1B verified against it).
    step = l1a_to_l1b(l1a.tree, "abs", version="V3")["metadata/processing/l1b"].attrs
    assert step["radiometric_capture_type"] == "wide" and step["radiometric_crop"] == "as_is"
    assert "wide_v3_abs" in step["radiometric_file"]


def test_as_is_radiometric_shape_mismatch_raises():
    from importlib.resources import files
    from hypso.calibration.correction import CalibrationShapeMismatchError, read_coeffs_from_file
    wide = files("hypso2_calibration").joinpath("data/h2_radiometric_calibration_matrix_wide_v3_abs.npz")
    with pytest.raises(CalibrationShapeMismatchError):
        # a nominal-sized AOI does not fit the pre-baked wide matrix
        read_coeffs_from_file(wide, "radiometric", x_start=428, x_stop=1508, y_start=0, y_stop=684,
                              bin_factor=9, crop_mode="as_is")
