"""hypso.processing: processing steps as tree -> tree functions (×37 step 2),
checked against the capture-object path on the real fixture capture."""
import warnings

import netCDF4 as nc
import numpy as np
import pytest

from conftest import COEFF_TYPE, L1A_PATH, requires_real_capture
from hypso.io.capture_tree import capture_to_tree
from hypso.io.product import read_product, write_product
from hypso.processing import capture_params, l1a_to_l1b, l1c_to_l1d
from test_capture_tree import assert_same_gcps
from test_product_io import FORMAT_2_ATTRS, assert_same_file

pytestmark = requires_real_capture


@pytest.fixture(scope="module")
def l1a_tree():
    return read_product(L1A_PATH)


@pytest.fixture(scope="module")
def old_l1b():
    """L1B from the capture-object path (no georeferencing - not needed)."""
    from hypso import Hypso
    satobj = Hypso(path=L1A_PATH, load_cube=True, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        satobj.generate_l1b_cube(coeff_type=COEFF_TYPE)
    return satobj


def test_capture_params_match_capture_object(l1a_tree, old_l1b):
    params = capture_params(l1a_tree)
    for name in ("capture_type", "x_start", "x_stop", "y_start", "y_stop", "bin_factor", "background_value",
                 "exposure", "frame_count", "image_height", "image_width"):
        assert getattr(params, name) == getattr(old_l1b, name), name
    assert params.spatial_dimensions == tuple(old_l1b.spatial_dimensions)


def test_l1a_to_l1b_matches_capture_path(l1a_tree, old_l1b, tmp_path):
    tree = l1a_to_l1b(l1a_tree, COEFF_TYPE)
    ds = tree.dataset
    assert ds.attrs["processing_level"] == "L1B"
    np.testing.assert_array_equal(ds["Lt"].values, old_l1b.l1b_cube.to_numpy())
    np.testing.assert_array_equal(ds["radiation_wavelength"].values, np.asarray(old_l1b.wavelengths, float))
    np.testing.assert_array_equal(ds["fwhm"].values, np.asarray(old_l1b.fwhm, float))

    old, new = tmp_path / "old.nc", tmp_path / "new.nc"
    write_product(capture_to_tree(old_l1b, "l1b"), old)
    write_product(tree, new)
    assert_same_file(old, new, ignore_attrs=FORMAT_2_ATTRS, ignore_groups=("/metadata/processing",))
    with nc.Dataset(new) as f:
        step = f["metadata/processing/l1b"]
        assert step.input_file == L1A_PATH.name and step.coeff_type == COEFF_TYPE
        assert "radiometric" in step.corrections_applied and len(step.radiometric_sha256) == 64


def test_l1a_to_l1b_leaves_input_unchanged(l1a_tree):
    before = l1a_tree.dataset["dn"].values.copy()
    corrections_attrs, root_attrs = dict(l1a_tree["metadata/corrections"].attrs), dict(l1a_tree.attrs)
    out = l1a_to_l1b(l1a_tree, "adjusted")
    assert out["metadata/corrections"].attrs["radiometric_coefficients_version"] == "adjusted"
    assert dict(l1a_tree["metadata/corrections"].attrs) == corrections_attrs
    assert dict(l1a_tree.attrs) == root_attrs and l1a_tree.encoding["source"] == str(L1A_PATH)
    assert l1a_tree.attrs["processing_level"] == "L1A"
    np.testing.assert_array_equal(l1a_tree.dataset["dn"].values, before)
    assert "processing" not in l1a_tree["metadata"].children


def test_l1a_to_l1b_rejects_other_levels(l1a_tree):
    with pytest.raises(ValueError):
        l1a_to_l1b(l1a_to_l1b(l1a_tree, COEFF_TYPE))


def test_capture_params_iso_time(satobj):
    assert capture_params(capture_to_tree(satobj, "l1c")).iso_time == satobj.iso_time


def test_l1c_to_l1d_matches_capture_path(satobj, tmp_path):
    # conftest's satobj: indirect georeferencing, L1B and L1D generated on
    # the capture object - the reference.
    tree = l1c_to_l1d(capture_to_tree(satobj, "l1c"))
    ds = tree.dataset
    assert ds.attrs["processing_level"] == "L1D"
    np.testing.assert_array_equal(ds["rhot"].values, satobj.l1d_cube.to_numpy())

    old, new = tmp_path / "old.nc", tmp_path / "new.nc"
    write_product(capture_to_tree(satobj, "l1d"), old)
    write_product(tree, new)
    assert_same_file(old, new, ignore_attrs=FORMAT_2_ATTRS, ignore_groups=("/metadata/processing",))
    with nc.Dataset(new) as f:
        step = f["metadata/processing/l1d"]
        assert step.ssi_source == "tsis" and step.solar_zenith == "solar_zenith"
        assert np.isclose(step.earth_sun_distance_au, f["metadata/srf"].earth_sun_distance_au)
        assert "l1c" in f["metadata/processing"].groups   # lineage from the L1C tree
