"""Spatial subsampling (capture_config ``spatial_subsampling``, older name
``sample_div``): readout keeps every N-th detector row along the slit, so a
1092-row AOI with subsampling 2 gives 546 samples. The per-pixel and per-row
corrections must then use the same detector rows, and direct georeferencing
their viewing angles."""
import logging

import numpy as np
import pytest

from conftest import requires_real_capture
from hypso.calibration.correction import CalibrationShapeMismatchError, read_coeffs_from_file
from hypso.calibration.keystone import keystone_positions
from hypso.processing.capture import spatial_subsampling, subsampled_rows

CONFIG = {"row_count": 1092}


def test_subsampled_rows_keep_the_first_row():
    assert subsampled_rows(1092, 1) == 1092
    assert subsampled_rows(1092, 2) == 546
    assert subsampled_rows(5, 2) == 3        # rows 0, 2, 4


def test_factor_from_either_attribute_name():
    assert spatial_subsampling({**CONFIG, "spatial_subsampling": 2}, 546) == (2, 1092)
    assert spatial_subsampling({**CONFIG, "sample_div": 2}, 546) == (2, 1092)
    assert spatial_subsampling({**CONFIG, "sample_div": 1, "spatial_subsampling": 1}, 1092) == (1, 1092)
    assert spatial_subsampling(CONFIG, 1092) == (1, 1092)
    assert spatial_subsampling({**CONFIG, "spatial_subsampling": 2}) == (2, 1092)   # no cube to check against
    # spatial_subsampling wins over the older sample_div (1 in every file so far)
    assert spatial_subsampling({**CONFIG, "sample_div": 1, "spatial_subsampling": 2}, 546) == (2, 1092)


def test_conflicting_attribute_values_are_warned_about(caplog):
    with caplog.at_level(logging.WARNING, logger="hypso.processing.capture"):
        assert spatial_subsampling({**CONFIG, "sample_div": 1, "spatial_subsampling": 2}, 546) == (2, 1092)
        assert "disagree" not in caplog.text
        assert spatial_subsampling({**CONFIG, "sample_div": 3, "spatial_subsampling": 2}, 546) == (2, 1092)
    assert "disagree" in caplog.text


def test_row_count_of_the_rows_read_out():
    # row_count 546 with subsampling 2 and a 546-sample cube: the AOI spans 1092 detector rows
    assert spatial_subsampling({"row_count": 546, "spatial_subsampling": 2}, 546) == (2, 1092)


def test_missing_factor_inferred_from_the_cube_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="hypso.processing.capture"):
        assert spatial_subsampling({**CONFIG, "sample_div": 1}, 546) == (2, 1092)
    assert "no spatial subsampling" in caplog.text


def test_inconsistent_sizes_raise():
    with pytest.raises(ValueError, match="implies 546"):
        spatial_subsampling({**CONFIG, "spatial_subsampling": 2}, 500)
    with pytest.raises(ValueError, match="implies 1092"):
        spatial_subsampling(CONFIG, 500)


def test_coefficient_matrices_keep_every_n_th_row(tmp_path):
    full = np.arange(20 * 12, dtype=float).reshape(20, 12)          # full detector: 20 rows, 12 columns
    path = tmp_path / "matrix.npz"
    np.savez(path, full)
    sub = read_coeffs_from_file(path, "radiometric", x_start=2, x_stop=8, y_start=3, y_stop=13, bin_factor=3,
                                row_step=2)
    binned = full[3:13, 2:8].reshape(10, 2, 3).mean(axis=2)
    np.testing.assert_array_equal(sub, binned[::2])                 # detector rows 3, 5, ..., 11

    # a pre-baked (as_is) matrix: for the AOI's detector rows, thinned; already thinned, as is
    for prebaked in (binned, binned[::2]):
        np.savez(path, prebaked)
        out = read_coeffs_from_file(path, "smile", x_start=2, x_stop=8, y_start=3, y_stop=13, bin_factor=3,
                                    crop_mode="as_is", row_step=2)
        np.testing.assert_array_equal(out, binned[::2])
    np.savez(path, binned[:7])
    with pytest.raises(CalibrationShapeMismatchError):
        read_coeffs_from_file(path, "smile", x_start=2, x_stop=8, y_start=3, y_stop=13, bin_factor=3,
                              crop_mode="as_is", row_step=2)


def test_keystone_positions_on_subsampled_rows():
    identity = [0, 0, 1, 0, 0, 0]                     # B(x, y) = y
    np.testing.assert_allclose(keystone_positions(identity, 4, 2, x_start=428, y_start=62, bin_factor=9,
                                                  row_step=2)[:, 0], [0, 1, 2, 3])
    # B = y + 1: a one-detector-row shift is half a subsampled sample
    b = [1.0, 0, 1, 0, 0, 0]
    np.testing.assert_allclose(keystone_positions(b, 3, 1, 0, 0, 1, row_step=2)[:, 0], [0.5, 1.5, 2.5])


def _subsampled_l1a(tree, factor=2):
    """The reference capture as if read out with spatial subsampling."""
    # every node on the samples dimension (the root cube, metadata/corrections' rad_matrix) is thinned
    out = tree.map_over_datasets(
        lambda ds: ds.isel(samples=slice(None, None, factor)) if "samples" in ds.dims else ds)
    out.attrs, out.encoding = dict(tree.attrs), dict(tree.encoding)
    config = out["metadata/capture_config"].to_dataset(inherit=False)
    config.attrs["spatial_subsampling"] = np.int64(factor)
    out["metadata/capture_config"].dataset = config
    return out


@requires_real_capture
@pytest.mark.parametrize("coeff_type, version", [("abs", "V3.1"), ("moved", "V2")])
def test_subsampled_calibration_equals_the_full_one_thinned(l1a, coeff_type, version):
    # radiometric, smile and destriping are per pixel or per row: a capture read out
    # every 2nd row calibrates to the full capture's every 2nd sample (V3.1 abs: the
    # pre-baked wide matrix, as_is; V2 moved: the full-detector matrix, cropped)
    from hypso.processing import capture_params, l1a_to_l1b
    sub_tree = _subsampled_l1a(l1a.tree)
    params = capture_params(sub_tree)
    assert (params.sample_div, params.image_height, params.capture_type) == (2, 546, "wide")
    np.testing.assert_array_equal(params.detector_rows[:3], [62, 64, 66])
    full = l1a_to_l1b(l1a.tree, coeff_type, version=version)
    sub = l1a_to_l1b(sub_tree, coeff_type, version=version)
    assert sub.dataset["Lt"].shape == (598, 546, 120)
    np.testing.assert_array_equal(sub.dataset["Lt"].values, full.dataset["Lt"].values[:, ::2])
    assert sub["metadata/capture_config"].attrs["spatial_subsampling"] == 2
    assert sub["metadata/processing/l1b"].attrs["corrections_applied"] == \
        full["metadata/processing/l1b"].attrs["corrections_applied"]


@requires_real_capture
def test_subsampled_direct_georeferencing_follows_the_detector_rows(l1a):
    from hypso.geometry import direct_georeference
    from hypso.processing import capture_params
    from hypso.processing.georeference import _framepose
    params = capture_params(l1a.tree)
    framepose = _framepose(l1a.tree, params)[:4]
    full_lat, full_lon, _ = direct_georeference(framepose, image_height=1092, aoi_offset=62)
    lat, lon, _ = direct_georeference(framepose, image_height=546, aoi_offset=62, row_step=2)
    naive_lat, naive_lon, _ = direct_georeference(framepose, image_height=546, aoi_offset=62)
    error = np.hypot(lat - full_lat[:, ::2], lon - full_lon[:, ::2])
    naive_error = np.hypot(naive_lat - full_lat[:, ::2], naive_lon - full_lon[:, ::2])
    assert error.max() < 1e-4                              # degrees: well under one pixel
    assert naive_error.max() > 100 * error.max()           # ignoring the subsampling squeezes the swath
