"""hypso.calibration.keystone: the keystone groundwork (×76). Synthetic only;
the opt-in calibrate(keystone=True) is tested with the real capture in
test_processing.py."""
import numpy as np

from hypso.calibration.keystone import correct_keystone, keystone_positions


def test_identity_model_leaves_the_cube_unchanged():
    identity = [0, 0, 1, 0, 0, 0]                     # B(x, y) = y
    cube = np.random.default_rng(0).random((4, 30, 5)).astype(np.float32)
    np.testing.assert_allclose(correct_keystone(cube, identity, x_start=428, y_start=62, bin_factor=9), cube)


def test_constant_shift_resamples_along_samples():
    shift = [1.5, 0, 1, 0, 0, 0]                      # B(x, y) = y + 1.5 everywhere
    cube = np.tile(np.arange(30, dtype=np.float64)[None, :, None], (2, 1, 3))
    out = correct_keystone(cube, shift, x_start=0, y_start=0, bin_factor=1)
    np.testing.assert_allclose(out[:, :28, :], cube[:, :28, :] + 1.5)   # linear data, linear resampling
    assert (out[:, -1, :] == 0).all()                                   # source outside the capture


def test_positions_use_full_frame_coordinates():
    # B depends on the full-frame column of the band centre (x) and row (y)
    b = [0, 0.001, 1, 0, 0, 0]                        # B = y + 0.001 x
    pos = keystone_positions(b, n_samples=3, n_bands=2, x_start=428, y_start=62, bin_factor=9)
    x_centres = 428 + np.array([0, 9]) + 4
    np.testing.assert_allclose(pos[0], 0.001 * x_centres)
    np.testing.assert_allclose(pos[:, 0] - pos[0, 0], [0, 1, 2])


def test_hypso2_model_is_shipped_disabled():
    from hypso2_calibration.main import optional_model
    path, entry = optional_model("keystone")
    assert entry["enabled"] is False and path.name == "h2_distortion_model_v1.npz"
    model = np.load(path)
    assert model["B_hat"].shape == (6,) and list(model["terms"]) == ["1", "x", "y", "xy", "x2", "y2"]
