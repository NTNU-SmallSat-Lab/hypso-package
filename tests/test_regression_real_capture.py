"""Golden-file regression against the real reference capture. The baseline
(tests/baseline/baseline.json) was recorded by tests/baseline/
capture_baseline.py running the original, pre-refactor hypso-package; this
runs the same chain through the public API (conftest's `products`:
calibrate(moved, V2) -> georeference -> toa_reflectance) and checks the
numbers match. The L1D signature was rescaled once for the Earth-Sun
distance fix (baseline.json `l1d_cube_earth_sun_rescale`).

Skipped automatically when the reference capture isn't present."""
import json

import numpy as np
import pytest

from conftest import BASELINE_PATH, requires_real_capture

pytestmark = requires_real_capture


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


def assert_matches_signature(name, arr, sig):
    a = np.asarray(arr, dtype=np.float64)
    assert list(a.shape) == sig["shape"], f"{name}: shape {list(a.shape)} != {sig['shape']}"
    flat = a.ravel()
    sample = flat[sig["sample_indices"]]
    assert np.allclose(sample, np.array(sig["sample_values"]), equal_nan=True), \
        f"{name}: sampled values differ"
    finite = flat[np.isfinite(flat)]
    assert finite.size, f"{name}: no finite values"
    assert np.isclose(float(np.mean(finite)), sig["mean"]), \
        f"{name}: mean {float(np.mean(finite))} != {sig['mean']}"


def test_identity(l1a, baseline):
    assert l1a.sat_id == baseline["sat_id"]


def test_wavelengths(l1b, baseline):
    assert np.allclose(l1b.wavelengths, baseline["wavelengths"])


@pytest.mark.parametrize("name, level, getter", [
    ("l1a_cube", "l1a", lambda p: p.cube.values),
    ("l1b_cube", "l1b", lambda p: p.cube.values),
    ("l1d_cube", "l1d", lambda p: p.cube.values),
    ("latitude", "l1d", lambda p: p.geometry["latitude"].values),
    ("longitude", "l1d", lambda p: p.geometry["longitude"].values),
])
def test_array_matches_baseline(products, baseline, name, level, getter):
    assert_matches_signature(name, getter(products[level]), baseline[name])
