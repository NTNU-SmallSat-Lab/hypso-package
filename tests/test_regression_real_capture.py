"""Plan §Verification item 1: golden-file regression against the real reference
capture. The baseline (tests/baseline/baseline.json) was recorded by
tests/baseline/capture_baseline.py running the UNMODIFIED pre-refactor code;
this test runs the current code through the identical pipeline and asserts the
numbers match. Calibration/georeferencing math wasn't the target of this
refactor, so these are strict signature checks (exact shapes, sampled values,
means), not loose tolerances.

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


def test_identity(satobj, baseline):
    assert satobj.sat_id == baseline["sat_id"]
    assert satobj.platform == baseline["platform"]


def test_wavelengths(satobj, baseline):
    wl = np.asarray(satobj.wavelengths, dtype=float).ravel()
    assert np.allclose(wl, baseline["wavelengths"])


@pytest.mark.parametrize("name,getter", [
    ("l1a_cube", lambda s: s.l1a_cube.to_numpy()),
    ("l1b_cube", lambda s: s.l1b_cube.to_numpy()),
    ("l1d_cube", lambda s: s.l1d_cube.to_numpy()),
    ("latitude", lambda s: s.latitudes),
    ("longitude", lambda s: s.longitudes),
])
def test_array_matches_baseline(satobj, baseline, name, getter):
    assert_matches_signature(name, getter(satobj), baseline[name])
