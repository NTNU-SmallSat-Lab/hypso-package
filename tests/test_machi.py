"""MACHI, the default L1D -> L2A method (×78): hypso.ac.machi vendored from
NTNU SmallSat Lab's MACHI (ebbc3e72) and wrapped as correct_atmosphere("machi")."""
import numpy as np
import pytest

import hypso
from conftest import requires_real_capture
from hypso.processing import correct_atmosphere


@requires_real_capture
def test_machi_on_the_reference_capture(l1d):
    l2a = hypso.Product(correct_atmosphere(l1d.tree))                 # machi is the default method
    assert l2a.level == "L2A" and l2a.atmospheric_correction == "machi" and l2a.cube.name == "rhos"
    step = l2a.provenance[-1]
    assert step["ac_tool"] == "machi" and float(step["est_min_R"]) == 0.05 and int(step["iterations"]) > 1
    model = l2a.tree["metadata/machi"].dataset
    T, A = model["T"].values, model["A"].values
    blank = np.flatnonzero(np.all(l1d.cube.values == 0, axis=(0, 1)))          # < 400 nm, zero after smile
    assert np.isnan(T[blank]).all() and np.isnan(l2a.cube.values[..., blank]).all()
    fitted = np.setdiff1d(np.arange(T.size), blank)
    assert ((T[fitted] > 0) & (T[fitted] <= 1)).all() and (A[fitted] >= 0).all()
    np.testing.assert_allclose(l2a.cube.values[..., fitted], (l1d.cube.values[..., fitted] - A[fitted]) / T[fitted],
                               rtol=1e-5)
    o2a = int(np.argmin(np.abs(l1d.wavelengths - 762)))
    assert T[o2a] < np.median(T[fitted])                                       # O2-A absorption shows in T


@requires_real_capture
def test_machi_options(l1d):
    tighter = hypso.Product(correct_atmosphere(l1d.tree, "machi", est_min_R=0.01, tol=0.05))
    assert float(tighter.provenance[-1]["est_min_R"]) == 0.01
    flagged = l1d.with_masks(cloud=np.zeros(l1d.cube.shape[:2], bool) | (np.arange(l1d.cube.shape[0]) < 50)[:, None])
    excluded = hypso.Product(correct_atmosphere(flagged.tree, "machi", fit_exclude_flags=["cloud"]))
    assert int(excluded.provenance[-1]["fit_pixels"]) == 548 * l1d.cube.shape[1]
    with pytest.raises(ValueError, match="unknown flags"):
        correct_atmosphere(flagged.tree, "machi", fit_exclude_flags=["land"])
