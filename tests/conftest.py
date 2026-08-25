"""Shared fixtures for the hypso-package test suite (plan §Verification).

Two kinds of tests live under tests/:

- Unit tests (test_sensors.py, test_io_schema_cf.py, test_ac_adapters.py,
  test_public_api.py) - no real data needed, run anywhere the package imports.
- Real-data tests (test_regression_real_capture.py, test_cf_format.py) - need
  the reference capture under /home/camerop/HYPSO_DATA_AOC/ and are skipped
  automatically when it isn't present (so the suite still runs in a fresh
  clone), per the approved plan.

The sys.path inserts mirror tests/baseline/compare_to_baseline.py so the suite
works both against an editable install (pip install -e hypso/) and a bare
checkout with the hypsoN_calibration siblings checked out next to hypso/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hypso"))
sys.path.insert(0, str(REPO_ROOT / "hypso1_calibration"))
sys.path.insert(0, str(REPO_ROOT / "hypso2_calibration"))

CAPTURE_DIR = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z")
L1A_PATH = CAPTURE_DIR / "aeronetvenice_2025-03-04T10-38-05Z-l1a.nc"
LATS_PATH = CAPTURE_DIR / "processing-temp" / "latitudes_indirectgeoref.dat"
LONS_PATH = CAPTURE_DIR / "processing-temp" / "longitudes_indirectgeoref.dat"
COEFF_TYPE = "moved"

BASELINE_PATH = Path(__file__).parent / "baseline" / "baseline.json"

requires_real_capture = pytest.mark.skipif(
    not L1A_PATH.is_file(),
    reason=f"reference capture not present at {L1A_PATH}",
)


@pytest.fixture(scope="session")
def satobj():
    """The reference capture processed L1A -> L1B -> L1D, exactly the pipeline
    tests/baseline/capture_baseline.py ran to record baseline.json (same
    coeff_type, same precomputed indirect-georeferencing lat/lon arrays), built
    once per test session - it takes on the order of a minute."""
    if not L1A_PATH.is_file():
        pytest.skip(f"reference capture not present at {L1A_PATH}")

    from hypso import Hypso

    obj = Hypso(path=L1A_PATH, load_cube=True, verbose=False)

    lats = np.fromfile(LATS_PATH, dtype=np.float32).reshape(obj.spatial_dimensions)
    lons = np.fromfile(LONS_PATH, dtype=np.float32).reshape(obj.spatial_dimensions)
    obj.run_georeferencing(latitudes=lats, longitudes=lons)

    # The deprecated in-place methods are deliberately used here: the baseline
    # was recorded through them, and the regression test must exercise the same
    # code path the production pipeline (hypso-processing-pipeline) still uses.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        obj.generate_l1b_cube(coeff_type=COEFF_TYPE)
        obj.generate_l1d_cube(use_direct_georef=False)

    return obj


@pytest.fixture(scope="session")
def written_nc_files(satobj, tmp_path_factory):
    """One written NetCDF per level (plus an L2A with a fabricated correction
    whose l2_variable_name is deliberately NOT the 'Rrs' fallback, to prove the
    dynamic product naming works), for the CF/format assertions."""
    from hypso.io.writer import write_level_nc, write_l2a_nc

    out_dir = tmp_path_factory.mktemp("nc_out")
    files = {}

    for level in ("l1b", "l1c", "l1d"):
        dst = out_dir / f"test-{level}.nc"
        write_level_nc(satobj, level=level, dst_nc=str(dst))
        files[level] = dst

    satobj.l2a_cube["testac"] = satobj.l1d_cube.to_numpy()
    satobj.l2a_cube["testac"].attrs["l2_variable_name"] = "chla"
    dst = out_dir / "test-l2a.nc"
    write_l2a_nc(satobj, correction="testac", dst_nc=str(dst))
    files["l2a"] = dst

    return files
