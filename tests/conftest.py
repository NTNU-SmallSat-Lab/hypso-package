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
# The baseline (tests/baseline/baseline.json) was recorded with the pre-V3
# "moved" coefficients, which are coeff_type "moved", version "V2" in the
# versioned catalog (hypso2_calibration data/coefficients.yaml, ×48).
COEFF_VERSION = "V2"

BASELINE_PATH = Path(__file__).parent / "baseline" / "baseline.json"

requires_real_capture = pytest.mark.skipif(
    not L1A_PATH.is_file(),
    reason=f"reference capture not present at {L1A_PATH}",
)


@pytest.fixture(scope="session")
def products():
    """The reference capture through the public API, L1A -> L1D - the same
    chain tests/baseline/capture_baseline.py recorded baseline.json with
    (coeff_type moved / version V2 = the pre-V3 "moved" set, the same
    precomputed indirect-georeferencing lat/lon files). Built once per
    session (about a minute)."""
    if not L1A_PATH.is_file():
        pytest.skip(f"reference capture not present at {L1A_PATH}")
    import hypso

    l1a = hypso.open(L1A_PATH)
    l1b = l1a.calibrate(COEFF_TYPE, version=COEFF_VERSION)
    l1c = l1b.georeference(latitudes=LATS_PATH, longitudes=LONS_PATH, direct=False)
    l1d = l1c.toa_reflectance()
    return {"l1a": l1a, "l1b": l1b, "l1c": l1c, "l1d": l1d}


@pytest.fixture(scope="session")
def l1a(products):
    return products["l1a"]


@pytest.fixture(scope="session")
def l1b(products):
    return products["l1b"]


@pytest.fixture(scope="session")
def l1c(products):
    return products["l1c"]


@pytest.fixture(scope="session")
def l1d(products):
    return products["l1d"]


@pytest.fixture(scope="session")
def written_nc_files(products, tmp_path_factory):
    """One file per level written by Product.write (L1A converted from the
    ground-station file, L1B-L1D, and a DPS L2A), for format assertions."""
    out_dir = tmp_path_factory.mktemp("nc_out")
    files = {level: product.write(out_dir=out_dir) for level, product in products.items()}
    files["l2a"] = products["l1d"].correct_atmosphere("dps").write(out_dir=out_dir)
    return files
