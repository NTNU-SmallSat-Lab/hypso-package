"""Run the CURRENT (possibly-refactored) code through the same L1A->L1D
pipeline as capture_baseline.py and compare against the saved baseline.json.
Ad hoc verification script for use during the refactor - the pytest-based
tests/test_regression_real_capture.py (added later) formalizes this check."""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "hypso"))
sys.path.insert(0, str(REPO_ROOT / "hypso1_calibration"))
sys.path.insert(0, str(REPO_ROOT / "hypso2_calibration"))

from hypso import Hypso  # noqa: E402

CAPTURE_DIR = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z")
L1A_PATH = CAPTURE_DIR / "aeronetvenice_2025-03-04T10-38-05Z-l1a.nc"
LATS_PATH = CAPTURE_DIR / "processing-temp" / "latitudes_indirectgeoref.dat"
LONS_PATH = CAPTURE_DIR / "processing-temp" / "longitudes_indirectgeoref.dat"
COEFF_TYPE = "moved"

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def compare_array(name, arr, expected_sig):
    a = np.asarray(arr, dtype=np.float64)
    ok = True
    if list(a.shape) != expected_sig["shape"]:
        print(f"[FAIL] {name}: shape {list(a.shape)} != expected {expected_sig['shape']}")
        ok = False
        return ok
    flat = a.ravel()
    sample = flat[expected_sig["sample_indices"]]
    expected_sample = np.array(expected_sig["sample_values"])
    if not np.allclose(sample, expected_sample, equal_nan=True):
        print(f"[FAIL] {name}: sample values differ. got={sample[:5]} expected={expected_sample[:5]}")
        ok = False
    finite = flat[np.isfinite(flat)]
    mean = float(np.mean(finite)) if finite.size else None
    if mean is None or not np.isclose(mean, expected_sig["mean"]):
        print(f"[FAIL] {name}: mean {mean} != expected {expected_sig['mean']}")
        ok = False
    if ok:
        print(f"[OK]   {name}: shape={list(a.shape)} mean={mean:.6f}")
    return ok


def main():
    baseline = json.loads(BASELINE_PATH.read_text())

    satobj = Hypso(path=L1A_PATH, load_cube=True, verbose=True)

    lats = np.fromfile(LATS_PATH, dtype=np.float32).reshape(satobj.spatial_dimensions)
    lons = np.fromfile(LONS_PATH, dtype=np.float32).reshape(satobj.spatial_dimensions)
    satobj.run_georeferencing(latitudes=lats, longitudes=lons)
    satobj.generate_l1b_cube(coeff_type=COEFF_TYPE)
    satobj.generate_l1d_cube(use_direct_georef=False)

    results = []
    results.append(satobj.sat_id == baseline["sat_id"])
    results.append(satobj.platform == baseline["platform"])
    print(f"[{'OK' if results[-1] else 'FAIL'}] sat_id/platform: {satobj.sat_id}/{satobj.platform}")

    wl = [float(w) for w in np.asarray(satobj.wavelengths).ravel()]
    wl_ok = np.allclose(wl, baseline["wavelengths"])
    results.append(wl_ok)
    print(f"[{'OK' if wl_ok else 'FAIL'}] wavelengths ({len(wl)} bands)")

    results.append(compare_array("l1a_cube", satobj.l1a_cube.to_numpy(), baseline["l1a_cube"]))
    results.append(compare_array("l1b_cube", satobj.l1b_cube.to_numpy(), baseline["l1b_cube"]))
    results.append(compare_array("l1d_cube", satobj.l1d_cube.to_numpy(), baseline["l1d_cube"]))
    results.append(compare_array("latitude", satobj.latitudes, baseline["latitude"]))
    results.append(compare_array("longitude", satobj.longitudes, baseline["longitude"]))

    if all(results):
        print("\nALL CHECKS PASSED - no numeric regression vs baseline.")
    else:
        print("\nSOME CHECKS FAILED - see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
