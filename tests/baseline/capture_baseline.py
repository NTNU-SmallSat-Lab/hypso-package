"""One-off script: run the CURRENT (pre-refactor) hypso-package L1A->L1D
pipeline against a real capture and save a compact numeric signature of the
results, for later regression comparison against the refactored code.

Not itself a pytest test - see tests/test_regression_real_capture.py, which
reads baseline.json written by this script. Run this script BEFORE making
any refactor changes; running it again afterward would just overwrite the
baseline with the new (possibly-buggy) code's own output, defeating the
point.
"""
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

OUT_PATH = Path(__file__).parent / "baseline.json"


def array_signature(arr) -> dict:
    """Compact numeric signature of an array: shape/dtype/summary stats plus
    a handful of fixed-index sample values - enough to catch a numeric
    regression without pickling multi-hundred-MB arrays into the repo."""
    a = np.asarray(arr, dtype=np.float64)
    flat = a.ravel()
    finite = flat[np.isfinite(flat)]
    sample_idx = np.linspace(0, len(flat) - 1, num=min(20, len(flat)), dtype=int)
    return {
        "shape": list(a.shape),
        "dtype": str(np.asarray(arr).dtype),
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "n_finite": int(finite.size),
        "n_total": int(flat.size),
        "sample_values": [float(x) for x in flat[sample_idx]],
        "sample_indices": [int(i) for i in sample_idx],
    }


def main():
    satobj = Hypso(path=L1A_PATH, load_cube=True, verbose=True)

    lats = np.fromfile(LATS_PATH, dtype=np.float32).reshape(satobj.spatial_dimensions)
    lons = np.fromfile(LONS_PATH, dtype=np.float32).reshape(satobj.spatial_dimensions)
    satobj.run_georeferencing(latitudes=lats, longitudes=lons)
    satobj.generate_l1b_cube(coeff_type=COEFF_TYPE)
    satobj.generate_l1d_cube(use_direct_georef=False)

    baseline = {
        "capture": L1A_PATH.name,
        "coeff_type": COEFF_TYPE,
        "sat_id": satobj.sat_id,
        "platform": satobj.platform,
        "wavelengths": [float(w) for w in np.asarray(satobj.wavelengths).ravel()],
        "l1a_cube": array_signature(satobj.l1a_cube.to_numpy()),
        "l1b_cube": array_signature(satobj.l1b_cube.to_numpy()),
        "l1d_cube": array_signature(satobj.l1d_cube.to_numpy()),
        "latitude": array_signature(satobj.latitudes),
        "longitude": array_signature(satobj.longitudes),
    }

    OUT_PATH.write_text(json.dumps(baseline, indent=2))
    print(f"Wrote baseline to {OUT_PATH}")


if __name__ == "__main__":
    main()
