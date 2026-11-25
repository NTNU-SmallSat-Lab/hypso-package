"""Compare the current code's output on the reference capture against
baseline.json, printing per-array differences (a debugging aid next to
tests/test_regression_real_capture.py, which asserts the same thing).
Run: python tests/baseline/compare_to_baseline.py"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for package in ("hypso", "hypso1_calibration", "hypso2_calibration"):
    sys.path.insert(0, str(REPO_ROOT / package))

import hypso  # noqa: E402

CAPTURE_DIR = Path("/home/camerop/HYPSO_DATA_AOC/aeronetvenice_2025-03-04T10-38-05Z")
BASELINE = json.loads((Path(__file__).parent / "baseline.json").read_text())


def report(name, arr):
    sig = BASELINE[name]
    a = np.asarray(arr, dtype=np.float64).ravel()
    sample = a[sig["sample_indices"]]
    mean = float(np.nanmean(a))
    print(f"{name:<10} shape {list(np.shape(arr)) == sig['shape']}  "
          f"max sample diff {np.nanmax(np.abs(sample - np.array(sig['sample_values']))):.3e}  "
          f"mean {mean:.10g} vs {sig['mean']:.10g}")


def main():
    l1a = hypso.open(CAPTURE_DIR / "aeronetvenice_2025-03-04T10-38-05Z-l1a.nc")
    l1b = l1a.calibrate("moved", version="V2")   # baseline = pre-V3 "moved"
    l1d = (l1b.georeference(latitudes=CAPTURE_DIR / "processing-temp" / "latitudes_indirectgeoref.dat",
                            longitudes=CAPTURE_DIR / "processing-temp" / "longitudes_indirectgeoref.dat",
                            direct=False)
              .toa_reflectance())
    report("l1a_cube", l1a.cube.values)
    report("l1b_cube", l1b.cube.values)
    report("l1d_cube", l1d.cube.values)
    report("latitude", l1d.geometry["latitude"].values)
    report("longitude", l1d.geometry["longitude"].values)


if __name__ == "__main__":
    main()
