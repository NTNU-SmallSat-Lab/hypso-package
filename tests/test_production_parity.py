"""L1A -> L1D through the refactor against production hypso-package files,
beyond the reference capture (REFACTOR_PROGRESS.md ×68): a second HYPSO-2
site from the reprocess dataset, and a HYPSO-1 capture. Each test is skipped
when its data is absent."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import hypso

CAPTURES = {
    # HYPSO-2, Kemigawa (Japan): reprocessed after the Earth-Sun distance fix,
    # so production matches through L1D.
    "kemigawa": (Path("/home/camerop/HYPSO_DATA_AOC_REPROCESS_AULUS2/kemigawa_2025-12-28T01-25-19Z"),
                 "kemigawa_2025-12-28T01-25-19Z", {"coeff_type": "moved", "version": "V2"}),
    # HYPSO-1 (Korean cal/val): produced 2026-09-14, before the Earth-Sun
    # distance fix (2026-10-23), so its L1D differs by (d_old / d_new)^2.
    "hypso1": (Path("/home/camerop/HYPSO_DATA_KOREANCALVAL/koreanCalValJL_2026-08-27T02-58-25Z_HYPSO_1"),
               "koreanCalValJL_2026-08-27T02-58-25Z", {"coeff_type": "moved"}),
}


def _process(name):
    directory, capture, calibration = CAPTURES[name]
    if not (directory / f"{capture}-l1a.nc").is_file():
        pytest.skip(f"{directory} not present")
    l1c = (hypso.open(directory / f"{capture}-l1a.nc").calibrate(**calibration)
           .georeference(latitudes=directory / "processing-temp" / "latitudes_indirectgeoref.dat",
                         longitudes=directory / "processing-temp" / "longitudes_indirectgeoref.dat", direct=False))
    production = {level: hypso.open(directory / f"{capture}-moved_unmasked-{level}.nc") for level in ("l1b", "l1c", "l1d")}
    return l1c, l1c.toa_reflectance(), production


@pytest.mark.parametrize("name", list(CAPTURES))
def test_l1b_l1c_match_production(name):
    l1c, _, production = _process(name)
    np.testing.assert_allclose(l1c.cube.values, production["l1b"].cube.values, rtol=1e-6)      # Lt (float32)
    np.testing.assert_array_equal(l1c.geometry["latitude"].values, production["l1c"].geometry["latitude"].values)
    np.testing.assert_allclose(l1c.geometry["solar_zenith"].values,
                               production["l1c"].geometry["solar_zenith"].values, rtol=1e-6)


def test_l1d_matches_production_hypso2():
    _, l1d, production = _process("kemigawa")
    np.testing.assert_allclose(l1d.cube.values, production["l1d"].cube.values, rtol=1e-6)


def test_l1d_hypso1_differs_by_the_distance_fix_only():
    _, l1d, production = _process("hypso1")
    doy = datetime(2026, 8, 27).timetuple().tm_yday
    d_old = 1 - 0.01672 * np.cos(0.9856 * (doy - 4))                 # the pre-fix formula (radians)
    d_new = 1 - 0.01672 * np.cos(np.radians(0.9856 * (doy - 4)))
    bright = l1d.cube.values > 0.01
    ratio = production["l1d"].cube.values[bright] / l1d.cube.values[bright]
    np.testing.assert_allclose(ratio, (d_old / d_new) ** 2, rtol=1e-6)
