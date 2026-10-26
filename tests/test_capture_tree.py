"""hypso.io.capture_tree: capture object -> product tree -> write_product
gives the old writer's file (io/writer.py) plus only the format-2 additions."""
import netCDF4 as nc
import numpy as np
import pytest

from conftest import L1A_PATH, requires_real_capture
from hypso.io.capture_tree import capture_to_tree
from hypso.io.product import write_product
from test_product_io import FORMAT_2_ATTRS, assert_same_file

pytestmark = requires_real_capture


@pytest.mark.parametrize("level, correction", [("l1b", None), ("l1c", None), ("l1d", None), ("l2a", "testac")])
def test_matches_old_writer(satobj, written_nc_files, tmp_path, level, correction):
    out = tmp_path / f"new-{level}.nc"
    write_product(capture_to_tree(satobj, level, correction=correction), out)
    assert_same_file(written_nc_files[level], out, ignore_attrs=FORMAT_2_ATTRS,
                     ignore_groups=("/metadata/processing",))


def test_provenance_recorded(satobj, tmp_path):
    out = tmp_path / "l1d.nc"
    write_product(capture_to_tree(satobj, "l1d"), out)
    with nc.Dataset(out) as f:
        step = f["metadata/processing/l1d"]
        assert step.input_file == L1A_PATH.name and step.coeff_type == "moved"
        assert step.georeferencing == "indirect" and step.ssi_source == "tsis"
        assert step.calibration_package == "hypso2_calibration"
        assert len(step.radiometric_sha256) == 64 and step.radiometric_file.endswith(".npz")
        assert f.history.splitlines()[-1].split(": ", 1)[1].startswith(f"l1d from {L1A_PATH.name}, hypso-package ")


def test_l1a_from_capture(satobj, tmp_path):
    out = tmp_path / "l1a.nc"
    write_product(capture_to_tree(satobj, "l1a"), out)
    with nc.Dataset(out) as f, nc.Dataset(L1A_PATH) as src:
        names = sorted((v for v in f.variables if v.startswith("dn_")), key=lambda v: int(f[v].band))
        assert len(names) == src["products/dn"].shape[-1] and f[names[0]].dtype == np.uint16
        assert all(np.array_equal(f[n][:], src["products/dn"][:, :, i]) for i, n in enumerate(names))
        assert not [k for k in f.ncattrs() if k.endswith("_file")]   # L1A is not calibrated


def test_masks_become_flags(satobj, tmp_path):
    land = np.zeros(satobj.spatial_dimensions, bool)
    land[:10] = True
    satobj.land_mask = land
    try:
        out = tmp_path / "l1d.nc"
        write_product(capture_to_tree(satobj, "l1d"), out)
        with nc.Dataset(out) as f:
            assert f["pixel_flags"].flag_meanings == "land"
            assert np.array_equal(f["pixel_flags"][:] == 1, land)
            assert not np.isnan(f["rhot_378"][:10]).any()   # product unmasked
    finally:
        satobj.land_mask = None
