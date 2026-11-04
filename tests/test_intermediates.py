"""The AC tool's intermediates (its other per-band outputs, e.g. OC-SMART's Lt,
Lr, Lrc, tg_sol): kept out of the L2A's variables, written by Product.write
as the L2A.AUX file next to the L2A (×98). Synthetic intermediates on a DPS
L2A here; test_api checks a real OC-SMART output end to end."""
import netCDF4
import numpy as np
import pytest
import xarray as xr

import hypso
from hypso.io.product import INTERMEDIATES_NODE
from hypso.processing.l2b import split_intermediates


@pytest.fixture(scope="module")
def l2a(l1d):
    """A DPS L2A with two synthetic intermediates (as _l2a_tree attaches them)
    and a synthetic tool chlorophyll (which goes to an L2B file)."""
    product = l1d.correct_atmosphere("dps")
    tree = product.tree
    shape = product.shape
    lt = np.linspace(0.01, 0.05, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    tree[INTERMEDIATES_NODE] = xr.DataTree(xr.Dataset({
        "Lt_560": (("lines", "samples"), lt, {"units": "sr-1", "long_name": "Lt (560 nm)", "wavelength": 560.0}),
        "tg_sol_560": (("lines", "samples"), np.full(shape, 0.97, np.float32), {"units": "1", "wavelength": 560.0}),
    }))
    root = tree.to_dataset(inherit=False)
    root["chlor_a_test"] = (("lines", "samples"), np.full(shape, 1.5, np.float32), {"units": "mg m-3"})
    tree.dataset = root
    return hypso.Product(tree)


def test_intermediates_product(l2a):
    aux = l2a.intermediates()
    assert aux.level == "L2A" and aux.shape == l2a.shape
    assert {"Lt_560", "tg_sol_560"} <= set(aux.variables.data_vars)
    assert "latitude" in aux.geometry and aux.flags.keys() == l2a.flags.keys()
    assert aux.filename() == "HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.AUX.dps.nc"
    assert aux.provenance[-1]["step"] == "l2a_aux_dps"
    with pytest.raises(ValueError, match="no band cube"):
        aux.cube
    assert "L2A AUX" in repr(aux)
    # the L2A itself is unchanged, and an L2A without intermediates has none
    assert INTERMEDIATES_NODE in l2a.tree.children
    assert split_intermediates(split_intermediates(l2a.tree)[0])[1] is None


def test_write_puts_intermediates_in_their_own_file(l2a, tmp_path):
    path = l2a.write(out_dir=tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.AUX.dps.nc",
                     "HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.RHOS.dps.nc",
                     "HYPSO2_HSI.20250304T103805.aeronetvenice.L2B.CHL.dps.nc"]
    with netCDF4.Dataset(path) as f:                      # not in the L2A file
        assert INTERMEDIATES_NODE not in f.groups and "Lt_560" not in f.variables
    for name in names:
        if ".L2B." in name:
            with netCDF4.Dataset(tmp_path / name) as f:       # nor in the L2B file
                assert INTERMEDIATES_NODE not in f.groups and "Lt_560" not in f.variables
    aux = hypso.open(tmp_path / names[0])
    assert aux.level == "L2A" and aux.tree.attrs["l2a_quantity"] == "AUX"
    expected = l2a.tree[INTERMEDIATES_NODE].dataset
    for name in ("Lt_560", "tg_sol_560"):
        np.testing.assert_allclose(aux.variables[name].values, expected[name].values, rtol=5e-5)   # 14 bits
        assert aux.variables[name].attrs["units"] == expected[name].attrs["units"]


def test_write_without_intermediates(l2a, tmp_path, caplog):
    l2a.write(out_dir=tmp_path, intermediates=False)
    assert not list(tmp_path.glob("*.L2A.AUX.*"))
    assert "intermediates not written" in caplog.text


def test_regrid_drops_intermediates(l2a, caplog):
    import logging
    with caplog.at_level(logging.INFO):
        gridded = l2a.regrid(resolution=300)
    assert INTERMEDIATES_NODE not in gridded.tree.children
    assert "drops the AC tool's intermediates" in caplog.text


def test_satpy_reads_the_aux_file(l2a, tmp_path):
    satpy = pytest.importorskip("satpy")
    l2a.write(out_dir=tmp_path)
    aux_file = next(tmp_path.glob("*.L2A.AUX.*"))
    scene = satpy.Scene(filenames=[str(aux_file)], reader="hypso_l2a")
    assert {"Lt_560", "tg_sol_560"} <= set(scene.available_dataset_names())
    scene.load(["Lt_560"])
    assert scene["Lt_560"].attrs["units"] == "sr-1"
