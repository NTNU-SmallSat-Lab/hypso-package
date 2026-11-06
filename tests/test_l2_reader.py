"""demo/hypso_l2_reader.py, the standalone L2 reader (numpy and xarray only, no
hypso import), on L2 files written by the package (x103): one Dataset with the
reflectance, concentrations and IOPs, per-band variables stacked into spectra."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hypso

DEMO = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO))
from hypso_l2_reader import find_l2_files, flag_mask, open_l2  # noqa: E402

IOP_BANDS, AOD_BANDS = (441, 490, 560), (441, 490)


def test_reader_does_not_import_hypso():
    tree = ast.parse((DEMO / "hypso_l2_reader.py").read_text())
    modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules <= {"argparse", "re", "collections", "pathlib", "numpy", "xarray"}, modules


@pytest.fixture(scope="module")
def capture(l1d, tmp_path_factory):
    """An L2A (DPS, rhos) with tool-style products: chla and SPM (-> L2B.BGC),
    per-band aph (-> L2B.IOP), per-band aod (stays in the L2A), and
    synthetic intermediates Lt and tg_sol (-> L2A.AUX; DPS has none of its own)."""
    from hypso.io.product import INTERMEDIATES_NODE
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:50] = True
    l2a = l1d.with_masks(land=land).correct_atmosphere("dps")
    shape = l2a.shape
    rng = np.random.default_rng(0)

    def band(value, units, nm):
        return xr.DataArray(np.full(shape, value, np.float32), attrs={"units": units, "wavelength": float(nm),
                                                                      "long_name": f"Test ({nm} nm)"})
    products = dict(chla=xr.DataArray(rng.uniform(0.1, 10, shape).astype(np.float32), attrs={"units": "mg m-3"}),
                    SPM=xr.DataArray(np.full(shape, 3.0, np.float32), attrs={"units": "g m-3"}))
    products.update({f"aph_{nm}": band(0.01 * i + 0.01, "m-1", nm) for i, nm in enumerate(IOP_BANDS)})
    products.update({f"aod_{nm}": band(0.1 * i + 0.1, "1", nm) for i, nm in enumerate(AOD_BANDS)})
    product = l2a.with_variables(**products)
    tree = product.tree
    tree[INTERMEDIATES_NODE] = xr.DataTree(xr.Dataset({
        f"{q}_{nm}": (("lines", "samples"), np.full(shape, value, np.float32), {"units": units, "wavelength": float(nm)})
        for q, value, units in (("Lt", 0.03, "sr-1"), ("tg_sol", 0.97, "1")) for nm in (560, 665)}))
    product = hypso.Product(tree)
    out = tmp_path_factory.mktemp("capture")
    path = product.write(out_dir=out)
    return out, path, product


def test_files_found(capture):
    out, path, _ = capture
    files = find_l2_files(out)
    assert set(files) == {"RHOS", "BGC", "IOP", "AUX"}, sorted(p.name for p in out.iterdir())
    assert files["RHOS"] == path


def test_one_dataset(capture):
    out, path, _ = capture
    ds = open_l2(out)
    written = hypso.open(path)
    # the reflectance: the package's cube as written, on one wavelength dimension
    assert ds["rhos"].dims == ("lines", "samples", "wavelength")
    np.testing.assert_allclose(ds["wavelength"].values, written.wavelengths, atol=1e-6)
    np.testing.assert_array_equal(ds["rhos"].values, written.cube.values)
    assert "fwhm" in ds.coords
    # concentrations (L2B.BGC), IOPs (L2B.IOP) and aod (in the L2A) in the same Dataset
    bgc = hypso.open(find_l2_files(out)["BGC"])
    np.testing.assert_array_equal(ds["chla"].values, bgc.variables["chla"].values)
    assert ds["SPM"].attrs["units"] == "g m-3"
    assert ds["aph"].dims == ("lines", "samples", "wavelength_iop")
    np.testing.assert_array_equal(ds["wavelength_iop"].values, IOP_BANDS)
    np.testing.assert_allclose(ds["aph"].sel(wavelength_iop=490).values, 0.02, rtol=1e-4)
    np.testing.assert_array_equal(ds["wavelength_aod"].values, AOD_BANDS)
    np.testing.assert_allclose(ds["aod"].sel(wavelength_aod=441).values, 0.1, rtol=1e-4)
    assert ds["aph"].attrs["long_name"] == "Test" and ds["aph"].attrs["units"] == "m-1"
    # geometry and flags once; no AUX unless asked for
    assert {"latitude", "longitude"} <= set(ds.coords) and "solar_zenith" in ds
    assert not any(name.startswith("Lt") for name in ds.variables)
    assert "aph_441" not in ds and "aph_441" in open_l2(out, spectra=False)


def test_aux_and_flags(capture, l1d):
    out, _, _ = capture
    ds = open_l2(out, aux=True)
    assert ds["Lt"].dims == ("lines", "samples", "wavelength_aux") and ds["tg_sol"].dims == ds["Lt"].dims
    np.testing.assert_array_equal(ds["wavelength_aux"].values, [560, 665])
    np.testing.assert_allclose(ds["tg_sol"].values, 0.97, rtol=1e-4)
    bits = ds["pixel_flags"].values.astype(int)
    assert (flag_mask(ds).values == (bits != 0)).all()
    land = flag_mask(ds, "land").values
    assert land[:50].all() and land.sum() == 50 * land.shape[1]
    with pytest.raises(KeyError, match="no pixel_flags"):
        flag_mask(ds.drop_vars("pixel_flags"))
    with pytest.raises(KeyError, match="unknown flags"):
        flag_mask(ds, "snow")


def test_lazy_and_merge(capture, tmp_path):
    out, _, _ = capture
    eager = open_l2(out)
    lazy = open_l2(out, lazy=True)
    assert hasattr(lazy["rhos"].data, "dask")
    np.testing.assert_array_equal(lazy["aph"].values, eager["aph"].values)
    merged = tmp_path / "merged.nc"
    eager.to_netcdf(merged)
    back = xr.open_dataset(merged)
    np.testing.assert_array_equal(back["rhos"].values, eager["rhos"].values)


def test_several_methods_need_a_choice(capture, tmp_path):
    out, path, _ = capture
    import shutil
    other = tmp_path / "two"
    shutil.copytree(out, other)
    shutil.copy(path, other / path.name.replace(".dps.", ".machi."))
    with pytest.raises(ValueError, match="several methods"):
        open_l2(other)
    assert set(open_l2(other, method="machi").data_vars) >= {"rhos"}
