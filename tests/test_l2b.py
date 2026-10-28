"""L2B products (TODO 4i step 1, ×87): geophysical products derived from an
L2A, 2D variables only, through naming, read/write, regrid, window and Satpy.
The products here are synthetic (a band ratio); the algorithms come later."""
import numpy as np
import pytest

import hypso
from hypso.processing import make_l2b

SITE = (45.3139, 12.5083)


@pytest.fixture(scope="module")
def l2a(l1d):
    return l1d.correct_atmosphere("dps")


@pytest.fixture(scope="module")
def l2b(l2a):
    cube = l2a.cube
    blue, green = (cube.isel(bands=int(np.argmin(np.abs(l2a.wavelengths - w)))).values for w in (490, 555))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (blue / green).astype(np.float32)
    variables = {"band_ratio": (ratio, {"units": "1", "long_name": "rhos(490) / rhos(555)"}),
                 "chlor_a": ((10 ** (0.3 - 2.0 * np.log10(ratio))).astype(np.float32),
                             {"units": "mg m-3", "long_name": "Chlorophyll-a concentration (synthetic)"})}
    return hypso.Product(make_l2b(l2a.tree, "CHL", "test", variables, parameters={"note": "synthetic"}))


def test_l2b_product(l2b, l2a):
    assert l2b.level == "L2B" and l2b.shape == l2a.shape
    assert set(l2b.variables.data_vars) == {"band_ratio", "chlor_a"}
    assert "latitude" in l2b.geometry and l2b.flags.keys() == l2a.flags.keys()
    assert l2b.filename() == "HYPSO2_HSI.20250304T103805.aeronetvenice.L2B.CHL.test.nc"
    step = l2b.provenance[-1]
    assert step["step"] == "l2b_chl_test" and step["note"] == "synthetic" and step["variables"] == "band_ratio chlor_a"
    with pytest.raises(ValueError, match="no band cube"):
        l2b.cube
    assert "L2B CHL" in repr(l2b)


def test_l2b_roundtrip(l2b, tmp_path):
    path = l2b.write(out_dir=tmp_path)
    back = hypso.open(path)
    assert back.level == "L2B" and back.filename() == path.name
    for name in ("band_ratio", "chlor_a"):
        np.testing.assert_allclose(back.variables[name].values, l2b.variables[name].values, rtol=1e-4,
                                   equal_nan=True)                       # 14-bit quantization
        assert back.variables[name].attrs["units"] == l2b.variables[name].attrs["units"]
    assert back.tree.attrs["l2b_group"] == "CHL" and back.tree.attrs["l2b_source"] == "test"
    lossless = hypso.open(l2b.write(out_dir=tmp_path, label="lossless", significant_bits=None))
    np.testing.assert_array_equal(lossless.variables["chlor_a"].values, l2b.variables["chlor_a"].values)


def test_l2b_regrid_window_satpy(l2b, tmp_path):
    from pyresample.geometry import AreaDefinition, SwathDefinition
    grid = l2b.regrid(resolution=3000, method="average")
    assert grid.level == "L2B" and grid.grid.endswith("_3000m") and "chlor_a" in grid.variables
    assert grid.filename().endswith(".L2B.CHL.test." + grid.grid + ".nc")
    window = l2b.window(*SITE, size=3)
    assert list(window["products"].values) == ["band_ratio", "chlor_a"]
    assert list(window["units"].values) == ["1", "mg m-3"] and window.attrs["l2b_group"] == "CHL"
    assert np.isfinite(window["median"].values).all()
    scene = l2b.to_satpy()
    assert isinstance(scene["chlor_a"].attrs["area"], SwathDefinition)
    satpy = pytest.importorskip("satpy")
    for product, area_type in ((l2b, SwathDefinition), (grid, AreaDefinition)):
        files = satpy.Scene(filenames=[str(product.write(out_dir=tmp_path))], reader="hypso_l2b")
        assert {"chlor_a", "band_ratio"} <= set(files.available_dataset_names())
        assert not {"latitude_direct", "pixel_flags", "x"} & set(files.available_dataset_names())
        files.load(["chlor_a"])
        assert isinstance(files["chlor_a"].attrs["area"], area_type)
        assert files["chlor_a"].attrs["units"] == "mg m-3"


def test_make_l2b_errors(l2a, l1d):
    good = {"x": (np.zeros(l2a.shape, np.float32), {"units": "1"})}
    with pytest.raises(ValueError, match="derived from L2A"):
        make_l2b(l1d.tree, "CHL", "test", good)
    with pytest.raises(ValueError, match="unknown L2B group"):
        make_l2b(l2a.tree, "CHLA", "test", good)
    with pytest.raises(ValueError, match="shape"):
        make_l2b(l2a.tree, "CHL", "test", {"x": (np.zeros((2, 2)), {})})
    with pytest.raises(ValueError, match="geometry or flag"):
        make_l2b(l2a.tree, "CHL", "test", {"latitude": (np.zeros(l2a.shape), {})})
    with pytest.raises(ValueError, match="may only contain"):
        make_l2b(l2a.tree, "CHL", "oc.4", good)


def test_tool_products_move_to_l2b(l2a, tmp_path):
    # TODO 4i step 2 (x88): an AC tool's own products go to L2B files, its
    # diagnostics stay in L2A, its flags are kept in L2A and copied to each L2B
    import xarray as xr
    from hypso.processing.l2b import tool_product_group
    shape = l2a.shape
    tool = l2a.with_variables(
        chla=xr.DataArray(np.full(shape, 1.5, np.float32), attrs={"units": "mg m-3"}),
        SPM=xr.DataArray(np.full(shape, 3.0, np.float32), attrs={"units": "g m-3"}),
        logfb=xr.DataArray(np.zeros(shape, np.float32), attrs={"units": "1"}),
        polymer_flags=xr.DataArray(np.zeros(shape, np.int32), attrs={"long_name": "Polymer flags"}))
    assert [tool_product_group(n) for n in ("chla", "chl_re_mishra", "chlor_a_oci", "ndci", "SPM_Nechad2010_664",
                                            "tsm_yoc", "Kd_490", "logfb", "Rgli", "polymer_flags")] == \
        ["CHL", "CHL", "CHL", "CHL", "TSM", "TSM", "KD", None, None, None]
    products = tool.tool_products()
    assert set(products) == {"CHL", "TSM"}
    assert set(products["CHL"].variables.data_vars) == {"chla", "polymer_flags"}
    assert set(products["TSM"].variables.data_vars) == {"SPM", "polymer_flags"}
    assert products["CHL"].filename().endswith(".L2B.CHL.dps.nc")
    assert l2a.tool_products() == {}

    path = tool.write(out_dir=tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.nc"))
    assert written == sorted([path.name, products["CHL"].filename(), products["TSM"].filename()])
    back = hypso.open(path)
    assert set(back.variables.data_vars) == {"logfb", "polymer_flags"}        # products moved out
    chl = hypso.open(tmp_path / products["CHL"].filename())
    np.testing.assert_allclose(chl.variables["chla"].values, 1.5)

    alone = tmp_path / "alone"
    alone.mkdir()
    tool.write(out_dir=alone, tool_products=False)
    assert len(list(alone.glob("*.nc"))) == 1


@pytest.fixture(scope="module")
def l2a_rrs(l2a):
    # An L2A.RRS product with one known spectrum everywhere: productive water
    # (green peak, red-edge peak), so OC4 and NDCI are in their valid range.
    tree = l2a.tree
    root = tree.to_dataset(inherit=False)
    wl = root["rhos"]["radiation_wavelength"].values
    spectrum = np.interp(wl, [400, 443, 490, 510, 560, 670, 705, 800],
                         [0.002, 0.0025, 0.004, 0.005, 0.008, 0.002, 0.004, 0.0005]).astype(np.float32)
    cube = root["rhos"].copy(data=np.broadcast_to(spectrum, root["rhos"].shape).copy())
    root = root.drop_vars("rhos")
    root["Rrs"] = cube
    root.attrs["l2a_quantity"] = "RRS"
    tree.dataset = root
    return hypso.Product(tree)


def test_derive_chlorophyll_and_tsm(l2a_rrs, l2a, tmp_path):
    # TODO 4i step 3 (x89): the package's own algorithms
    from hypso.processing.l2b_algorithms import NECHAD2010_665, OC4_OLCI, band_at
    chl = l2a_rrs.derive("CHL")
    assert chl.filename().endswith(".L2B.CHL.oci.nc")
    assert set(chl.variables.data_vars) == {"chlor_a", "chl_oc4", "chl_ci"}
    cube, wl = l2a_rrs.cube.values[:1, :1], l2a_rrs.wavelengths
    blue = max(band_at(cube, wl, w)[0, 0] for w in (443, 490, 510))
    x = np.log10(blue / band_at(cube, wl, 560)[0, 0])
    expected_oc4 = 10 ** sum(a * x ** i for i, a in enumerate(OC4_OLCI))
    np.testing.assert_allclose(chl.variables["chl_oc4"].values[0, 0], expected_oc4, rtol=1e-5)
    # productive water: CI is above the 0.35 mg m-3 blend edge, so chlor_a = OC4
    assert chl.variables["chl_ci"].values[0, 0] > 0.35
    np.testing.assert_allclose(chl.variables["chlor_a"].values[0, 0], expected_oc4, rtol=1e-5)
    assert chl.variables["chlor_a"].attrs["standard_name"] == "mass_concentration_of_chlorophyll_a_in_sea_water"
    assert chl.provenance[-1]["step"] == "l2b_chl_oci" and "O'Reilly" in chl.provenance[-1]["reference"]

    tsm = l2a_rrs.derive("TSM")
    rhow = np.pi * band_at(cube, wl, 665)[0, 0]
    c = NECHAD2010_665
    np.testing.assert_allclose(tsm.variables["spm_nechad2010_665"].values[0, 0],
                               c["A"] * rhow / (1 - rhow / c["C"]), rtol=1e-5)
    assert tsm.filename().endswith(".L2B.TSM.nechad2010.nc")

    ndci = l2a.derive("CHL", "ndci")                     # NDCI also accepts rhos (an index)
    assert {"chl_ndci", "ndci"} == set(ndci.variables.data_vars)
    back = hypso.open(chl.write(out_dir=tmp_path))
    np.testing.assert_allclose(back.variables["chlor_a"].values, chl.variables["chlor_a"].values, rtol=1e-4)

    with pytest.raises(ValueError, match="needs an L2A with Rrs"):
        l2a.derive("CHL")                                # OCI on surface reflectance
    with pytest.raises(ValueError, match="no algorithm"):
        l2a_rrs.derive("CHL", "oc3")
    with pytest.raises(ValueError, match="no package algorithm"):
        l2a_rrs.derive("IOP")
