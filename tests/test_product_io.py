"""hypso.io.product: one reader/writer for every level (DataTree model)."""
import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from conftest import requires_real_capture
from hypso.io.product import PRODUCT_DIMS, read_product, write_product

LINES, SAMPLES, BANDS = 4, 3, 5
WAVELENGTHS = np.array([400.04, 450.12, 500.27, 550.33, 600.49])


def _walk(group, prefix=""):
    out = {f"{prefix}/{name}": var for name, var in group.variables.items()}
    for name, sub in group.groups.items():
        out.update(_walk(sub, f"{prefix}/{name}"))
    return out


def _group_attrs(group, prefix=""):
    out = {prefix or "/": {k: group.getncattr(k) for k in group.ncattrs()}}
    for name, sub in group.groups.items():
        out.update(_group_attrs(sub, f"{prefix}/{name}"))
    return out


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def assert_same_file(path_a, path_b, ignore_attrs=("history",)):
    """Same variables (values, dtype, dims, attrs) and group/global attrs."""
    with nc.Dataset(path_a) as a, nc.Dataset(path_b) as b:
        va, vb = _walk(a), _walk(b)
        assert sorted(va) == sorted(vb)
        for name in va:
            x, y = va[name], vb[name]
            assert x.dtype == y.dtype, name
            assert x.dimensions == y.dimensions, name
            assert _same(x[:], y[:]), name
            ax = {k: x.getncattr(k) for k in x.ncattrs()}
            ay = {k: y.getncattr(k) for k in y.ncattrs()}
            assert sorted(ax) == sorted(ay), name
            assert all(_same(ax[k], ay[k]) for k in ax), name
        ga, gb = _group_attrs(a), _group_attrs(b)
        assert sorted(ga) == sorted(gb)
        for g in ga:
            x = {k: v for k, v in ga[g].items() if k not in ignore_attrs}
            y = {k: v for k, v in gb[g].items() if k not in ignore_attrs}
            assert sorted(x) == sorted(y), g
            assert all(_same(x[k], y[k]) for k in x), (g, [k for k in x if not _same(x[k], y[k])])


def _tree(prefix="Lt", level="L1C", dtype=np.float32, geometry=True):
    rng = np.random.default_rng(0)
    cube = (rng.random((LINES, SAMPLES, BANDS)) * 1000).astype(dtype)
    root = xr.Dataset(
        {prefix: (PRODUCT_DIMS, cube)},
        coords={"wavelength": ("bands", np.around(WAVELENGTHS, 1)),
                "radiation_wavelength": ("bands", WAVELENGTHS),
                "fwhm": ("bands", np.full(BANDS, 5.46))},
        attrs={"processing_level": level, "sat_id": "HYPSO-2"})
    if geometry:
        root["latitude"] = (("lines", "samples"), rng.random((LINES, SAMPLES)).astype(np.float32) + 45)
        root["longitude"] = (("lines", "samples"), rng.random((LINES, SAMPLES)).astype(np.float32) + 12)
        root["solar_zenith"] = (("lines", "samples"), np.full((LINES, SAMPLES), 30, np.float32))
    tree = xr.DataTree(root)
    tree["metadata/capture_config"] = xr.DataTree(xr.Dataset(attrs={"exposure": 40.0, "frame_count": LINES}))
    tree["metadata/timing"] = xr.DataTree(xr.Dataset({"timestamps": ("lines", np.arange(LINES, dtype=float))}))
    return tree


# --- synthetic: no real data needed ---

def test_round_trip_is_exact(tmp_path):
    first, second = tmp_path / "a.nc", tmp_path / "b.nc"
    write_product(_tree(), first)
    write_product(read_product(first), second)
    assert_same_file(first, second)


def test_written_layout(tmp_path):
    path = tmp_path / "a.nc"
    write_product(_tree(), path)
    with nc.Dataset(path) as f:
        bands = [v for v in f.variables if v.startswith("Lt_")]
        assert bands == ["Lt_400", "Lt_450", "Lt_500", "Lt_550", "Lt_600"]
        var = f["Lt_450"]
        assert var.parameter == "Lt_450" and var.wave_name == "450" and var.band == 1
        assert var.wavelength == 450.1 and var.radiation_wavelength == 450.12
        assert var.coordinates == "latitude longitude" and var.grid_mapping == "crs_wgs84"
        assert "_FillValue" not in var.ncattrs()
        assert var.filters()["zlib"] and var.chunking() == [LINES, SAMPLES]
        assert f["latitude"].units == "degrees_north" and "crs_wgs84" in f.variables
        assert f.Conventions.startswith("CF-") and f.processing_level == "L1C"
        assert set(f.groups) == {"metadata"}


def test_without_geometry_has_no_geolocation_references(tmp_path):
    path = tmp_path / "a.nc"
    write_product(_tree(level="L1B", geometry=False), path)
    with nc.Dataset(path) as f:
        assert "coordinates" not in f["Lt_400"].ncattrs()
        assert "crs_wgs84" not in f.variables and "latitude" not in f.variables


def test_dn_is_written_as_uint16(tmp_path):
    path = tmp_path / "a.nc"
    write_product(_tree(prefix="dn", level="L1A", dtype=np.uint16, geometry=False), path)
    with nc.Dataset(path) as f:
        assert f["dn_400"].dtype == np.uint16
    assert read_product(path).dataset["dn"].dtype == np.uint16


def _write_legacy(path, ground_station_l1a=False):
    """A pre-refactor grouped file: /products + /geometry + legacy attrs."""
    rng = np.random.default_rng(1)
    with nc.Dataset(path, "w") as f:
        f.sat_id = "HYPSO-2"
        for dim, size in (("lines", LINES), ("samples", SAMPLES), ("bands", BANDS), ("gcps", 2)):
            f.createDimension(dim, size)
        products = f.createGroup("products")
        corrections = f.createGroup("metadata/corrections")
        wl = corrections.createVariable("wavelengths", "f4", ("bands",))
        wl[:] = WAVELENGTHS
        if ground_station_l1a:
            dn = products.createVariable("dn", "u2", ("lines", "samples", "bands"))
            dn[:] = rng.integers(0, 4000, (LINES, SAMPLES, BANDS))
            return
        f.processing_level = "L1D"
        for band, wave in enumerate(np.around(WAVELENGTHS, 1)):
            var = products.createVariable(f"rhot_{int(wave)}", "f4", ("lines", "samples"))
            var[:] = rng.random((LINES, SAMPLES))
            var.units, var.wave, var.band = "", wave, band
            var.wavelength, var.radiation_wavelength, var.fwhm = wave, (WAVELENGTHS[band],), 5.46
            var.coordinates = "/geometry/longitude /geometry/latitude"
        geometry = f.createGroup("geometry")
        lat = geometry.createVariable("latitude", "f4", ("lines", "samples"))
        lat[:] = 45.0
        lat.units, lat.valid_min, lat.valid_max = "degrees", -180.0, 180.0
        lon = geometry.createVariable("longitude", "f4", ("lines", "samples"))
        lon[:] = 12.0
        gcp = f.createGroup("metadata/gcp")
        gcp.createVariable("latitude", "f4", ("gcps",))[:] = [45.0, 45.1]


def test_legacy_grouped_file_is_normalised(tmp_path):
    legacy, out = tmp_path / "legacy.nc", tmp_path / "out.nc"
    _write_legacy(legacy)
    tree = read_product(legacy)
    assert tree.dataset["rhot"].dims == PRODUCT_DIMS
    np.testing.assert_allclose(tree.dataset["radiation_wavelength"], WAVELENGTHS)
    write_product(tree, out)
    with nc.Dataset(out) as f:
        assert "products" not in f.groups and "geometry" not in f.groups
        assert f["latitude"].units == "degrees_north" and f["latitude"].valid_max == 90
        assert f["rhot_400"].units == "1" and "wave" not in f["rhot_400"].ncattrs()
        assert f["rhot_400"].coordinates == "latitude longitude"
        assert set(f["metadata/gcp"].variables) == {"gcp_latitude"}


def test_ground_station_l1a(tmp_path):
    source, out = tmp_path / "l1a.nc", tmp_path / "out.nc"
    _write_legacy(source, ground_station_l1a=True)
    tree = read_product(source)
    ds = tree.dataset
    assert ds.attrs["processing_level"] == "L1A" and ds["dn"].dtype == np.uint16
    assert np.all(ds["fwhm"].values > 0)  # from the HYPSO-2 sensor profile
    write_product(tree, out)
    with nc.Dataset(source) as a, nc.Dataset(out) as b:
        for band in range(BANDS):
            name = f"dn_{int(np.around(WAVELENGTHS, 1)[band])}"
            assert np.array_equal(b[name][:], a["products/dn"][:, :, band])
        assert not [k for k in b.ncattrs() if k.endswith("_file")]


# --- real capture: the current writer's files survive read + write unchanged ---

@requires_real_capture
@pytest.mark.parametrize("level", ["l1b", "l1c", "l1d", "l2a"])
def test_real_files_round_trip(written_nc_files, tmp_path, level):
    out = tmp_path / f"rt-{level}.nc"
    write_product(read_product(written_nc_files[level]), out)
    assert_same_file(written_nc_files[level], out)
