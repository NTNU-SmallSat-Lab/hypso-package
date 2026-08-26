"""Tests for hypso.resample's generalized cube/product resampling.
resample_l1a_cube/l1b_cube/l1c_cube/l1d_cube were four independent,
near-identical functions (each differing only in which satobj.l1X_cube
attribute it read) - now thin wrappers over one resample_cube(); this also
makes resample_products() (previously a stub - its only body was a
commented-out block referencing the since-deleted DataArrayDict) a real,
working implementation for free, since it just loops resample_cube() over
satobj.products (hypso.containers.DatasetDict).

Needs real geolocation (satobj.latitudes/resolution) and a real
pyresample.AreaDefinition, so every test here is gated on the reference
capture, like the other real-data tests in this suite."""
import numpy as np
import pytest

from conftest import requires_real_capture

pytestmark = requires_real_capture


@pytest.fixture(scope="module")
def area_def(satobj):
    from pyresample.geometry import AreaDefinition

    lats = np.asarray(satobj.latitudes)
    lons = np.asarray(satobj.longitudes)
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    return AreaDefinition(
        "test_area", "test", "test_proj",
        {"proj": "longlat", "datum": "WGS84"},
        40, 30,
        (lon_min, lat_min, lon_max, lat_max),
    )


@pytest.mark.parametrize("level", ["l1a", "l1b", "l1c", "l1d"])
def test_resample_level_wrappers_share_one_implementation(satobj, area_def, level):
    from hypso import resample as resample_mod

    fn = getattr(resample_mod, f"resample_{level}_cube")
    resampled_data, resampled_lat, resampled_lon = fn(satobj, area_def)

    assert resampled_data.shape[:2] == (30, 40)
    assert resampled_data.shape[2] == getattr(satobj, f"{level}_cube").shape[2]
    assert resampled_lat.shape == (30, 40)
    assert resampled_lon.shape == (30, 40)
    # some overlap between the swath and the target area is expected -
    # if literally nothing resampled, something is structurally broken
    assert np.isfinite(resampled_data.values).any()


def test_resample_cube_use_indirect_georef_does_not_crash(satobj, area_def):
    # Regression test: use_indirect_georef=True used to read
    # satobj.latitudes_indirect/longitudes_indirect/resolution_indirect -
    # names never set anywhere (only _direct-suffixed names exist for the
    # OTHER georeferencing method) - so this raised AttributeError. Fixed to
    # read the same satobj.latitudes/longitudes/resolution the default path
    # uses (run_georeferencing is what actually populates those - the
    # "indirect", externally-supplied-lat/lon path).
    from hypso.resample import resample_cube

    resampled_data, resampled_lat, resampled_lon = resample_cube(
        satobj.l1a_cube, satobj, area_def, use_indirect_georef=True)

    assert resampled_data.shape[:2] == (30, 40)
    assert resampled_lat.shape == (30, 40)
    assert resampled_lon.shape == (30, 40)


def test_resample_cube_is_the_shared_implementation(satobj, area_def):
    from hypso.resample import resample_cube, resample_l1a_cube

    direct = resample_cube(satobj.l1a_cube, satobj, area_def)
    via_wrapper = resample_l1a_cube(satobj, area_def)

    assert np.array_equal(direct[0].values, via_wrapper[0].values, equal_nan=True)
    assert np.array_equal(direct[1], via_wrapper[1])
    assert np.array_equal(direct[2], via_wrapper[2])


def test_resample_products_is_a_real_implementation_not_a_stub(satobj, area_def):
    # resample_products used to just print an error and return None - this
    # pins that it now actually resamples every registered product into a
    # real xr.Dataset.
    from hypso.resample import resample_products
    import xarray as xr

    fake_chla = np.random.rand(*satobj.spatial_dimensions).astype(np.float64)
    satobj.products['chla'] = fake_chla
    try:
        ds = resample_products(satobj, area_def)

        assert isinstance(ds, xr.Dataset)
        assert "chla" in ds.data_vars
        assert ds["chla"].shape == (30, 40)
        assert "latitude" in ds.coords and "longitude" in ds.coords
        assert ds.coords["latitude"].shape == (30, 40)
        assert np.isfinite(ds["chla"].values).any()
    finally:
        del satobj.products['chla']  # keep the shared session fixture clean


def test_resample_products_empty_when_no_products_registered(satobj, area_def):
    from hypso.resample import resample_products

    assert list(satobj.products) == []  # confirm clean before asserting on empty result
    ds = resample_products(satobj, area_def)
    assert list(ds.data_vars) == []
