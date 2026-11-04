"""RGB exposure adjustment (hypso.rgb.adjust_exposure, ×91): one gain for all
channels, in stops or automatic, optional gamma; synthetic images."""
import numpy as np
import pytest
import xarray as xr

from hypso.rgb import adjust_exposure, exposure_gain


@pytest.fixture
def dark():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(5, 40, (50, 60, 3)).astype(np.float32)
    rgb[..., 2] *= 1.5                                    # bluish, like water
    return rgb


def test_ev_is_a_gain_in_stops_and_keeps_colour(dark):
    out = adjust_exposure(dark, ev=1.0)
    np.testing.assert_allclose(out, np.clip(dark * 2, 0, 255), rtol=1e-6)
    ratio_in, ratio_out = dark[..., 2] / dark[..., 0], out[..., 2] / out[..., 0]
    np.testing.assert_allclose(ratio_out, ratio_in, rtol=1e-5)          # colour unchanged (nothing clipped)
    np.testing.assert_allclose(adjust_exposure(dark, ev=-1.0), dark / 2, rtol=1e-6)
    assert adjust_exposure(dark, ev=6.0).max() == 255                     # clipped at full scale


def test_automatic_exposure(dark):
    out = adjust_exposure(dark, target=0.5)          # low enough that no channel saturates
    assert out.max() < 255
    luma = out @ np.array([0.2126, 0.7152, 0.0722])
    assert np.percentile(luma, 99) == pytest.approx(0.5 * 255, rel=1e-3)
    assert exposure_gain(dark) == pytest.approx(0.9 * 255 / np.percentile(dark @ np.array([0.2126, 0.7152, 0.0722]), 99))
    with pytest.raises(ValueError, match="no positive"):
        exposure_gain(np.zeros((4, 4, 3)))


def test_gamma_normalize_and_types(dark):
    linear = adjust_exposure(dark, ev=1.0, normalize=True)
    lifted = adjust_exposure(dark, ev=1.0, normalize=True, gamma=2.2)
    np.testing.assert_allclose(lifted, linear ** (1 / 2.2), rtol=1e-5)
    assert linear.max() <= 1 and lifted.mean() > linear.mean()
    da = xr.DataArray(dark, dims=("y", "x", "channel"), coords={"channel": ["R", "G", "B"]}, attrs={"source": "x.bmp"})
    out = adjust_exposure(da, ev=0.5)
    assert out.attrs["exposure_ev"] == pytest.approx(0.5) and out.attrs["source"] == "x.bmp"
    ds = xr.Dataset({c: (("y", "x"), dark[..., i].round().astype(np.uint8)) for i, c in enumerate("RGB")})
    out = adjust_exposure(ds, ev=1.0)
    assert out["R"].dtype == np.float32 and out.attrs["exposure_gain"] == pytest.approx(2.0)
    np.testing.assert_allclose(out["G"].values, ds["G"].values * 2.0)
    hsi_like = dark / 255                                                 # 0-1 image, e.g. HSI2RGB
    np.testing.assert_allclose(adjust_exposure(hsi_like, ev=1.0, full_scale=1.0), np.clip(hsi_like * 2, 0, 1),
                               rtol=1e-6)


def test_plot_rgb_map_exposure(dark):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from hypso.rgb import plot_rgb_map
    from hypso.rgb.maps import _rgb_and_grid
    import pyproj
    ds = xr.Dataset({c: (("y", "x"), dark[..., i]) for i, c in enumerate("RGB")},
                    coords={"x": np.arange(60) * 30.0 + 4.5e5, "y": 3.64e6 - np.arange(50) * 30.0})
    ds["crs"] = ((), 0, pyproj.CRS("EPSG:32652").to_cf())
    stretched, *_ = _rgb_and_grid(ds, (2, 98))
    exposed, *_ = _rgb_and_grid(ds, (2, 98), {"ev": 1.0})
    np.testing.assert_allclose(exposed, np.clip(dark * 2 / 255, 0, 1), rtol=1e-6)
    assert not np.allclose(stretched, exposed)
    plot_rgb_map(ds, exposure={"gamma": 2.2}, graticule=False, scale_bar=False, north_arrow=False)


def test_black_point(dark):
    from hypso.rgb import black_level
    levels = black_level(dark, "channel")
    np.testing.assert_allclose(levels, np.percentile(dark, 1, axis=(0, 1)), rtol=1e-6)
    np.testing.assert_allclose(black_level(dark, "common"), np.full(3, levels.min()))
    out = adjust_exposure(dark, ev=1.0, black_point="channel")
    np.testing.assert_allclose(out, np.clip((dark - levels) * 2, 0, 255), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(adjust_exposure(dark, ev=0.0, black_point=5.0), np.clip(dark - 5, 0, 255), atol=1e-5)
    da = xr.DataArray(dark, dims=("y", "x", "channel"))
    assert len(adjust_exposure(da, black_point="common").attrs["exposure_black"]) == 3
    with pytest.raises(ValueError, match="black_point"):
        adjust_exposure(dark, black_point="haze")
