"""RGB camera registration and maps (hypso.rgb.registration / maps, ×84)."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hypso

CAPTURE = Path("/home/camerop/HYPSO_DATA_AOC/ariake_2025-04-04T02-27-25Z")
pytestmark = pytest.mark.skipif(not CAPTURE.is_dir(), reason=f"{CAPTURE} not present")


@pytest.fixture(scope="module")
def registered():
    from hypso.rgb import find_camera_image, read_camera_image, register_camera
    l1d = hypso.open(CAPTURE / "ariake_2025-04-04T02-27-25Z-moved_unmasked-l1d.nc", wavelengths=[640, 550, 460])
    camera = read_camera_image(find_camera_image(CAPTURE))
    return l1d, camera, register_camera(camera, l1d)


def test_registration_passes_quality_check(registered):
    _, _, reg = registered
    assert reg.ok and reg.reason == ""
    assert reg.inliers >= 100 and reg.rms_m < 300 and reg.coverage > 0.5


def test_georeference_and_pixel_mappings(registered):
    from hypso.rgb import georeference_camera, hsi_to_camera_pixels
    l1d, camera, reg = registered
    geo = georeference_camera(camera, reg, l1d)
    assert geo["latitude"].shape == camera.shape[:2] and geo.attrs["registration_ok"] == "true"
    assert 32.0 < float(geo["latitude"].mean()) < 33.6 and 129.6 < float(geo["longitude"].mean()) < 131.1
    assert float(geo["in_hsi_swath"].mean()) > 0.8
    # camera pixel -> its lat/lon -> HSI pixel -> back to the camera: within a few camera pixels
    rows, cols = hsi_to_camera_pixels(reg, l1d, camera.shape[:2])
    r, c = 1374, 1920
    line, sample = l1d.pixel(float(geo["latitude"][r, c]), float(geo["longitude"][r, c]))
    assert abs(rows[line, sample] - r) < 30 and abs(cols[line, sample] - c) < 30     # an HSI pixel spans ~28 x 3 camera px


def test_regrid_camera_and_map(registered, tmp_path):
    from hypso.rgb import georeference_camera, plot_rgb_map, regrid_camera
    l1d, camera, reg = registered
    geo = georeference_camera(camera, reg)
    hsi_grid = l1d.regrid(resolution=300)
    shared = regrid_camera(geo, hsi_grid.area_definition())
    assert shared["R"].shape == hsi_grid.cube.shape[:2] and shared.attrs["grid_method"] == "average"
    own = regrid_camera(geo)
    assert own.attrs["grid_method"] == "nearest" and own.attrs["grid"].startswith("utm52n_")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = plot_rgb_map(shared, graticule=True, scale_bar=True, north_arrow=True, title="test")
    ax.figure.savefig(tmp_path / "map.png")
    plot_rgb_map(hsi_grid)                                   # a regridded product works too


def test_unregistrable_image_is_rejected(registered):
    from hypso.rgb import register_camera
    l1d, camera, _ = registered
    noise = xr.DataArray(np.random.default_rng(0).random((687, 960, 3)).astype(np.float32) * 255,
                         dims=("y", "x", "channel"), coords={"channel": ["R", "G", "B"]})
    reg = register_camera(noise, l1d, downsample=1.0)
    assert not reg.ok and reg.reason
