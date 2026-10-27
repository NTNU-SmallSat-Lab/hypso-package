"""RGB camera products: camera_product / write_camera / open_camera and the
hypso_rgb_l1a Satpy reader (×85). Synthetic image and registration."""
import numpy as np
import pytest
import xarray as xr

from hypso.rgb import CameraRegistration, camera_filename, camera_product, open_camera, write_camera

UTM52N = "EPSG:32652"


@pytest.fixture
def camera():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, (60, 80, 3)).astype(np.float32)
    return xr.DataArray(rgb, dims=("y", "x", "channel"), coords={"channel": ["R", "G", "B"]},
                        attrs={"instrument": "RGB", "acquisition_time": "2025-04-04T02:27:23Z",
                               "bayer_pattern": "RGGB", "source": "ariake_2025_04_04T02_27_23.bmp",
                               "orientation": "rotated 180 degrees"})


@pytest.fixture
def registration():
    # 30 m pixels, north up, from a corner near Ariake
    h = np.array([[30.0, 0, 450000], [0, -30.0, 3640000], [0, 0, 1]])
    return CameraRegistration(ok=True, reason="", homography=h, crs=UTM52N, matches=100, inliers=90,
                              rms_m=150.0, coverage=0.7, settings={"resolution": 150})


def test_unregistered_product(camera, tmp_path):
    ds = camera_product(camera)
    assert ds["R"].dtype == np.uint8 and "latitude" not in ds
    assert ds.attrs["georeferenced"] == "false" and ds.attrs["location_description"] == "ariake"
    assert camera_filename(ds) == "HYPSO2_RGB.20250404T022723.ariake.L1A.nc"
    path = write_camera(ds, out_dir=tmp_path)
    back = open_camera(path)
    assert path.name == camera_filename(ds) and "date_created" in back.attrs
    np.testing.assert_array_equal(back["G"].values, np.round(camera.sel(channel="G").values))


def test_failed_registration_is_recorded(camera, registration):
    from dataclasses import replace
    failed = replace(registration, ok=False, reason="too few inliers", homography=None)
    ds = camera_product(camera, failed, target="somewhere")
    assert "latitude" not in ds and ds.attrs["registration_ok"] == "false"
    assert ds.attrs["registration_reason"] == "too few inliers" and ds.attrs["location_description"] == "somewhere"


def test_registered_product_roundtrip(camera, registration, tmp_path):
    ds = camera_product(camera, registration)
    assert ds.attrs["georeferenced"] == "true" and ds["latitude"].dtype == np.float32
    assert ds["R"].attrs["coordinates"] == "latitude longitude"
    back = open_camera(write_camera(ds, out_dir=tmp_path, label="test"))
    assert back.attrs["registration_inliers"] == 90
    np.testing.assert_array_equal(back["latitude"].values, ds["latitude"].values)
    assert 32 < float(back["latitude"].mean()) < 33.5


def test_regridded_product_name(camera, registration):
    from hypso.rgb import regrid_camera
    grid = regrid_camera(camera_product(camera, registration), resolution=60)
    assert camera_filename(grid).startswith("HYPSO2_RGB.20250404T022723.ariake.L1A.utm52n_60m")


def test_satpy_reads_camera_product(camera, registration, tmp_path):
    satpy = pytest.importorskip("satpy")
    path = write_camera(camera_product(camera, registration), out_dir=tmp_path)
    scene = satpy.Scene(filenames=[str(path)], reader="hypso_rgb_l1a")
    assert {"R", "G", "B", "latitude", "longitude"} <= set(scene.available_dataset_names())
    scene.load(["true_color"])
    assert scene["true_color"].shape == (3, 60, 80)
    assert scene["true_color"].attrs["sensor"] == "hypso2_rgb"


def test_camera_names_match_only_rgb_reader():
    import fnmatch
    from pathlib import Path
    import yaml
    from trollsift import globify
    import hypso.satpy
    readers = Path(hypso.satpy.__file__).parent / "etc" / "readers"
    config = yaml.load((readers / "hypso_rgb_l1a.yaml").read_text(), Loader=yaml.BaseLoader)
    patterns = [p for ft in config["file_types"].values() for p in ft["file_patterns"]]
    name = "HYPSO2_RGB.20250404T022723.ariake.L1A.nc"
    assert any(fnmatch.fnmatch(name, globify(p)) for p in patterns)
    assert not any(fnmatch.fnmatch(name.replace("_RGB.", "_HSI."), globify(p)) for p in patterns)
