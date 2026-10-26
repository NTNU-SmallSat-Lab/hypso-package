"""hypso.open / hypso.Product - the public API (REFACTOR_PROGRESS.md ×50)."""
import json
from pathlib import Path

import numpy as np
import pytest

import hypso
from conftest import COEFF_TYPE, COEFF_VERSION, L1A_PATH, LATS_PATH, LONS_PATH, requires_real_capture

pytestmark = requires_real_capture


def test_product_is_immutable(l1a):
    l1b = l1a.calibrate()
    assert l1a.level == "L1A" and l1b.level == "L1B"
    with pytest.raises(AttributeError):
        l1a.level = "L1B"
    tree = l1a.tree
    tree.attrs["processing_level"] = "X"
    assert l1a.level == "L1A"


def test_contents(l1d):
    assert l1d.level == "L1D" and l1d.sat_id == "HYPSO-2" and l1d.target == "aeronetvenice"
    assert l1d.cube.name == "rhot" and l1d.cube.dims == ("lines", "samples", "bands")
    assert {"latitude", "longitude", "solar_zenith"} <= set(l1d.geometry.data_vars)
    assert l1d.flags == {}
    assert [s["step"] for s in l1d.provenance] == ["l1b", "l1c", "l1d"]
    assert l1d.provenance[0]["coefficient_release"] == COEFF_VERSION
    assert "rhot" in repr(l1d)


def test_wrong_level_explains(l1a):
    with pytest.raises(ValueError, match=r'needs an L1C product, this one is L1A or \.to\("L1D"\)'):
        l1a.toa_reflectance()


def test_to_runs_missing_steps(l1a, l1d):
    via_to = l1a.to("L1D", calibrate={"coeff_type": COEFF_TYPE, "version": COEFF_VERSION},
                    georeference={"latitudes": LATS_PATH, "longitudes": LONS_PATH, "direct": False})
    np.testing.assert_array_equal(via_to.cube.values, l1d.cube.values)
    with pytest.raises(ValueError):
        l1d.to("L1B")


def test_write_names_and_round_trips(l1d, tmp_path):
    path = l1d.write(out_dir=tmp_path)
    assert path.name == "HYPSO2_HSI.20250304T103805.aeronetvenice.L1D.nc"
    reopened = hypso.open(path)
    assert reopened.level == "L1D"
    np.testing.assert_array_equal(reopened.cube.values, l1d.cube.values.astype(np.float32))
    assert [s["step"] for s in reopened.provenance] == ["l1b", "l1c", "l1d"]
    assert path.with_name(l1d.filename(label="test")).name.endswith(".L1D.test.nc")


def test_dps(l1d):
    # Dark pixel subtraction: per-band minimum subtracted, clipped at 0 (the
    # removed ac_dark_pixel_subtraction's math; identical before its removal).
    l2a = l1d.correct_atmosphere("dps")
    cube = l1d.cube.values
    np.testing.assert_array_equal(l2a.cube.values, np.clip(cube - cube.min(axis=(0, 1)), 0, None))
    assert l2a.level == "L2A" and l2a.cube.name == "Rrs" and l2a.atmospheric_correction == "dps"
    assert l2a.filename() == "HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.dps.nc"
    assert [s["step"] for s in l2a.provenance][-1] == "l2a_dps"
    assert "latitude" in l2a.geometry


def test_dps_masked_ignores_flagged_pixels(l1d):
    import xarray as xr
    from hypso.masks.flags import FLAGS_NAME, encode_flags
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:100] = True
    tree = l1d.tree
    root = tree.to_dataset(inherit=False)
    root[FLAGS_NAME] = encode_flags({"land": land})
    tree.dataset = root
    flagged = hypso.Product(tree)
    l2a = flagged.correct_atmosphere("dps", masked=True)
    assert np.isnan(l2a.cube.values[:100]).all()
    assert not np.isnan(l2a.cube.values[100:]).any()
    dark = np.nanmin(np.where(land[..., None], np.nan, l1d.cube.values), axis=(0, 1))
    np.testing.assert_allclose(l2a.cube.values[100:], np.clip(l1d.cube.values[100:] - dark, 0, None))
    with pytest.raises(ValueError, match="pixel_flags"):
        l1d.correct_atmosphere("dps", masked=True)


def test_correct_atmosphere_level_and_method_errors(l1a, l1d):
    with pytest.raises(ValueError, match="needs an L1D"):
        l1a.correct_atmosphere("dps")
    with pytest.raises(ValueError, match="unknown method"):
        l1d.correct_atmosphere("magic")


def _fake_polymer(monkeypatch, calls):
    """Stands in for the Polymer subprocess: checks the staged input and
    writes a v2-style output with 10 of the bands."""
    import netCDF4 as nc
    import hypso.processing.atmospheric as atmospheric

    def run(python_path, driver_module, config, tool_name):
        assert driver_module == "hypso.ac.adapters._polymer_driver" and tool_name == "polymer"
        staged = hypso.open(config["polymer_l1_input_nc_file"])
        calls.append({"config": config, "staged": staged})
        wavelengths = staged.wavelengths[::12][:10] + 0.2
        values = np.random.default_rng(0).random(staged.cube.shape[:2] + (len(wavelengths),)).astype(np.float32)
        out = Path(config["polymer_output_dir"]) / "raw.polymer.nc"
        with nc.Dataset(out, "w") as f:
            f.createDimension("y", values.shape[0]); f.createDimension("x", values.shape[1])
            f.createDimension("band", values.shape[2])
            f.createVariable("rho_w", "f4", ("y", "x", "band"))[:] = values
            f.createVariable("bands", "f4", ("band",))[:] = wavelengths
        calls[-1]["values"] = values
        return {"status": "ok", "output_file": str(out)}

    monkeypatch.setattr(atmospheric, "run_subprocess_driver", run)


POLYMER_CONFIG = {"polymer": {"base_path": "/opt/polymer", "polymer_path": "/opt/polymer/polymer", "version": "v2",
                              "settings": {"band_cloudmask": 790}}}


def test_polymer_mocked(l1c, monkeypatch, tmp_path):
    calls = []
    _fake_polymer(monkeypatch, calls)
    l2a = l1c.correct_atmosphere("polymer", config=POLYMER_CONFIG, settings={"bands_corr": [448, 660]},
                                         work_dir=tmp_path / "work")
    staged = calls[0]["staged"]
    assert staged.level == "L1C"
    assert any(n.startswith("Band_") for n in staged.tree["metadata/srf"].dataset.data_vars)   # SRFs added
    assert calls[0]["config"]["polymer_settings"] == {"band_cloudmask": 790, "bands_corr": [448, 660]}
    assert (tmp_path / "work").is_dir()                                                           # kept

    assert l2a.level == "L2A" and l2a.cube.name == "rho_w" and l2a.atmospheric_correction == "polymer"
    mapped = list(range(0, 120, 12))[:10]
    np.testing.assert_array_equal(l2a.cube.values[:, :, mapped], calls[0]["values"])
    assert np.isnan(np.delete(l2a.cube.values, mapped, axis=2)).all()
    step = l2a.provenance[-1]
    assert step["step"] == "l2a_polymer" and step["ac_tool"] == "polymer" and step["ac_version"] == "v2"
    assert '"band_cloudmask": 790' in step["settings_effective"] and "bands_corr" in step["settings_user"]
    assert step["input_level"] == "L1C" and step["bands_returned"] == 10
    assert l2a.filename().endswith(".L2A.polymer.nc")


def test_polymer_masked_input_and_temp_cleanup(l1d, monkeypatch):
    import xarray as xr
    from pathlib import Path
    from hypso.masks.flags import FLAGS_NAME, encode_flags
    calls = []
    _fake_polymer(monkeypatch, calls)
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:50] = True
    tree = l1d.tree
    root = tree.to_dataset(inherit=False)
    root[FLAGS_NAME] = encode_flags({"land": land})
    tree.dataset = root
    l2a = hypso.Product(tree).correct_atmosphere("polymer", masked=True, config=POLYMER_CONFIG)
    staged = calls[0]["staged"]
    assert staged.level == "L1D"
    assert np.isnan(staged.cube.values[:50]).all() and not np.isnan(staged.cube.values[50:]).any()
    assert not Path(calls[0]["config"]["polymer_output_dir"]).exists()   # temporary work_dir removed
    assert l2a.flags["land"].values[:50].all()


def test_polymer_needs_configuration(l1d, monkeypatch):
    monkeypatch.delenv("HYPSO_AC_CONFIG", raising=False)
    with pytest.raises(ValueError, match="HYPSO_AC_CONFIG"):
        l1d.correct_atmosphere("polymer")
    with pytest.raises(ValueError, match="no 'polymer' section"):
        l1d.correct_atmosphere("polymer", config={"acolite": {}})


def _fake_acolite(monkeypatch, calls):
    """Stands in for the ACOLITE subprocess: writes L2R/L2W files with 8
    bands named as ACOLITE does."""
    import netCDF4 as nc
    import hypso.processing.atmospheric as atmospheric

    def run(python_path, driver_module, config, tool_name):
        assert driver_module == "hypso.ac.adapters._acolite_driver" and tool_name == "acolite"
        overrides = config["settings_overrides"]
        staged = hypso.open(overrides["inputfile"])
        calls.append({"config": config, "staged": staged})
        wavelengths = staged.wavelengths[5::15][:8] + 0.3
        rng = np.random.default_rng(1)
        values = {}
        for suffix, prefix in (("L2R", "rhos"), ("L2W", "Rrs")):
            data = rng.random(staged.cube.shape[:2] + (len(wavelengths),)).astype(np.float32)
            values[suffix] = data
            with nc.Dataset(Path(overrides["output"]) / f"HYPSO2_2025_03_04_10_38_05_{suffix}.nc", "w") as f:
                f.createDimension("y", data.shape[0]); f.createDimension("x", data.shape[1])
                for i, wl in enumerate(wavelengths):
                    var = f.createVariable(f"{prefix}_{int(round(wl))}", "f4", ("y", "x"))
                    var[:] = data[:, :, i]
                    var.wavelength = float(wl)
        calls[-1]["values"] = values
        return {"status": "ok", "settings": {**overrides, "dsf_wave_range": "450,800", "sensor": config["sensor"]}}

    monkeypatch.setattr(atmospheric, "run_subprocess_driver", run)


ACOLITE_CONFIG = {"acolite": {"path": "/opt/acolite", "settings": {"dsf_aot_estimate": "tiled"}}}


def test_acolite_mocked(l1c, monkeypatch, tmp_path):
    calls = []
    _fake_acolite(monkeypatch, calls)
    work = tmp_path / "work"
    l2w = l1c.correct_atmosphere("acolite", config=ACOLITE_CONFIG, work_dir=work)
    config = calls[0]["config"]
    assert config["sensor"] == "HYPSO2" and calls[0]["staged"].level == "L1C"
    assert config["settings_overrides"]["dsf_aot_estimate"] == "tiled"        # config settings applied
    assert config["settings_overrides"]["l2w_parameters"][0] == "Rrs_*"       # adapter defaults kept
    mapped = list(range(5, 120, 15))[:8]
    assert l2w.cube.name == "Rrs" and l2w.atmospheric_correction == "acolite_l2w"
    np.testing.assert_array_equal(l2w.cube.values[:, :, mapped], calls[0]["values"]["L2W"])
    assert np.isnan(np.delete(l2w.cube.values, mapped, axis=2)).all()
    step = l2w.provenance[-1]
    assert step["step"] == "l2a_acolite_l2w" and step["ac_tool"] == "acolite" and step["ac_version"] == "git checkout"
    assert '"dsf_wave_range": "450,800"' in step["settings_effective"]        # what ACOLITE reported
    assert l2w.filename().endswith(".L2A.acolite_l2w.nc")

    # L2R from the same run: no second ACOLITE run
    l2r = l1c.correct_atmosphere("acolite", output="l2r", config=ACOLITE_CONFIG, work_dir=work, reuse_output=True)
    assert len(calls) == 1
    assert l2r.cube.name == "rhos" and l2r.atmospheric_correction == "acolite_l2r"
    np.testing.assert_array_equal(l2r.cube.values[:, :, mapped], calls[0]["values"]["L2R"])
    assert l2r.provenance[-1]["reused_output"] == "true"
    with pytest.raises(ValueError, match="output must be"):
        l1c.correct_atmosphere("acolite", output="l3", config=ACOLITE_CONFIG)


FAKE_OCSMART = r'''
import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
cfg = dict(line.split(" = ", 1) for line in Path("OCSMART_Input.txt").read_text().splitlines() if " = " in line)
l1b, l2 = Path(cfg["l1b_path"].strip()), Path(cfg["l2_path"].strip())
staged = sorted(l1b.glob("*.nc"))
assert len(staged) == 1 and staged[0].name.startswith("HYPSO_HSI_"), staged
with nc.Dataset(staged[0]) as f:
    lines, samples = len(f.dimensions["lines"]), len(f.dimensions["samples"])
print("fake OC-SMART processing", staged[0].name)
out = l2 / staged[0].name.replace(".nc", "_L2_OCSMART.h5")
rng = np.random.default_rng(2)
with nc.Dataset(out, "w") as f:
    f.createDimension("y", lines); f.createDimension("x", samples)
    f.createVariable("Latitude", "f4", ("y", "x"))[:] = 45.0
    for group in ("Lr", "Lrc", "Lt", "Rrs"):
        g = f.createGroup(group)
        for wl in (412, 443, 490, 555, 670):
            g.createVariable(f"{group}_{wl}", "f4", ("y", "x"))[:] = rng.random((lines, samples))
'''


def test_ocsmart_with_fake_installation(l1d, tmp_path):
    import netCDF4 as nc
    install = tmp_path / "OCSMART_Linux_v2.6.3"
    install.mkdir()
    (install / "OCSMART.py").write_text(FAKE_OCSMART)
    (install / "OCSMART_Input.txt").write_text("original contents\n")
    config = {"ocsmart": {"dir": str(install), "python": __import__("sys").executable}}

    l2a = l1d.correct_atmosphere("ocsmart", config=config, work_dir=tmp_path / "work")
    assert (install / "OCSMART_Input.txt").read_text() == "original contents\n"     # restored
    assert (install / ".OCSMART_Input.lock").exists()                              # pipeline's lock file
    output = tmp_path / "work" / "ocsmart" / "HYPSO_HSI_aeronetvenice_2025-03-04T10-38-05Z-l1d_L2_OCSMART.h5"
    assert output.is_file()
    with nc.Dataset(output) as f:
        expected = np.stack([f["Rrs"][f"Rrs_{wl}"][:] for wl in (412, 443, 490, 555, 670)], axis=-1)
    index = [int(np.argmin(np.abs(l1d.wavelengths - wl))) for wl in (412, 443, 490, 555, 670)]
    np.testing.assert_array_equal(l2a.cube.values[:, :, index], expected)
    assert np.isnan(np.delete(l2a.cube.values, index, axis=2)).all()
    assert l2a.cube.name == "Rrs" and l2a.atmospheric_correction == "ocsmart"
    step = l2a.provenance[-1]
    assert step["ac_tool"] == "ocsmart" and step["ac_version"] == "v2.6.3" and step["bands_returned"] == 5
    assert l2a.filename().endswith(".L2A.ocsmart.nc")


def test_ocsmart_errors(l1d, tmp_path):
    from hypso.ac.adapters.base import ACRunError
    install = tmp_path / "ocsmart_install"
    install.mkdir()
    (install / "OCSMART.py").write_text("print('no output written')\n")
    config = {"ocsmart": {"dir": str(install), "python": __import__("sys").executable}}
    with pytest.raises(ValueError, match="contains '.'"):
        l1d.correct_atmosphere("ocsmart", config=config, work_dir=tmp_path / "work.v2")
    with pytest.raises(ACRunError, match="no OC-SMART output"):
        l1d.correct_atmosphere("ocsmart", config=config, work_dir=tmp_path / "work")
    (install / "OCSMART.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(ACRunError):
        l1d.correct_atmosphere("ocsmart", config=config, work_dir=tmp_path / "work2")


def test_with_masks_and_variables_round_trip(l1d, tmp_path):
    import xarray as xr
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:20] = True
    cloud = np.zeros_like(land)
    cloud[:, :30] = True
    masked = l1d.with_masks(land=land, descriptions={"land": "test land"}).with_masks(cloud=cloud)
    assert list(masked.flags) == ["land", "cloud"] and l1d.flags == {}
    np.testing.assert_array_equal(masked.flags["land"].values, land)
    with_var = masked.with_variables(ndwi=xr.DataArray(np.ones(land.shape, np.float32), dims=("lines", "samples"),
                                                       attrs={"units": "1", "long_name": "test index"}))
    back = hypso.open(with_var.write(out_dir=tmp_path))
    assert list(back.variables.data_vars) == ["ndwi"] and back.variables["ndwi"].attrs["long_name"] == "test index"
    assert list(back.flags) == ["land", "cloud"]
    assert "test land" in back.tree.dataset["pixel_flags"].attrs["comment"]
    with pytest.raises(ValueError, match="already"):
        l1d.with_variables(latitude=np.zeros(land.shape))
    with pytest.raises(ValueError, match="shape"):
        l1d.with_variables(bad=np.zeros((2, 2)))


def test_spectrum_pixel_resolution(l1d):
    line, sample = l1d.pixel(45.314, 12.508)
    spectrum = l1d.spectrum(45.314, 12.508)
    assert spectrum.dims == ("bands",) and "radiation_wavelength" in spectrum.coords
    np.testing.assert_array_equal(spectrum.values, l1d.cube.values[line, sample])
    np.testing.assert_array_equal(l1d.spectrum(line=3, sample=4).values, l1d.cube.values[3, 4])
    # same ground resolution the capture-object API computed (876.38 m on this capture)
    assert abs(l1d.resolution - 876.3778) < 0.01


def test_resample_and_satpy(l1d, tmp_path):
    from pyresample.geometry import AreaDefinition
    from satpy import Scene
    area = AreaDefinition("t", "t", "t", "EPSG:4326", 60, 40, (12.3, 45.2, 12.7, 45.45))
    land = np.zeros(l1d.cube.shape[:2], bool)
    land[:20] = True
    ds = l1d.with_masks(land=land).resample(area)
    assert ds["rhot"].shape == (40, 60, 120) and ds["flag_land"].dtype == bool
    assert np.isfinite(ds["rhot"].values[:, :, 50]).all()
    assert len(l1d.to_satpy().keys()) == 120
    # the Satpy reader plugin reads current-format file names
    scene = Scene(filenames=[str(l1d.write(out_dir=tmp_path))], reader="hypso_l1d")
    assert "rhot_413" in scene.available_dataset_names()


def test_hsi2rgb_on_l1d(l1d):
    from hypso.rgb import HSI2RGB
    rgb = HSI2RGB(l1d.wavelengths, l1d.cube.values)
    assert rgb.shape == l1d.cube.shape[:2] + (3,)
    assert np.isfinite(rgb).all() and 0 <= rgb.min() and rgb.max() <= 1
