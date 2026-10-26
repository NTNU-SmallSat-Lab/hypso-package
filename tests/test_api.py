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


# Production outputs for the reference capture (HYPSO_DATA_AOC_REPROCESS_AULUS2):
# the fake AC runners below return these real tool outputs, and the L2A
# products are compared against the production L2A files made from them.
REPROCESS_DIR = Path("/home/camerop/HYPSO_DATA_AOC_REPROCESS_AULUS2/aeronetvenice_2025-03-04T10-38-05Z")
POLYMER_OUTPUT = REPROCESS_DIR / "polymer" / "aeronetvenice_2025-03-04T10-38-05Z-moved_unmasked-l1c.polymer.nc"
ACOLITE_OUTPUTS = {s: REPROCESS_DIR / "acolite" / f"HYPSO2_2025_03_04_10_38_05_{s}.nc" for s in ("L2R", "L2W")}
OCSMART_OUTPUT = REPROCESS_DIR / "HYPSO_HSI_aeronetvenice_2025-03-04T10-38-05Z-l1d_L2_OCSMART.h5"
requires_ac_outputs = pytest.mark.skipif(not REPROCESS_DIR.is_dir(), reason=f"AC outputs not present at {REPROCESS_DIR}")


def _production_l2a(method):
    return hypso.open(REPROCESS_DIR / f"aeronetvenice_2025-03-04T10-38-05Z-moved_unmasked-l2a-{method}.nc")


def _assert_same_cube(l2a, production):
    np.testing.assert_array_equal(l2a.wavelengths, production.wavelengths)
    np.testing.assert_allclose(l2a.cube.values, production.cube.values, rtol=1e-6, equal_nan=True)


def test_dps(l1d):
    # Dark pixel subtraction: per-band minimum subtracted, clipped at 0, then
    # divided by pi so the cube is Rrs (sr-1) like the other methods
    # (2026-10-26; before, the reflectance difference itself was stored).
    l2a = l1d.correct_atmosphere("dps")
    cube = l1d.cube.values
    np.testing.assert_allclose(l2a.cube.values, np.clip(cube - cube.min(axis=(0, 1)), 0, None) / np.pi, rtol=1e-6)
    assert l2a.level == "L2A" and l2a.cube.name == "Rrs" and l2a.atmospheric_correction == "dps"
    assert l2a.filename() == "HYPSO2_HSI.20250304T103805.aeronetvenice.L2A.dps.nc"
    assert [s["step"] for s in l2a.provenance][-1] == "l2a_dps"
    assert "latitude" in l2a.geometry


def test_dps_masked_ignores_flagged_pixels(l1d):
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
    np.testing.assert_allclose(l2a.cube.values[100:], np.clip(l1d.cube.values[100:] - dark, 0, None) / np.pi,
                               rtol=1e-6)
    with pytest.raises(ValueError, match="pixel_flags"):
        l1d.correct_atmosphere("dps", masked=True)


def test_l2a_written_attributes(l1d, tmp_path):
    import netCDF4 as nc
    path = l1d.correct_atmosphere("dps").write(out_dir=tmp_path)
    with nc.Dataset(path) as f:
        band = f["Rrs_378"] if "Rrs_378" in f.variables else f[next(v for v in f.variables if v.startswith("Rrs_"))]
        assert band.units == "sr-1" and band.long_name.startswith("Remote Sensing Reflectance")
        assert band.standard_name == ("surface_ratio_of_upwelling_radiance_emerging_from_sea_water_"
                                      "to_downwelling_radiative_flux_in_air")


def test_correct_atmosphere_level_and_method_errors(l1a, l1d):
    with pytest.raises(ValueError, match="needs an L1D"):
        l1a.correct_atmosphere("dps")
    with pytest.raises(ValueError, match="unknown method"):
        l1d.correct_atmosphere("magic")


def _fake_polymer(monkeypatch, calls):
    """Stands in for the Polymer subprocess: checks the staged input and
    returns the real Polymer v2 output for this capture."""
    import shutil
    import hypso.processing.atmospheric as atmospheric

    def run(python_path, driver_module, config, tool_name):
        assert driver_module == "hypso.ac.adapters._polymer_driver" and tool_name == "polymer"
        calls.append({"config": config, "staged": hypso.open(config["polymer_l1_input_nc_file"])})
        out = Path(config["polymer_output_dir"]) / "raw.polymer.nc"
        shutil.copy(POLYMER_OUTPUT, out)
        return {"status": "ok", "output_file": str(out)}

    monkeypatch.setattr(atmospheric, "run_subprocess_driver", run)


POLYMER_CONFIG = {"polymer": {"base_path": "/opt/polymer", "polymer_path": "/opt/polymer/polymer", "version": "v2",
                              "settings": {"band_cloudmask": 790}}}


@requires_ac_outputs
def test_polymer_mocked(l1c, monkeypatch, tmp_path):
    import netCDF4 as nc
    from hypso.processing.atmospheric import POLYMER_FLAGS
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
    _assert_same_cube(l2a, _production_l2a("polymer"))
    step = l2a.provenance[-1]
    assert step["step"] == "l2a_polymer" and step["ac_tool"] == "polymer" and step["ac_version"] == "v2"
    assert '"band_cloudmask": 790' in step["settings_effective"] and "bands_corr" in step["settings_user"]
    assert step["input_level"] == "L1C" and step["bands_returned"] == 113
    assert l2a.filename().endswith(".L2A.polymer.nc")

    # 2D products: chla = 10**logchl (linear, mg m-3) and Polymer's own flags
    variables = l2a.variables
    assert {"chla", "logfb", "SPM", "Rgli", "Rnir", "polymer_flags"} <= set(variables)
    with nc.Dataset(POLYMER_OUTPUT) as f:
        logchl, flags = f["logchl"][:], f["flags"][:]
    np.testing.assert_allclose(variables["chla"].values, np.ma.filled(10.0 ** logchl.astype(np.float64), np.nan),
                               rtol=1e-5, equal_nan=True)
    assert variables["chla"].attrs["units"] == "mg m-3"
    np.testing.assert_array_equal(variables["polymer_flags"].values, flags)
    assert variables["polymer_flags"].attrs["flag_meanings"].split() == list(POLYMER_FLAGS)
    np.testing.assert_allclose(variables["chla"].values, _production_l2a("polymer").variables["chla"].values,
                               rtol=1e-5, equal_nan=True)

    # and they survive writing
    reread = hypso.open(l2a.write(out_dir=tmp_path))
    assert reread.variables["polymer_flags"].dtype == flags.dtype
    assert reread.variables["polymer_flags"].attrs["flag_meanings"] == variables["polymer_flags"].attrs["flag_meanings"]
    assert reread.variables["chla"].attrs["standard_name"] == "mass_concentration_of_chlorophyll_a_in_sea_water"


@requires_ac_outputs
def test_polymer_masked_input_and_temp_cleanup(l1d, monkeypatch):
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
    """Stands in for the ACOLITE subprocess: returns the real L2R/L2W
    outputs for this capture."""
    import shutil
    import hypso.processing.atmospheric as atmospheric

    def run(python_path, driver_module, config, tool_name):
        assert driver_module == "hypso.ac.adapters._acolite_driver" and tool_name == "acolite"
        overrides = config["settings_overrides"]
        calls.append({"config": config, "staged": hypso.open(overrides["inputfile"])})
        for path in ACOLITE_OUTPUTS.values():
            shutil.copy(path, Path(overrides["output"]) / path.name)
        return {"status": "ok", "settings": {**overrides, "dsf_wave_range": "450,800", "sensor": config["sensor"]}}

    monkeypatch.setattr(atmospheric, "run_subprocess_driver", run)


ACOLITE_CONFIG = {"acolite": {"path": "/opt/acolite", "settings": {"dsf_aot_estimate": "tiled"}}}


@requires_ac_outputs
def test_acolite_mocked(l1c, monkeypatch, tmp_path):
    import netCDF4 as nc
    calls = []
    _fake_acolite(monkeypatch, calls)
    work = tmp_path / "work"
    l2w = l1c.correct_atmosphere("acolite", config=ACOLITE_CONFIG, work_dir=work)
    config = calls[0]["config"]
    assert config["sensor"] == "HYPSO2" and calls[0]["staged"].level == "L1C"
    assert config["settings_overrides"]["dsf_aot_estimate"] == "tiled"        # config settings applied
    assert config["settings_overrides"]["l2w_parameters"][0] == "Rrs_*"       # adapter defaults kept
    assert l2w.cube.name == "Rrs" and l2w.atmospheric_correction == "acolite_l2w"
    _assert_same_cube(l2w, _production_l2a("acolite_l2w"))
    step = l2w.provenance[-1]
    assert step["step"] == "l2a_acolite_l2w" and step["ac_tool"] == "acolite" and step["ac_version"] == "git checkout"
    assert '"dsf_wave_range": "450,800"' in step["settings_effective"]        # what ACOLITE reported
    assert l2w.filename().endswith(".L2A.acolite_l2w.nc")

    # L2W's 2D products and flags, with CF standard names where ACOLITE's aren't
    variables = l2w.variables
    assert {"acolite_l2_flags", "SPM_Nechad2010_664", "chl_re_mishra", "ndci"} <= set(variables)
    with nc.Dataset(ACOLITE_OUTPUTS["L2W"]) as f:
        np.testing.assert_array_equal(variables["acolite_l2_flags"].values, f["l2_flags"][:])
    chl = variables["chl_re_mishra"].attrs
    assert chl["standard_name"] == "mass_concentration_of_chlorophyll_a_in_sea_water"
    assert chl["source_standard_name"] == "mass_concentration_of_chlorophyll_a_in_water"
    assert "standard_name" not in variables["ndci"].attrs                       # not a CF name
    np.testing.assert_allclose(variables["chl_re_mishra"].values,
                               _production_l2a("acolite_l2w").variables["chl_re_mishra"].values,
                               rtol=1e-6, equal_nan=True)

    # L2R from the same run: no second ACOLITE run, no 2D products
    l2r = l1c.correct_atmosphere("acolite", output="l2r", config=ACOLITE_CONFIG, work_dir=work, reuse_output=True)
    assert len(calls) == 1
    assert l2r.cube.name == "rhos" and l2r.atmospheric_correction == "acolite_l2r"
    _assert_same_cube(l2r, _production_l2a("acolite_l2r"))
    assert l2r.provenance[-1]["reused_output"] == "true"
    with pytest.raises(ValueError, match="output must be"):
        l1c.correct_atmosphere("acolite", output="l3", config=ACOLITE_CONFIG)


# Stands in for OCSMART.py: checks the staged input against the
# OCSMART_Input.txt paths and copies the real OC-SMART output into l2_path.
FAKE_OCSMART = r"""
import shutil
from pathlib import Path
cfg = dict(line.split(" = ", 1) for line in Path("OCSMART_Input.txt").read_text().splitlines() if " = " in line)
l1b, l2 = Path(cfg["l1b_path"].strip()), Path(cfg["l2_path"].strip())
staged = sorted(l1b.glob("*.nc"))
assert len(staged) == 1 and staged[0].name.startswith("HYPSO_HSI_"), staged
print("fake OC-SMART processing", staged[0].name)
shutil.copy(REAL_OUTPUT, l2 / staged[0].name.replace(".nc", "_L2_OCSMART.h5"))
"""


@requires_ac_outputs
def test_ocsmart_with_fake_installation(l1d, tmp_path):
    install = tmp_path / "OCSMART_Linux_v2.6.3"
    install.mkdir()
    (install / "OCSMART.py").write_text(f"REAL_OUTPUT = {str(OCSMART_OUTPUT)!r}\n" + FAKE_OCSMART)
    (install / "OCSMART_Input.txt").write_text("original contents\n")
    config = {"ocsmart": {"dir": str(install), "python": __import__("sys").executable}}

    l2a = l1d.correct_atmosphere("ocsmart", config=config, work_dir=tmp_path / "work")
    assert (install / "OCSMART_Input.txt").read_text() == "original contents\n"     # restored
    assert (install / ".OCSMART_Input.lock").exists()                              # pipeline's lock file
    output = tmp_path / "work" / "ocsmart" / "HYPSO_HSI_aeronetvenice_2025-03-04T10-38-05Z-l1d_L2_OCSMART.h5"
    assert output.is_file()
    assert l2a.cube.name == "Rrs" and l2a.atmospheric_correction == "ocsmart"
    _assert_same_cube(l2a, _production_l2a("ocsmart"))
    step = l2a.provenance[-1]
    assert step["ac_tool"] == "ocsmart" and step["ac_version"] == "v2.6.3"
    assert step["bands_returned"] == int((~np.isnan(l2a.cube.values).all(axis=(0, 1))).sum())
    assert l2a.filename().endswith(".L2A.ocsmart.nc")

    variables = l2a.variables
    assert {"chlor_a_oci", "chlor_a_yoc", "ocsmart_l2_flags"} <= set(variables)
    assert not {"Latitude", "Solar_zenith"} & set(variables)
    assert variables["chlor_a_oci"].attrs["units"] == "mg m-3"
    assert variables["ocsmart_l2_flags"].dtype.kind == "i"
    production = _production_l2a("ocsmart").variables
    for name in ("chlor_a_oci", "chlor_a_yoc"):
        np.testing.assert_allclose(variables[name].values, production[name].values, rtol=1e-6, equal_nan=True)


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
    # Plausible brightness and colour balance. With the inverted sRGB gamma
    # (fixed 2026-10-26) this scene rendered dark and green: mean 0.081,
    # channel means R/G/B 0.065/0.107/0.071; correct: 0.244, 0.282/0.267/0.184.
    red, green, blue = rgb.mean(axis=(0, 1))
    assert 0.15 < rgb.mean() < 0.5
    assert green <= red
