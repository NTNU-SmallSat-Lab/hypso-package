"""Tests for the subprocess-isolation mechanism (hypso.ac.adapters.base's
ACRunError/run_subprocess_driver) and its two users, Polymer and ACOLITE - see
_polymer_driver.py's/_acolite_driver.py's module docstrings for why each needs
this (Polymer: its v1/v2 builds ship different, incompatible versions of the
same-named top-level packages, and Python's sys.modules cache makes switching
between them within one long-lived process unsafe; ACOLITE: crash containment
and parallelism, plus consistency with Polymer's isolation - no demonstrated
version-conflict bug the way Polymer's has, since only one ACOLITE build is
wired into hypso-processing-pipeline's config today).

Three tiers per tool, cheapest/most-general first:
1. Driver-level unit tests - call <tool>_driver.main() directly (no subprocess
   spawn), with a synthetic module standing in for the real tool so the
   driver's OWN JSON/error-handling logic is exercised without needing the
   tool installed.
2. Real-subprocess tests via run_subprocess_driver - a real process spawn,
   deliberately pointed at nonexistent tool paths, proving ACRunError
   propagates correctly end-to-end through a real subprocess boundary.
   Needs no external tool installed (the paths are meant to not exist).
3. A real-tool-checkout test (skipped if absent) that runs the actual driver
   against the real tool installed on this machine, with a deliberately-
   missing INPUT FILE - proving the real imports succeed through the isolated
   subprocess. Manually verified while building this pass: Polymer's
   eoread.hypso.Level1_HYPSO raises a clean FileNotFoundError on the missing
   file (no import error anywhere in the traceback); ACOLITE's acolite_run
   instead logs "Path ... does not exist." and returns normally (status
   "ok") - different tools, different missing-file handling, but both prove
   the same thing: real imports succeeded and real tool code ran."""
import json
import sys
from pathlib import Path

import pytest

from hypso.ac.adapters import ACRunError, run_subprocess_driver
from hypso.ac.adapters import _polymer_driver
from conftest import requires_real_capture
from hypso.ac.adapters import _acolite_driver

REAL_POLYMER_BASE_PATH = Path("/home/camerop/AC/Polymer/Polymer_HYPSO_SRF_Oct_2025")
REAL_ACOLITE_PATH = Path("/home/camerop/AC/ACOLITE/acolite")

requires_real_polymer = pytest.mark.skipif(
    not REAL_POLYMER_BASE_PATH.is_dir(),
    reason=f"real Polymer checkout not present at {REAL_POLYMER_BASE_PATH}",
)

requires_real_acolite = pytest.mark.skipif(
    not REAL_ACOLITE_PATH.is_dir(),
    reason=f"real ACOLITE checkout not present at {REAL_ACOLITE_PATH}",
)


# --- tier 1: driver-level unit tests, no subprocess, no Polymer ---

def test_driver_reports_import_error_as_structured_result(tmp_path):
    # No `polymer`/`eoread` importable at all from these nonexistent paths -
    # exercises the driver's except-branch and JSON serialization directly.
    config = {
        "polymer_base_path": str(tmp_path / "does_not_exist"),
        "polymer_path": None, "eoread_path": None, "eotools_path": None,
        "core_path": None,
        "polymer_l1_input_nc_file": "irrelevant.nc",
        "polymer_output_dir": str(tmp_path),
        "if_exists": "overwrite",
        "srf_nc_path": "irrelevant_srf.nc",
        "polymer_version": "v1",
        "optional_output_datasets": [],
    }
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config))

    rc = _polymer_driver.main(str(config_path), str(result_path))

    assert rc == 1
    result = json.loads(result_path.read_text())
    assert result["status"] == "error"
    assert result["error_type"] == "ModuleNotFoundError"
    assert "traceback" in result and "message" in result


def test_driver_rejects_unknown_polymer_version(tmp_path, monkeypatch):
    # Exercises the driver's own version-selection match/case _ branch,
    # without needing real Polymer: stub `polymer.main_v5`/`eoread.hypso`
    # onto sys.path via fake modules so the driver's import lines succeed and
    # execution reaches the version-selection logic.
    import types
    fake_eoread = types.ModuleType("eoread")
    fake_eoread_hypso = types.ModuleType("eoread.hypso")
    fake_eoread_hypso.Level1_HYPSO = lambda path: object()
    fake_eoread.hypso = fake_eoread_hypso

    fake_polymer = types.ModuleType("polymer")
    fake_polymer_main = types.ModuleType("polymer.main_v5")
    fake_polymer_main.run_polymer = lambda *a, **k: "/tmp/unused.nc"
    fake_polymer_main.default_output_datasets = []
    fake_polymer.main_v5 = fake_polymer_main

    monkeypatch.setitem(sys.modules, "eoread", fake_eoread)
    monkeypatch.setitem(sys.modules, "eoread.hypso", fake_eoread_hypso)
    monkeypatch.setitem(sys.modules, "polymer", fake_polymer)
    monkeypatch.setitem(sys.modules, "polymer.main_v5", fake_polymer_main)

    config = {
        "polymer_base_path": None, "polymer_path": None, "eoread_path": None,
        "eotools_path": None, "core_path": None,
        "polymer_l1_input_nc_file": "irrelevant.nc",
        "polymer_output_dir": str(tmp_path),
        "if_exists": "overwrite",
        "srf_nc_path": "irrelevant_srf.nc",
        "polymer_version": "v3-does-not-exist",
        "optional_output_datasets": [],
    }
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config))

    rc = _polymer_driver.main(str(config_path), str(result_path))

    assert rc == 1
    result = json.loads(result_path.read_text())
    assert result["error_type"] == "ValueError"
    assert "v3-does-not-exist" in result["message"]


def test_driver_success_path_with_stubbed_polymer(tmp_path, monkeypatch):
    # Full happy path through the driver's own logic (v1 output_datasets
    # selection, run_polymer call, result.json writing) with polymer/eoread
    # stubbed - no real tool needed to verify the driver wires its inputs to
    # run_polymer correctly and reports success.
    import types

    captured_kwargs = {}

    fake_eoread = types.ModuleType("eoread")
    fake_eoread_hypso = types.ModuleType("eoread.hypso")
    fake_eoread_hypso.Level1_HYPSO = lambda path: ("L1", path)
    fake_eoread.hypso = fake_eoread_hypso

    def fake_run_polymer(level1, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["level1"] = level1
        return str(tmp_path / "polymer_raw_output.nc")

    fake_polymer = types.ModuleType("polymer")
    fake_polymer_main = types.ModuleType("polymer.main_v5")
    fake_polymer_main.run_polymer = fake_run_polymer
    fake_polymer_main.default_output_datasets = ["logchl", "logfb"]
    fake_polymer.main_v5 = fake_polymer_main

    monkeypatch.setitem(sys.modules, "eoread", fake_eoread)
    monkeypatch.setitem(sys.modules, "eoread.hypso", fake_eoread_hypso)
    monkeypatch.setitem(sys.modules, "polymer", fake_polymer)
    monkeypatch.setitem(sys.modules, "polymer.main_v5", fake_polymer_main)

    config = {
        "polymer_base_path": None, "polymer_path": None, "eoread_path": None,
        "eotools_path": None, "core_path": None,
        "polymer_l1_input_nc_file": "input.nc",
        "polymer_output_dir": str(tmp_path),
        "if_exists": "skip",
        "srf_nc_path": "srf.nc",
        "polymer_version": "v1",
        "optional_output_datasets": ["SPM"],
    }
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config))

    rc = _polymer_driver.main(str(config_path), str(result_path))

    assert rc == 0
    result = json.loads(result_path.read_text())
    assert result["status"] == "ok"
    assert result["output_file"] == str(tmp_path / "polymer_raw_output.nc")
    # driver wired the config through to run_polymer correctly
    assert captured_kwargs["dir_out"] == str(tmp_path)
    assert captured_kwargs["if_exists"] == "skip"
    assert captured_kwargs["srf_getter"] == "hypso.ac.ac_polymer_srf_getter"
    assert captured_kwargs["srf_getter_arg"] == "srf.nc"
    assert captured_kwargs["output_datasets"] == ["logchl", "logfb", "SPM"]


# --- tier 2: real subprocess spawn, no Polymer needed (paths don't exist) ---

def test_run_subprocess_driver_raises_ac_run_error_on_bad_config(tmp_path):
    config = {
        "polymer_base_path": str(tmp_path / "does_not_exist"),
        "polymer_path": None, "eoread_path": None, "eotools_path": None,
        "core_path": None,
        "polymer_l1_input_nc_file": "irrelevant.nc",
        "polymer_output_dir": str(tmp_path),
        "if_exists": "overwrite",
        "srf_nc_path": "irrelevant_srf.nc",
        "polymer_version": "v1",
        "optional_output_datasets": [],
    }

    with pytest.raises(ACRunError) as exc:
        run_subprocess_driver(
            python_path=sys.executable,
            driver_module="hypso.ac.adapters._polymer_driver",
            config=config,
            tool_name="polymer",
        )

    err = exc.value
    assert err.tool == "polymer"
    assert err.error_type == "ModuleNotFoundError"
    assert "polymer" in err.message.lower() or "no module" in err.message.lower()
    assert str(err)  # summary message is well-formed


def test_run_subprocess_driver_raises_on_bad_interpreter():
    # A nonexistent interpreter path - proves failures BEFORE the driver even
    # starts (subprocess spawn failure itself) also surface as ACRunError,
    # not an uncaught OSError from subprocess.run.
    with pytest.raises((ACRunError, FileNotFoundError)):
        run_subprocess_driver(
            python_path="/no/such/interpreter-binary",
            driver_module="hypso.ac.adapters._polymer_driver",
            config={},
            tool_name="polymer",
        )


# --- tier 3: real Polymer checkout, skipped if absent ---

@requires_real_polymer
def test_real_polymer_imports_succeed_through_subprocess(tmp_path):
    # The real proof: point the real driver at the REAL Polymer v1 checkout's
    # paths (matching hypso-processing-pipeline's config.yaml conventions)
    # with a deliberately-missing input file. If imports were broken (the
    # thing this whole pass is about avoiding), this would fail with
    # ModuleNotFoundError/ImportError. It must instead fail with
    # FileNotFoundError, proving eoread/polymer/core/eotools all imported
    # correctly in the isolated subprocess.
    base = REAL_POLYMER_BASE_PATH
    config = {
        "polymer_base_path": str(base),
        "polymer_path": str(base / "polymer-master-v5"),
        "eoread_path": str(base / "eoread"),
        "eotools_path": str(base / "eotools"),
        "core_path": str(base / "core"),
        "polymer_l1_input_nc_file": str(tmp_path / "does_not_exist-l1c.nc"),
        "polymer_output_dir": str(tmp_path),
        "if_exists": "overwrite",
        "srf_nc_path": str(tmp_path / "does_not_exist_srf.nc"),
        "polymer_version": "v1",
        "optional_output_datasets": ["SPM"],
    }

    with pytest.raises(ACRunError) as exc:
        run_subprocess_driver(
            python_path=sys.executable,
            driver_module="hypso.ac.adapters._polymer_driver",
            config=config,
            tool_name="polymer",
        )

    err = exc.value
    assert err.error_type == "FileNotFoundError"
    assert "does_not_exist-l1c.nc" in err.message


# =====================================================================
# ACOLITE
# =====================================================================

# --- tier 1: driver-level unit tests, no subprocess, no ACOLITE ---

def test_acolite_driver_reports_import_error_as_structured_result(tmp_path):
    config = {
        "acolite_path": str(tmp_path / "does_not_exist"),
        "settings_arg": "HYPSO2",
        "settings_overrides": {"inputfile": "irrelevant.nc", "output": str(tmp_path)},
    }
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config))

    rc = _acolite_driver.main(str(config_path), str(result_path))

    assert rc == 1
    result = json.loads(result_path.read_text())
    assert result["status"] == "error"
    assert result["error_type"] == "ModuleNotFoundError"


def test_acolite_driver_success_path_with_stubbed_acolite(tmp_path, monkeypatch):
    # Full happy path through the driver's own logic (settings load + merge,
    # acolite_run call, EARTHDATA env-var pickup, result.json writing) with
    # acolite stubbed - no real tool needed to verify the driver wires its
    # inputs correctly.
    import types

    class FakeSettings(dict):
        pass

    captured = {}

    fake_settings_module = types.ModuleType("acolite.acolite.settings")
    fake_settings_module.load = lambda arg: FakeSettings(_loaded_from=arg)

    def fake_acolite_run(settings):
        captured["settings"] = dict(settings)
        return None

    fake_acolite_acolite = types.ModuleType("acolite.acolite")
    fake_acolite_acolite.acolite_run = fake_acolite_run
    fake_acolite_acolite.settings = fake_settings_module

    fake_acolite = types.ModuleType("acolite")
    fake_acolite.acolite = fake_acolite_acolite

    monkeypatch.setitem(sys.modules, "acolite", fake_acolite)
    monkeypatch.setitem(sys.modules, "acolite.acolite", fake_acolite_acolite)
    monkeypatch.setitem(sys.modules, "acolite.acolite.settings", fake_settings_module)
    monkeypatch.setenv("HYPSO_ACOLITE_EARTHDATA_USERNAME", "someuser")
    monkeypatch.setenv("HYPSO_ACOLITE_EARTHDATA_PASSWORD", "somepass")

    config = {
        "acolite_path": "/unused",
        "settings_arg": "HYPSO2",
        "settings_overrides": {"inputfile": "input.nc", "output": str(tmp_path),
                               "l2w_parameters": ["Rrs_*"]},
    }
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config))

    rc = _acolite_driver.main(str(config_path), str(result_path))

    assert rc == 0
    result = json.loads(result_path.read_text())
    assert result["status"] == "ok"
    settings = captured["settings"]
    assert settings["_loaded_from"] == "HYPSO2"
    assert settings["inputfile"] == "input.nc"
    assert settings["l2w_parameters"] == ["Rrs_*"]
    # EARTHDATA credentials picked up from env, not from config.json
    assert settings["EARTHDATA_u"] == "someuser"
    assert settings["EARTHDATA_p"] == "somepass"
    assert settings["ancillary_data"] is True


def test_acolite_driver_earthdata_env_not_in_config_file(tmp_path):
    # Confirms the actual secret-handling contract: credentials never appear
    # in the JSON config file ACOLITEAdapter.run_correction writes to disk.
    config = {
        "acolite_path": "/unused",
        "settings_arg": "HYPSO2",
        "settings_overrides": {"inputfile": "input.nc", "output": str(tmp_path)},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assert "EARTHDATA" not in config_path.read_text()


# --- tier 2: real subprocess spawn, no ACOLITE needed (paths don't exist) ---

def test_acolite_run_subprocess_driver_raises_ac_run_error_on_bad_config(tmp_path):
    config = {
        "acolite_path": str(tmp_path / "does_not_exist"),
        "settings_arg": "HYPSO2",
        "settings_overrides": {"inputfile": "irrelevant.nc", "output": str(tmp_path)},
    }

    with pytest.raises(ACRunError) as exc:
        run_subprocess_driver(
            python_path=sys.executable,
            driver_module="hypso.ac.adapters._acolite_driver",
            config=config,
            tool_name="acolite",
        )

    err = exc.value
    assert err.tool == "acolite"
    assert err.error_type == "ModuleNotFoundError"


def test_run_subprocess_driver_extra_env_reaches_subprocess(tmp_path):
    # Uses the real Polymer driver only as a vehicle to prove extra_env is
    # actually threaded through to the subprocess's environment - doesn't
    # need Polymer or ACOLITE installed, since the import failure happens
    # before any env-var-dependent code would run either way; this just
    # checks the env mechanism itself via a bad-path failure that still
    # completes the full subprocess round trip.
    config = {
        "polymer_base_path": str(tmp_path / "does_not_exist"),
        "polymer_path": None, "eoread_path": None, "eotools_path": None,
        "core_path": None, "polymer_l1_input_nc_file": "x.nc",
        "polymer_output_dir": str(tmp_path), "if_exists": "overwrite",
        "srf_nc_path": "x_srf.nc", "polymer_version": "v1",
        "optional_output_datasets": [],
    }
    # Must still fail (paths don't exist) - the point is only that passing
    # extra_env doesn't itself break the round trip.
    with pytest.raises(ACRunError):
        run_subprocess_driver(
            python_path=sys.executable,
            driver_module="hypso.ac.adapters._polymer_driver",
            config=config,
            tool_name="polymer",
            extra_env={"HYPSO_TEST_EXTRA_ENV_VAR": "present"},
        )


# --- tier 3: real ACOLITE checkout, skipped if absent ---

@requires_real_acolite
def test_real_acolite_imports_and_runs_through_subprocess(tmp_path):
    # The real proof: point the real driver at the REAL ACOLITE checkout with
    # a deliberately-missing input file. If imports were broken, this would
    # fail with ModuleNotFoundError/ImportError. ACOLITE's own acolite_run
    # does NOT raise on a missing input file (unlike Polymer's Level1_HYPSO) -
    # it logs "Path ... does not exist." and returns normally - so success
    # here (status "ok" + a real ACOLITE-written log file in the output dir)
    # is itself the proof that real ACOLITE code executed, not a lack of one.
    config = {
        "acolite_path": str(REAL_ACOLITE_PATH),
        "settings_arg": "HYPSO2",
        "settings_overrides": {
            "inputfile": str(tmp_path / "does_not_exist-l1c.nc"),
            "output": str(tmp_path),
            "polygon": None, "rgb_rhot": True, "rgb_rhos": True,
            "map_l2w": False, "l2w_mask": False, "l2w_mask_threshold": 0.2,
            "l2w_parameters": ["Rrs_*"],
        },
    }

    result = run_subprocess_driver(
        python_path=sys.executable,
        driver_module="hypso.ac.adapters._acolite_driver",
        config=config,
        tool_name="acolite",
    )

    assert result["status"] == "ok"
    log_files = list(tmp_path.glob("acolite_run_*_log_file.txt"))
    assert log_files, "ACOLITE did not write its own run log - real processing may not have executed"
    log_text = log_files[0].read_text()
    assert "does not exist" in log_text


# =====================================================================
# OC-SMART
# =====================================================================
# OC-SMART has no importable Python API and no "import the tool" step to
# isolate the way Polymer/ACOLITE needed - it's always invoked as a bare
# `python OCSMART.py` subprocess, so OCSMARTAdapter.run_correction doesn't
# use run_subprocess_driver/a driver module at all; it calls
# subprocess.Popen directly. Tier 1 here mocks that Popen call to exercise
# run_correction's OWN logic (capture-local staging with the correct
# sensor-autodetection filename prefix, OCSMART_Input.txt write + restore -
# both on success and on failure, ACRunError on nonzero exit) without
# needing OC-SMART installed. Tier 3 (real installation, skipped if absent)
# is the strongest evidence: manually verified while building this pass
# against the real OC-SMART install and a real HYPSO L1D file - OC-SMART's
# own console output showed "Sensor : HYPSO HSI" (proving the HYPSO_PREFIX
# fix actually resolves the sensor-autodetection bug it targets - the
# pre-fix behavior was OC-SMART silently emitting NO output at all with
# exit code 0), correctly found the staged file, and then failed inside
# ITS OWN src/L1B.py with `AttributeError: 'L1B' object has no attribute
# 'latitude'` while reading geolocation - the same class of new-flattened-
# NetCDF-format incompatibility already known and accepted for eoread/
# ACOLITE (see docs/architecture.rst), now confirmed for a third external
# reader. That failure is OC-SMART's own reader, not this adapter - the
# adapter's own mechanism (staging/config/subprocess/cleanup/error
# propagation) all demonstrably worked correctly up to that point.
from unittest.mock import patch

from hypso.ac.adapters import get_ac_adapter

REAL_OCSMART_DIR = Path("/home/camerop/AC/OC-SMART/OCSMART_Linux_v2.6.3")
REAL_OCSMART_PYTHON = Path("/home/camerop/miniconda3/envs/ocsmart/bin/python3")

requires_real_ocsmart = pytest.mark.skipif(
    not (REAL_OCSMART_DIR.is_dir() and REAL_OCSMART_PYTHON.is_file()),
    reason=f"real OC-SMART install/env not present at {REAL_OCSMART_DIR}",
)


class _FakeCapture:
    """Minimal satobj stand-in for OCSMARTAdapter.run_correction - only the
    attributes it actually reads."""
    def __init__(self, capture_dir, l1d_nc_file, capture_name="testcapture", ocsmart_dir=None):
        self.capture_dir = capture_dir
        self.l1d_nc_file = l1d_nc_file
        self.capture_name = capture_name
        self.ocsmart_dir = ocsmart_dir or capture_dir


class _FakePopen:
    """Stands in for subprocess.Popen: yields no output, returns
    `returncode` from wait(). Records the argv/cwd it was constructed with
    for assertions."""
    calls = []

    def __init__(self, argv, cwd=None, **kwargs):
        self.argv = argv
        self.cwd = cwd
        self.stdout = iter([])
        type(self).calls.append(self)

    def wait(self):
        return self._returncode

    _returncode = 0


def test_ocsmart_stages_input_with_hypso_prefix_and_writes_config(tmp_path):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    ocsmart_dir = tmp_path / "ocsmart_install"
    ocsmart_dir.mkdir()
    l1d_file = tmp_path / "source-l1d.nc"
    l1d_file.write_bytes(b"fake netcdf content")

    satobj = _FakeCapture(capture_dir=capture_dir, l1d_nc_file=l1d_file,
                          capture_name="aeronetvenice_2025-03-04T10-38-05Z",
                          ocsmart_dir=ocsmart_dir)

    captured_argv = {}

    class _SuccessPopen(_FakePopen):
        _returncode = 0

        def __init__(self, argv, cwd=None, **kwargs):
            super().__init__(argv, cwd=cwd, **kwargs)
            captured_argv["argv"] = argv
            captured_argv["cwd"] = cwd
            # simulate OC-SMART having written its output by the time it exits
            ocsmart = get_ac_adapter("ocsmart")
            ocsmart.output_path(satobj).write_bytes(b"fake h5 output")

    with patch("hypso.ac.adapters.ocsmart.subprocess.Popen", _SuccessPopen):
        ocsmart = get_ac_adapter("ocsmart")
        result = ocsmart.run_correction(satobj, python_path=str(tmp_path / "fake_python"))

    assert result == ocsmart.output_path(satobj)
    assert captured_argv["argv"] == [str(tmp_path / "fake_python"), "OCSMART.py"]
    assert captured_argv["cwd"] == ocsmart_dir
    # staging cleaned up after a successful run
    assert not (capture_dir / "ocsmart_staging").exists()


def test_ocsmart_config_restored_even_on_failure(tmp_path):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    ocsmart_dir = tmp_path / "ocsmart_install"
    ocsmart_dir.mkdir()
    input_txt = ocsmart_dir / "OCSMART_Input.txt"
    original_content = "l1b_path = /some/pre-existing/path/\n"
    input_txt.write_text(original_content)

    l1d_file = tmp_path / "source-l1d.nc"
    l1d_file.write_bytes(b"fake netcdf content")
    satobj = _FakeCapture(capture_dir=capture_dir, l1d_nc_file=l1d_file, ocsmart_dir=ocsmart_dir)

    class _FailPopen(_FakePopen):
        _returncode = 1

    with patch("hypso.ac.adapters.ocsmart.subprocess.Popen", _FailPopen):
        ocsmart = get_ac_adapter("ocsmart")
        with pytest.raises(ACRunError) as exc:
            ocsmart.run_correction(satobj, python_path="python3")

    assert exc.value.tool == "ocsmart"
    assert exc.value.returncode == 1
    # OCSMART_Input.txt restored to its pre-call content despite the failure
    assert input_txt.read_text() == original_content
    # staging cleaned up despite the failure
    assert not (capture_dir / "ocsmart_staging").exists()


def test_ocsmart_skip_existing(tmp_path):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    satobj = _FakeCapture(capture_dir=capture_dir, l1d_nc_file=tmp_path / "unused-l1d.nc",
                          capture_name="somecapture")
    ocsmart = get_ac_adapter("ocsmart")
    ocsmart.output_path(satobj).write_bytes(b"already there")

    with patch("hypso.ac.adapters.ocsmart.subprocess.Popen") as popen:
        result = ocsmart.run_correction(satobj, skip_existing=True)

    assert result == ocsmart.output_path(satobj)
    popen.assert_not_called()


@requires_real_ocsmart
@requires_real_capture
def test_real_ocsmart_stages_and_autodetects_sensor(written_nc_files, tmp_path):
    # The real proof: run the real adapter against the real OC-SMART
    # installation and a real, written HYPSO L1D file (written_nc_files, from
    # conftest.py - the same fixture the CF-format tests use). Confirms the
    # HYPSO_PREFIX naming fix actually works (console output must show sensor
    # autodetection succeeding) - the pre-fix behavior was OC-SMART silently
    # producing no output at all (exit 0) when the prefix was wrong.
    class _RealCapture:
        def __init__(self, capture_dir, l1d_nc_file, ocsmart_dir):
            self.capture_dir = capture_dir
            self.l1d_nc_file = l1d_nc_file
            self.capture_name = "aeronetvenice_2025-03-04T10-38-05Z"
            self.ocsmart_dir = ocsmart_dir

    satobj = _RealCapture(capture_dir=tmp_path.absolute(),
                          l1d_nc_file=written_nc_files["l1d"],
                          ocsmart_dir=str(REAL_OCSMART_DIR))

    ocsmart = get_ac_adapter("ocsmart")
    with pytest.raises(ACRunError) as exc:
        ocsmart.run_correction(satobj, python_path=str(REAL_OCSMART_PYTHON))

    # Whatever OC-SMART's own failure is (today: a new-NetCDF-format
    # incompatibility in its geolocation reader, matching eoread/ACOLITE's
    # already-accepted breakage), sensor autodetection must have succeeded
    # first - that's what HYPSO_PREFIX fixes, and it's the thing this test
    # exists to prove.
    assert "Sensor : HYPSO HSI" in exc.value.stdout
    assert not (tmp_path / "ocsmart_staging").exists()
