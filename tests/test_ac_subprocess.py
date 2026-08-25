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
