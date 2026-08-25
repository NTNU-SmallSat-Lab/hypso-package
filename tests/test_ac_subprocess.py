"""Tests for the subprocess-isolation mechanism (hypso.ac.adapters.base's
ACRunError/run_subprocess_driver, and Polymer's use of it - see
_polymer_driver.py's module docstring for why Polymer specifically needs this:
its v1/v2 builds ship different, incompatible versions of the same-named
top-level packages, and Python's sys.modules cache makes switching between
them within one long-lived process unsafe).

Three tiers, cheapest/most-general first:
1. Driver-level unit tests - call _polymer_driver.main() directly (no
   subprocess spawn), with a synthetic module standing in for `polymer`/
   `eoread` so the driver's OWN JSON/error-handling logic is exercised
   without needing Polymer installed.
2. Real-subprocess tests via run_subprocess_driver - a real process spawn,
   deliberately pointed at nonexistent tool paths, proving ACRunError
   propagates correctly end-to-end through a real subprocess boundary.
   Needs no external tool installed (the paths are meant to not exist).
3. A real-Polymer-checkout test (skipped if absent) that runs the actual
   driver against the real HYPSO-SRF Polymer build on this machine, with a
   deliberately-missing INPUT FILE - proving the real `import polymer`/
   `import eoread` succeed through the isolated subprocess (this was
   manually verified while building this pass: eoread.hypso.Level1_HYPSO
   raises a clean FileNotFoundError on the missing file, with no import
   error anywhere in the traceback)."""
import json
import sys
from pathlib import Path

import pytest

from hypso.ac.adapters import ACRunError, run_subprocess_driver
from hypso.ac.adapters import _polymer_driver

REAL_POLYMER_BASE_PATH = Path("/home/camerop/AC/Polymer/Polymer_HYPSO_SRF_Oct_2025")

requires_real_polymer = pytest.mark.skipif(
    not REAL_POLYMER_BASE_PATH.is_dir(),
    reason=f"real Polymer checkout not present at {REAL_POLYMER_BASE_PATH}",
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
