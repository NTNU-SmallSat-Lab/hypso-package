"""Shared interface for atmospheric-correction tool adapters (see this
subpackage's __init__.py docstring). Also holds get_inferred_wavelength_band_map,
the wavelength-matching helper every adapter's open_output path uses to map a
tool's output bands back onto HYPSO band indices - previously
HypsoBase._get_inferred_wavelength_band_map (zero external callers, confirmed by
grep, so it moved here outright with no wrapper kept on HypsoBase).

Also holds the subprocess-isolation helper (ACRunError, run_subprocess_driver)
used by adapters that need to run their external tool in a fresh process rather
than in-process - see PolymerAdapter.run_correction / _polymer_driver.py for the
concrete, demonstrated reason Polymer needs this: its v1 (HYPSO-SRF-patched) and
v2 (stock) builds ship different, incompatible versions of the same-named
top-level packages (confirmed live: v2's `core` needs core.process.blockwise,
absent from v1's `core` checkout), and Python's sys.modules import cache pins
whichever version is imported first for the rest of the process's lifetime -
unsafe if a long-lived process (e.g. hypso-processing-pipeline) ever runs a v1
correction and a v2 correction without restarting. A fresh subprocess per run
sidesteps this entirely."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class ACRunError(RuntimeError):
    """A subprocess-isolated AC tool run failed. Carries the subprocess's raw
    stdout/stderr and, if its driver script produced a structured result.json
    before failing, that error's type/message/traceback from the tool's own
    process - so a caller sees *why* (e.g. "eoread not on this interpreter's
    path") instead of just "subprocess exited 1"."""

    def __init__(self, tool: str, returncode: int, stdout: str, stderr: str,
                error_type: str = None, message: str = None):
        self.tool = tool
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.error_type = error_type
        self.message = message

        summary = f"{error_type}: {message}" if error_type else "no structured error reported"
        super().__init__(
            f"{tool} subprocess failed (exit {returncode}): {summary}\n"
            f"--- subprocess stderr ---\n{stderr}"
        )


def run_subprocess_driver(python_path: str, driver_module: str, config: dict, tool_name: str,
                          extra_env: dict = None) -> dict:
    """Run `driver_module` (a hypso.ac.adapters._*_driver module with a
    `python -m`-invocable __main__ block) in a fresh subprocess under
    `python_path`, passing it `config` as JSON and reading back its JSON
    result. Config/result files live in a per-call TemporaryDirectory (not a
    shared/fixed path) so concurrent runs never collide - the exact class of
    bug hypso-processing-pipeline had to work around by hand for OC-SMART's
    staging directory.

    `python_path` defaults to sys.executable at the call site (the same
    interpreter this process is running under) when the caller doesn't need a
    genuinely separate environment - that interpreter must have `hypso`
    importable for any driver whose tool resolves a hook by dotted string name
    back into this package (e.g. Polymer's srf_getter).

    `extra_env`, if given, is merged into the subprocess's environment (on top
    of this process's own os.environ) - the mechanism for passing secrets
    (e.g. ACOLITE's EARTHDATA credentials) to a driver WITHOUT writing them
    into config.json, which lives on disk (even briefly, even in a private
    TemporaryDirectory) for the run's duration.

    Raises ACRunError on any failure (non-zero exit, missing/unparseable
    result.json, or a result.json reporting status != "ok"); returns the
    parsed result dict on success.
    """
    with tempfile.TemporaryDirectory(prefix=f"hypso_ac_{tool_name}_") as tmp:
        config_path = Path(tmp, "config.json")
        result_path = Path(tmp, "result.json")
        config_path.write_text(json.dumps(config, default=str))

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        proc = subprocess.run(
            [python_path, "-m", driver_module, str(config_path), str(result_path)],
            capture_output=True, text=True, env=env,
        )

        result = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                result = None

        if proc.returncode != 0 or result is None or result.get("status") != "ok":
            raise ACRunError(
                tool=tool_name, returncode=proc.returncode,
                stdout=proc.stdout, stderr=proc.stderr,
                error_type=(result or {}).get("error_type"),
                message=(result or {}).get("message"),
            )

        return result


class ACAdapter:
    """One adapter per external atmospheric-correction tool, behind a shared
    run_correction/open_output interface. This pass is *organizational* (the
    approved plan's "prepare the AC functions to be refactored"): every method
    body is today's HypsoBase method body relocated verbatim - same subprocess/
    sys.path/external-tool-parsing logic, not rewritten - so a future rewrite of
    one tool's internals has a clean, isolated target that doesn't touch the
    other tools or HypsoBase.

    Adapters are stateless (all per-capture state lives on the satobj passed to
    every call), so the module-level instances in this subpackage are shared
    safely between captures.
    """

    #: Registry key, e.g. "polymer" - also the l2a_cube correction key convention.
    key: str = None

    def run_correction(self, satobj, **kwargs):
        """Run the external AC tool on this capture's L1 product."""
        raise NotImplementedError

    def open_output(self, satobj, **kwargs):
        """Read the tool's output file(s) into satobj.l2a_cube."""
        raise NotImplementedError


def get_inferred_wavelength_band_map(satobj, inferred_wavelengths):

    # Map inferred wavelengths to HYPSO wavelengths
    A = np.array(inferred_wavelengths, dtype=float)
    B = np.array(satobj.wavelengths, dtype=float)

    index_map = {}
    indices_unique = []

    for a in A:
        ix = np.argmin(np.abs(B - a))
        if ix not in index_map: # ensure uniqueness
            index_map[ix] = a
            indices_unique.append(ix)
        else:
            print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

    wl_band_map = np.array(indices_unique, dtype=int)


    return wl_band_map
