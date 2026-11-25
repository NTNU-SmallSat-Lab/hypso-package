"""Running an external atmospheric-correction tool in its own process
(ACRunError, run_subprocess_driver), used by hypso.processing.atmospheric.

Why a subprocess: Polymer's v1 (HYPSO-SRF-patched) and v2 (stock) builds ship
different, incompatible versions of the same-named top-level packages
(confirmed live: v2's `core` needs core.process.blockwise, absent from v1's
`core` checkout), and Python's sys.modules import cache pins whichever version
is imported first for the rest of the process's lifetime - unsafe if a
long-lived process ever runs a v1 correction and a v2 correction. A fresh
subprocess per run sidesteps this, and contains crashes in the tools' native
code."""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _package_roots() -> list:
    """Directories containing the imported hypso / hypsoN_calibration
    packages, for a driver subprocess's PYTHONPATH."""
    import importlib
    roots = []
    for name in ("hypso", "hypso1_calibration", "hypso2_calibration"):
        try:
            root = str(Path(importlib.import_module(name).__file__).resolve().parent.parent)
        except Exception:
            continue
        if root not in roots:
            roots.append(root)
    return roots


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
        # The driver imports hypso: make this process's hypso and calibration
        # packages importable in the child even when they are not installed
        # (a checkout on sys.path), and run it in `tmp` - `python -m` puts the
        # working directory first on sys.path, and from a repo root the bare
        # hypso/ project directory would shadow the package.
        env["PYTHONPATH"] = os.pathsep.join(_package_roots() + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        if extra_env:
            env.update(extra_env)

        proc = subprocess.run(
            [python_path, "-m", driver_module, str(config_path), str(result_path)],
            capture_output=True, text=True, env=env, cwd=tmp,
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
