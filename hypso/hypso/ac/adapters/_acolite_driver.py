"""Subprocess entry point for running ACOLITE atmospheric correction in an
isolated process. See ACOLITEAdapter.run_correction and base.py's
ACRunError/run_subprocess_driver docstrings for the mechanism.

Unlike Polymer, ACOLITE has no DEMONSTRATED version-conflict bug driving this
(only one ACOLITE build is wired into hypso-processing-pipeline's config
today - both the HYPSO-stage and PACE-stage runners point at the same
config.tools.acolite_path). The justification here is crash containment
(ACOLITE's gdal/pyresample/cartopy-heavy stack taking down the whole host
process on a segfault or hang) and parallelism (no two captures' ACOLITE runs
can safely overlap in one process today), plus consistency with the pattern
Polymer's isolation already established - and a future ACOLITE build split
(e.g. a dedicated PACE build, mirroring Polymer's HYPSO-vs-PACE split) would
face the identical sys.modules risk Polymer's already hit.

Not meant to be imported directly - invoked via
`<python_path> -m hypso.ac.adapters._acolite_driver <config.json> <result.json>`
by base.py's run_subprocess_driver(), which ACOLITEAdapter.run_correction calls.

EARTHDATA credentials are deliberately NOT part of config.json (that file sits
on disk, even briefly, in a TemporaryDirectory) - they're read from
HYPSO_ACOLITE_EARTHDATA_USERNAME/HYPSO_ACOLITE_EARTHDATA_PASSWORD environment
variables instead, which the adapter sets only in this subprocess's own
environment (run_subprocess_driver's extra_env).

config.json keys (all written by ACOLITEAdapter.run_correction):
    acolite_path: sys.path entry for the ACOLITE installation.
    settings_arg: positional arg to acolite.acolite.settings.load() - either
        a settings file path or a platform name string (e.g. "HYPSO2") - see
        run_correction's docstring for why passing the platform name
        explicitly matters (load() only applies a sensor's own
        config/defaults/<name>.txt when given that name; it does not
        auto-detect the sensor from the input file).
    settings_overrides: dict merged into the loaded settings object
        (input/output paths, l2w product list, etc) via settings[key] = value.

result.json: {"status": "ok"} or
{"status": "error", "error_type": ..., "message": ..., "traceback": ...}.
"""
import json
import os
import sys
import traceback
from pathlib import Path


def main(config_path: str, result_path: str) -> int:
    config = json.loads(Path(config_path).read_text())

    try:
        sys.path.append(config["acolite_path"])

        from acolite.acolite.settings import load
        from acolite.acolite import acolite_run

        settings = load(config["settings_arg"])

        for key, value in config["settings_overrides"].items():
            settings[key] = value

        earthdata_u = os.environ.get("HYPSO_ACOLITE_EARTHDATA_USERNAME")
        earthdata_p = os.environ.get("HYPSO_ACOLITE_EARTHDATA_PASSWORD")
        if earthdata_u and earthdata_p:
            settings['EARTHDATA_u'] = earthdata_u
            settings['EARTHDATA_p'] = earthdata_p
            settings['ancillary_data'] = True

        acolite_run(settings=settings)

        Path(result_path).write_text(json.dumps({"status": "ok"}))
        return 0

    except Exception as exc:
        Path(result_path).write_text(json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
