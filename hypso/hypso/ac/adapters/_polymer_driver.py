"""Subprocess entry point for running Polymer atmospheric correction in an
isolated process. See base.py's module docstring for the concrete, demonstrated
reason: Polymer's v1 (HYPSO-SRF-patched) and v2 (stock) builds ship different,
incompatible versions of the same-named top-level packages (`core` at least),
and Python's sys.modules cache makes switching between them within one
long-lived process unsafe.

Not meant to be imported directly - invoked via
`<python_path> -m hypso.ac.adapters._polymer_driver <config.json> <result.json>`
by base.py's run_subprocess_driver(), which PolymerAdapter.run_correction calls.
Everything here that doesn't need Polymer itself imported (path resolution,
output-file renaming) stays in PolymerAdapter.run_correction, in the parent
process - this driver's job is only the part that must happen after Polymer is
imported: resolving the version-specific output_datasets/outputs_names kwargs
(needs polymer.main_v5.default_output_datasets for v1) and calling run_polymer.

config.json keys (all written by PolymerAdapter.run_correction):
    polymer_base_path, polymer_path, eoread_path, eotools_path, core_path:
        sys.path entries (base_path inserted last, so it wins on name clashes -
        matches the original in-process code's insertion order). Any may be
        null/omitted.
    polymer_l1_input_nc_file, polymer_output_dir, if_exists, srf_nc_path:
        passed straight through to run_polymer.
    polymer_version: "v1" or "v2" - selects output_datasets vs outputs_names.
    optional_output_datasets: appended to either selection list.

result.json: {"status": "ok", "output_file": "..."} or
{"status": "error", "error_type": ..., "message": ..., "traceback": ...}.
"""
import json
import sys
import traceback
from pathlib import Path

from hypso.ac.ac_polymer import SRF_GETTER_PATH


def main(config_path: str, result_path: str) -> int:
    config = json.loads(Path(config_path).read_text())

    try:
        for key in ("polymer_path", "eotools_path", "eoread_path", "core_path"):
            value = config.get(key)
            if value:
                sys.path.insert(0, value)
        if config.get("polymer_base_path"):
            sys.path.insert(0, config["polymer_base_path"])

        from eoread.hypso import Level1_HYPSO
        from polymer.main_v5 import run_polymer, default_output_datasets

        optional_output_datasets = config["optional_output_datasets"]

        match config["polymer_version"]:
            case "v1":
                output_selection_kwargs = {
                    "output_datasets": default_output_datasets + optional_output_datasets,
                }
            case "v2":
                output_selection_kwargs = {
                    "outputs": "named",
                    "outputs_names": [
                        "latitude", "longitude", "rho_w", "logchl", "logfb",
                        "Rgli", "Rnir", "flags",
                    ] + optional_output_datasets,
                }
            case other:
                raise ValueError(f"Unknown polymer_version: {other!r}")

        output_file = run_polymer(
            Level1_HYPSO(config["polymer_l1_input_nc_file"]),
            dir_out=config["polymer_output_dir"],
            if_exists=config["if_exists"],
            srf_getter=SRF_GETTER_PATH,
            srf_getter_arg=config["srf_nc_path"],
            **output_selection_kwargs,
        )

        Path(result_path).write_text(json.dumps({
            "status": "ok",
            "output_file": str(output_file),
        }))
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
