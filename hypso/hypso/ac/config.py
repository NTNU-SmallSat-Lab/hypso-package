"""Where the external atmospheric-correction tools are installed and how to
run them: one YAML file, one section per tool (REFACTOR_PROGRESS.md ×51).

The package has no built-in tool locations. The file is found from, in
order: the `config` argument (a path or an already-loaded dict), or the
HYPSO_AC_CONFIG environment variable. Example (only the sections for the
tools you use are needed):

    polymer:
      python: /path/to/envs/ac/bin/python      # optional, default: this interpreter
      base_path: /path/to/Polymer
      polymer_path: /path/to/Polymer/polymer
      eoread_path: /path/to/Polymer/eoread
      eotools_path: /path/to/Polymer/eotools
      core_path: /path/to/Polymer/core
      version: v2                              # which Polymer build: v1 | v2
      ancillary_source: nasa                   # nasa | era5
      settings:                                # Polymer Params overrides
        bands_corr: [448, 459, ...]

Per-call options override the file's values; `settings` dicts are merged
key by key.
"""
import os
from pathlib import Path

import yaml

ENV_VAR = "HYPSO_AC_CONFIG"


def load_ac_config(config=None) -> dict:
    if isinstance(config, dict):
        return config
    path = config or os.environ.get(ENV_VAR)
    if not path:
        raise ValueError(f"no AC tool configuration: pass config=<path or dict> or set {ENV_VAR} "
                         f"(see hypso.ac.config for the format)")
    return yaml.safe_load(Path(path).read_text()) or {}


def tool_config(tool: str, config=None, overrides: dict = None) -> dict:
    """The `tool` section, with `overrides` applied (settings merged)."""
    loaded = load_ac_config(config)
    if tool not in loaded:
        raise ValueError(f"the AC tool configuration has no {tool!r} section")
    section = dict(loaded[tool] or {})
    for key, value in (overrides or {}).items():
        if key == "settings" and isinstance(value, dict):
            section["settings"] = {**(section.get("settings") or {}), **value}
        else:
            section[key] = value
    return section
