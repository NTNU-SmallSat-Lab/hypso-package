"""Processing provenance carried in every product.

Each processing step adds a `metadata/processing/<step>` node (one per
step, kept when the product is read and written again, so a file holds its
whole lineage) and one line to the global `history` attribute:

    2026-10-26T10:02:11Z: l1d from HYPSO2_HSI.20250612T095802.aeronetvenice.L1C.nc,
        hypso-package 26.10.0 (git 25f4c3bf)

Node attributes: date_utc, software / software_version /
software_git_commit / software_git_dirty, input_file, the step's parameters
(scalars as attributes, anything else as JSON), and optionally:

- calibration: package / version / git commit, and `<kind>_file` +
  `<kind>_sha256` for every coefficient file loaded (versions alone don't
  identify coefficients - see docs/versioning.rst);
- ac: tool, version, git commit, install path, `settings_effective` (full
  settings actually used, JSON) and `settings_user` (only what the user
  set). Credential-like keys are removed and listed in
  `settings_redacted_keys`.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.calibration.manifest import sha256_file
from hypso.versioning import package_info

PRODUCT_FORMAT_VERSION = "2.0"   # files without the attribute are format 1

# EARTHDATA_u / EARTHDATA_p (ACOLITE), *_password, *_token, ...
_SECRET_KEY = re.compile(r"(_u|_p|user(name)?|pass(word)?|token|secret|credential|api_?key)$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def attr_value(value):
    """A NetCDF-storable attribute value: bools as "true"/"false", None as
    "", containers as JSON, numbers and strings unchanged."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def redact(settings: dict):
    """(settings without credential-like keys, sorted removed keys)."""
    removed = sorted(k for k in settings if _SECRET_KEY.search(str(k)))
    return {k: v for k, v in settings.items() if k not in removed}, removed


def _package_attrs(prefix: str, package: str, label: str = None) -> dict:
    info = package_info(package)
    dirty = info["git_dirty"]
    return {
        prefix: label or package,
        f"{prefix}_version": info["version"],
        f"{prefix}_git_commit": info["git_commit"] or "",
        f"{prefix}_git_dirty": "unknown" if dirty is None else attr_value(dirty),
    }


def software_attrs() -> dict:
    return _package_attrs("software", "hypso", label="hypso-package")


def calibration_attrs(package: str, files: dict) -> dict:
    """`files`: {kind: path}, e.g. {"radiometric": ..., "smile": ...}; None
    entries (a step not applied) are skipped."""
    attrs = _package_attrs("calibration_package", package)
    for kind, path in files.items():
        if path is None:
            continue
        attrs[f"{kind}_file"] = Path(path).name
        attrs[f"{kind}_sha256"] = sha256_file(path)
    return attrs


def ac_attrs(tool: str, version: str = None, git_commit: str = None, install_path=None,
             settings_effective: dict = None, settings_user: dict = None) -> dict:
    attrs = {"ac_tool": tool, "ac_version": version or "unknown", "ac_git_commit": git_commit or "",
             "ac_install_path": str(install_path or "")}
    redacted = set()
    for name, settings in (("settings_effective", settings_effective), ("settings_user", settings_user)):
        if settings is None:
            continue
        clean, removed = redact(settings)
        redacted.update(removed)
        attrs[name] = json.dumps(clean, default=str, sort_keys=True)
    if redacted:
        attrs["settings_redacted_keys"] = ", ".join(sorted(redacted))
    return attrs


def add_step(tree: xr.DataTree, step: str, *, input_file=None, parameters: dict = None,
             calibration: dict = None, ac: dict = None) -> None:
    """Record a processing step on `tree` (in place). `calibration` /
    `ac` are dicts from calibration_attrs() / ac_attrs()."""
    date = utc_now()
    attrs = {"date_utc": date, **software_attrs()}
    if input_file is not None:
        attrs["input_file"] = Path(input_file).name
    for key, value in (parameters or {}).items():
        attrs[key] = attr_value(value)
    attrs.update(calibration or {})
    attrs.update(ac or {})
    tree[f"metadata/processing/{step}"] = xr.DataTree(xr.Dataset(attrs=attrs))

    line = f"{date}: {step}"
    if input_file is not None:
        line += f" from {Path(input_file).name}"
    commit = attrs["software_git_commit"][:8]
    line += f", hypso-package {attrs['software_version']}"
    if commit:
        line += f" (git {commit}{', uncommitted changes' if attrs['software_git_dirty'] == 'true' else ''})"
    if ac:
        line += f", {ac['ac_tool']} {ac['ac_version']}"
    history = tree.attrs.get("history")
    tree.attrs["history"] = f"{history}\n{line}" if history else line
