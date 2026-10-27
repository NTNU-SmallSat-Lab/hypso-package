"""Coefficient selection from a calibration package's data/coefficients.yaml.

Shared by hypso1_calibration and hypso2_calibration: each package holds an
identical copy of this file (they cannot depend on each other or on hypso,
which depends on them); tests/test_coefficients.py checks the copies match.
Edit both together.

A catalog has, per correction (radiometric, smile, destriping, spectral),
the file each version uses - radiometric per variant (coeff_type) - and
`releases` naming the version of each correction a release selects. Any
file entry may carry `capture_types: {<type>: {file, crop, description}}`
to use another file for that capture type; `crop: as_is` uses it without
cropping or binning. An optional `known_capture_types` list makes any other
capture type get no files (HYPSO-1's long-standing behaviour). See each
package's data/coefficients.yaml.
"""
import logging
import warnings
from importlib.resources import files

import yaml

logger = logging.getLogger(__name__)

CORRECTIONS = ("radiometric", "smile", "destriping", "spectral", "spectral_full_frame")


def load(package: str) -> dict:
    """A calibration package's coefficient catalog."""
    return yaml.safe_load(files(package).joinpath("data/coefficients.yaml").read_text())


def _version_key(version: str, available) -> str:
    # "v3" and "V3" both name release/version V3.
    for key in available:
        if str(key).upper() == str(version).upper():
            return key
    raise ValueError(f"Unknown version {version!r}. Available: {sorted(available)}")


def _kind(correction: str) -> str:
    return "spectral" if correction == "spectral_full_frame" else correction


def resolve(catalog: dict, coeff_type: str = None, version=None) -> dict:
    """The coefficient selection for a coeff_type and version, without file
    paths: {"coeff_type", "release", "radiometric": "<version>/<variant>",
    and the version of each other correction}.

    coeff_type: radiometric variant; default the catalog's
        default_coeff_type. Deprecated names (`aliases`) map to a variant and
        release, with a DeprecationWarning.
    version: a release name (default the catalog's default_release), or
        {correction: version} to mix; corrections missing from the dict come
        from the default release.
    """
    if coeff_type is None:
        coeff_type = catalog["default_coeff_type"]
    aliases = catalog.get("aliases") or {}
    if coeff_type in aliases:
        new_type, release = aliases[coeff_type]
        warnings.warn(f"coeff_type={coeff_type!r} is deprecated; use coeff_type={new_type!r}, "
                      f"version={release!r}", DeprecationWarning, stacklevel=3)
        coeff_type = new_type
        if version is None:
            version = release

    corrections = [c for c in CORRECTIONS if _kind(c) in catalog]
    releases = catalog["releases"]
    if version is None or isinstance(version, str):
        release = _version_key(version or catalog["default_release"], releases)
        selection = dict(releases[release])
    else:
        unknown = sorted(set(version) - set(corrections))
        if unknown:
            raise ValueError(f"Unknown corrections {unknown}. Known: {corrections}")
        release = "custom"
        selection = dict(releases[catalog["default_release"]])
        selection.update(version)

    for correction in corrections:
        selection[correction] = _version_key(selection[correction], catalog[_kind(correction)])

    variants = catalog["radiometric"][selection["radiometric"]]
    if coeff_type not in variants:
        raise ValueError(f"Invalid coeff_type: {coeff_type!r} for radiometric {selection['radiometric']}. "
                         f"Available: {sorted(variants)}")
    note = catalog.get("spectral_note")
    if note and selection["spectral"] not in note["versions"] and release != "V1":
        warnings.warn(f"Spectral calibration {selection['spectral']}: {note['message']}", UserWarning, stacklevel=3)

    out = {"coeff_type": coeff_type, "release": release,
           "radiometric": f"{selection['radiometric']}/{coeff_type}"}
    out.update({c: selection[c] for c in corrections if c != "radiometric"})
    return out


def calibration_files(package: str, catalog: dict, capture_type=None, coeff_type=None, version=None) -> dict:
    """Absolute paths of the selected coefficient files.

    Returns {"radiometric", "smile", "destriping", "spectral",
    "spectral_full_frame": path or None, "crop_modes": {kind: crop} for
    capture-type files that declare one, "selection": resolve(...) (plus
    "radiometric_capture_type" when a capture-type radiometric file was
    chosen)}.
    """
    selection = resolve(catalog, coeff_type, version)
    known = catalog.get("known_capture_types")
    if known is not None and capture_type not in known:
        return {**{c: None for c in CORRECTIONS}, "crop_modes": {}, "selection": selection}
    radiometric_version, variant = selection["radiometric"].split("/")
    crop_modes = {}

    def path(name):
        return files(package).joinpath(f"data/{name}") if name else None

    def entry(kind, item):
        override = (item.get("capture_types") or {}).get(capture_type)
        if override is None:
            return item.get("file")
        if kind == "radiometric":
            selection["radiometric_capture_type"] = capture_type
        if "crop" in override:
            crop_modes[kind] = override["crop"]
        return override.get("file")

    out = {"radiometric": path(entry("radiometric", catalog["radiometric"][radiometric_version][variant]))}
    for correction in ("smile", "destriping", "spectral", "spectral_full_frame"):
        version_key = selection.get(correction)
        out[correction] = (path(entry(_kind(correction), catalog[_kind(correction)][version_key]))
                           if version_key is not None else None)
    out["crop_modes"] = crop_modes
    out["selection"] = selection
    logger.debug("%s coefficients: %s", package, selection)
    return out


def optional_model(package: str, catalog: dict, kind: str, version: str = None):
    """An optional, release-independent model (e.g. ``keystone``): ``(path,
    entry)`` for ``version`` (default the kind's ``default`` version), or
    ``(None, None)`` if the catalog has none. ``entry["enabled"]`` says whether
    it has been reviewed for use; callers decide what to do when it is false.
    Releases never select these."""
    models = catalog.get("optional", {}).get(kind)
    if not models:
        return None, None
    key = _version_key(version or models["default"], [k for k in models if k != "default"])
    entry = models[key]
    return files(package).joinpath(f"data/{entry['file']}"), {"version": key, **entry}
