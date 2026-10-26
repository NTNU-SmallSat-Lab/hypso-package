"""L1C/L1D -> L2A: atmospheric correction.

correct_atmosphere(tree, method, ...) returns an L2A tree: the method's
product cube on the input's bands, the input's geometry, pixel_flags and
metadata, an `atmospheric_correction` global attribute (used in file
names), and an `l2a_<method>` provenance step.

Methods:
- "dps": dark pixel subtraction, in-process on the L1D cube.
- "polymer": Polymer in a subprocess (hypso.ac.adapters._polymer_driver),
  from L1C (default) or L1D. Installation and settings come from the AC
  tool configuration (hypso.ac.config).
- ACOLITE, OC-SMART: to be added.

External tools read a file: the input is written (masked if requested,
with metadata/srf added if missing) into `work_dir`, which is a temporary
directory removed afterwards unless given.
"""
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.ac.adapters.base import run_subprocess_driver
from hypso.ac.config import tool_config

from hypso.io.product import PRODUCT_DIMS, product_name, write_product
from hypso.io.provenance import ac_attrs, add_step
from hypso.masks.flags import FLAGS_NAME, apply_flags

logger = logging.getLogger(__name__)

_INPUT_LEVELS = {"dps": ("L1D",), "polymer": ("L1C", "L1D")}


def _input_cube(tree: xr.DataTree, masked: bool, mask_flags) -> xr.DataArray:
    cube = tree.dataset[product_name(tree)]
    if not masked:
        return cube
    if FLAGS_NAME not in tree.dataset:
        raise ValueError("masked=True but the product has no pixel_flags")
    return apply_flags(cube, tree.dataset[FLAGS_NAME], mask_flags)


def _l2a_tree(tree: xr.DataTree, method: str, variable: str, cube: np.ndarray, parameters: dict,
              ac: dict) -> xr.DataTree:
    root = tree.to_dataset(inherit=False)
    source = root[product_name(tree)]
    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out[variable] = xr.DataArray(cube, dims=PRODUCT_DIMS, coords={
        name: ("bands", source[name].values) for name in ("wavelength", "radiation_wavelength", "fwhm")})
    out.attrs["processing_level"] = "L2A"
    out.attrs["atmospheric_correction"] = method

    result = tree.copy()
    result.dataset = out
    input_source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, f"l2a_{method}", input_file=input_source or f"{tree.attrs['processing_level']} (in memory)",
             parameters=parameters, ac=ac)
    return result


def _dps(tree, masked, mask_flags, statistic="min", percentile=None):
    # hypso.ac.ac_dark_pixel_subtraction's math, with NaN-aware statistics so
    # masked (NaN) pixels are ignored instead of making the whole dark
    # spectrum NaN; identical on unmasked input.
    cube = _input_cube(tree, masked, mask_flags).values
    if statistic == "min":
        dark_spectrum = np.nanmin(cube, axis=(0, 1))
    elif statistic == "percentile":
        percentile = 1 if percentile is None else percentile
        dark_spectrum = np.nanpercentile(cube, percentile, axis=(0, 1))
    else:
        raise ValueError("statistic must be 'min' or 'percentile'")
    corrected = np.clip(cube - dark_spectrum, 0, None)
    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "statistic": statistic, "dark_spectrum": [float(v) for v in dark_spectrum]}
    if percentile is not None:
        parameters["percentile"] = percentile
    return _l2a_tree(tree, "dps", "Rrs", corrected, parameters,
                     ac_attrs("dps", version="hypso-package", settings_effective={"statistic": statistic,
                                                                                  "percentile": percentile}))


def _band_map(source_wavelengths, tool_wavelengths) -> np.ndarray:
    """Index of the nearest source band for each tool band, unique (a tool
    band whose nearest source band is already taken is dropped) - as
    hypso.ac.adapters.base.get_inferred_wavelength_band_map."""
    source = np.asarray(source_wavelengths, dtype=float)
    seen, indices = set(), []
    for wavelength in np.asarray(tool_wavelengths, dtype=float):
        index = int(np.argmin(np.abs(source - wavelength)))
        if index in seen:
            logger.warning("Duplicate prevented: %s mapped to %s", wavelength, index)
            continue
        seen.add(index)
        indices.append(index)
    return np.array(indices, dtype=int)


def _git_commit(path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _stage_input(tree: xr.DataTree, masked: bool, mask_flags, work_dir: Path) -> Path:
    """Write the tool's input file into work_dir."""
    from hypso.api import Product
    from hypso.processing.toa import srf_node

    staged = tree.copy()
    if masked:
        root = staged.to_dataset(inherit=False)
        root[product_name(tree)] = _input_cube(tree, masked, mask_flags)
        staged.dataset = root
    has_srfs = ("metadata" in staged.children and "srf" in staged["metadata"].children
                and any(name.startswith("Band_") for name in staged["metadata/srf"].dataset.data_vars))
    if not has_srfs:
        # Polymer reads per-band SRFs from metadata/srf (its srf_getter); an
        # L1C straight from georeferencing has none yet.
        staged["metadata/srf"] = xr.DataTree(srf_node(staged))
    path = work_dir / Product(staged).filename()
    write_product(staged, path)
    return path


def _polymer(tree, masked, mask_flags, work_dir, config=None, **options):
    from hypso.load import load_polymer_l2_v1_nc, load_polymer_l2_v2_nc

    cfg = tool_config("polymer", config, options)
    version = cfg.get("version", "v2")
    input_file = _stage_input(tree, masked, mask_flags, work_dir)
    output_dir = work_dir / "polymer"
    output_dir.mkdir(parents=True, exist_ok=True)
    driver_config = {
        "polymer_base_path": cfg.get("base_path"),
        "polymer_path": cfg.get("polymer_path"),
        "eoread_path": cfg.get("eoread_path"),
        "eotools_path": cfg.get("eotools_path"),
        "core_path": cfg.get("core_path"),
        "polymer_l1_input_nc_file": str(input_file),
        "polymer_output_dir": str(output_dir),
        "if_exists": "overwrite",
        "srf_nc_path": "",   # SRFs are read from the input file's metadata/srf
        "polymer_version": version,
        "optional_output_datasets": list(cfg.get("optional_outputs", ["SPM"])),
        "polymer_settings": cfg.get("settings") or {},
        "ancillary_source": cfg.get("ancillary_source", "nasa"),
        "era5_allow_preliminary": cfg.get("era5_allow_preliminary", True),
    }
    result = run_subprocess_driver(python_path=cfg.get("python") or sys.executable,
                                   driver_module="hypso.ac.adapters._polymer_driver",
                                   config=driver_config, tool_name="polymer")
    output = (load_polymer_l2_v1_nc if version == "v1" else load_polymer_l2_v2_nc)(Path(result["output_file"]))
    rho_w = output["rho_w"]
    source_wavelengths = tree.dataset["radiation_wavelength"].values
    # The loader returns `bands` (1D) as a plain masked array, not a DataArray.
    index = _band_map(source_wavelengths, np.ma.getdata(np.asarray(output["bands"])))
    cube = np.full(tree.dataset[product_name(tree)].shape, np.nan)
    cube[:, :, index] = np.asarray(rho_w)[:, :, :len(index)]

    settings = {k: driver_config[k] for k in ("polymer_version", "optional_output_datasets", "polymer_settings",
                                              "ancillary_source", "era5_allow_preliminary")}
    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "input_level": tree.attrs["processing_level"], "bands_returned": int(len(index))}
    ac = ac_attrs("polymer", version=version, git_commit=_git_commit(cfg.get("polymer_path") or cfg.get("base_path")),
                  install_path=cfg.get("base_path"), settings_effective=settings,
                  settings_user={**(options.get("settings") or {}), **{k: v for k, v in options.items() if k != "settings"}})
    return _l2a_tree(tree, "polymer", "rho_w", cube, parameters, ac)


_METHODS = {"dps": _dps, "polymer": _polymer}
_EXTERNAL = {"polymer"}


def correct_atmosphere(tree: xr.DataTree, method: str, masked: bool = False, mask_flags=None,
                       work_dir=None, config=None, **options) -> xr.DataTree:
    """Atmospheric correction of an L1C/L1D tree with `method`.

    masked: give the method NaN where pixel_flags are set (all flags, or
        only `mask_flags`, e.g. ["land", "cloud"]). The input product itself
        stays unmasked.
    work_dir: external tools only - where the staged input and the tool's
        raw output go; default a temporary directory, removed afterwards.
    config: external tools only - AC tool configuration (path or dict, see
        hypso.ac.config); default the HYPSO_AC_CONFIG environment variable.
    options: method-specific - dps: statistic ("min" | "percentile"),
        percentile; polymer: any key of the configuration's polymer section
        (e.g. version, settings={...}, ancillary_source), overriding it.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; available: {sorted(_METHODS)}")
    level = tree.attrs.get("processing_level")
    if level not in _INPUT_LEVELS[method]:
        raise ValueError(f"{method} needs an {' or '.join(_INPUT_LEVELS[method])} product, got {level}")
    if method not in _EXTERNAL:
        return _METHODS[method](tree, masked, mask_flags, **options)
    keep = work_dir is not None
    work_dir = Path(work_dir) if keep else Path(tempfile.mkdtemp(prefix=f"hypso_{method}_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _METHODS[method](tree, masked, mask_flags, work_dir, config=config, **options)
    finally:
        if not keep:
            shutil.rmtree(work_dir, ignore_errors=True)
