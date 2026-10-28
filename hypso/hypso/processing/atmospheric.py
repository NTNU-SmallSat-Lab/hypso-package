"""L1C or L1D to L2A: atmospheric correction.

:func:`correct_atmosphere` returns an L2A tree: the method's product cube on
the input's bands, its 2D products and flags, the input's geometry,
``pixel_flags`` and metadata, an ``atmospheric_correction`` global attribute
(used in file names), and an ``l2a_<method>`` provenance step.

Methods:

- "machi" (default): MACHI, Minimal Atmospheric Compensation for Hyperspectral
  Imagers (hypso.ac.machi, vendored from NTNU SmallSat Lab), in-process on the
  L1D cube: one scattering and one transmission spectrum for the whole image,
  fitted so corrected spectra are smooth; surface reflectance ``rhos``.
- "dps": dark pixel subtraction, in-process on the L1D cube; surface
  reflectance ``rhos``.
- "polymer": Polymer in a subprocess (hypso.ac.adapters._polymer_driver),
  from L1C (default) or L1D; Polymer's water reflectance divided by pi, Rrs.
  Installation and settings come from the AC
  tool configuration (hypso.ac.config).
- "acolite": ACOLITE in a subprocess (hypso.ac.adapters._acolite_driver),
  from L1C (default) or L1D; output="l2w" (Rrs, default) or "l2r" (rhos).
  EARTHDATA credentials (for ancillary data) are read from the
  HYPSO_ACOLITE_EARTHDATA_USERNAME/_PASSWORD environment variables, never
  from the configuration file.
- "ocsmart": OC-SMART (`python OCSMART.py` in its installation directory),
  from L1D; Rrs. OC-SMART reads its paths from a shared OCSMART_Input.txt
  in that directory: the file is written and restored under the same lock
  hypso-processing-pipeline uses (.OCSMART_Input.lock), so concurrent runs
  on one installation wait for each other.

The product variable is ``Rrs`` or ``rhos`` and gives the quantity field of
the file name (hypso.io.naming: ``L2A.RRS.<method>`` or ``L2A.RHOS.<method>``).

External tools read a file: the input is written (masked if requested,
with metadata/srf added if missing) into `work_dir`, which is a temporary
directory removed afterwards unless given.
"""
import fcntl
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.ac.adapters.base import ACRunError, run_subprocess_driver
from hypso.ac.config import tool_config

from hypso.io.product import PRODUCT_DIMS, SPATIAL_DIMS, product_name, write_product
from hypso.io.naming import quantity_token
from hypso.io.provenance import ac_attrs, add_step
from hypso.masks.flags import FLAGS_NAME, apply_flags

logger = logging.getLogger(__name__)

_INPUT_LEVELS = {"machi": ("L1D",), "dps": ("L1D",), "polymer": ("L1C", "L1D"), "acolite": ("L1C", "L1D"), "ocsmart": ("L1D",)}


def _input_cube(tree: xr.DataTree, masked: bool, mask_flags) -> xr.DataArray:
    cube = tree.dataset[product_name(tree)]
    if not masked:
        return cube
    if FLAGS_NAME not in tree.dataset:
        raise ValueError("masked=True but the product has no pixel_flags")
    return apply_flags(cube, tree.dataset[FLAGS_NAME], mask_flags)


def _l2a_tree(tree: xr.DataTree, method: str, variable: str, cube: np.ndarray, parameters: dict,
              ac: dict, variables: dict = None) -> xr.DataTree:
    root = tree.to_dataset(inherit=False)
    source = root[product_name(tree)]
    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out[variable] = xr.DataArray(np.asarray(cube, dtype=np.float32), dims=PRODUCT_DIMS, coords={
        name: ("bands", source[name].values) for name in ("wavelength", "radiation_wavelength", "fwhm")})
    for name, (values, attrs) in (variables or {}).items():
        out[name] = xr.Variable(SPATIAL_DIMS, values, attrs)
    out.attrs["processing_level"] = "L2A"
    out.attrs["atmospheric_correction"] = method
    out.attrs["l2a_quantity"] = quantity_token(variable)

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
    # Subtracting the dark spectrum removes the path signal only (no
    # transmittance, glint or sky-reflection correction), so the result is a
    # surface reflectance, rhos (user decision 2026-10-28, reversing the
    # 2026-10-26 one that divided by pi and stored it as Rrs).
    corrected = np.clip(cube - dark_spectrum, 0, None)
    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "statistic": statistic, "dark_spectrum": [float(v) for v in dark_spectrum]}
    if percentile is not None:
        parameters["percentile"] = percentile
    return _l2a_tree(tree, "dps", "rhos", corrected, parameters,
                     ac_attrs("dps", version="hypso-package", settings_effective={"statistic": statistic,
                                                                                  "percentile": percentile}))


def _machi(tree, masked, mask_flags, est_min_R=0.05, tol=0.01, approach_rate=1.0, fit_exclude_flags=None,
           fit_subsample=1):
    # MACHI (hypso.ac.machi, vendored unchanged from upstream ebbc3e72): fit one
    # scattering (A) and one transmission (T) spectrum on the image's finite
    # pixels (all of them, as upstream - user decision 2026-10-27 - or without
    # those flagged in fit_exclude_flags), then rhos = (rhot - A) / T for every
    # pixel. The input is reflectance, so MACHI's solar spectrum is ones.
    from hypso.ac import machi
    cube = _input_cube(tree, masked, mask_flags).values
    lines, samples, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    fit = np.isfinite(pixels).all(axis=1)
    if fit_exclude_flags:
        if FLAGS_NAME not in tree.dataset:
            raise ValueError("fit_exclude_flags given but the product has no pixel_flags")
        from hypso.masks.flags import decode_flags
        flags = decode_flags(tree.dataset[FLAGS_NAME])
        unknown = sorted(set(fit_exclude_flags) - set(flags))
        if unknown:
            raise ValueError(f"unknown flags {unknown}; the product has {sorted(flags)}")
        for name in fit_exclude_flags:
            fit &= ~flags[name].values.ravel()
    if fit.sum() < 2:
        raise ValueError("MACHI needs finite pixels to fit; none are left")
    # Bands without signal (all zero: HYPSO's smile correction blanks < 400 nm)
    # make MACHI's update divide by zero and stop at once; fit the others only.
    rows = np.flatnonzero(fit)
    used = np.flatnonzero([np.any(pixels[rows, b] != 0) for b in range(bands)])
    if fit_subsample > 1:
        # Every Nth fit pixel, plus each band's darkest one: MACHI starts A from
        # (and caps it at) the per-band minimum, and a plain subsample misses it
        # (A then shifts by ~1e-3, ~4% of dark water). With the minima kept, a
        # 1-in-16 fit changes rhos by ~7e-5 for 1/12 of the time and 1/4 of the
        # memory (reference capture, REFACTOR_PROGRESS.md ×80).
        darkest = [rows[int(np.argmin(pixels[rows, b]))] for b in used]
        rows = np.union1d(rows[::int(fit_subsample)], darkest)
    t_fit, a_fit, objective = machi.atm_correction(pixels[rows][:, used].astype(np.float64), solar=np.ones(used.size),
                                                   tol=tol, est_min_R=est_min_R, approach_rate=approach_rate)
    T = np.full(bands, np.nan)
    A = np.full(bands, np.nan)
    T[used], A[used] = t_fit, a_fit
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.where(T > 0, (cube - A) / T, np.nan)
    settings = {"est_min_R": est_min_R, "tol": tol, "approach_rate": approach_rate,
                "fit_exclude_flags": list(fit_exclude_flags or []), "fit_subsample": int(fit_subsample)}
    parameters = {**settings, "masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "fit_pixels": int(rows.size), "bands_fitted": int(used.size),
                  "bands_without_signal": [int(b) for b in np.setdiff1d(np.arange(bands), used)],
                  "iterations": len(objective) - 1,
                  "objective_final": float(objective[-1])}
    result = _l2a_tree(tree, "machi", "rhos", corrected, parameters,
                       ac_attrs("machi", version=MACHI_VERSION, settings_effective=settings))
    wavelengths = tree.dataset["radiation_wavelength"].values
    result["metadata/machi"] = xr.DataTree(xr.Dataset(
        {"T": ("bands", np.asarray(T, dtype="f8"), {"long_name": "Atmospheric transmission fitted by MACHI",
                                                   "units": "1"}),
         "A": ("bands", np.asarray(A, dtype="f8"), {"long_name": "Atmospheric scattering (path reflectance) "
                                                                 "fitted by MACHI", "units": "1"}),
         "objective": ("iteration", np.asarray(objective, dtype="f8"),
                       {"long_name": "MACHI objective (mean squared spectral derivative) per iteration"})},
        coords={"wavelength": ("bands", np.asarray(wavelengths, dtype="f8"), {"units": "nm"})},
        attrs={"model": "rhos = (rhot - A) / T", "source": MACHI_VERSION}))
    return result


# Vendored MACHI version (hypso/ac/machi.py).
MACHI_VERSION = "MACHI ebbc3e72 (vendored)"


def _band_map(source_wavelengths, tool_wavelengths) -> tuple:
    """(source indices, tool indices): each tool band paired with the
    nearest source band. A tool band whose nearest source band is already
    taken is dropped. Returned as pairs so a dropped band cannot shift the
    ones after it (the removed capture-object adapters assigned the first N
    tool bands, misplacing every band after a dropped one)."""
    source = np.asarray(source_wavelengths, dtype=float)
    seen, source_idx, tool_idx = set(), [], []
    for t, wavelength in enumerate(np.asarray(tool_wavelengths, dtype=float)):
        index = int(np.argmin(np.abs(source - wavelength)))
        if index in seen:
            logger.warning("Duplicate prevented: %s mapped to %s", wavelength, index)
            continue
        seen.add(index)
        source_idx.append(index)
        tool_idx.append(t)
    return np.array(source_idx, dtype=int), np.array(tool_idx, dtype=int)


def _onto_bands(tree, tool_cube, tool_wavelengths) -> tuple:
    """The tool's (y, x, n) cube on the input product's bands (NaN where it
    has none), and how many bands it filled."""
    source_idx, tool_idx = _band_map(tree.dataset["radiation_wavelength"].values, tool_wavelengths)
    cube = np.full(tree.dataset[product_name(tree)].shape, np.nan, dtype=np.float32)
    cube[:, :, source_idx] = np.asarray(tool_cube)[:, :, tool_idx]
    return cube, int(len(source_idx))


def _tool_variables(path, keep, renames: dict = None, overrides: dict = None, shape=None) -> dict:
    """{name: (2D array, attrs)} for an AC tool's 2D output variables for
    which keep(name) is true, read with their own dtype (flags stay
    integer; masked float values become NaN), renamed per `renames`, with
    attributes made CF-valid (io.cf.tool_variable_attrs) and `overrides`
    applied. Variables of another shape than `shape` are skipped."""
    import netCDF4
    from hypso.io import cf
    out = {}
    with netCDF4.Dataset(path) as f:
        for name, var in f.variables.items():
            if var.ndim != 2 or not keep(name):
                continue
            data = var[:]
            data = (np.ma.filled(np.ma.asarray(data, dtype=np.float32), np.nan) if var.dtype.kind == "f"
                    else np.ma.getdata(data))
            if shape is not None and data.shape != tuple(shape):
                logger.warning("Skipping %s from %s: shape %s, product is %s", name, Path(path).name, data.shape, shape)
                continue
            attrs = cf.tool_variable_attrs({k: var.getncattr(k) for k in var.ncattrs()})
            attrs["source_variable"] = name
            new_name = (renames or {}).get(name, name)
            attrs.update((overrides or {}).get(new_name, {}))
            out[new_name] = (data, attrs)
    return out


def _git_commit(path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _stage_input(tree: xr.DataTree, masked: bool, mask_flags, work_dir: Path, add_srfs: bool = True,
                 name: str = None) -> Path:
    """Write the tool's input file into work_dir (named `name`, default the
    product's own file name)."""
    from hypso.api import Product
    from hypso.processing.toa import srf_node

    staged = tree.copy()
    if masked:
        root = staged.to_dataset(inherit=False)
        root[product_name(tree)] = _input_cube(tree, masked, mask_flags)
        staged.dataset = root
    has_srfs = ("metadata" in staged.children and "srf" in staged["metadata"].children
                and any(name.startswith("Band_") for name in staged["metadata/srf"].dataset.data_vars))
    if add_srfs and not has_srfs:
        # Polymer reads per-band SRFs from metadata/srf (its srf_getter); an
        # L1C straight from georeferencing has none yet.
        staged["metadata/srf"] = xr.DataTree(srf_node(staged))
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / (name or Product(staged).filename())
    # Lossless: AC amplifies input errors about tenfold in Rrs (×71).
    write_product(staged, path, significant_bits=None)
    return path


# Polymer's flag bits (polymer/common.py: L2FLAGS and L2FLAGS_POLYMER). Its
# output's own flag_meanings attribute lists only the first three.
POLYMER_FLAGS = {"LAND": 1, "CLOUD_BASE": 2, "L1_INVALID": 4, "OUT_OF_BOUNDS": 16, "EXCEPTION": 32,
                 "THICK_AEROSOL": 64, "HIGH_AIR_MASS": 128, "EXTERNAL_MASK": 512, "CASE2": 1024,
                 "INCONSISTENCY": 2048, "ANOMALY_RWMOD_BLUE": 4096}


def _polymer_variables(path, shape) -> dict:
    """Polymer's 2D outputs: chla (10**logchl, user decision 2026-10-26),
    logfb, SPM, Rgli, Rnir and its flags (polymer_flags, with Polymer's full
    flag table)."""
    from hypso.io import cf
    variables = _tool_variables(path, keep=lambda n: n not in ("latitude", "longitude") and not n.startswith("rho_w"),
                                renames={"flags": "polymer_flags"}, shape=shape)
    if "logchl" in variables:
        logchl, attrs = variables.pop("logchl")
        variables["chla"] = (np.power(10.0, logchl).astype(np.float32),
                             {"units": "mg m-3", "long_name": "Chlorophyll-a concentration (Polymer, 10**logchl)",
                              "standard_name": cf.CHLA_STANDARD_NAME, "source_variable": "logchl"})
    descriptions = {"logfb": ("1", "log10 of the particle backscattering factor (Polymer)"),
                    "SPM": (None, "Suspended particulate matter (Polymer)"),
                    "Rgli": ("1", "Sun glint reflectance (Polymer)"),
                    "Rnir": ("1", "Reflectance at the NIR band used for aerosol correction (Polymer)")}
    for name, (units, long_name) in descriptions.items():
        if name in variables:
            data, attrs = variables[name]
            attrs.setdefault("long_name", long_name)
            if units:
                attrs.setdefault("units", units)
    if "polymer_flags" in variables:
        data, attrs = variables["polymer_flags"]
        attrs.update(long_name="Polymer flags", flag_masks=np.array(list(POLYMER_FLAGS.values()), dtype=data.dtype),
                     flag_meanings=" ".join(POLYMER_FLAGS))
    return variables


def _polymer(tree, masked, mask_flags, work_dir, config=None, **options):
    from hypso.ac.polymer_output import read_polymer_reflectance

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
    output_file = Path(result["output_file"])
    tool_cube, tool_wavelengths = read_polymer_reflectance(output_file)
    cube, n_bands = _onto_bands(tree, tool_cube, tool_wavelengths)
    variables = _polymer_variables(output_file, tree.dataset[product_name(tree)].shape[:2])

    settings = {k: driver_config[k] for k in ("polymer_version", "optional_output_datasets", "polymer_settings",
                                              "ancillary_source", "era5_allow_preliminary")}
    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "input_level": tree.attrs["processing_level"], "bands_returned": n_bands,
                  "output_file": output_file.name}
    # Polymer returns water-leaving reflectance rho_w = pi Rrs; stored as Rrs
    # (sr-1) like the other water products (user decision 2026-10-28).
    cube = cube / np.pi
    parameters["converted"] = "Rrs = rho_w / pi"
    ac = ac_attrs("polymer", version=version, git_commit=_git_commit(cfg.get("polymer_path") or cfg.get("base_path")),
                  install_path=cfg.get("base_path"), settings_effective=settings,
                  settings_user={**(options.get("settings") or {}), **{k: v for k, v in options.items() if k != "settings"}})
    return _l2a_tree(tree, "polymer", "Rrs", cube, parameters, ac, variables)


# hypso.ac.adapters.acolite.ACOLITEAdapter's settings, applied over the
# sensor defaults and the configuration's settings_file.
ACOLITE_SETTINGS = {
    "polygon": None, "rgb_rhot": True, "rgb_rhos": True,
    "map_l2w": False,  # produces blank .pngs
    "l2w_mask": False, "l2w_mask_threshold": 0.2,
    "l2w_parameters": ["Rrs_*", "spm_nechad2010", "spm_nechad2016", "chl_re_mishra", "chl_oc2", "chl_oc3",
                       "chl_re_moses3b", "chl_re_moses3b740", "fai", "fai_rhot", "fait", "ndci"],
}
_ACOLITE_OUTPUTS = {"l2w": ("L2W", "Rrs"), "l2r": ("L2R", "rhos")}


def _acolite_version(path) -> str:
    # ACOLITE's own rule (acolite/__init__.py): config/config.txt's
    # `version`, otherwise it is a git clone.
    config_txt = Path(path or "", "config", "config.txt")
    if config_txt.is_file():
        for line in config_txt.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "version" and value.strip():
                return value.strip()
    return "git checkout"


def _acolite(tree, masked, mask_flags, work_dir, config=None, output="l2w", reuse_output=False, **options):
    from hypso.ac import load_acolite_l2r_nc, load_acolite_l2w_nc

    if output not in _ACOLITE_OUTPUTS:
        raise ValueError(f"output must be one of {sorted(_ACOLITE_OUTPUTS)}, got {output!r}")
    suffix, variable = _ACOLITE_OUTPUTS[output]
    cfg = tool_config("acolite", config, options)
    output_dir = work_dir / "acolite"
    existing = sorted(output_dir.glob(f"*_{suffix}.nc")) if output_dir.is_dir() else []
    sensor = str(tree.attrs.get("sat_id", "")).replace("-", "").upper()     # HYPSO-2 -> HYPSO2
    settings = {**ACOLITE_SETTINGS, **(cfg.get("settings") or {})}

    effective = None   # the settings ACOLITE reports it used (None when reusing an earlier run)
    if reuse_output and existing:
        logger.info("Reusing ACOLITE output %s", existing[-1])
    else:
        input_file = _stage_input(tree, masked, mask_flags, work_dir, add_srfs=False)
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        result = run_subprocess_driver(
            python_path=cfg.get("python") or sys.executable, driver_module="hypso.ac.adapters._acolite_driver",
            config={"acolite_path": cfg.get("path"), "sensor": sensor, "settings_file": cfg.get("settings_file"),
                    "settings_overrides": {**settings, "inputfile": str(input_file), "output": str(output_dir)}},
            tool_name="acolite")
        existing = sorted(output_dir.glob(f"*_{suffix}.nc"))
        effective = result.get("settings")
    if len(existing) != 1:
        raise RuntimeError(f"expected one ACOLITE *_{suffix}.nc in {output_dir}, found {len(existing)}")

    data = (load_acolite_l2w_nc if output == "l2w" else load_acolite_l2r_nc)(existing[0])[variable]
    cube, n_bands = _onto_bands(tree, data.values, data["band"].values)
    # L2W carries ACOLITE's water products (chlorophyll, SPM, NDCI, ... - the
    # set depends on l2w_parameters) and l2_flags.
    variables = {}
    if output == "l2w":
        variables = _tool_variables(existing[0], keep=lambda n: n not in ("lat", "lon") and not n.startswith("Rrs"),
                                    renames={"l2_flags": "acolite_l2_flags"},
                                    overrides={"acolite_l2_flags": {"long_name": "ACOLITE L2W flags (l2_flags)"}},
                                    shape=tree.dataset[product_name(tree)].shape[:2])

    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "input_level": tree.attrs["processing_level"], "output": output, "output_file": existing[0].name,
                  "bands_returned": n_bands, "reused_output": bool(reuse_output and existing)}
    settings_file = cfg.get("settings_file")
    ac = ac_attrs("acolite", version=_acolite_version(cfg.get("path")), git_commit=_git_commit(cfg.get("path")),
                  install_path=cfg.get("path"),
                  settings_effective=effective if effective is not None else
                  {"sensor": sensor, "settings_file": Path(settings_file).name if settings_file else "", **settings,
                   "note": "reused an earlier run's output; requested settings, not ACOLITE's report"},
                  settings_user={**(options.get("settings") or {}),
                                 **{k: v for k, v in options.items() if k != "settings"}})
    return _l2a_tree(tree, f"acolite_{output}", variable, cube, parameters, ac, variables)


# OC-SMART's sensor autodetection (src/sensorinfo.py) only recognises HYPSO
# input files by this prefix (no satellite digit; HYPSO-1 and -2 share one
# coefficient file) - hypso.ac.adapters.ocsmart.OCSMARTAdapter.HYPSO_PREFIX.
OCSMART_PREFIX = "HYPSO_HSI"


# OC-SMART output variables that repeat the input's geometry.
_OCSMART_GEOMETRY = {"Latitude", "Longitude", "Relative_azimuth", "Sensor_zenith", "Solar_zenith"}


def _ocsmart_version(install_dir) -> str:
    # Installations are named like OCSMART_Linux_v2.6.3.
    match = re.search(r"v\d+(\.\d+)+", Path(install_dir).name)
    return match.group(0) if match else "unknown"


@contextmanager
def _ocsmart_lock(install_dir):
    # Same lock file as hypso-processing-pipeline's ocsmart_config_lock.
    lock_path = Path(install_dir) / ".OCSMART_Input.lock"
    with open(lock_path, "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Waiting for another OC-SMART run to release %s", lock_path)
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _ocsmart(tree, masked, mask_flags, work_dir, config=None, **options):
    from hypso.api import Product
    from hypso.ac import load_ocsmart_h5

    cfg = tool_config("ocsmart", config, options)
    install_dir = Path(cfg["dir"]).absolute()
    if "." in str(work_dir.absolute()):
        # OCSMART.py replaces every "." in its configured path values with its
        # own directory, so these paths must not contain one.
        raise ValueError(f"OC-SMART cannot use a work_dir whose path contains '.': {work_dir.absolute()}")
    l2_prod = cfg.get("l2_prod", "Lt,Lr,Lrc,rrs,chl,tg_sol,tg_sen,Lwp")
    solz_limit, senz_limit = cfg.get("solz_limit", 70.0), cfg.get("senz_limit", 70.0)

    capture_id = _capture_id(tree)
    staging_dir = work_dir.absolute() / "ocsmart_staging" / "l1b"
    output_dir = work_dir.absolute() / "ocsmart"
    output_dir.mkdir(parents=True, exist_ok=True)
    _stage_input(tree, masked, mask_flags, staging_dir, add_srfs=False, name=f"{OCSMART_PREFIX}_{capture_id}-l1d.nc")
    output_file = output_dir / f"{OCSMART_PREFIX}_{capture_id}-l1d_L2_OCSMART.h5"

    input_txt = install_dir / "OCSMART_Input.txt"
    lines = []
    with _ocsmart_lock(install_dir):
        original = input_txt.read_text() if input_txt.is_file() else None
        try:
            input_txt.write_text(f"l1b_path = {staging_dir}/\nl2_path = {output_dir}/\nl2_prod = {l2_prod}\n"
                                 f"solz_limit = {solz_limit}\nsenz_limit = {senz_limit}\n")
            process = subprocess.Popen([str(cfg.get("python") or "python3"), "OCSMART.py"], cwd=install_dir,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                logger.info("OC-SMART: %s", line.rstrip())
                lines.append(line)
            returncode = process.wait()
        finally:
            if original is not None:
                input_txt.write_text(original)
    if returncode != 0:
        raise ACRunError(tool="ocsmart", returncode=returncode, stdout="".join(lines), stderr="")
    if not output_file.is_file():
        raise ACRunError(tool="ocsmart", returncode=0, stdout="".join(lines), stderr="",
                         error_type="FileNotFoundError",
                         message=f"no OC-SMART output at {output_file} after a zero-exit run - it likely did not "
                                 f"recognise the sensor from the staged file name; see its output above")

    rrs = load_ocsmart_h5(output_file)["Rrs"]
    cube, n_bands = _onto_bands(tree, rrs.values, rrs["band"].values)
    # OC-SMART writes no attributes; units/names per its documentation.
    from hypso.io import cf
    chl = {"units": "mg m-3", "standard_name": cf.CHLA_STANDARD_NAME}
    variables = _tool_variables(
        output_file, keep=lambda n: n not in _OCSMART_GEOMETRY,
        renames={"L2_flags": "ocsmart_l2_flags", "chlor_a(oci)": "chlor_a_oci", "chlor_a(yoc)": "chlor_a_yoc",
                 "tsm(yoc)": "tsm_yoc"},
        overrides={"ocsmart_l2_flags": {"long_name": "OC-SMART L2 flags (L2_flags)"},
                   "chlor_a_oci": {**chl, "long_name": "Chlorophyll-a concentration, OCI algorithm (OC-SMART)"},
                   "chlor_a_yoc": {**chl, "long_name": "Chlorophyll-a concentration, YOC algorithm (OC-SMART)"},
                   "tsm_yoc": {"units": "g m-3", "standard_name": cf.SPM_STANDARD_NAME,
                               "long_name": "Total suspended matter, YOC algorithm (OC-SMART)"}},
        shape=tree.dataset[product_name(tree)].shape[:2])

    parameters = {"masked": masked, "mask_flags": list(mask_flags) if mask_flags else "all" if masked else "",
                  "input_level": "L1D", "bands_returned": n_bands, "output_file": output_file.name}
    settings = {"l2_prod": l2_prod, "solz_limit": solz_limit, "senz_limit": senz_limit}
    ac = ac_attrs("ocsmart", version=_ocsmart_version(install_dir), install_path=install_dir,
                  settings_effective=settings, settings_user=options)
    return _l2a_tree(tree, "ocsmart", "Rrs", cube, parameters, ac, variables)


def _capture_id(tree: xr.DataTree) -> str:
    """{target}_{YYYY-MM-DDTHH-MM-SSZ}, the capture directory name."""
    from datetime import datetime
    from hypso.io.naming import product_name as make_name
    from .capture import acquisition_time
    time = datetime.fromisoformat(acquisition_time(tree.attrs))
    return make_name(tree.attrs.get("sat_id"), time, tree.attrs.get("location_description"), "L1D").capture_id


_METHODS = {"machi": _machi, "dps": _dps, "polymer": _polymer, "acolite": _acolite, "ocsmart": _ocsmart}
_EXTERNAL = {"polymer", "acolite", "ocsmart"}


def correct_atmosphere(tree: xr.DataTree, method: str = "machi", masked: bool = False, mask_flags=None,
                       work_dir=None, config=None, **options) -> xr.DataTree:
    """Atmospheric correction of an L1C or L1D tree.

    Parameters
    ----------
    tree : xarray.DataTree
        L1C or L1D product tree in sensor geometry (not regridded).
    method : {"machi", "dps", "polymer", "acolite", "ocsmart"}, default "machi"
        See the module description.
    masked : bool, default False
        Give the method NaN where ``pixel_flags`` are set (all flags, or only
        ``mask_flags``). The input product itself stays unmasked.
    mask_flags : list of str, optional
        Flags to mask with, e.g. ``["land", "cloud"]``.
    work_dir : str or path, optional
        External tools only: where the staged input and the tool's raw
        output go (kept); default a temporary directory, removed afterwards.
    config : str, path or dict, optional
        External tools only: AC tool configuration (:mod:`hypso.ac.config`);
        default the ``HYPSO_AC_CONFIG`` environment variable.
    **options
        Method-specific. machi: ``est_min_R`` (expected minimum surface
        reflectance, default 0.05), ``tol`` (default 0.01), ``approach_rate``
        (default 1), ``fit_exclude_flags`` (pixel_flags names left out of the
        fit; default none, all finite pixels), ``fit_subsample`` (fit on every
        Nth pixel plus each band's darkest; default 1, all). dps: ``statistic`` (``"min"`` or
        ``"percentile"``), ``percentile``. polymer, acolite, ocsmart: any key
        of the tool's configuration section (e.g. ``settings={...}``),
        overriding it; acolite also ``output`` (``"l2w"`` or ``"l2r"``) and
        ``reuse_output``.

    Returns
    -------
    xarray.DataTree
        L2A tree.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; available: {sorted(_METHODS)}")
    level = tree.attrs.get("processing_level")
    if level not in _INPUT_LEVELS[method]:
        raise ValueError(f"{method} needs an {' or '.join(_INPUT_LEVELS[method])} product, got {level}")
    if tree.attrs.get("grid"):
        # The AC tools read sensor geometry; correct first, then regrid.
        raise ValueError(f"correct_atmosphere needs a product in sensor geometry, this one is on grid "
                         f"{tree.attrs['grid']!r}: run it before regrid()")
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
