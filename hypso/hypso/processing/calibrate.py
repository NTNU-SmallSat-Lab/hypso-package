"""L1A -> L1B: radiometric, smile, destriping and spectral calibration.

Same coefficient selection and math as hypso.calibration.pipeline's
set_calibration_coeff_files / load_calibration_coeff_files /
run_calibration, which take a capture object; this takes and returns a
product tree."""
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.calibration.correction import CalibrationShapeMismatchError, read_coeffs_from_file
from hypso.calibration.destriping import run_destriping_correction
from hypso.calibration.radiometric import run_radiometric_calibration
from hypso.calibration.registry import get_custom_calibration_coeffs
from hypso.calibration.smile import run_smile_correction
from hypso.calibration.summary import format_calibration_summary
from hypso.io.product import PRODUCT_DIMS, product_name
from hypso.io.provenance import add_step, calibration_attrs

from .capture import CaptureParams, capture_params

logger = logging.getLogger(__name__)

_FILE_KINDS = ("radiometric", "smile", "destriping", "spectral")


def _calibration_files(params: CaptureParams, coeff_type: str, version, coeff_files: dict):
    # set_calibration_coeff_files: explicit files, then a registered custom
    # set, then the sensor's built-in coefficients. Returns (files, selection);
    # selection is the versioned choice when the sensor provides one.
    if coeff_files is not None:
        return {key: coeff_files.get(key) for key in _FILE_KINDS}, None
    custom = get_custom_calibration_coeffs(params.sat_id, coeff_type)
    if custom is not None:
        return {key: custom.get(key) for key in _FILE_KINDS}, None
    kwargs = {k: v for k, v in (("coeff_type", coeff_type), ("version", version)) if v is not None}
    files = params.sensor_profile.calibration_files(params.capture_type, **kwargs)
    return {key: files.get(key) for key in _FILE_KINDS}, files.get("selection")


def _load(path, kind: str, params: CaptureParams, bin_factor: int = None, crop_mode: str = None):
    # load_calibration_coeff_files: a missing or unreadable file skips that
    # correction, except a shape mismatch, which is an error.
    kwargs = {} if crop_mode is None else {"crop_mode": crop_mode}
    try:
        return read_coeffs_from_file(path, kind, params.x_start, params.x_stop, params.y_start, params.y_stop,
                                     params.bin_factor if bin_factor is None else bin_factor, **kwargs)
    except CalibrationShapeMismatchError:
        raise
    except Exception:
        return None


def l1a_to_l1b(tree: xr.DataTree, coeff_type: str = None, version=None, coeff_files: dict = None,
               radiometric: bool = True, smile: bool = True, destripe: bool = True,
               spectral: bool = True) -> xr.DataTree:
    """Calibrate an L1A tree to L1B (TOA radiance `Lt`).

    coeff_type: radiometric variant - for HYPSO-2 "abs" (default), "moved",
        "lab" - or a set registered with
        hypso.calibration.register_calibration_coeffs. None uses the file's
        metadata/corrections radiometric_coefficients_version if present,
        else the sensor's default.
    version: coefficient release ("V3" default, "V2", "V1"), or a
        per-correction dict to mix versions, e.g. {"spectral": "V3"} (see
        hypso2_calibration's data/coefficients.yaml).
    coeff_files: explicit {kind: path} instead of coeff_type/version.
    radiometric/smile/destripe/spectral: switch individual steps off.
    """
    if tree.attrs.get("processing_level") != "L1A":
        raise ValueError(f"l1a_to_l1b needs an L1A tree, got {tree.attrs.get('processing_level')}")
    params = capture_params(tree)
    corrections = tree["metadata/corrections"].to_dataset(inherit=False)
    if coeff_type is None:
        coeff_type = corrections.attrs.get("radiometric_coefficients_version")
    else:
        corrections.attrs["radiometric_coefficients_version"] = str(coeff_type).lower()

    files, selection = _calibration_files(params, coeff_type, version, coeff_files)
    if selection is not None:
        coeff_type = selection["coeff_type"]
    crop_modes = params.sensor_profile.capture_mode_crop_modes.get(params.capture_type, {})
    rad_coeffs = _load(files["radiometric"], "radiometric", params)
    smile_coeffs = _load(files["smile"], "smile", params, crop_mode=crop_modes.get("smile", "crop_and_bin"))
    destriping_coeffs = _load(files["destriping"], "destriping", params,
                              crop_mode=crop_modes.get("destriping", "crop_and_bin"))
    spectral_coeffs = _load(files["spectral"], "spectral", params)
    spectral_coeffs_unbinned = _load(files["spectral"], "spectral", params, bin_factor=1)

    root = tree.to_dataset(inherit=False)
    cube = np.asarray(root[product_name(tree)].values, dtype="double")   # load_l1a_nc_cube reads doubles
    applied = []
    if rad_coeffs is not None and radiometric:
        cube = run_radiometric_calibration(cube=cube, background_value=params.background_value,
                                           exp=params.exposure, image_height=params.image_height,
                                           image_width=params.image_width, frame_count=params.frame_count,
                                           bin_factor=params.bin_factor, rad_coeffs=rad_coeffs)
        applied.append("radiometric")
    if smile_coeffs is not None and smile:
        cube = run_smile_correction(cube=cube, smile_coeffs=smile_coeffs)
        applied.append("smile")
    if destriping_coeffs is not None and destripe:
        cube = run_destriping_correction(cube=cube, destriping_coeffs=destriping_coeffs)
        applied.append("destriping")

    wavelengths = root["radiation_wavelength"].values
    if spectral_coeffs is not None and spectral:
        wavelengths = np.asarray(spectral_coeffs, dtype=float)
        applied.append("spectral")

    out = root.drop_vars([product_name(tree), "wavelength", "radiation_wavelength", "fwhm"])
    out["Lt"] = xr.DataArray(cube, dims=PRODUCT_DIMS, coords={
        "wavelength": ("bands", np.around(wavelengths, 1)),
        "radiation_wavelength": ("bands", wavelengths),
        "fwhm": ("bands", params.fwhm(wavelengths))})
    out.attrs["processing_level"] = "L1B"
    for kind in _FILE_KINDS:
        out.attrs[f"{kind}_file"] = str(Path(files[kind]).name) if files[kind] is not None else "None"

    # metadata/corrections as io/writer.py records it after calibration.
    if rad_coeffs is not None:
        corrections["rad_matrix"] = (("radrows", "radcols"), np.asarray(rad_coeffs, dtype="f4"))
    if spectral_coeffs is not None:
        corrections["wavelengths"] = ("bands", np.asarray(spectral_coeffs, dtype="f8"))
        corrections["spec_coeffs"] = ("specrows", np.asarray(spectral_coeffs, dtype="f8"))
    if spectral_coeffs_unbinned is not None:
        corrections["wavelengths_unbinned"] = ("bands_unbinned", np.asarray(spectral_coeffs_unbinned, dtype="f8"))
        corrections["spec_coeffs_unbinned"] = ("specrows_unbinned",
                                               np.asarray(spectral_coeffs_unbinned, dtype="f8"))

    result = tree.copy()
    result.dataset = out
    result["metadata/corrections"] = xr.DataTree(corrections)
    logger.info("%s", format_calibration_summary(params.sat_id, params.capture_type, files, applied, selection))
    parameters = {"coeff_type": coeff_type or "custom", "capture_type": params.capture_type,
                  "corrections_applied": applied}
    if selection is not None:
        parameters["coefficient_release"] = selection["release"]
        for kind in ("radiometric", "smile", "spectral"):
            parameters[f"{kind}_version"] = selection[kind]
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, "l1b", input_file=source or "L1A (in memory)",
             parameters=parameters,
             calibration=calibration_attrs("hypso{}_calibration".format(params.sat_id.split("-")[-1]), files))
    return result
