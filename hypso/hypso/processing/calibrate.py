"""L1A to L1B: radiometric, smile, destriping and spectral calibration.

Coefficient files come from the sensor's calibration package
(``hypso1_calibration``, ``hypso2_calibration``); see the calibration guide
for the variants and releases.
"""
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
    selected = {key: files.get(key) for key in _FILE_KINDS}
    selected["crop_modes"] = files.get("crop_modes") or {}
    return selected, files.get("selection")


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
    """Calibrate an L1A tree to L1B (top-of-atmosphere radiance ``Lt``).

    Parameters
    ----------
    tree : xarray.DataTree
        L1A product tree.
    coeff_type : str, optional
        Radiometric variant: ``"abs"``, ``"moved"`` or ``"lab"`` (default
        ``"abs"`` for HYPSO-2, ``"moved"`` for HYPSO-1), or a set registered with
        ``hypso.calibration.register_calibration_coeffs``. None uses the
        file's ``radiometric_coefficients_version`` if present, else the
        sensor's default.
    version : str or dict, optional
        Coefficient release (HYPSO-2: ``"V3"`` default, ``"V2"``, ``"V1"``;
        HYPSO-1: ``"V2"`` default, ``"V1"``), or ``{correction: release}`` to
        mix them, e.g. ``{"spectral": "V3"}`` (the calibration package's
        ``data/coefficients.yaml``).
    coeff_files : dict, optional
        Explicit ``{kind: path}`` instead of ``coeff_type`` and ``version``.
    radiometric, smile, destripe, spectral : bool, default True
        False skips that correction.

    Returns
    -------
    xarray.DataTree
        L1B tree. The files used, their checksums and the package versions
        are recorded in ``metadata/processing/l1b``, and a summary is logged.
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
    # Per capture type (sensor profile), overridden by what the selected file
    # itself declares (calibration catalog, e.g. the V3 wide matrix).
    crop_modes = {**params.sensor_profile.capture_mode_crop_modes.get(params.capture_type, {}),
                  **files.pop("crop_modes", {})}
    rad_coeffs = _load(files["radiometric"], "radiometric", params,
                       crop_mode=crop_modes.get("radiometric", "crop_and_bin"))
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

    # metadata/corrections: the coefficients applied, kept with the product.
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
        if "radiometric_capture_type" in selection:
            parameters["radiometric_capture_type"] = selection["radiometric_capture_type"]
    parameters["radiometric_crop"] = crop_modes.get("radiometric", "crop_and_bin")
    source = tree.encoding.get("source")
    result.encoding.pop("source", None)
    add_step(result, "l1b", input_file=source or "L1A (in memory)",
             parameters=parameters,
             calibration=calibration_attrs("hypso{}_calibration".format(params.sat_id.split("-")[-1]), files))
    return result
