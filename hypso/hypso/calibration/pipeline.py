"""Per-capture calibration orchestration, extracted from HypsoCapture
(self.calibration composition - part of the HypsoCapture breakup called for in the
approved refactor plan, see REFACTOR_PROGRESS.md). Bodies are moved verbatim from
HypsoCapture.py, not rewritten - same math, same behavior, just relocated - each
function takes `satobj` explicitly (matching the pattern already used by hypso.ac's
free functions and hypso.georeferencing.geo, see geo.py's module docstring for why: these read
many HypsoCapture attributes, so taking satobj as a parameter avoids either
duplicating that state or coupling this module tightly to HypsoCapture's internals via
a stored reference).

HypsoCapture's private _set_calibration_coeff_files/_run_calibration/
_load_calibration_coeff_files had no external callers (confirmed by grep before
moving), so they moved here outright with no wrapper kept on HypsoCapture - unlike
hypso.georeferencing.geo's run_georeferencing()/run_direct_georeferencing(), which stayed as
methods because those specific names are called externally.
"""
import logging

import numpy as np

from .correction import read_coeffs_from_file, CalibrationShapeMismatchError
from .radiometric import run_radiometric_calibration
from .smile import run_smile_correction
from .destriping import run_destriping_correction
from .registry import get_custom_calibration_coeffs
from hypso.capture_types import _get_fwhm, _get_fwhm_unbinned

logger = logging.getLogger(__name__)


def set_calibration_coeff_files(satobj, coeff_type: str = 'moved', coeff_files: dict = None, **kwargs) -> None:
    """
    Set the absolute path for the calibration coefficients (radiometric, smile,
    destriping, spectral) for this capture's sensor. Three ways to supply them,
    checked in order:

    1. `coeff_files` - an explicit dict (radiometric/smile/destriping/spectral/
       spectral_full_frame -> path) for a true one-off set, no registration needed.
    2. `coeff_type` matching a name previously registered via
       hypso.calibration.register_calibration_coeffs(sat_id, name, files) - a
       custom, reusable, named set, plugged in without touching the bundled
       hypsoN_calibration packages.
    3. `coeff_type` falling through to satobj.sensor_profile's calibration_files
       resolver (the sensor's built-in presets, e.g. "moved"/"adjusted"/
       "original" for HYPSO-1/-2).

    :return: None.
    """
    if satobj.sensor_profile is None:
        raise AttributeError(
            "set_calibration_coeff_files requires satobj.sensor_profile to be set - "
            "either construct this capture with a SensorProfile (see hypso.sensors), "
            "or override this method in a subclass."
        )

    capture_type = satobj.capture_type

    if coeff_files is not None:
        logger.debug("Using explicitly-supplied calibration coefficient files (coeff_files=...)")
        calibration_files = {key: coeff_files.get(key) for key in
                              ("radiometric", "smile", "destriping", "spectral", "spectral_full_frame")}
    else:
        custom = get_custom_calibration_coeffs(satobj.sat_id, coeff_type)
        if custom is not None:
            logger.debug("Using registered custom calibration coefficient set %r", coeff_type)
            calibration_files = custom
        else:
            logger.debug("Setting calibration coefficient files with coeff_type: %s", coeff_type)
            calibration_files = satobj.sensor_profile.calibration_files(capture_type, coeff_type=coeff_type)

    satobj.coeff_type = coeff_type if coeff_type is not None else "custom"
    satobj.rad_coeff_file = calibration_files['radiometric']
    satobj.smile_coeff_file = calibration_files['smile']
    satobj.destriping_coeff_file = calibration_files['destriping']
    satobj.spectral_coeff_file = calibration_files['spectral']

    # Per-capture_type crop strategy for smile/destriping (see
    # SensorProfile.capture_mode_crop_modes and calibration/correction.py's
    # read_coeffs_from_file) - defaults to "crop_and_bin" for any
    # capture_type/coeff_type not explicitly declared "as_is", which is also
    # the correct behavior for coeff_files=/registered custom sets (neither
    # has ever supported "as_is" pre-baked files, so this preserves their
    # existing behavior unchanged).
    crop_modes = satobj.sensor_profile.capture_mode_crop_modes.get(capture_type, {})
    satobj.smile_coeff_crop_mode = crop_modes.get('smile', 'crop_and_bin')
    satobj.destriping_coeff_crop_mode = crop_modes.get('destriping', 'crop_and_bin')

    return None


def load_calibration_coeff_files(satobj) -> None:
    """
    Load the calibration coefficients included in the package. This includes radiometric,
    smile and destriping correction.

    :return: None.
    """
    try:
        satobj.rad_coeffs = read_coeffs_from_file(satobj.rad_coeff_file, 'radiometric', satobj.x_start, satobj.x_stop, satobj.y_start, satobj.y_stop, satobj.bin_factor)
    except Exception:
        satobj.rad_coeffs = None

    # getattr defaults, not a plain attribute read: run_calibration(set_coeffs=
    # False) skips set_calibration_coeff_files (the only place these get set)
    # but still calls this function unconditionally - crop_mode may genuinely
    # not exist yet the first time that happens.
    smile_crop_mode = getattr(satobj, 'smile_coeff_crop_mode', 'crop_and_bin')
    destriping_crop_mode = getattr(satobj, 'destriping_coeff_crop_mode', 'crop_and_bin')

    try:
        satobj.smile_coeffs = read_coeffs_from_file(satobj.smile_coeff_file, 'smile', satobj.x_start, satobj.x_stop, satobj.y_start, satobj.y_stop, satobj.bin_factor, crop_mode=smile_crop_mode)
    except CalibrationShapeMismatchError:
        raise
    except Exception:
        satobj.smile_coeffs = None

    try:
        satobj.destriping_coeffs = read_coeffs_from_file(satobj.destriping_coeff_file, 'destriping', satobj.x_start, satobj.x_stop, satobj.y_start, satobj.y_stop, satobj.bin_factor, crop_mode=destriping_crop_mode)
    except CalibrationShapeMismatchError:
        raise
    except Exception:
        satobj.destriping_coeffs = None

    try:
        satobj.spectral_coeffs = read_coeffs_from_file(satobj.spectral_coeff_file, 'spectral', satobj.x_start, satobj.x_stop, satobj.y_start, satobj.y_stop, satobj.bin_factor)
    except Exception:
        satobj.spectral_coeffs = None

    try:
        satobj.spectral_coeffs_unbinned = read_coeffs_from_file(satobj.spectral_coeff_file, 'spectral', satobj.x_start, satobj.x_stop, satobj.y_start, satobj.y_stop, 1)
    except Exception:
        satobj.spectral_coeffs_unbinned = None

    return None


def run_calibration(satobj,
                    radiometric: bool = True,
                    smile: bool = True,
                    destripe: bool = True,
                    spectral: bool = True,
                    set_coeffs: bool = True,
                    coeff_type: str = None,
                    **kwargs) -> np.ndarray:
    """
    Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
        Assumes all coefficients has been adjusted to the frame size (cropped and
        binned), and that the data cube contains 12-bit values.

    :return: None
    """
    if satobj.VERBOSE:
        logger.info('Running calibration routines...')

    if coeff_type is None:
        try:
            coeff_type = satobj.metadata.corrections.attrs['radiometric_coefficients_version']
        except Exception:
            pass
    else:
        satobj.metadata.corrections.attrs['radiometric_coefficients_version'] = str(coeff_type).lower()

    if set_coeffs:
        set_calibration_coeff_files(satobj, coeff_type=coeff_type, **kwargs)

    load_calibration_coeff_files(satobj)

    calibrated_cube = satobj.l1a_cube.to_numpy()

    if satobj.rad_coeffs is not None:
        if radiometric:
            if satobj.VERBOSE:
                logger.info("Running radiometric calibration...")

            calibrated_cube = run_radiometric_calibration(cube=calibrated_cube,
                                            background_value=satobj.background_value,
                                            exp=satobj.exposure,
                                            image_height=satobj.image_height,
                                            image_width=satobj.image_width,
                                            frame_count=satobj.frame_count,
                                            bin_factor=satobj.bin_factor,
                                            rad_coeffs=satobj.rad_coeffs
                                            )

    if satobj.smile_coeffs is not None:
        if smile:
            if satobj.VERBOSE:
                logger.info("Running smile correction...")

            calibrated_cube = run_smile_correction(cube=calibrated_cube,
                                            smile_coeffs=satobj.smile_coeffs)

    if satobj.destriping_coeffs is not None:
        if destripe:
            if satobj.VERBOSE:
                logger.info("Running destriping correction...")

            calibrated_cube = run_destriping_correction(cube=calibrated_cube,
                                                destriping_coeffs=satobj.destriping_coeffs)

    if satobj.spectral_coeffs is not None:
        if spectral:
            if satobj.VERBOSE:
                logger.info("Running spectral correction (binned)...")

            satobj.wavelengths = satobj.spectral_coeffs
            # fwhm is derived from wavelengths via the sensor's lookup table
            # (see io/dispatch.py's set_hypso_attributes) - resync it here too,
            # or it stays stale (computed from the pre-calibration wavelengths
            # set at load time) for the rest of this capture's life. Confirmed
            # via a real written L1B file before this fix: bands whose
            # spectral-calibration-refined wavelength crossed a fwhm_lookup_wl
            # boundary kept the load-time fwhm value instead of the correct one.
            if hasattr(satobj, 'fwhm_lookup_wl') and hasattr(satobj, 'fwhm_lookup_fwhm'):
                satobj.fwhm = _get_fwhm(satobj)

    if satobj.spectral_coeffs_unbinned is not None:
        if spectral:
            if satobj.VERBOSE:
                logger.info("Running spectral correction (unbinned)...")

            satobj.wavelengths_unbinned = satobj.spectral_coeffs_unbinned
            if hasattr(satobj, 'fwhm_lookup_wl') and hasattr(satobj, 'fwhm_lookup_fwhm'):
                satobj.fwhm_unbinned = _get_fwhm_unbinned(satobj)

    return calibrated_cube
