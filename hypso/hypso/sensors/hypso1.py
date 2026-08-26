"""HYPSO-1 sensor profile. fwhm/fwhm_lookup_wl/fwhm_lookup_fwhm values moved verbatim from
the old hypso.hypso1.Hypso1.__init__ (unchanged - this is a relocation, not
a recalibration)."""
import numpy as np

from hypso1_calibration import get_hypso1_calibration_files

from . import SensorProfile, register_sensor

FWHM = np.array([9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6,
                  9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 9.6, 6.6, 6.6, 6.6, 6.6, 6.6,
                  6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 6.6, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2,
                  8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8,
                  5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8,
                  5.8, 5.8, 5.8, 5.8, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1,
                  4.1, 4.1, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                  4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

# HYPSO-1, based on pre-launch calibration modelled for unbinned spectrum.
# Based on https://ieeexplore.ieee.org/abstract/document/9843655
FWHM_LOOKUP_WL = np.array([435.85, 546.07, 696.54, 706.72, 738.40, 763.51])
FWHM_LOOKUP_FWHM = np.array([4.3, 4.3, 4.05, 3.65, 3.60, 3.85])


def _calibration_files(capture_type: str, coeff_type: str = "moved") -> dict:
    return get_hypso1_calibration_files(capture_type, coeff_type=coeff_type)


# Moved here verbatim from io/dispatch.py's check_capture_type (see
# REFACTOR_PROGRESS.md's capture-dimensions audit) - same three values,
# now declared per-sensor instead of one hardcoded chain shared (and
# unverified) across every registered sensor.
CAPTURE_TYPE_THRESHOLDS = (
    ("nominal", "frame_count", 956),
    ("moon", "frame_count", 106),
    ("wide", "image_height", 1092),
)


HYPSO1_PROFILE = SensorProfile(
    key="hypso1",
    sat_id="HYPSO-1",
    sensor="hypso1_hsi",
    platform="hypso1",
    fwhm=FWHM,
    fwhm_lookup_wl=FWHM_LOOKUP_WL,
    fwhm_lookup_fwhm=FWHM_LOOKUP_FWHM,
    calibration_files=_calibration_files,
    capture_type_thresholds=CAPTURE_TYPE_THRESHOLDS,
)

register_sensor(HYPSO1_PROFILE)


def get_hypso1_wavelengths(aoi_x=428, column_count=1080, bin_factor=9):
    """Unchanged from hypso.hypso1.get_hypso1_wavelengths - kept here since
    it's sensor data/calibration logic, not HypsoCapture orchestration. Also
    re-exported from hypso.hypso1 for backward compatibility (confirmed
    public export, hypso/__init__.py)."""
    from hypso.calibration import read_coeffs_from_file

    calibration_files = get_hypso1_calibration_files()
    spectral_coeff_file = calibration_files["spectral"]

    x_start = aoi_x
    x_stop = aoi_x + column_count

    return read_coeffs_from_file(
        coeff_path=spectral_coeff_file, coeff_type="spectral",
        x_start=x_start, x_stop=x_stop, bin_factor=bin_factor,
    )
