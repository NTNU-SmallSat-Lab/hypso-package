"""HYPSO-2 sensor profile. fwhm/fwhm_lookup_wl/fwhm_lookup_fwhm values moved verbatim from
the old hypso.hypso2.Hypso2.__init__ (unchanged - this is a relocation, not
a recalibration)."""
import numpy as np

from hypso2_calibration import get_hypso2_calibration_files

from . import SensorProfile, register_sensor

FWHM = np.array([5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                  5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46,
                  5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 5.46, 3.34,
                  3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                  3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                  3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                  3.34, 3.34, 3.34, 3.34, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                  3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29, 3.29,
                  3.29, 3.29, 3.29, 3.29, 3.29, 3.32, 3.32, 3.32, 3.32, 3.32, 3.32,
                  3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.54, 3.54, 3.54, 3.54,
                  3.58, 3.58, 3.58, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59, 3.59])

FWHM_LOOKUP_WL = np.array([435.84, 546.07, 696.54, 706.72, 738.4, 751.46, 763.51, 772.38, 811.53, 826.45, 842.46, 871.68, 912])
FWHM_LOOKUP_FWHM = np.array([5.46, 3.34, 3.29, 3.32, 3.42, 3.54, 3.58, 3.59, 4.16, 4.06, 4.66, 4.47, 5.06])


def _calibration_files(capture_type: str, coeff_type: str = "moved") -> dict:
    return get_hypso2_calibration_files(capture_type, coeff_type=coeff_type)


HYPSO2_PROFILE = SensorProfile(
    key="hypso2",
    sat_id="HYPSO-2",
    sensor="hypso2_hsi",
    platform="hypso2",
    fwhm=FWHM,
    fwhm_lookup_wl=FWHM_LOOKUP_WL,
    fwhm_lookup_fwhm=FWHM_LOOKUP_FWHM,
    calibration_files=_calibration_files,
)

register_sensor(HYPSO2_PROFILE)


def get_hypso2_wavelengths(aoi_x=428, column_count=1080, bin_factor=9):
    """Unchanged from hypso.hypso2.get_hypso2_wavelengths - kept here since
    it's sensor data/calibration logic, not HypsoCapture orchestration. Also
    re-exported from hypso.hypso2 for backward compatibility (confirmed
    public export, hypso/__init__.py)."""
    from hypso.calibration import read_coeffs_from_file

    calibration_files = get_hypso2_calibration_files()
    spectral_coeff_file = calibration_files["spectral"]

    x_start = aoi_x
    x_stop = aoi_x + column_count

    return read_coeffs_from_file(
        coeff_path=spectral_coeff_file, coeff_type="spectral",
        x_start=x_start, x_stop=x_stop, bin_factor=bin_factor,
    )
