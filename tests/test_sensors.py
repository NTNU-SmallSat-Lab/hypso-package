"""Plan §Verification item 3 (part): hypso.sensors registry unit tests -
profile lookup by both keys, required-field completeness for both HYPSO-1 and
HYPSO-2. No real data needed."""
import numpy as np
import pytest

from hypso.sensors import get_sensor_profile, registered_sensors, SensorProfile


@pytest.mark.parametrize("sat_id,key", [("HYPSO-1", "hypso1"), ("HYPSO-2", "hypso2")])
def test_lookup_by_either_key(sat_id, key):
    by_sat_id = get_sensor_profile(sat_id)
    by_key = get_sensor_profile(key)
    assert by_sat_id is by_key
    assert by_sat_id.sat_id == sat_id
    assert by_sat_id.key == key


def test_unknown_sensor_raises_keyerror_listing_known():
    with pytest.raises(KeyError) as exc:
        get_sensor_profile("HYPSO-99")
    assert "HYPSO-1" in str(exc.value) and "HYPSO-2" in str(exc.value)


def test_registered_sensors_deduplicated():
    profiles = registered_sensors()
    assert len(profiles) == len({p.sat_id for p in profiles})
    assert {p.sat_id for p in profiles} >= {"HYPSO-1", "HYPSO-2"}


@pytest.mark.parametrize("key", ["hypso1", "hypso2"])
def test_profile_field_completeness(key):
    p = get_sensor_profile(key)
    assert isinstance(p, SensorProfile)
    # every field the processing code reads off the profile must be populated
    for field in ("key", "sat_id", "sensor", "platform"):
        assert isinstance(getattr(p, field), str) and getattr(p, field), field
    for field in ("fwhm", "fwhm_lookup_wl", "fwhm_lookup_fwhm"):
        arr = np.asarray(getattr(p, field))
        assert arr.size > 0, field
        assert np.all(np.isfinite(arr)), field
    assert callable(p.calibration_files)
    # fwhm_lookup_wl/fwhm_lookup_fwhm are paired reference arrays for nearest-neighbor lookup
    assert np.asarray(p.fwhm_lookup_wl).shape == np.asarray(p.fwhm_lookup_fwhm).shape


def test_profiles_are_frozen():
    p = get_sensor_profile("hypso1")
    with pytest.raises(Exception):
        p.sat_id = "changed"


@pytest.mark.parametrize("key", ["hypso1", "hypso2"])
def test_capture_type_thresholds_well_formed(key):
    # capture-dimensions audit (REFACTOR_PROGRESS.md): check_capture_type
    # used to be one hardcoded chain shared by every sensor - each profile
    # now declares its own (capture_type, attr, expected_value) rules.
    p = get_sensor_profile(key)
    assert p.capture_type_thresholds
    for capture_type, attr, expected_value in p.capture_type_thresholds:
        assert isinstance(capture_type, str) and capture_type
        assert attr in ("frame_count", "image_height")
        assert isinstance(expected_value, int)


@pytest.mark.parametrize("frame_count,image_height,expected", [
    (956, 684, "nominal"),
    (106, 684, "moon"),
    (500, 1092, "wide"),
    (12345, 12345, "custom"),
])
def test_capture_type_classified_from_sensor_profile(frame_count, image_height, expected):
    from hypso.processing import classify_capture_type
    profile = get_sensor_profile("hypso1")
    assert classify_capture_type(profile, {"frame_count": frame_count, "image_height": image_height}) == expected


def test_hypso1_moon_capture_type_gets_real_calibration_files():
    # Regression test for a confirmed live bug: hypso1_calibration's
    # get_hypso1_calibration_files() had no "moon" case in its match
    # statement, so it fell through to the catch-all and returned every
    # coefficient file as None - which calibration/pipeline.py's
    # load_calibration_coeff_files silently swallows (bare try/except),
    # so a moon capture (frame_count==106) got NO calibration at all with
    # no error. radiometric_calibration_matrix_HYPSO-1_moon.npz already
    # shipped in hypso1_calibration's data/ unused before this fix.
    p = get_sensor_profile("hypso1")
    files = p.calibration_files("moon", coeff_type="moved")
    assert files["radiometric"] is not None
    assert "moon" in str(files["radiometric"])
    assert files["spectral"] is not None


# --- Capture-dimensions plan, Fix 3: imaging-mode schema (YAML-driven) ---

@pytest.mark.parametrize("key", ["hypso1", "hypso2"])
def test_capture_mode_crop_modes_well_formed(key):
    p = get_sensor_profile(key)
    assert isinstance(p.capture_mode_crop_modes, dict)
    for capture_type, modes in p.capture_mode_crop_modes.items():
        assert isinstance(capture_type, str) and capture_type
        assert isinstance(modes, dict)
        for coeff_type, crop_mode in modes.items():
            assert coeff_type in ("smile", "destriping")
            assert crop_mode in ("as_is", "crop_and_bin")


def test_hypso1_capture_mode_crop_modes_match_expected():
    # nominal/wide are pre-baked (as_is); moon/custom aren't declared at all,
    # meaning callers get the "crop_and_bin" default - see
    # calibration/pipeline.py's set_calibration_coeff_files.
    p = get_sensor_profile("hypso1")
    assert p.capture_mode_crop_modes["nominal"] == {"smile": "as_is", "destriping": "as_is"}
    assert p.capture_mode_crop_modes["wide"] == {"smile": "as_is", "destriping": "as_is"}
    assert p.capture_mode_crop_modes.get("moon", {}) == {}


def test_hypso1_get_calibration_files_matches_pre_yaml_golden_values():
    # Regression test for the match-case -> YAML-lookup refactor in
    # hypso1_calibration/main.py: confirms every capture_type still resolves
    # to the exact same filenames as the old hardcoded match statement -
    # golden values captured from that code before the refactor.
    from hypso1_calibration import get_hypso1_calibration_files
    from pathlib import Path

    golden = {
        "custom": {"radiometric": "h1_radiometric_calibration_matrix_full_moved.npz",
                   "smile": "spectral_calibration_matrix_HYPSO-1_full_v1.npz",
                   "destriping": None,
                   "spectral": "spectral_array_calibrated_poly_full.npz"},
        "nominal": {"radiometric": "h1_radiometric_calibration_matrix_full_moved.npz",
                    "smile": "smile_correction_matrix_HYPSO-1_nominal_v1.npz",
                    "destriping": "destriping_matrix_HYPSO-1_nominal_v1.npz",
                    "spectral": "spectral_array_calibrated_poly_full.npz"},
        "wide": {"radiometric": "h1_radiometric_calibration_matrix_full_moved.npz",
                 "smile": "smile_correction_matrix_HYPSO-1_wide_v1.npz",
                 "destriping": "destriping_matrix_HYPSO-1_wide_v1.npz",
                 "spectral": "spectral_array_calibrated_poly_full.npz"},
        "moon": {"radiometric": "radiometric_calibration_matrix_HYPSO-1_moon.npz",
                 "smile": "spectral_calibration_matrix_HYPSO-1_full_v1.npz",
                 "destriping": None,
                 "spectral": "spectral_array_calibrated_poly_full.npz"},
        "totally_unknown_mode": {"radiometric": None, "smile": None, "destriping": None, "spectral": None},
    }
    for capture_type, expected in golden.items():
        files = get_hypso1_calibration_files(capture_type, coeff_type="moved")
        for coeff_type, expected_name in expected.items():
            actual = files[coeff_type]
            actual_name = Path(actual).name if actual is not None else None
            assert actual_name == expected_name, (capture_type, coeff_type)


def _nominal_smile(y_stop):
    from hypso.calibration.correction import read_coeffs_from_file
    p = get_sensor_profile("hypso1")
    files = p.calibration_files("nominal", coeff_type="moved")
    crop_mode = p.capture_mode_crop_modes["nominal"]["smile"]
    assert crop_mode == "as_is"
    return read_coeffs_from_file(files["smile"], "smile", x_start=0, x_stop=1080, y_start=0, y_stop=y_stop,
                                 bin_factor=9, crop_mode=crop_mode)


def test_calibration_shape_mismatch_raises_not_silently_none():
    # A pre-baked ("as_is") coefficient file whose shape doesn't match the
    # capture's AOI/bin_factor must fail loudly, not silently skip the
    # correction. nominal's real row_count is 684 - deliberately wrong here.
    from hypso.calibration.correction import CalibrationShapeMismatchError
    with pytest.raises(CalibrationShapeMismatchError):
        _nominal_smile(y_stop=500)


def test_calibration_matching_shape_does_not_raise():
    assert _nominal_smile(y_stop=684).shape == (684, 120)
