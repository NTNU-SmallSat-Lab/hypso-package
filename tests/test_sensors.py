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
    # every field HypsoCapture reads off the profile must be populated
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


class _FakeCapture:
    VERBOSE = False

    def __init__(self, sensor_profile, frame_count, image_height):
        self.sensor_profile = sensor_profile
        self.frame_count = frame_count
        self.image_height = image_height


@pytest.mark.parametrize("frame_count,image_height,expected", [
    (956, 684, "nominal"),
    (106, 684, "moon"),
    (500, 1092, "wide"),
    (12345, 12345, "custom"),
])
def test_check_capture_type_classifies_from_sensor_profile(frame_count, image_height, expected):
    from hypso.io.dispatch import check_capture_type

    satobj = _FakeCapture(get_sensor_profile("hypso1"), frame_count, image_height)
    check_capture_type(satobj)
    assert satobj.capture_type == expected


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
