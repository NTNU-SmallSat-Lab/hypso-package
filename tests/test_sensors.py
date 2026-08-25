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
    # every field HypsoBase reads off the profile must be populated
    for field in ("key", "sat_id", "sensor", "platform"):
        assert isinstance(getattr(p, field), str) and getattr(p, field), field
    for field in ("fwhm", "srf_wl", "srf_fwhm"):
        arr = np.asarray(getattr(p, field))
        assert arr.size > 0, field
        assert np.all(np.isfinite(arr)), field
    assert callable(p.calibration_files)
    # srf_wl/srf_fwhm are paired reference arrays for nearest-neighbor lookup
    assert np.asarray(p.srf_wl).shape == np.asarray(p.srf_fwhm).shape


def test_profiles_are_frozen():
    p = get_sensor_profile("hypso1")
    with pytest.raises(Exception):
        p.sat_id = "changed"
