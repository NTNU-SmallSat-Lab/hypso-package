"""Tests for hypso.georeferencing.geo_state (GeoAngles/TrackGeometry) - the
consolidation of HypsoCapture's 17 flat georeferencing-angle/track
attributes into two small frozen dataclasses, held as
self.angles/self.angles_direct/self.track/self.track_direct.

Regression coverage for two real hasattr-trap bugs found and fixed as part
of this consolidation: HypsoCapture._generate_l1d_cube_impl and
capture_types._spawn_l1d both had `hasattr(satobj, 'solar_zenith_angles_direct')`
checks that would have silently and permanently broken (always True) once
that name became a property - fixed to `angles_direct.solar_zenith is not
None`, verified here by exercising use_direct_georef=True on a capture that
has only run indirect (standard) georeferencing.
"""
import dataclasses
import warnings

import numpy as np
import pytest

from conftest import requires_real_capture
from hypso import capture_types
from hypso.georeferencing import GeoAngles, TrackGeometry

pytestmark = requires_real_capture


def test_geo_angles_and_track_geometry_are_frozen():
    angles = GeoAngles(sensor_zenith=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        angles.sensor_zenith = 2

    track = TrackGeometry(bbox=(0, 0, 1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        track.bbox = (1, 1, 2, 2)


def test_satobj_angles_populated_after_georeferencing(satobj):
    assert isinstance(satobj.angles, GeoAngles)
    assert isinstance(satobj.angles_direct, GeoAngles)
    assert isinstance(satobj.track, TrackGeometry)
    assert isinstance(satobj.track_direct, TrackGeometry)

    # satobj fixture only ever runs indirect (standard) georeferencing.
    assert satobj.angles.solar_zenith is not None
    assert satobj.angles.sensor_zenith is not None
    assert satobj.angles.relative_azimuth is not None
    assert satobj.angles_direct.solar_zenith is None
    assert satobj.track.bbox is not None


def test_compat_properties_resolve_through_angles(satobj):
    assert satobj.sat_zenith_angles is satobj.angles.sensor_zenith
    assert satobj.sat_azimuth_angles is satobj.angles.sensor_azimuth
    assert satobj.solar_zenith_angles is satobj.angles.solar_zenith
    assert satobj.solar_azimuth_angles is satobj.angles.solar_azimuth
    assert satobj.relative_azimuth_angles is satobj.angles.relative_azimuth
    assert satobj.sat_zenith_angles_direct is satobj.angles_direct.sensor_zenith
    assert satobj.solar_zenith_angles_direct is satobj.angles_direct.solar_zenith


def test_compat_properties_are_read_only(satobj):
    with pytest.raises(AttributeError):
        satobj.solar_zenith_angles = np.zeros(satobj.spatial_dimensions)


def test_use_direct_georef_falls_back_when_direct_not_run_deprecated_path(satobj):
    # Regression test for the fixed HypsoCapture._generate_l1d_cube_impl bug.
    assert satobj.angles_direct.solar_zenith is None

    # Isolated same-type copy so this doesn't mutate the shared session-scoped
    # satobj fixture other tests rely on - see test_masking.py's isolated
    # fixture for the same pattern.
    isolated = capture_types.spawn_as(satobj, type(satobj))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        isolated.generate_l1d_cube(use_direct_georef=True)

    assert isolated.l1d_cube is not None
    assert not np.isnan(isolated.l1d_cube.to_numpy()).all()


def test_use_direct_georef_falls_back_when_direct_not_run_spawn_path(satobj):
    # Regression test for the fixed capture_types._spawn_l1d bug - this one
    # would previously raise AttributeError (no such property existed on
    # L1BCapture at all), not just silently misbehave.
    assert satobj.angles_direct.solar_zenith is None

    l1b = satobj.to_l1b(coeff_type="moved")
    l1d = l1b.to_l1d(use_direct_georef=True)

    assert l1d.cube is not None
    assert not np.isnan(l1d.cube.to_numpy()).all()
