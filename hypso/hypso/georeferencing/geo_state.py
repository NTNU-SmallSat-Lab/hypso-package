"""Consolidates 9 of HypsoCapture's 17 flat georeferencing-angle/track
attributes (sat_zenith_angles, sat_zenith_angles_direct, ...,
along_track_gsd, along_track_gsd_direct, ...) into two small dataclasses,
each held in two instances - self.angles/self.angles_direct and
self.track/self.track_direct.

framepose (the 17th name) is deliberately NOT included - it has no direct
variant (run_direct_georeferencing reuses the same satobj.framepose rather
than computing a separate one, see georeferencing/geo.py), so folding it
into either dataclass would misrepresent that asymmetry. It stays a plain
top-level attribute, untouched by this consolidation.

Frozen rather than mutable (unlike hypso.io.metadata's CaptureMetadata):
hypso.capture_types.spawn_as does a shallow __dict__ update when spawning a
new capture object, so a mutable GeoAngles/TrackGeometry instance would
become the SAME object shared between the spawned object and its source -
an in-place field mutation on one would silently corrupt the other. Frozen
makes that mistake raise dataclasses.FrozenInstanceError immediately
instead of silently corrupting state, and costs nothing: every writer
(georeferencing/geo.py,
io/dispatch.py) already rebinds a whole new instance rather than mutating
fields in place.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GeoAngles:
    sensor_zenith: Optional[np.ndarray] = None
    sensor_azimuth: Optional[np.ndarray] = None
    solar_zenith: Optional[np.ndarray] = None
    solar_azimuth: Optional[np.ndarray] = None
    relative_azimuth: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TrackGeometry:
    bbox: Optional[tuple] = None
    along_track_gsd: Optional[np.ndarray] = None
    across_track_gsd: Optional[np.ndarray] = None
