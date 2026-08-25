"""Plan §Verification item 3 (part): hypso.io.schema per-level schemas and
hypso.io.cf attribute builders. Each cf assertion maps to one of the confirmed
pre-refactor bugs (see cf.py's docstrings), so a change that reintroduces one
fails loudly. No real data needed."""
import numpy as np
import pytest

from hypso.io import cf
from hypso.io.schema import get_schema, SCHEMAS_BY_LEVEL


# --- schema ---

def test_all_levels_present():
    assert set(SCHEMAS_BY_LEVEL) == {"L1A", "L1B", "L1C", "L1D", "L2A"}


@pytest.mark.parametrize("level,expected", [
    ("L1A", False), ("L1B", False),  # pre-georeferencing: structurally no geometry
    ("L1C", True), ("L1D", True), ("L2A", True),
])
def test_has_geometry_per_level(level, expected):
    # L1B has_geometry=False is the structural fix for the original
    # copy-pasted dangling /geometry reference bug in l1b_nc_writer.py.
    assert get_schema(level).has_geometry is expected


def test_get_schema_case_insensitive_and_unknown():
    assert get_schema("l1c") is get_schema("L1C")
    with pytest.raises(KeyError):
        get_schema("l3")


def test_spatial_dims_default():
    # The future-L3 seam: every current (swath) schema keeps the default.
    for schema in SCHEMAS_BY_LEVEL.values():
        assert schema.spatial_dims == ("lines", "samples")


def test_source_cube_attrs():
    # L1C deliberately shares L1B's cube (georeferencing adds no new cube).
    assert get_schema("L1C").source_cube_attr == "l1b_cube"
    assert get_schema("L1B").source_cube_attr == "l1b_cube"
    assert get_schema("L1D").source_cube_attr == "l1d_cube"


# --- cf builders: one assertion per confirmed pre-refactor bug ---

def test_latitude_attrs_fixed():
    # Bug: units="degrees" and valid range [-180,180] copy-pasted from longitude.
    attrs = cf.latitude_attrs()
    assert attrs["units"] == "degrees_north"
    assert attrs["valid_min"] == -90.0 and attrs["valid_max"] == 90.0
    assert attrs["standard_name"] == "latitude"


def test_longitude_attrs():
    attrs = cf.longitude_attrs()
    assert attrs["units"] == "degrees_east"
    assert attrs["valid_min"] == -180.0 and attrs["valid_max"] == 180.0
    assert attrs["standard_name"] == "longitude"


def test_zenith_and_azimuth_ranges_differentiated():
    # Bug: every angle variable used to get the same blanket [-180,180].
    zen = cf.zenith_angle_attrs("Sensor Zenith Angle")
    azi = cf.azimuth_angle_attrs("Sensor Azimuth Angle")
    assert (zen["valid_min"], zen["valid_max"]) == (0.0, 180.0)
    assert (azi["valid_min"], azi["valid_max"]) == (-180.0, 180.0)


def test_global_attrs_have_conventions():
    # Bug: no Conventions attribute existed anywhere pre-refactor.
    attrs = cf.global_attrs("L1C", "some title")
    assert attrs["Conventions"].startswith("CF-")
    assert attrs["processing_level"] == "L1C"
    assert attrs["title"] == "some title"


def test_band_attrs_wavelengths():
    # Bugs: radiation_wavelength was a 1-element tuple (trailing comma), and a
    # third redundant `wave` attribute duplicated `wavelength`.
    attrs = cf.band_attrs(long_name="Top-of-Atmosphere Radiance",
                          units="W m-2 um-1 sr-1",
                          wavelength_nm=378.5,
                          radiation_wavelength_nm=378.54673723,
                          fwhm=3.33, wave_name="Lt_378", band_index=0,
                          include_geolocation=True)
    assert np.ndim(attrs["radiation_wavelength"]) == 0
    assert isinstance(attrs["radiation_wavelength"], float)
    assert "wave" not in attrs
    # wavelength and radiation_wavelength are genuinely different values
    # (nominal label vs precise as-calibrated) - both must survive.
    assert attrs["wavelength"] == 378.5
    assert attrs["radiation_wavelength"] == 378.54673723
    assert attrs["band"] == 0


def test_band_attrs_geolocation_toggle():
    # include_geolocation=False is the L1B path: no coordinates/grid_mapping
    # reference may be present (nothing to dangle).
    with_geo = cf.band_attrs("x", "1", 500.0, 500.0, 3.3, "rhot_500", 1,
                             include_geolocation=True)
    without_geo = cf.band_attrs("x", "1", 500.0, 500.0, 3.3, "rhot_500", 1,
                                include_geolocation=False)
    assert with_geo["coordinates"] == "latitude longitude"
    assert with_geo["grid_mapping"] == "crs_wgs84"
    assert "coordinates" not in without_geo
    assert "grid_mapping" not in without_geo


def test_crs_wgs84_attrs():
    attrs = cf.crs_wgs84_attrs()
    assert attrs["grid_mapping_name"] == "latitude_longitude"
