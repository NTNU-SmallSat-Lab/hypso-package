"""hypso.io.naming - product file names (REFACTOR_PROGRESS.md x42)."""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from hypso.io.naming import parse_product_name, product_name

T = datetime(2025, 6, 12, 9, 58, 2)


@pytest.mark.parametrize("level, ac, label, expected", [
    ("L1A", None, None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1A.nc"),
    ("L1D", None, None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.nc"),
    ("L2A", "polymer", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.polymer.nc"),
    ("L1D", None, "adjusted", "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.adjusted.nc"),
    ("L2A", "acolite_l2r", "adjusted", "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.acolite_l2r.adjusted.nc"),
])
def test_compose_and_parse(level, ac, label, expected):
    name = product_name("HYPSO-2", T, "aeronetvenice", level, ac=ac, label=label)
    assert name.filename() == expected
    parsed = parse_product_name(expected)
    assert parsed == name
    assert parsed.capture_id == "aeronetvenice_2025-06-12T09-58-02Z"


def test_aware_time_is_converted_to_utc():
    aware = datetime(2025, 6, 12, 11, 58, 2, tzinfo=timezone.utc).astimezone()
    name = product_name("HYPSO-1", aware, "frohavet", "L1B")
    assert name.filename() == "HYPSO1_HSI.20250612T115802.frohavet.L1B.nc"


def test_sorts_by_satellite_then_time():
    names = [product_name(s, t, tgt, "L1D").filename() for s, t, tgt in (
        ("HYPSO-2", datetime(2025, 1, 2), "zeebrugge"),
        ("HYPSO-1", datetime(2025, 6, 1), "aeronetvenice"),
        ("HYPSO-2", datetime(2024, 12, 31), "annapolis"),
    )]
    assert [n.split(".")[0] + "." + n.split(".")[1][:8] for n in sorted(names)] == [
        "HYPSO1_HSI.20250601", "HYPSO2_HSI.20241231", "HYPSO2_HSI.20250102"]


@pytest.mark.parametrize("kwargs", [
    dict(level="L2A"),                      # L2A without AC
    dict(level="L1D", ac="polymer"),        # AC on a non-L2A level
    dict(level="L1D", label="bad.label"),   # '.' inside a field
    dict(level="L1D", target="venice lagoon"),
])
def test_invalid_names_rejected(kwargs):
    fields = dict(sat_id="HYPSO-2", time=T, target="aeronetvenice")
    fields.update(kwargs)
    with pytest.raises(ValueError):
        product_name(**fields).filename()


@pytest.mark.parametrize("legacy, level, ac, legacy_label", [
    ("aeronetvenice_2025-06-12T09-58-02Z-l1a.nc", "L1A", None, None),
    ("aeronetvenice_2025-06-12T09-58-02Z-moved_unmasked-l1d.nc", "L1D", None, "moved_unmasked"),
    ("aeronetvenice_2025-06-12T09-58-02Z-moved_unmasked-l2a-acolite_l2r.nc", "L2A", "acolite_l2r", "moved_unmasked"),
])
def test_legacy_names_still_parse(legacy, level, ac, legacy_label):
    parsed = parse_product_name(legacy)
    assert parsed.style == "legacy" and parsed.sat_id is None
    assert (parsed.level, parsed.ac, parsed.legacy_label) == (level, ac, legacy_label)
    assert parsed.capture_id == "aeronetvenice_2025-06-12T09-58-02Z"
    # A legacy name converts to the current style once sat_id is known.
    current = replace(parsed, sat_id="HYPSO-2").with_level(level, ac)
    assert current.filename().startswith("HYPSO2_HSI.20250612T095802.aeronetvenice.")


def test_unrelated_names_rejected():
    for name in ("HYPSO2_HSI.notatime.aeronetvenice.L1D.nc", "readme.txt", "HYPSO-2_moved_srf.nc"):
        with pytest.raises(ValueError):
            parse_product_name(name)


@pytest.mark.parametrize("level, ac, grid, label, expected", [
    ("L1D", None, "utm33n_876m", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.utm33n_876m.nc"),
    ("L2A", "polymer", "epsg4326_0p0079deg", "test",
     "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.polymer.epsg4326_0p0079deg.test.nc"),
    ("L1C", None, "custom_30x40m", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1C.custom_30x40m.nc"),
])
def test_grid_field(level, ac, grid, label, expected):
    name = product_name("HYPSO-2", T, "aeronetvenice", level, ac=ac, grid=grid, label=label)
    assert name.filename() == expected
    assert parse_product_name(expected) == name


def test_grid_field_errors():
    with pytest.raises(ValueError, match="invalid grid"):
        product_name("HYPSO-2", T, "aeronetvenice", "L1D", grid="utm33n").filename()
    with pytest.raises(ValueError, match="read back as a grid"):
        product_name("HYPSO-2", T, "aeronetvenice", "L1D", label="utm33n_30m").filename()
