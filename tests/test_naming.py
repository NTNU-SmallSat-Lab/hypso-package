"""hypso.io.naming - product file names (REFACTOR_PROGRESS.md x42)."""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from hypso.io.naming import parse_product_name, product_name

T = datetime(2025, 6, 12, 9, 58, 2)


@pytest.mark.parametrize("level, quantity, ac, label, expected", [
    ("L1A", None, None, None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1A.nc"),
    ("L1D", None, None, None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.nc"),
    ("L2A", "RRS", "polymer", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.RRS.polymer.nc"),
    ("L1D", None, None, "adjusted", "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.adjusted.nc"),
    ("L2A", "RHOS", "acolite", "adjusted", "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.RHOS.acolite.adjusted.nc"),
])
def test_compose_and_parse(level, quantity, ac, label, expected):
    name = product_name("HYPSO-2", T, "aeronetvenice", level, ac=ac, label=label, quantity=quantity)
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
    dict(level="L2A", ac="polymer"),        # L2A without quantity
    dict(level="L2A", ac="polymer", quantity="CHL"),   # unknown quantity
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
    current = replace(parsed, sat_id="HYPSO-2").with_level(level, ac, "RHOS" if ac else None)
    assert current.filename().startswith("HYPSO2_HSI.20250612T095802.aeronetvenice.")


def test_unrelated_names_rejected():
    for name in ("HYPSO2_HSI.notatime.aeronetvenice.L1D.nc", "readme.txt", "HYPSO-2_moved_srf.nc"):
        with pytest.raises(ValueError):
            parse_product_name(name)


@pytest.mark.parametrize("level, ac, grid, label, expected", [
    ("L1D", None, "utm33n_876m", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.utm33n_876m.nc"),
    ("L2A", "polymer", "epsg4326_0p0079deg", "test",
     "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.RRS.polymer.epsg4326_0p0079deg.test.nc"),
    ("L1C", None, "custom_30x40m", None, "HYPSO2_HSI.20250612T095802.aeronetvenice.L1C.custom_30x40m.nc"),
])
def test_grid_field(level, ac, grid, label, expected):
    name = product_name("HYPSO-2", T, "aeronetvenice", level, ac=ac, grid=grid, label=label,
                        quantity="RRS" if ac else None)
    assert name.filename() == expected
    assert parse_product_name(expected) == name


def test_grid_field_errors():
    with pytest.raises(ValueError, match="invalid grid"):
        product_name("HYPSO-2", T, "aeronetvenice", "L1D", grid="utm33n").filename()
    with pytest.raises(ValueError, match="read back as a grid"):
        product_name("HYPSO-2", T, "aeronetvenice", "L1D", label="utm33n_30m").filename()


def test_instrument_field():
    # "HSI" is the instrument (Hyperspectral Imager), like NASA's PACE_OCI
    from hypso.io.naming import INSTRUMENTS, parse_platform_token, platform_token
    assert INSTRUMENTS["HSI"] == "Hyperspectral Imager"
    assert platform_token("HYPSO-2") == "HYPSO2_HSI"
    assert parse_platform_token("HYPSO1_HSI") == ("HYPSO-1", "HSI")
    name = parse_product_name("HYPSO2_HSI.20250612T095802.aeronetvenice.L1D.nc")
    assert (name.sat_id, name.instrument) == ("HYPSO-2", "HSI")
    with pytest.raises(ValueError, match="unknown instrument"):
        product_name("HYPSO-2", T, "aeronetvenice", "L1D", instrument="XYZ").filename()
    with pytest.raises(ValueError, match="not a HYPSO product file name"):
        parse_product_name("HYPSO2_XYZ.20250612T095802.aeronetvenice.L1D.nc")   # unknown instrument


def test_l2a_quantity_field():
    # L2A names carry the quantity (RRS, RHOS) before the method (2026-10-28)
    from hypso.io.naming import name_method, quantity_token
    assert quantity_token("Rrs") == "RRS" and quantity_token("rhos") == "RHOS" and quantity_token("rho_w") == "RHOW"
    assert name_method("acolite_l2w") == "acolite" and name_method("polymer") == "polymer"
    # the product variable is accepted in place of the token
    assert product_name("HYPSO-2", T, "aeronetvenice", "L2A", ac="machi", quantity="rhos").filename() == \
        "HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.RHOS.machi.nc"
    with pytest.raises(ValueError, match="no quantity token"):
        quantity_token("chla")


@pytest.mark.parametrize("old, ac, grid, label", [
    ("HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.polymer.nc", "polymer", None, None),
    ("HYPSO2_HSI.20250612T095802.aeronetvenice.L2A.acolite_l2r.utm33n_880m.test.nc", "acolite_l2r", "utm33n_880m", "test"),
])
def test_names_before_quantity_field_still_parse(old, ac, grid, label):
    parsed = parse_product_name(old)
    assert (parsed.level, parsed.quantity, parsed.ac, parsed.grid, parsed.label) == ("L2A", None, ac, grid, label)


def test_l2b_names():
    # L2B: product group and source in the quantity/method fields (TODO 4i, x87)
    from hypso.io.naming import L2B_GROUPS
    assert {"CHL", "IOP", "TSM", "KD"} <= set(L2B_GROUPS)
    name = product_name("HYPSO-2", T, "aeronetvenice", "L2B", ac="oc4", quantity="CHL", grid="utm33n_880m")
    assert name.filename() == "HYPSO2_HSI.20250612T095802.aeronetvenice.L2B.CHL.oc4.utm33n_880m.nc"
    assert parse_product_name(name.filename()) == name
    for bad in (dict(quantity="RRS", ac="oc4"), dict(ac="oc4"), dict(quantity="CHL")):
        with pytest.raises(ValueError):
            product_name("HYPSO-2", T, "aeronetvenice", "L2B", **bad).filename()
    with pytest.raises(ValueError):
        parse_product_name("HYPSO2_HSI.20250612T095802.aeronetvenice.L2B.oc4.nc")     # no group
