"""CF (Climate and Forecast) attribute builders shared by every NetCDF level
writer (see io/writer.py). Centralizing these fixes several confirmed bugs
that existed because each level's writer set these attributes inline,
independently, with no shared source of truth - see ARCHITECTURE_PROPOSAL.md
and REFACTOR_PROGRESS.md for how each one was found.
"""
from typing import Optional

CONVENTIONS = "CF-1.10"


def global_attrs(processing_level: str, title: str) -> dict:
    """Global attributes CF recommends and that were confirmed absent from
    every level's output before this refactor (no `Conventions` attribute
    anywhere meant CF-aware tooling had no signal to even attempt CF-aware
    parsing)."""
    return {
        "Conventions": CONVENTIONS,
        "title": title,
        "source": "HYPSO Hyperspectral Imager",
        "processing_level": processing_level,
        # "history"/"references" are appended-to/filled in by the caller
        # (write_level_nc), which knows the actual source file and package
        # version - not fixed values this module can supply.
    }


def latitude_attrs() -> dict:
    """Fixes a confirmed bug: the previous geometry writer set
    units="degrees" (not CF's required "degrees_north" for a variable to be
    recognized as a latitude coordinate) and valid_min/valid_max=[-180,180]
    (copy-pasted from longitude - latitude's real range is [-90,90])."""
    return {
        "long_name": "Latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
        "valid_min": -90.0,
        "valid_max": 90.0,
    }


def longitude_attrs() -> dict:
    return {
        "long_name": "Longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
        "valid_min": -180.0,
        "valid_max": 180.0,
    }


def zenith_angle_attrs(long_name: str, standard_name: Optional[str] = None) -> dict:
    """Zenith angles are physically [0, 180] (nadir to anti-nadir) - the
    previous writer gave every angle variable (zenith AND azimuth alike) the
    same blanket valid_min/valid_max=[-180,180]."""
    attrs = {"long_name": long_name, "units": "degree", "valid_min": 0.0, "valid_max": 180.0}
    if standard_name:
        attrs["standard_name"] = standard_name
    return attrs


def azimuth_angle_attrs(long_name: str, standard_name: Optional[str] = None) -> dict:
    """Azimuth angles are physically [0, 360) (or equivalently [-180,180]
    with a wrap) - kept as [-180,180] here since that's the range HYPSO's
    own angle computation (hypso.geometry) already produces, just no longer
    sharing zenith's range by copy-paste."""
    attrs = {"long_name": long_name, "units": "degree", "valid_min": -180.0, "valid_max": 180.0}
    if standard_name:
        attrs["standard_name"] = standard_name
    return attrs


def crs_wgs84_attrs() -> dict:
    """Unchanged from the previous writer - this one was already correctly
    CF-formed (grid_mapping_name="latitude_longitude" + the WGS84
    ellipsoid parameters)."""
    return {
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "geographic_crs_name": "WGS84",
    }


def geolocation_ref_attrs() -> dict:
    """coordinates/grid_mapping attrs for a per-band product variable, valid
    now that geometry variables (latitude/longitude/crs_wgs84) live in the
    same (root) group as the product variables referencing them - CF's
    group-relative name resolution only walks up to ancestor groups, never
    sideways to siblings, so this only works because io/writer.py no longer
    nests products/geometry into separate groups (see the NetCDF group
    structure decision in REFACTOR_PROGRESS.md)."""
    return {
        "coordinates": "latitude longitude",
        "grid_mapping": "crs_wgs84",
    }


def band_attrs(long_name: str, units: str, wavelength_nm: float, radiation_wavelength_nm: float,
                fwhm: float, wave_name: str, band_index: int, include_geolocation: bool) -> dict:
    """Per-band product variable attrs (e.g. for Lt_378 or rhot_378).

    wavelength vs. radiation_wavelength: kept as two distinct attributes,
    not collapsed into one - they carry genuinely different values in real
    files (wavelength is the nominal/rounded band-center label,
    radiation_wavelength is the precise as-calibrated value; e.g. 378.5 vs
    378.54673723 for the same band). radiation_wavelength is the CF-flavored
    name (maps to CF's "sensor_band_central_radiation_wavelength" concept,
    per Unidata's EC-netCDF-CF swath convention) - but standard_name is a
    property CF only defines for a *variable*, not for one attribute among
    several on a variable, so it's not set here: this Lt_378/rhot_378
    variable's own physical quantity is radiance/reflectance, not
    wavelength, and no standard_name is confidently known for those (see
    REFACTOR_PROGRESS.md's "Research: SNAP/CF wavelength convention"
    section) - left unset rather than guessed. `wave` (a third,
    truly-redundant duplicate of `wavelength`) is dropped.
    """
    attrs = {
        "long_name": long_name,
        "units": units,
        "wavelength_units": "nanometers",
        "wavelength": float(wavelength_nm),
        "radiation_wavelength": float(radiation_wavelength_nm),
        "radiation_wavelength_unit": "nm",
        "fwhm": float(fwhm),
        "parameter": f"{wave_name}",
        "wave_name": wave_name,
        "band": band_index,
    }
    if include_geolocation:
        attrs.update(geolocation_ref_attrs())
    return attrs
