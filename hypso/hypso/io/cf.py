"""CF (Climate and Forecast) attribute builders for every product level,
used by the one writer (hypso.io.product.write_product). Keeping them in
one place fixed several bugs of the earlier per-level writers, which each
set these attributes inline (wrong latitude units and ranges, a tuple
radiation_wavelength, missing Conventions).
"""
from typing import Optional

CONVENTIONS = "CF-1.10"


REFERENCES = "https://hypso.space"


def global_attrs(processing_level: str, title: str) -> dict:
    """Global attributes CF recommends and that were confirmed absent from
    every level's output before this refactor (no `Conventions` attribute
    anywhere meant CF-aware tooling had no signal to even attempt CF-aware
    parsing).

    "history" is deliberately NOT included here - unlike the other globals,
    CF wants it to be an append-only provenance trail (one line per
    processing step, not a value that gets overwritten each time a new
    level is written), which needs the *existing* value from whatever was
    already on the source product - see hypso.io.provenance.add_step,
    which builds it from satobj.metadata.global_attrs and appends this
    write's own entry before writing it."""
    return {
        "Conventions": CONVENTIONS,
        "title": title,
        "source": "HYPSO Hyperspectral Imager",
        "processing_level": processing_level,
        "references": REFERENCES,
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
    sideways to siblings, so this only works because the writer no longer
    nests products and geometry into separate groups (docs/architecture.rst,
    "File format")."""
    return {
        "coordinates": "latitude longitude",
        "grid_mapping": "crs_wgs84",
    }


def band_attrs(long_name: str, units: str, wavelength_nm: float, radiation_wavelength_nm: float,
                fwhm: float, wave_name: str, band_index: int, include_geolocation: bool,
                *, var_name: str, standard_name: Optional[str] = None) -> dict:
    """Per-band product variable attrs (e.g. for Lt_378 or rhot_378).

    wavelength vs. radiation_wavelength: kept as two distinct attributes,
    not collapsed into one - they carry genuinely different values in real
    files (wavelength is the nominal/rounded band-center label,
    radiation_wavelength is the precise as-calibrated value; e.g. 378.5 vs
    378.54673723 for the same band). radiation_wavelength is the CF-flavored
    name (maps to CF's "sensor_band_central_radiation_wavelength" concept,
    per Unidata's EC-netCDF-CF swath convention) - but standard_name is a
    property CF only defines for a *variable*, not for one attribute among
    several on a variable. The variable's own standard_name (its radiance or
    reflectance quantity) comes from `standard_name`, set per level in
    io/schema.py from the CF standard name table. `wave` (a third,
    truly-redundant duplicate of `wavelength`) is dropped.

    var_name is the variable's own name (e.g. "Lt_378"), written as
    `parameter` as the pre-refactor writers did; wave_name is the bare
    wavelength label (e.g. "378").
    """
    attrs = {
        "long_name": long_name,
        "units": units,
        "wavelength_units": "nanometers",
        "wavelength": float(wavelength_nm),
        "radiation_wavelength": float(radiation_wavelength_nm),
        "radiation_wavelength_unit": "nm",
        "fwhm": float(fwhm),
        "parameter": var_name,
        "wave_name": wave_name,
        "band": band_index,
    }
    if standard_name:
        attrs["standard_name"] = standard_name
    if include_geolocation:
        attrs.update(geolocation_ref_attrs())
    return attrs


# Standard names used in products, all checked against the CF standard name
# table (v95, cfconventions.org).
RRS_STANDARD_NAME = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
CHLA_STANDARD_NAME = "mass_concentration_of_chlorophyll_a_in_sea_water"
SPM_STANDARD_NAME = "mass_concentration_of_suspended_matter_in_sea_water"

# L2A product cube attributes by variable name (the product's quantity
# depends on the atmospheric-correction method, so it is not a per-level
# schema constant).
L2A_PRODUCTS = {
    "Rrs": {"units": "sr-1", "long_name": "Remote Sensing Reflectance", "standard_name": RRS_STANDARD_NAME},
    "rho_w": {"units": "1", "long_name": "Water-Leaving Reflectance (pi Rrs)"},
    "rhos": {"units": "1", "long_name": "Surface Reflectance"},
}

# AC tools write some standard_names that are not CF names; their CF
# equivalents. Others not in CF are kept only as source_standard_name.
TOOL_STANDARD_NAMES = {
    "mass_concentration_of_chlorophyll_a_in_water": CHLA_STANDARD_NAME,
    "concentration_of_suspended_matter_in_water": SPM_STANDARD_NAME,
}
CF_STANDARD_NAMES = {RRS_STANDARD_NAME, CHLA_STANDARD_NAME, SPM_STANDARD_NAME, "latitude", "longitude",
                     "toa_outgoing_radiance_per_unit_wavelength", "toa_bidirectional_reflectance"}


def tool_variable_attrs(attrs: dict) -> dict:
    """An AC tool's variable attributes with its standard_name made CF-valid:
    mapped to the CF equivalent, or moved to source_standard_name."""
    attrs = {k: v for k, v in attrs.items() if k not in ("_FillValue", "coordinates", "grid_mapping")}
    name = attrs.get("standard_name")
    if name is not None and name not in CF_STANDARD_NAMES:
        attrs["source_standard_name"] = attrs.pop("standard_name")
        if name in TOOL_STANDARD_NAMES:
            attrs["standard_name"] = TOOL_STANDARD_NAMES[name]
    return attrs
