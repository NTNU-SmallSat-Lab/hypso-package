"""Per-level NetCDF schema. Replaces the previous design of one ~480-line
writer file (and one loader file) per product level - confirmed near-
duplicates of each other differing mostly by find/replace of the level name
and product variable name (see REFACTOR_PROGRESS.md) - with one shared
writer/reader (io/writer.py, io/reader.py) parametrized by a LevelSchema.

The single most load-bearing thing this schema captures: whether a level
has geometry at all. L1A/L1B are pre-georeferencing and genuinely have no
geometry data - encoding that as `has_geometry=False` here means the writer
structurally cannot write a dangling `coordinates`/`grid_mapping` reference
for those levels (the confirmed bug this whole refactor started from), the
same way every other per-level writer file could each independently forget
to check.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelSchema:
    processing_level: str          # e.g. "L1B" - written as the processing_level global attr
    product_prefix: str            # "Lt" (radiance) or "rhot" (reflectance)
    product_units: str             # CF units string, "" for dimensionless reflectance
    product_long_name: str         # e.g. "Top-of-Atmosphere Radiance"
    source_cube_attr: str          # satobj attribute holding the source cube, e.g. "l1b_cube"
    has_geometry: bool             # whether this level's capture has been georeferenced yet
    title: str                     # used for the global `title` attribute
    spatial_dims: tuple = ("lines", "samples")
    # Every level implemented so far is swath-shaped: (lines, samples) tied to the
    # capture's along-track/across-track geometry. This field exists so a future
    # gridded Level 3 product (regular lat/lon grid, mosaicked/composited across
    # captures - not implemented yet, no generation code exists for it) can reuse
    # io/writer.py's product/geometry-writing code by declaring spatial_dims=
    # ("lat", "lon") instead, rather than needing a parallel writer. Not otherwise
    # exercised today - every current schema keeps the default.


L1A_SCHEMA = LevelSchema(
    processing_level="L1A",
    product_prefix="dn",
    product_units="1",
    product_long_name="Raw Digital Numbers",
    source_cube_attr="l1a_cube",
    has_geometry=False,
    title="HYPSO Level 1A Raw Digital Numbers",
)

L1B_SCHEMA = LevelSchema(
    processing_level="L1B",
    product_prefix="Lt",
    product_units="W m-2 um-1 sr-1",
    product_long_name="Top-of-Atmosphere Radiance",
    source_cube_attr="l1b_cube",
    has_geometry=False,  # confirmed: L1B is pre-georeferencing - see the bug this fixes above
    title="HYPSO Level 1B Top-of-Atmosphere Radiance",
)

L1C_SCHEMA = LevelSchema(
    processing_level="L1C",
    product_prefix="Lt",
    product_units="W m-2 um-1 sr-1",
    product_long_name="Top-of-Atmosphere Radiance",
    source_cube_attr="l1b_cube",  # L1c is L1b data + georeferencing, see HypsoBase.l1c_cube's docstring
    has_geometry=True,
    title="HYPSO Level 1C Top-of-Atmosphere Radiance (georeferenced)",
)

L1D_SCHEMA = LevelSchema(
    processing_level="L1D",
    product_prefix="rhot",
    product_units="1",
    product_long_name="Top-of-Atmosphere Reflectance",
    source_cube_attr="l1d_cube",
    has_geometry=True,
    title="HYPSO Level 1D Top-of-Atmosphere Reflectance",
)

# L2A varies by which AC method produced it: the actual product variable name
# comes from satobj.l2a_cube[correction].attrs['l2_variable_name'] (set by each
# AC adapter - "chla"/"Rrs"/etc, not a fixed name), not a schema constant -
# io/writer.py's write_l2a_nc reads it per-call and overrides product_prefix on
# this schema (dataclasses.replace) before writing. product_prefix/product_units
# here are only the fallback used if a correction's cube is missing that attr
# (matches write/l2a_nc_writer.py's original except-fallback to "Rrs").
L2A_SCHEMA = LevelSchema(
    processing_level="L2",
    product_prefix="Rrs",
    product_units="1",
    product_long_name="Bottom-of-Atmosphere Reflectance",
    source_cube_attr="l2a_cube",
    has_geometry=True,
    title="HYPSO Level 2 Bottom-of-Atmosphere Reflectance",
)

SCHEMAS_BY_LEVEL = {
    "L1A": L1A_SCHEMA,
    "L1B": L1B_SCHEMA,
    "L1C": L1C_SCHEMA,
    "L1D": L1D_SCHEMA,
    "L2A": L2A_SCHEMA,
}


def get_schema(level: str) -> LevelSchema:
    try:
        return SCHEMAS_BY_LEVEL[level.upper()]
    except KeyError:
        raise KeyError(f"No LevelSchema for {level!r}. Known levels: {sorted(SCHEMAS_BY_LEVEL)}") from None
