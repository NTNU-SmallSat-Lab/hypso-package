"""Per-level NetCDF schema: what differs between product levels (product
variable name, units, CF standard name, title, whether there is geometry),
so one reader and writer (hypso.io.product) serve every level.

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
    """What differs between product levels: product variable name, units,
    CF standard name, title and whether the level has geometry."""
    processing_level: str          # e.g. "L1B" - written as the processing_level global attr
    product_prefix: str            # "Lt" (radiance) or "rhot" (reflectance)
    product_units: str             # CF units string, "" for dimensionless reflectance
    product_long_name: str         # e.g. "Top-of-Atmosphere Radiance"
    has_geometry: bool             # whether this level's capture has been georeferenced yet
    title: str                     # used for the global `title` attribute
    # CF standard name of the product variable, checked against the CF
    # standard name table (v95); None where no name fits.
    product_standard_name: str = None
    spatial_dims: tuple = ("lines", "samples")
    # Unused: gridded products (hypso.grid) are recognised by their (y, x)
    # dimensions (hypso.io.product.spatial_dims), not by a schema.


L1A_SCHEMA = LevelSchema(
    processing_level="L1A",
    product_prefix="dn",
    product_units="1",
    product_long_name="Raw Digital Numbers",
    has_geometry=False,
    title="HYPSO Level 1A Raw Digital Numbers",
)

L1B_SCHEMA = LevelSchema(
    processing_level="L1B",
    product_standard_name="toa_outgoing_radiance_per_unit_wavelength",
    product_prefix="Lt",
    product_units="W m-2 um-1 sr-1",
    product_long_name="Top-of-Atmosphere Radiance",
    has_geometry=False,  # confirmed: L1B is pre-georeferencing - see the bug this fixes above
    title="HYPSO Level 1B Top-of-Atmosphere Radiance",
)

L1C_SCHEMA = LevelSchema(
    processing_level="L1C",
    product_standard_name="toa_outgoing_radiance_per_unit_wavelength",
    product_prefix="Lt",
    product_units="W m-2 um-1 sr-1",
    product_long_name="Top-of-Atmosphere Radiance",
    has_geometry=True,
    title="HYPSO Level 1C Top-of-Atmosphere Radiance (georeferenced)",
)

L1D_SCHEMA = LevelSchema(
    processing_level="L1D",
    product_standard_name="toa_bidirectional_reflectance",
    product_prefix="rhot",
    product_units="1",
    product_long_name="Top-of-Atmosphere Reflectance",
    has_geometry=True,
    title="HYPSO Level 1D Top-of-Atmosphere Reflectance",
)

# L2A's product variable depends on the AC method (Rrs, rho_w, rhos); the
# writer takes its units and names from io.cf.L2A_PRODUCTS by variable name.
# The values here are the fallback for any other variable.
L2A_SCHEMA = LevelSchema(
    processing_level="L2",
    product_prefix="Rrs",
    product_units="1",
    product_long_name="Bottom-of-Atmosphere Reflectance",
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
    """The :class:`LevelSchema` of a level ("L1A" ... "L1D", "L2A"/"L2")."""
    try:
        return SCHEMAS_BY_LEVEL[level.upper()]
    except KeyError:
        raise KeyError(f"No LevelSchema for {level!r}. Known levels: {sorted(SCHEMAS_BY_LEVEL)}") from None
