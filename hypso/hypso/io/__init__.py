"""Schema-driven NetCDF I/O, replacing the previous one-writer/one-loader-file-per-level
design (see io/schema.py's module docstring and REFACTOR_PROGRESS.md for why)."""
from .schema import LevelSchema, get_schema, SCHEMAS_BY_LEVEL
from . import cf
from .writer import write_level_nc, write_l2a_nc
