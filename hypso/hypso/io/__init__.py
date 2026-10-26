"""Product file I/O: one reader and one writer for every level
(io/product.py), file names (io/naming.py), provenance (io/provenance.py),
and the CF attribute builders and per-level schema they use."""
from . import cf
from .schema import LevelSchema, get_schema, SCHEMAS_BY_LEVEL
from .product import read_product, write_product, convert_product, list_band_variables
from .naming import parse_product_name, product_name
