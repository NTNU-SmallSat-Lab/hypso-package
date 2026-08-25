# write_l1b/l1c/l1d/l2a_nc_file now come from hypso.io.writer (the schema-driven
# writer replacing this package's old per-level writer files - l1b_nc_writer.py
# etc. are unused but kept in place, not deleted, in case anything imports their
# internals directly). Names/signatures are unchanged - confirmed imported
# directly by hypso-processing-pipeline. write_products_nc_file is NOT migrated
# (tied to the products/_products property, intentionally left untouched).
from .utils import set_or_create_attr
from hypso.io.writer import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file, write_l2a_nc_file
from .products_writer import write_products_nc_file
from .metadata_srf_group_writer import metadata_srf_group_writer


