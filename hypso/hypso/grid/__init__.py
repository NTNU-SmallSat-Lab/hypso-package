"""Map grids for hypso products: swath and grid definitions (pyresample),
resampling onto any target as a bare Dataset, and regridding a product onto
a map grid as a product of the same level (Product.regrid)."""
from .areas import (area_definition, grid_area, grid_name, resolution, source_definition, swath_definition,
                    utm_crs)
from .regrid import OUTSIDE_FLAG, regrid
from .resample import Resampler, resample_product
