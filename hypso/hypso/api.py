"""The public API: one immutable Product per capture and level
(REFACTOR_PROGRESS.md ×50).

    import hypso

    l1d = (hypso.open("…L1A.nc")
              .calibrate()                                   # abs / V3
              .georeference(latitudes=lats, longitudes=lons)
              .toa_reflectance())
    l1d.write()                                              # NASA-style name

Every processing method returns a new Product and leaves the one it was
called on unchanged. Underneath are the tree functions in hypso.processing
and the product tree from hypso.io.product (`Product.tree`), which stay
available for direct use.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.io.naming import product_name
from hypso.io.product import PRODUCT_DIMS, product_name as cube_name, read_product, write_product
from hypso.masks.flags import FLAGS_NAME, decode_flags

LEVELS = ("L1A", "L1B", "L1C", "L1D")

_GEOMETRY = ("latitude", "longitude", "sensor_zenith", "sensor_azimuth", "solar_zenith", "solar_azimuth",
             "relative_azimuth")


class Product:
    """One capture at one processing level. Create with hypso.open(path)
    or Product(tree)."""

    __slots__ = ("_tree",)

    def __init__(self, tree: xr.DataTree):
        level = tree.attrs.get("processing_level")
        if level not in LEVELS + ("L2A",):
            raise ValueError(f"not a HYPSO product tree (processing_level={level!r})")
        object.__setattr__(self, "_tree", tree)

    def __setattr__(self, name, value):
        raise AttributeError("Product is immutable; processing methods return a new Product")

    @classmethod
    def open(cls, path) -> "Product":
        """Read a product file of any level and any supported layout."""
        return cls(read_product(path))

    # -- contents -----------------------------------------------------------

    @property
    def tree(self) -> xr.DataTree:
        """The underlying product tree (a copy: changing it does not change
        this Product)."""
        return self._tree.copy()

    @property
    def level(self) -> str:
        return self._tree.attrs["processing_level"]

    @property
    def sat_id(self) -> str:
        return self._tree.attrs.get("sat_id")

    @property
    def target(self) -> str:
        return self._tree.attrs.get("location_description")

    @property
    def cube(self) -> xr.DataArray:
        """The product cube (dn, Lt, rhot, or the L2A variable), dims
        (lines, samples, bands), with wavelength/fwhm band coordinates."""
        return self._tree.dataset[cube_name(self._tree)]

    @property
    def wavelengths(self) -> np.ndarray:
        """Band centre wavelengths (nm), as calibrated."""
        return self._tree.dataset["radiation_wavelength"].values

    @property
    def geometry(self) -> xr.Dataset:
        """Latitude, longitude and angles (empty before georeferencing)."""
        ds = self._tree.dataset
        names = [n for n in ds.data_vars
                 if n in _GEOMETRY or (n.endswith("_direct") and n[:-len("_direct")] in _GEOMETRY)]
        return ds[names]

    @property
    def flags(self) -> dict:
        """{name: boolean DataArray} decoded from pixel_flags ({} if none)."""
        ds = self._tree.dataset
        return decode_flags(ds[FLAGS_NAME]) if FLAGS_NAME in ds else {}

    @property
    def provenance(self) -> list:
        """One dict per processing step, oldest first (from
        metadata/processing)."""
        if "metadata" not in self._tree.children or "processing" not in self._tree["metadata"].children:
            return []
        steps = [{"step": name, **dict(node.attrs)} for name, node in self._tree["metadata/processing"].children.items()]
        return sorted(steps, key=lambda s: s.get("date_utc", ""))

    def __repr__(self):
        cube = self.cube
        return (f"<hypso.Product {self.sat_id} {self.target} {self.level}: {cube.name} "
                f"{dict(cube.sizes)}>")

    # -- processing (each returns a new Product) ----------------------------

    def _require(self, level: str, method: str):
        if self.level != level:
            hint = f' or .to("{_NEXT[level]}")' if level in _NEXT else ""
            raise ValueError(f"{method}() needs an {level} product, this one is {self.level}{hint}")

    def calibrate(self, coeff_type: str = None, version=None, **kwargs) -> "Product":
        """L1A -> L1B. coeff_type: radiometric variant ("abs" default,
        "moved", "lab"); version: "V3" (default), "V2", "V1" or a
        per-correction dict - see docs/calibration.rst."""
        from hypso.processing import l1a_to_l1b
        self._require("L1A", "calibrate")
        return Product(l1a_to_l1b(self._tree, coeff_type=coeff_type, version=version, **kwargs))

    def georeference(self, latitudes=None, longitudes=None, direct: bool = True) -> "Product":
        """L1B -> L1C. latitudes/longitudes: indirect coordinates (arrays or
        paths to raw float32 files); omit to use direct georeferencing."""
        from hypso.processing import l1b_to_l1c
        self._require("L1B", "georeference")
        return Product(l1b_to_l1c(self._tree, latitudes=latitudes, longitudes=longitudes, direct=direct))

    def toa_reflectance(self, use_direct_georef: bool = False, ssi_source: str = "tsis",
                        use_unbinned: bool = True) -> "Product":
        """L1C -> L1D (TOA reflectance)."""
        from hypso.processing import l1c_to_l1d
        self._require("L1C", "toa_reflectance")
        return Product(l1c_to_l1d(self._tree, use_direct_georef=use_direct_georef, ssi_source=ssi_source,
                                  use_unbinned=use_unbinned))

    def correct_atmosphere(self, method: str, masked: bool = False, mask_flags=None, work_dir=None,
                           config=None, **options) -> "Product":
        """L1C/L1D -> L2A with `method` ("dps", "polymer"). masked=True gives
        the method NaN where pixel_flags are set (all, or `mask_flags`).
        External tools: `config` (AC tool configuration, see
        hypso.ac.config; default HYPSO_AC_CONFIG), `work_dir` (kept if
        given), and options overriding the tool's configuration."""
        from hypso.processing import correct_atmosphere
        return Product(correct_atmosphere(self._tree, method, masked=masked, mask_flags=mask_flags,
                                          work_dir=work_dir, config=config, **options))

    @property
    def atmospheric_correction(self):
        """The L2A method, e.g. "polymer" (None below L2A)."""
        return self._tree.attrs.get("atmospheric_correction")

    def to(self, level: str, calibrate: dict = None, georeference: dict = None,
           toa_reflectance: dict = None) -> "Product":
        """Run the steps from this level up to `level`, each with its own
        options, e.g. to("L1D", calibrate={"version": "V2"},
        georeference={"latitudes": ..., "longitudes": ...})."""
        level = level.upper()
        if level not in LEVELS or LEVELS.index(level) < LEVELS.index(self.level):
            raise ValueError(f"cannot go from {self.level} to {level}")
        options = {"calibrate": calibrate or {}, "georeference": georeference or {},
                   "toa_reflectance": toa_reflectance or {}}
        product = self
        while product.level != level:
            step = _STEPS[product.level]
            product = getattr(product, step)(**options[step])
        return product

    # -- output -------------------------------------------------------------

    def filename(self, label: str = None, ac: str = None) -> str:
        """The NASA-style file name for this product (hypso.io.naming)."""
        from hypso.processing.capture import acquisition_time
        from datetime import datetime
        time = datetime.fromisoformat(acquisition_time(self._tree.attrs))
        if ac is None and self.level == "L2A":
            ac = self.atmospheric_correction
        return product_name(self.sat_id, time, self.target, self.level, ac=ac, label=label).filename()

    def write(self, path=None, out_dir=None, label: str = None, datacube: bool = False) -> Path:
        """Write in the current format. Without `path`, the file is named by
        filename() in `out_dir` (default: the current directory)."""
        if path is None:
            path = Path(out_dir or ".") / self.filename(label=label)
        path = Path(path)
        write_product(self._tree, path, datacube=datacube)
        return path


_STEPS = {"L1A": "calibrate", "L1B": "georeference", "L1C": "toa_reflectance"}
_NEXT = {"L1A": "L1B", "L1B": "L1C", "L1C": "L1D"}


def open(path) -> Product:  # noqa: A001 - mirrors xarray.open_dataset / builtin-style entry point
    """Read a HYPSO product file (any level, any supported layout)."""
    return Product.open(path)
