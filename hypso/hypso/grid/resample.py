"""Resampling between pyresample geometries: one Resampler per source and
target pair (neighbour lookup done once, applied to every variable), an
Averager for coarser grids (mean of the source pixels in each cell), and
resample_product - a product's contents on any target, as a bare Dataset.
"""
import numpy as np
import xarray as xr
from pyresample import kd_tree
from pyresample.bilinear import NumpyBilinearResampler

from hypso.io.product import has_cube, product_name
from hypso.masks.flags import FLAGS_NAME, decode_flags

from .areas import METRES_PER_DEGREE, geolocation, resolution as product_resolution, source_definition

METHODS = ("nearest", "bilinear", "average")


class Resampler:
    """Nearest-neighbour and (optionally) bilinear resampling from `source`
    to `target`; radius_of_influence in metres."""

    def __init__(self, source, target, radius_of_influence: float, bilinear: bool):
        self.shape = target.shape
        self._source_shape = source.shape
        self._neighbours = kd_tree.get_neighbour_info(source, target, radius_of_influence, neighbours=1)[:3]
        self._bilinear = None
        if bilinear:
            self._bilinear = NumpyBilinearResampler(source, target, radius_of_influence)
            self._bilinear.get_bil_info()

    def nearest(self, data: np.ndarray, fill_value=np.nan) -> np.ndarray:
        """(rows, cols[, bands]); integer data keeps its dtype (fill 0)."""
        if data.dtype.kind in "iub":
            fill_value = 0
        return kd_tree.get_sample_from_neighbour_info("nn", self.shape, data, *self._neighbours,
                                                      fill_value=fill_value)

    def bilinear(self, data: np.ndarray) -> np.ndarray:
        if self._bilinear is None:
            raise ValueError("Resampler was made without bilinear=True")

        def one(band):
            # pyresample's bilinear solver divides by zero for degenerate
            # neighbour quads and discards those results itself.
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(self._bilinear.get_sample_from_bil_info(band, fill_value=np.nan,
                                                                          output_shape=self.shape))
        if data.ndim == 2:
            return one(data)
        return np.stack([one(data[:, :, i]) for i in range(data.shape[2])], axis=-1)

    def __call__(self, data: np.ndarray, method: str) -> np.ndarray:
        """Float data with `method`, output in the input's float dtype."""
        out = self.bilinear(data) if method == "bilinear" else self.nearest(data)
        return out.astype(data.dtype if data.dtype.kind == "f" else np.float32, copy=False)

    def coverage(self, method: str) -> np.ndarray:
        """True where the target has data from the source with `method`."""
        ones = np.ones(self._source_shape, dtype=np.float32)
        return np.isfinite(self(ones, method))


class Averager:
    """Cell averages on a grid: each target cell gets the mean of the
    source pixels whose centres fall in it (NaN ignored per band) - an
    unweighted approximation of area-weighted averaging, close when source
    pixels are much smaller than the cells. Only for targets coarser than
    the source; finer ones leave empty cells inside the swath."""

    def __init__(self, latitudes, longitudes, target):
        if not hasattr(target, "get_array_indices_from_lonlat"):
            raise ValueError("method='average' needs a grid (AreaDefinition) target, not a swath")
        self.shape = target.shape
        cols, rows = target.get_array_indices_from_lonlat(np.ravel(longitudes), np.ravel(latitudes))
        inside = ~np.ma.getmaskarray(cols) & ~np.ma.getmaskarray(rows)
        self._inside = inside
        self._cells = (np.ma.getdata(rows)[inside] * self.shape[1] + np.ma.getdata(cols)[inside]).astype(np.int64)
        self._size = self.shape[0] * self.shape[1]
        #: Source pixels per cell.
        self.count = np.bincount(self._cells, minlength=self._size).reshape(self.shape)

    def _pixels(self, data):
        return np.asarray(data).reshape(self._inside.size, -1)[self._inside]

    def mean(self, data: np.ndarray) -> np.ndarray:
        """Float (rows, cols[, bands]) -> cell means in the input's float dtype."""
        pixels = self._pixels(data).astype(np.float64)
        out = np.full((self._size, pixels.shape[1]), np.nan)
        for band in range(pixels.shape[1]):
            values = pixels[:, band]
            valid = np.isfinite(values)
            sums = np.bincount(self._cells[valid], weights=values[valid], minlength=self._size)
            counts = np.bincount(self._cells[valid], minlength=self._size)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, band] = np.where(counts > 0, sums / counts, np.nan)
        dtype = data.dtype if data.dtype.kind == "f" else np.float32
        return out.reshape(self.shape + np.shape(data)[2:]).astype(dtype)

    def bitwise_or(self, data: np.ndarray) -> np.ndarray:
        """Integer flags (rows, cols) -> per cell, every flag set in any of
        its pixels (0 for empty cells)."""
        out = np.zeros(self._size, dtype=data.dtype)
        np.bitwise_or.at(out, self._cells, self._pixels(data)[:, 0])
        return out.reshape(self.shape)


def default_radius(source_metres: float, target_metres: float = 0.0) -> float:
    """Radius of influence (m): 1.5 times the coarser of source and target
    pixel size, so every target pixel inside the swath finds neighbours."""
    return 1.5 * max(source_metres, target_metres)


def target_resolution_metres(target) -> float:
    """Pixel size (m) of an AreaDefinition; 0 for a swath target."""
    if not hasattr(target, "pixel_size_x"):
        return 0.0
    size = max(abs(target.pixel_size_x), abs(target.pixel_size_y))
    return size * METRES_PER_DEGREE if target.crs.is_geographic else size


def is_flag_variable(name: str, attrs) -> bool:
    """Bit-flag variables (CF flag_masks, or AC tools' *flags* variables)."""
    return "flag_masks" in attrs or "flag" in name.lower()


class GridOperators:
    """How each kind of variable goes from a product onto `target` with
    `method`, so resample_product and regrid treat variables alike:

    - floats(): the cube and float variables - nearest, bilinear, or cell
      average; azimuths (by name) through their sine and cosine unless
      nearest, so 0/360 does not average to 180;
    - integers(): nearest neighbour, except bit flags with "average", which
      are OR-ed over the cell;
    - covered(): target pixels that received data.
    """

    def __init__(self, tree: xr.DataTree, target, method: str, radius_of_influence: float = None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        source_metres, target_metres = product_resolution(tree), target_resolution_metres(target)
        self.radius = radius_of_influence or default_radius(source_metres, target_metres)
        self._tree, self._target = tree, target
        self._resampler = self.averager = None
        if method == "average":
            if not hasattr(target, "get_array_indices_from_lonlat"):
                raise ValueError("method='average' needs a grid (AreaDefinition) target, not a swath")
            if target_metres < 0.99 * source_metres:   # 1%: default grids round the resolution
                raise ValueError(f"method='average' needs a grid at least as coarse as the product "
                                 f"({source_metres:.0f} m); this one is {target_metres:.0f} m - use nearest")
            self.averager = Averager(*geolocation(tree), target)
        else:
            self._resampler = Resampler(source_definition(tree), target, self.radius,
                                        bilinear=method == "bilinear")

    @property
    def resampler(self) -> Resampler:
        # With "average", made only if a non-flag integer variable needs nearest.
        if self._resampler is None:
            self._resampler = Resampler(source_definition(self._tree), self._target, self.radius, bilinear=False)
        return self._resampler

    def _floats(self, values):
        return self.averager.mean(values) if self.averager else self.resampler(values, self.method)

    def floats(self, values: np.ndarray, name: str = "") -> np.ndarray:
        if "azimuth" in name and self.method != "nearest":
            radians = np.deg2rad(values)
            degrees = np.rad2deg(np.arctan2(self._floats(np.sin(radians)), self._floats(np.cos(radians))))
            return np.mod(degrees, 360.0).astype(values.dtype)
        return self._floats(values)

    def integers(self, values: np.ndarray, flags: bool) -> np.ndarray:
        if self.averager is not None and flags:
            return self.averager.bitwise_or(values)
        return self.resampler.nearest(values)

    def covered(self) -> np.ndarray:
        return self.averager.count > 0 if self.averager is not None else self.resampler.coverage(self.method)

    def parameters(self) -> dict:
        """What was done, for provenance."""
        out = {"method": self.method,
               "flags_method": "bitwise OR over the cell" if self.averager is not None else "nearest",
               "azimuth_method": "nearest" if self.method == "nearest" else f"sin/cos {self.method}"}
        if self.averager is None:
            out["radius_of_influence_m"] = float(self.radius)
        return out


def resample_product(tree: xr.DataTree, target, method: str = "nearest",
                     radius_of_influence: float = None) -> xr.Dataset:
    """A product's contents on any pyresample target, as a bare Dataset.

    Parameters
    ----------
    tree : xarray.DataTree
        A georeferenced product tree.
    target : pyresample.geometry.AreaDefinition or SwathDefinition
    method : {"nearest", "bilinear", "average"}, default "nearest"
        Per kind of variable as in :class:`GridOperators`; ``"average"`` only
        onto grids coarser than the product.
    radius_of_influence : float, optional
        In m; default :func:`default_radius`.

    Returns
    -------
    xarray.Dataset
        The cube ``(y, x, band)`` with the source band wavelengths, each
        derived variable ``(y, x)``, each flag as boolean ``flag_<name>``,
        and the target's latitude and longitude.
    """
    from hypso.io.product import _extra_variables
    ds = tree.dataset
    ops = GridOperators(tree, target, method, radius_of_influence)
    cube = ds[product_name(tree)] if has_cube(tree) else None      # L2B: 2D variables only

    out = {cube.name: (("y", "x", "band"), ops.floats(cube.values))} if cube is not None else {}
    for name in _extra_variables(ds, [cube.name] if cube is not None else []):
        var = ds[name]
        values = (ops.integers(var.values, is_flag_variable(name, var.attrs)) if var.dtype.kind in "iub"
                  else ops.floats(var.values, name))
        out[name] = (("y", "x"), values, dict(var.attrs))
    if FLAGS_NAME in ds:
        flags = xr.DataArray(ops.integers(ds[FLAGS_NAME].values, flags=True), dims=("y", "x"),
                             attrs=dict(ds[FLAGS_NAME].attrs))
        for name, flag in decode_flags(flags).items():
            out[f"flag_{name}"] = (("y", "x"), flag.values)

    grid_lon, grid_lat = target.get_lonlats()
    coords = {"latitude": (("y", "x"), grid_lat), "longitude": (("y", "x"), grid_lon)}
    if cube is not None:
        coords["wavelength"] = ("band", cube["radiation_wavelength"].values)
    result = xr.Dataset(out, coords=coords)
    result.attrs.update(sat_id=tree.attrs.get("sat_id"), target=tree.attrs.get("location_description"),
                        processing_level=tree.attrs.get("processing_level"), resample_method=method,
                        radius_of_influence_m=float(ops.radius))
    return result
