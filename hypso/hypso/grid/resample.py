"""Resampling between pyresample geometries: one Resampler per source and
target pair (neighbour lookup done once, applied to every variable), and
resample_product - a product's contents on any target, as a bare Dataset.
"""
import numpy as np
import xarray as xr
from pyresample import kd_tree
from pyresample.bilinear import NumpyBilinearResampler

from hypso.io.product import product_name
from hypso.masks.flags import FLAGS_NAME, decode_flags

from .areas import METRES_PER_DEGREE, resolution as product_resolution, source_definition

METHODS = ("nearest", "bilinear")


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


def resample_product(tree: xr.DataTree, target, method: str = "nearest",
                     radius_of_influence: float = None) -> xr.Dataset:
    """xr.Dataset on `target` (pyresample AreaDefinition or SwathDefinition):
    the product cube (y, x, band, with the source band wavelengths), each
    derived variable (y, x), each flag as boolean flag_<name>, and the
    target's latitude/longitude. method applies to the cube and float
    variables; flags and integer variables are always nearest neighbour.
    radius_of_influence (m) defaults to default_radius()."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    from hypso.io.product import _extra_variables
    ds = tree.dataset
    source = source_definition(tree)
    radius = radius_of_influence or default_radius(product_resolution(tree), target_resolution_metres(target))
    cube = ds[product_name(tree)]
    resampler = Resampler(source, target, radius, bilinear=method == "bilinear")

    out = {cube.name: (("y", "x", "band"), resampler(cube.values, method))}
    for name in _extra_variables(ds, [cube.name]):
        values = ds[name].values
        out[name] = (("y", "x"), resampler.nearest(values) if values.dtype.kind in "iub"
                     else resampler(values, method), dict(ds[name].attrs))
    if FLAGS_NAME in ds:
        flags = xr.DataArray(resampler.nearest(ds[FLAGS_NAME].values), dims=("y", "x"),
                             attrs=dict(ds[FLAGS_NAME].attrs))
        for name, flag in decode_flags(flags).items():
            out[f"flag_{name}"] = (("y", "x"), flag.values)

    grid_lon, grid_lat = target.get_lonlats()
    result = xr.Dataset(out, coords={"latitude": (("y", "x"), grid_lat), "longitude": (("y", "x"), grid_lon),
                                     "wavelength": ("band", cube["radiation_wavelength"].values)})
    result.attrs.update(sat_id=tree.attrs.get("sat_id"), target=tree.attrs.get("location_description"),
                        processing_level=tree.attrs.get("processing_level"), resample_method=method,
                        radius_of_influence_m=float(radius))
    return result
