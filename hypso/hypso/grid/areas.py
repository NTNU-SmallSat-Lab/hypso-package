"""pyresample geometry for hypso products: the swath a product was imaged
in, the grid a regridded product is on, and new grids to regrid onto.

Grids from grid_area() have their extent snapped to whole multiples of the
pixel size, so two captures regridded to the same CRS and resolution share
pixel boundaries and can be compared pixel by pixel.
"""
import numpy as np
import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition, SwathDefinition

from hypso.io.product import GRID_DIMS, GRID_MAPPING, spatial_dims

# Metres per degree of latitude (WGS84 mean), to compare a geographic grid's
# pixel size with ground distances.
METRES_PER_DEGREE = 111_320.0


def geolocation(tree: xr.DataTree, direct: bool = False) -> tuple:
    """(latitude, longitude) arrays of a product; ValueError if it has none."""
    ds = tree.dataset
    suffix = "_direct" if direct else ""
    if f"latitude{suffix}" not in ds:
        kind = "direct " if direct else ""
        raise ValueError(f"{tree.attrs.get('processing_level')} product has no {kind}geolocation "
                         f"(georeference it first)")
    return ds[f"latitude{suffix}"].values, ds[f"longitude{suffix}"].values


def swath_definition(tree: xr.DataTree, direct: bool = False) -> SwathDefinition:
    """SwathDefinition of a product's per-pixel geolocation (direct or not)."""
    latitudes, longitudes = geolocation(tree, direct)
    attrs = {"sensor": tree.attrs.get("sat_id"), "name": "swath"}
    return SwathDefinition(lons=xr.DataArray(longitudes, dims=("y", "x"), attrs=attrs),
                           lats=xr.DataArray(latitudes, dims=("y", "x"), attrs=attrs))


def source_definition(tree: xr.DataTree):
    """What to resample from: a gridded product's AreaDefinition, else its
    swath (as numpy arrays, which pyresample's kd-tree needs)."""
    if spatial_dims(tree.dataset) == GRID_DIMS:
        return area_definition(tree)
    latitudes, longitudes = geolocation(tree)
    return SwathDefinition(lons=longitudes, lats=latitudes)


def resolution(tree: xr.DataTree) -> float:
    """Ground resolution (m): a gridded product's pixel size (converted from
    degrees for a geographic grid), else the larger of the mean along- and
    across-track GSD."""
    if tree.attrs.get("grid"):
        crs = pyproj.CRS.from_cf(dict(tree.dataset[GRID_MAPPING].attrs))
        size = float(tree.attrs["grid_resolution"])
        return size * METRES_PER_DEGREE if crs.is_geographic else size
    from hypso.geometry import compute_gsd, compute_resolution
    latitudes, longitudes = geolocation(tree)
    along, across = compute_gsd(frame_count=latitudes.shape[0], image_height=latitudes.shape[1],
                                latitudes=latitudes, longitudes=longitudes)
    return compute_resolution(along_track_gsd=along, across_track_gsd=across)


def utm_crs(latitudes, longitudes) -> pyproj.CRS:
    """The WGS84 UTM CRS of the zone containing the scene centre.

    EPSG 326xx (north) or 327xx (south); standard 6-degree zones, without the
    Norway and Svalbard exceptions.

    Parameters
    ----------
    latitudes, longitudes : array_like

    Returns
    -------
    pyproj.CRS
    """
    lat = float(np.nanmean(latitudes))
    lon = float(np.nanmean(longitudes))
    zone = min(int((lon + 180) // 6) + 1, 60)
    return pyproj.CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def grid_area(crs, resolution: float, latitudes, longitudes, area_id: str = "hypso_grid") -> AreaDefinition:
    """A grid covering the given coordinates, snapped to the pixel size.

    Parameters
    ----------
    crs : str, int or pyproj.CRS
        Anything pyproj accepts, e.g. ``"EPSG:32633"``.
    resolution : float
        Square pixel size in the CRS's units.
    latitudes, longitudes : array_like
        Coordinates to cover (NaN ignored).
    area_id : str, default "hypso_grid"

    Returns
    -------
    pyresample.geometry.AreaDefinition
        Extent snapped to whole multiples of ``resolution``, so grids of two
        captures in the same CRS and resolution share pixel boundaries.
    """
    crs = pyproj.CRS.from_user_input(crs)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float))
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError(f"no coordinates can be projected to {crs.name}")
    x, y = x[finite], y[finite]
    extent = (np.floor(x.min() / resolution) * resolution, np.floor(y.min() / resolution) * resolution,
              np.ceil(x.max() / resolution) * resolution, np.ceil(y.max() / resolution) * resolution)
    width = max(int(round((extent[2] - extent[0]) / resolution)), 1)
    height = max(int(round((extent[3] - extent[1]) / resolution)), 1)
    return AreaDefinition(area_id, f"{crs.name}, {resolution:g} pixels", area_id, crs, width, height, extent)


def area_definition(tree: xr.DataTree) -> AreaDefinition:
    """AreaDefinition of a gridded product, from its x/y coordinates
    (pixel centres) and `crs` grid-mapping variable."""
    ds = tree.dataset
    if spatial_dims(ds) != GRID_DIMS:
        raise ValueError("not a gridded product (regrid it first)")
    return area_from_grid(dict(ds[GRID_MAPPING].attrs), ds["x"].values, ds["y"].values,
                          float(tree.attrs["grid_resolution_x"]), float(tree.attrs["grid_resolution_y"]),
                          tree.attrs.get("grid", "hypso_grid"))


def area_from_grid(crs_attrs: dict, x, y, size_x: float, size_y: float, name: str = "hypso_grid") -> AreaDefinition:
    """AreaDefinition from a CF grid-mapping variable's attributes, the x/y
    pixel-centre coordinates and the pixel size - what a gridded product
    file holds (also used by the Satpy readers)."""
    crs = pyproj.CRS.from_cf(dict(crs_attrs))
    x, y = np.asarray(x), np.asarray(y)
    extent = (x.min() - size_x / 2, y.min() - size_y / 2, x.max() + size_x / 2, y.max() + size_y / 2)
    return AreaDefinition(name, name, name, crs, x.size, y.size, extent)


def grid_name(crs, size_x: float, size_y: float) -> str:
    """File-name token for a grid (io/naming.py GRID_TOKEN): utm33n_880m,
    epsg4326_0p01deg, custom_30m; "AxB" for non-square pixels."""
    crs = pyproj.CRS.from_user_input(crs)
    zone = crs.utm_zone
    epsg = crs.to_epsg()
    prefix = f"utm{zone.lower()}" if zone else (f"epsg{epsg}" if epsg else "custom")

    def number(value):
        return f"{round(value, 6):g}".replace(".", "p")

    size = number(size_x) if np.isclose(size_x, size_y) else f"{number(size_x)}x{number(size_y)}"
    return f"{prefix}_{size}{'deg' if crs.is_geographic else 'm'}"


def coordinate_attrs(crs) -> tuple:
    """CF attributes for the x and y coordinate variables of a grid in `crs`."""
    axes = pyproj.CRS.from_user_input(crs).cs_to_cf()
    x = next(a for a in axes if a.get("axis") == "X" or a.get("standard_name") in ("projection_x_coordinate",
                                                                                     "longitude"))
    y = next(a for a in axes if a.get("axis") == "Y" or a.get("standard_name") in ("projection_y_coordinate",
                                                                                     "latitude"))
    return dict(x), dict(y)
