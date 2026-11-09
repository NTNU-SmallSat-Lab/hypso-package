"""Footprints, shapefiles, RGB images and maps of HYPSO products.

- :func:`footprint`: a product's ground footprint as a shapely polygon in
  WGS84 longitude/latitude: the outline of the whole swath, or of the pixels
  in a mask (e.g. those with data).
- :func:`footprint_attributes`: the capture's metadata for a footprint
  (capture, satellite, time, level, georeferencing, calibration, AC method).
- :func:`write_footprints`: one or several footprints as an ESRI Shapefile
  (.shp/.shx/.dbf/.prj/.cpg, EPSG:4326) or GeoJSON, one feature per product.
- :func:`rgb_image` and :func:`save_rgb`: an RGB rendering (bands nearest 640,
  550 and 460 nm; percentile stretch, per channel for L1, one gain for L2A;
  display gamma) in sensor geometry, as an array or a PNG (in the HYPSO
  servers' orientation by default); from any product with a cube (L1B-L1D
  radiance or rhot, L2A Rrs or rhos).
- :func:`plot_map`: a product variable (a band of the cube, a 2D variable such
  as ``chlor_a``) or an RGB rendering on a map grid (:meth:`hypso.Product.regrid`),
  with the footprint outline, coastlines, graticule, scale bar, north arrow
  and colour bar.
- :func:`plot_overview`: the footprints of several captures on one map.

Optional dependencies: shapely (footprints), pyshp (shapefiles), matplotlib
(figures), cartopy and its Natural Earth data (coastlines); pillow writes the
PNGs.
"""
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pyproj

from hypso.rgb.maps import RGB_WAVELENGTHS, decorate_map

#: Shapefile attribute fields (names at most 10 characters): (name, type, size, decimals).
FOOTPRINT_FIELDS = (
    ("capture", "C", 64, 0),      # <target>_<YYYY-MM-DDTHH-MM-SSZ>, as on the HYPSO servers
    ("satellite", "C", 16, 0),
    ("target", "C", 48, 0),
    ("start_utc", "C", 32, 0),    # acquisition start, ISO 8601
    ("level", "C", 8, 0),
    ("product", "C", 24, 0),      # e.g. L1D, L2A.RRS.ocsmart, L2B.BGC.ocsmart
    ("georef", "C", 12, 0),       # indirect (GCP-corrected), direct, unknown
    ("calib", "C", 80, 0),        # radiometric calibration file
    ("ac_method", "C", 24, 0),
    ("lines", "N", 6, 0),
    ("samples", "N", 6, 0),
    ("area_km2", "N", 12, 1),
    ("centre_lat", "N", 10, 5),
    ("centre_lon", "N", 10, 5),
    ("source", "C", 128, 0),      # file name, when read from a file
)

# 2D variables shown on a logarithmic colour scale by default (concentrations and IOPs).
_LOG_VARIABLES = re.compile(r"^(chl|chlor|tsm|spm|tur|t_|a_|aph|adg|apg|bbp|bb_|bp_|kd|ndci)", re.IGNORECASE)


def _product(source, **open_kwargs):
    """A hypso.Product from a Product or a path (opened with open_kwargs)."""
    from hypso.api import Product
    if isinstance(source, Product):
        return source
    import hypso
    return hypso.open(source, **open_kwargs)


def _lonlat(product):
    geometry = product.geometry
    if "latitude" not in geometry:
        raise ValueError(f"{product.level} product has no latitude/longitude: georeference it first")
    lat = geometry["latitude"].values.astype(float)
    lon = geometry["longitude"].values.astype(float)
    if np.nanmax(lon) - np.nanmin(lon) > 180:          # across the antimeridian: 0-360
        lon = np.mod(lon, 360.0)
    return lat, lon


def georeferencing(product) -> str:
    """How the product's latitude/longitude were made: "indirect" (ground
    processing's GCP-corrected coordinates), "direct" (from attitude and
    position), or "unknown" (no record, e.g. a product from older software)."""
    product = _product(product)
    for step in reversed(product.provenance):
        if "georeferencing" in step:
            return str(step["georeferencing"])
    return "unknown"


def footprint(product, mask=None, step: int = 10, block: int = 8):
    """The product's ground footprint, in WGS84 longitude/latitude.

    Parameters
    ----------
    product : hypso.Product or path
        Any georeferenced product (L1C and later), in sensor geometry.
    mask : array_like of bool or "data", optional
        Outline only these pixels (lines x samples) instead of the whole
        swath; ``"data"`` for pixels with data (any finite band of the cube,
        else any finite 2D variable). Pixels are taken in blocks of
        ``block`` x ``block``: a block is in when any of its pixels is.
    step : int, default 10
        Swath outline: a vertex every ``step`` pixels along the edges (and at
        the corners).
    block : int, default 8
        Block size for ``mask``.

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        Longitudes in -180..180, or 0..360 for a footprint across the
        antimeridian.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    product = _product(product)
    if product.grid is not None:
        raise ValueError("footprint needs a product in sensor geometry, not a gridded one")
    lat, lon = _lonlat(product)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if mask is None:
        rows = np.flatnonzero(finite.any(axis=1))
        cols = np.flatnonzero(finite.any(axis=0))
        if rows.size == 0:
            raise ValueError("the product has no finite latitude/longitude")
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]

        def along(start, stop):
            return np.unique(np.r_[np.arange(start, stop, step), stop])
        rr, cc = along(r0, r1), along(c0, c1)
        ring = ([(r0, c) for c in cc] + [(r, c1) for r in rr[1:]]
                + [(r1, c) for c in cc[::-1][1:]] + [(r, c0) for r in rr[::-1][1:]])
        points = [(lon[r, c], lat[r, c]) for r, c in ring if finite[r, c]]
        polygon = Polygon(points)
        return polygon if polygon.is_valid else make_valid(polygon)

    if isinstance(mask, str):
        if mask != "data":
            raise ValueError(f"mask must be a boolean array or 'data', not {mask!r}")
        mask = _data_mask(product)
    mask = np.asarray(mask, bool) & finite
    if mask.shape != lat.shape:
        raise ValueError(f"mask has shape {mask.shape}, expected {lat.shape}")
    if not mask.any():
        raise ValueError("the mask selects no pixels")
    n_lines, n_samples = lat.shape
    quads = []
    for r in range(0, n_lines, block):
        r1 = min(r + block, n_lines - 1)
        for c in range(0, n_samples, block):
            if not mask[r:r + block, c:c + block].any():
                continue
            c1 = min(c + block, n_samples - 1)
            corners = [(r, c), (r, c1), (r1, c1), (r1, c)]
            if not all(finite[i, j] for i, j in corners):
                continue
            quad = Polygon([(lon[i, j], lat[i, j]) for i, j in corners])
            quads.append(quad if quad.is_valid else quad.buffer(0))
    if not quads:
        raise ValueError("no block of the mask has finite corner coordinates")
    # a tiny buffer closes the slivers between neighbouring quads (float rounding)
    eps = 1e-7
    return make_valid(unary_union([q.buffer(eps) for q in quads]).buffer(-eps))


def _data_mask(product) -> np.ndarray:
    from hypso.io.product import has_cube
    if has_cube(product.tree):
        return np.isfinite(product.cube.values).any(axis=-1)
    variables = product.variables
    names = [n for n in variables.data_vars if np.issubdtype(variables[n].dtype, np.floating)]
    if not names:
        raise ValueError("the product has neither a cube nor float variables to take 'data' from")
    return np.logical_or.reduce([np.isfinite(variables[n].values) for n in names])


def _product_label(product) -> str:
    """e.g. L1D, L2A.RRS.ocsmart, L2B.BGC.ocsmart (the file name's product fields)."""
    name = product.filename()
    stem = name[:-3] if name.endswith(".nc") else name
    parts = stem.split(".")
    i = next((k for k, p in enumerate(parts) if p == product.level), None)
    return ".".join(parts[i:]) if i is not None else product.level


def footprint_attributes(product, source=None) -> dict:
    """The capture's metadata for a footprint feature (:data:`FOOTPRINT_FIELDS`).

    Parameters
    ----------
    product : hypso.Product
    source : str or path, optional
        The file the product was read from (its name is recorded).
    """
    from hypso.processing.capture import acquisition_time
    attrs = product.tree.attrs
    start = datetime.fromisoformat(acquisition_time(attrs))
    lat, lon = _lonlat(product)
    level = product.level
    ac = attrs.get("atmospheric_correction") if level in ("L2A", "L2B") else None
    if level == "L2B":
        ac = attrs.get("l2b_source", ac)
    return {
        "capture": f"{product.target}_{start.strftime('%Y-%m-%dT%H-%M-%SZ')}",
        "satellite": product.sat_id,
        "target": product.target,
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": level,
        "product": _product_label(product),
        "georef": georeferencing(product),
        "calib": str(attrs.get("radiometric_file", "")),
        "ac_method": str(ac or ""),
        "lines": int(lat.shape[0]),
        "samples": int(lat.shape[1]),
        "area_km2": 0.0,                                  # filled in from the polygon
        "centre_lat": float(np.nanmean(lat)),
        "centre_lon": float(np.nanmean(lon)),
        "source": Path(source).name if source is not None else "",
    }


def _area_km2(geometry) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    return abs(geod.geometry_area_perimeter(geometry)[0]) / 1e6


def write_footprints(sources, path, mask=None, step: int = 10, block: int = 8) -> Path:
    """Write footprints to an ESRI Shapefile or GeoJSON, one feature per product.

    Parameters
    ----------
    sources : hypso.Product, path, or a list of them
        Products in sensor geometry; paths are opened (all their bands are
        read, so prefer the smallest file of a capture, e.g. the L2B.BGC or
        the L1D opened with a few bands).
    path : str or path
        ``*.shp`` (writes .shp, .shx, .dbf, .prj and .cpg; needs pyshp) or
        ``*.geojson``/``*.json``.
    mask, step, block
        As in :func:`footprint`; ``mask`` must be ``None`` or ``"data"`` here.

    Returns
    -------
    pathlib.Path
        The .shp or GeoJSON file.
    """
    from shapely.geometry import mapping
    from shapely.geometry.polygon import orient
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    if mask not in (None, "data"):
        raise ValueError("write_footprints takes mask=None or 'data'; for an array use footprint()")
    features = []
    for source in sources:
        product = _product(source)
        geometry = footprint(product, mask=mask, step=step, block=block)
        record = footprint_attributes(product, source=None if hasattr(source, "tree") else source)
        record["area_km2"] = round(_area_km2(geometry), 1)
        features.append((geometry, record))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".geojson", ".json"):
        collection = {"type": "FeatureCollection",
                      "features": [{"type": "Feature", "geometry": mapping(g), "properties": r} for g, r in features]}
        path.write_text(json.dumps(collection, indent=1))
        return path
    if path.suffix.lower() != ".shp":
        raise ValueError(f"{path.name}: write a .shp or .geojson file")
    try:
        import shapefile
    except ImportError as error:
        raise ImportError("writing shapefiles needs pyshp (pip install pyshp)") from error
    with shapefile.Writer(str(path.with_suffix("")), shapeType=shapefile.POLYGON, encoding="utf-8") as writer:
        for name, kind, size, decimals in FOOTPRINT_FIELDS:
            writer.field(name, kind, size=size, decimal=decimals)
        for geometry, record in features:
            parts = []
            for polygon in getattr(geometry, "geoms", [geometry]):
                if polygon.geom_type != "Polygon":
                    continue
                polygon = orient(polygon, sign=-1.0)      # shapefile: outer rings clockwise, holes anticlockwise
                parts.append([list(c) for c in polygon.exterior.coords])
                parts.extend([list(c) for c in ring.coords] for ring in polygon.interiors)
            writer.poly(parts)
            writer.record(**{name: record[name] for name, *_ in FOOTPRINT_FIELDS})
    path.with_suffix(".prj").write_text(pyproj.CRS.from_epsg(4326).to_wkt(pyproj.enums.WktVersion.WKT1_ESRI))
    path.with_suffix(".cpg").write_text("UTF-8")
    return path


# -- RGB images in sensor geometry ----------------------------------------------------------------

def _stretch(rgb, stretch=(2, 98), per_channel: bool = True, gamma: float = 2.2):
    """Scale an (..., 3) image to 0-1 (NaN kept).

    per_channel: each channel from its own low to high percentile (TOA
    reflectance: one gain for all channels renders land pink); otherwise one
    gain for all channels from 0 to the high percentile of all values (keeps
    the colours of surface reflectance). Then ``** (1 / gamma)``.
    """
    rgb = np.asarray(rgb, float)
    out = np.full(rgb.shape, np.nan)
    if per_channel:
        for k in range(3):
            channel = rgb[..., k]
            finite = np.isfinite(channel)
            if not finite.any():
                continue
            low, high = np.percentile(channel[finite], stretch)
            out[..., k] = np.clip((channel - low) / max(high - low, 1e-12), 0, 1)
    else:
        finite = np.isfinite(rgb)
        if finite.any():
            high = np.percentile(rgb[finite], stretch[1])
            out = np.clip(rgb / max(high, 1e-12), 0, 1)
    return out ** (1.0 / gamma) if gamma else out


def _rgb_settings(product, stretch, per_channel, gamma) -> dict:
    """Defaults by level. L1 (TOA): per-channel (2, 98) percentile stretch
    and gamma 2.2 (one gain for all channels renders TOA land pink). L2A
    (surface reflectance): one gain for all channels to the 99.5th
    percentile, linear, which keeps the true colour of water (gamma or a
    per-channel stretch wash it out to flat cyan or distort it)."""
    surface = product.level in ("L2A", "L2B")
    return {"stretch": stretch if stretch is not None else ((0, 99.5) if surface else (2, 98)),
            "per_channel": per_channel if per_channel is not None else not surface,
            "gamma": gamma if gamma is not None else (1.0 if surface else 2.2)}


def rgb_image(product, stretch=None, wavelengths=RGB_WAVELENGTHS, per_channel: bool = None,
              gamma: float = None) -> np.ndarray:
    """An RGB rendering of a product's cube, in sensor geometry.

    Parameters
    ----------
    product : hypso.Product or path
        A product with a cube (L1B-L1D, L2A); a path is opened with only the
        three bands.
    stretch : tuple of float, optional
        Percentiles over the pixels with data that map to 0 and 1 (with one
        gain for all channels, only the upper one is used: 0 maps to 0).
        Default (2, 98) for L1, (0, 99.5) for L2A.
    wavelengths : tuple of float, default (640, 550, 460)
        The bands (nearest) for red, green and blue.
    per_channel : bool, optional
        Stretch each channel on its own (default for L1 products: one gain
        for all channels renders TOA land pink) or use one gain for all
        channels from 0 (default for L2A surface reflectance, whose colours
        a per-channel stretch distorts).
    gamma : float, optional
        Display gamma (``image ** (1 / gamma)``): default 2.2 for L1, 1
        (linear) for L2A.

    Returns
    -------
    numpy.ndarray
        (lines, samples, 3) in 0-1, NaN where the product has no data (e.g.
        masked pixels of an L2A).
    """
    from hypso.io.product import has_cube
    product = _product(product, wavelengths=list(wavelengths))
    if not has_cube(product.tree):
        raise ValueError(f"a {product.level} product without a cube has no bands for an RGB image")
    bands = [int(np.argmin(np.abs(product.wavelengths - w))) for w in wavelengths]
    rgb = product.cube.values[..., bands].astype(float)
    return _stretch(rgb, **_rgb_settings(product, stretch, per_channel, gamma))


def save_rgb(product, path, stretch=None, wavelengths=RGB_WAVELENGTHS, per_channel: bool = None,
             gamma: float = None, orientation: str = "server", bin_samples: int = 1) -> Path:
    """Write :func:`rgb_image` as a PNG (transparent where there is no data).

    Parameters
    ----------
    product, stretch, wavelengths, per_channel, gamma
        As in :func:`rgb_image`.
    path : str or path
    orientation : {"server", "product"}, default "server"
        "server": as the HYPSO servers' quicklooks (lines across, samples
        up); "product": the arrays as stored (lines down, samples across).
    bin_samples : int, default 1
        Average this many neighbouring samples (the servers' ``-bin3``
        images use 3, for pixels closer to square on the ground).
    """
    from PIL import Image
    image = rgb_image(product, stretch=stretch, wavelengths=wavelengths, per_channel=per_channel, gamma=gamma)
    if bin_samples > 1:
        n = image.shape[1] // bin_samples * bin_samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN bins
            image = np.nanmean(image[:, :n].reshape(image.shape[0], -1, bin_samples, 3), axis=2)
    alpha = np.isfinite(image).all(axis=-1)
    rgba = np.dstack([np.nan_to_num(image) * 255, alpha * 255]).round().astype(np.uint8)
    if orientation == "server":
        rgba = rgba.transpose(1, 0, 2)[::-1]
    elif orientation != "product":
        raise ValueError(f"orientation must be 'server' or 'product', not {orientation!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


# -- maps -----------------------------------------------------------------------------------------

def _cube_units(product) -> tuple:
    """(units, long_name) of the product's cube, as written to files (the
    cube read back from per-band variables carries no attributes)."""
    from hypso.io import cf
    from hypso.io.product import _level_key
    from hypso.io.schema import get_schema
    name = product.cube.name
    if product.level == "L2A" and name in cf.L2A_PRODUCTS:
        return cf.L2A_PRODUCTS[name]["units"], cf.L2A_PRODUCTS[name]["long_name"]
    schema = get_schema(_level_key(product.level))
    return schema.product_units, schema.product_long_name


def _subset(product, variable, wavelength):
    """The product reduced to what the map needs (regridding all bands of a
    cube or every IOP variable would be slow and large), and what is plotted:
    "rgb", "band" or "variable"."""
    import xarray as xr
    from hypso.api import Product
    from hypso.io.product import has_cube
    from hypso.masks.flags import FLAGS_NAME
    cube = has_cube(product.tree)
    if variable is None:                              # RGB
        if not cube:
            raise ValueError(f"a {product.level} product without a cube has no bands for an RGB map")
        product = product.select_bands(wavelengths=list(RGB_WAVELENGTHS))
        keep, mode = set(), "rgb"
    elif cube and variable == product.cube.name:
        if wavelength is None:
            raise ValueError(f"give the wavelength of {variable!r} to map")
        product = product.select_bands(wavelengths=[wavelength])
        keep, mode = set(), "band"
    else:
        if variable not in product.variables:
            raise KeyError(f"{variable!r} is not a variable of the product; it has "
                           f"{sorted(product.variables.data_vars)}")
        if cube:
            product = product.select_bands(wavelengths=[float(product.wavelengths[0])])   # one band, any subset
        keep, mode = {variable}, "variable"
    tree = product.tree.copy()
    root = tree.to_dataset(inherit=False)
    extras = set(product.variables.data_vars) - keep
    root = root.drop_vars([n for n in extras if n != FLAGS_NAME and n in root])
    tree.dataset = root
    return Product(tree), mode


def plot_map(source, variable: str = None, wavelength: float = None, ax=None, resolution: float = None,
             cmap: str = None, log: bool = None, vmin: float = None, vmax: float = None,
             percentiles=(2, 98), stretch=None, per_channel: bool = None, gamma: float = None,
             outline: bool = True, water: bool = True, coastlines: bool = True,
             graticule: bool = True, scale_bar: bool = True, north_arrow: bool = True, title: str = None,
             colorbar: bool = True):
    """Plot a product variable, a band, or an RGB rendering on a map.

    The product is regridded (:meth:`hypso.Product.regrid`, nearest
    neighbour, default the UTM zone of the scene at the product's own
    resolution) after reducing it to what is plotted.

    Parameters
    ----------
    source : hypso.Product or path
        A product in sensor geometry or already gridded.
    variable : str, optional
        A 2D variable (e.g. ``"chlor_a_oci"``) or the cube's name (``"Rrs"``,
        ``"rhos"``, ``"rhot"``) with ``wavelength``; default an RGB rendering
        of the cube (bands nearest 640, 550 and 460 nm).
    wavelength : float, optional
        For a band of the cube (nearest).
    ax : matplotlib.axes.Axes, optional
    resolution : float, optional
        Grid pixel size in m; default the product's.
    cmap : str, optional
        Default "viridis" ("turbo" is avoided: not perceptually uniform).
    log : bool, optional
        Logarithmic colour scale; default on for concentrations and IOPs
        (``chl*``, ``tsm*``, ``a_*``, ``bbp_*``, ``kd*``...).
    vmin, vmax : float, optional
        Colour range; default the ``percentiles`` of the data.
    percentiles : tuple of float, default (2, 98)
    stretch, per_channel, gamma
        RGB rendering, as in :func:`rgb_image`.
    outline : bool, default True
        Draw the swath footprint (needs shapely).
    water : bool, default True
        Shade the sea and lakes (Natural Earth) light blue under the data;
        land is light grey.
    coastlines, graticule, scale_bar, north_arrow
        Map elements (:func:`hypso.rgb.decorate_map`).
    title : str, optional
        Default: target, time, product and variable.
    colorbar : bool, default True

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    product = _product(source)
    gridded = product.grid is not None
    shape = None if gridded else product
    reduced, mode = _subset(product, variable, wavelength)
    if not gridded:
        reduced = reduced.regrid(resolution=resolution)
    area = reduced.area_definition()
    x, y = (np.asarray(v) for v in area.get_proj_vectors())
    crs = area.crs
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]
    origin = "upper" if y[0] > y[-1] else "lower"
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 8))
    ax.set_facecolor(LAND_COLOR)                      # land (and no data off the water)
    ink, coast = "black", "#333333"
    if mode == "rgb":
        bands = [int(np.argmin(np.abs(reduced.wavelengths - w))) for w in RGB_WAVELENGTHS]
        rgb = _stretch(reduced.cube.values[..., bands].astype(float),
                       **_rgb_settings(reduced, stretch, per_channel, gamma))
        alpha = np.isfinite(rgb).all(axis=-1)
        ax.imshow(np.dstack([np.nan_to_num(rgb), alpha]), extent=extent, origin=origin, interpolation="nearest",
                  zorder=0)
        label = "RGB"
    else:
        if mode == "band":
            data = reduced.cube.values[..., 0].astype(float)
            units, long_name = _cube_units(reduced)
            attrs = {"units": units, "long_name": f"{long_name} at {reduced.wavelengths[0]:.0f} nm"}
            label = f"{variable} {reduced.wavelengths[0]:.0f} nm"
        else:
            data = reduced.variables[variable].values.astype(float)
            attrs = dict(reduced.variables[variable].attrs)
            label = variable
        finite = data[np.isfinite(data)]
        use_log = log if log is not None else mode == "variable" and bool(_LOG_VARIABLES.match(variable))
        if use_log:
            finite = finite[finite > 0]
        if finite.size:
            low, high = np.percentile(finite, percentiles)
        else:
            low, high = (1e-3, 1.0) if use_log else (0.0, 1.0)
        low, high = vmin if vmin is not None else low, vmax if vmax is not None else high
        norm = LogNorm(low, high) if use_log else Normalize(low, high)
        image = ax.imshow(np.ma.masked_invalid(np.where(data > 0, data, np.nan) if use_log else data),
                          extent=extent, origin=origin, interpolation="nearest", cmap=cmap or "viridis", norm=norm)
        if colorbar:
            units = attrs.get("units", "")
            name = attrs.get("long_name", label)
            bar = plt.colorbar(image, ax=ax, shrink=0.7, pad=0.02, extend="both")
            bar.set_label(f"{name} [{units}]" if units else name)
    if outline and shape is not None:
        try:
            polygon = footprint(shape)
            to_map = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            for part in getattr(polygon, "geoms", [polygon]):
                if part.geom_type == "Polygon":
                    px, py = to_map.transform(*np.asarray(part.exterior.coords).T)
                    ax.plot(px, py, color="black", lw=0.8)
        except Exception as error:                    # shapely missing
            warnings.warn(f"footprint outline skipped: {error}", UserWarning, stacklevel=2)
    decorate_map(ax, extent, crs, graticule=graticule, scale_bar=scale_bar, north_arrow=north_arrow,
                 coastlines=coastlines, ink=ink, coast_color=coast, water=water, water_color=WATER_COLOR)
    if title is None:
        record = footprint_attributes(product)
        title = f"{record['capture']}\n{record['product']}: {label}"
    if title:
        ax.set_title(title, fontsize=10)
    return ax


LAND_COLOR, WATER_COLOR = "#e3e3e3", "#c6dbef"
_OVERVIEW_COLOURS = ("#2a78b5", "#d8552b", "#3a9a5b", "#8a5cc2", "#c79a14", "#1d9fa6", "#c2477f", "#6b6b6b")


def plot_overview(sources, ax=None, labels=None, margin: float = 0.1, coastlines: bool = True,
                  graticule: bool = True, scale_bar: bool = True, north_arrow: bool = True, title: str = None,
                  crs=None, water: bool = True, fill: bool = False):
    """Plot the footprints of several products on one map.

    Parameters
    ----------
    sources : list of hypso.Product, paths or shapely polygons
        Polygons in WGS84 longitude/latitude (e.g. from :func:`footprint`)
        are drawn as they are.
    ax : matplotlib.axes.Axes, optional
    labels : list of str, optional
        Legend labels; default each product's acquisition time (UTC),
        after its target when the products have several targets.
    margin : float, default 0.1
        Space around the footprints, as a fraction of their extent.
    coastlines, graticule, scale_bar, north_arrow
        Map elements (:func:`hypso.rgb.decorate_map`).
    title : str, optional
    crs : optional
        Map projection; default the UTM zone of the footprints' centre.
    water : bool, default True
        Shade the sea and lakes light blue; land is light grey.
    fill : bool, default False
        Tint the inside of each footprint in its colour (outlines only by
        default).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from hypso.grid.areas import utm_crs
    polygons, records = [], []
    for k, source in enumerate(sources):
        if hasattr(source, "geom_type"):
            polygons.append(source)
            records.append({"target": "", "start_utc": "", "name": f"footprint {k + 1}"})
            continue
        product = _product(source)
        polygons.append(footprint(product))
        records.append(footprint_attributes(product))
    several_targets = len({r["target"] for r in records if r["target"]}) > 1
    names = [r.get("name") or (f"{r['target']}  " if several_targets else "")
             + r["start_utc"][:16].replace("T", " ") + " UTC" for r in records]
    labels = labels or names
    if len(polygons) > len(_OVERVIEW_COLOURS):
        warnings.warn(f"{len(polygons)} footprints and {len(_OVERVIEW_COLOURS)} colours: colours repeat",
                      UserWarning, stacklevel=2)
    bounds = np.array([p.bounds for p in polygons])
    centre_lon = (bounds[:, 0].min() + bounds[:, 2].max()) / 2
    centre_lat = (bounds[:, 1].min() + bounds[:, 3].max()) / 2
    crs = utm_crs([centre_lat], [centre_lon]) if crs is None else pyproj.CRS.from_user_input(crs)
    to_map = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(LAND_COLOR if water else "white")
    xs, ys = [], []
    for k, (polygon, label) in enumerate(zip(polygons, labels)):
        colour = _OVERVIEW_COLOURS[k % len(_OVERVIEW_COLOURS)]
        for i, part in enumerate(getattr(polygon, "geoms", [polygon])):
            if part.geom_type != "Polygon":
                continue
            px, py = to_map.transform(*np.asarray(part.exterior.coords).T)
            if fill:
                ax.fill(px, py, facecolor=colour, alpha=0.12, edgecolor="none")
            ax.plot(px, py, color=colour, lw=1.8, label=label if i == 0 else None, zorder=3)
            xs.extend(px)
            ys.extend(py)
    xs, ys = np.asarray(xs), np.asarray(ys)
    pad = margin * max(np.ptp(xs), np.ptp(ys))
    extent = [xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad]
    decorate_map(ax, extent, crs, graticule=graticule, scale_bar=scale_bar, north_arrow=north_arrow,
                 coastlines=coastlines, ink="black", coast_color="#555555", water=water,
                 water_color=WATER_COLOR)
    ax.set_aspect("equal")
    # outside the map, so it never hides a footprint
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
