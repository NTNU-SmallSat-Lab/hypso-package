"""RGB images on map grids, and map figures.

- :func:`regrid_camera`: a georeferenced RGB camera image
  (:func:`hypso.rgb.georeference_camera`) on a map grid - the capture's UTM
  zone, any CRS, or another product's grid - by cell averaging (default; the
  camera's ~30 m pixels are finer than most grids) or nearest neighbour.
- :func:`plot_rgb_map`: a gridded RGB image (from :func:`regrid_camera`, or a
  regridded hypso.Product) as a map, with optional map elements: latitude/
  longitude graticule, scale bar, north arrow, coastlines and title.

matplotlib is needed for the figures, and cartopy (with its Natural Earth
data) for coastlines; neither is a dependency of the package.
"""
import warnings

import numpy as np
import pyproj
import xarray as xr

from hypso.grid.areas import coordinate_attrs, grid_area, grid_name, utm_crs

CHANNELS = ("R", "G", "B")
#: HSI bands used for an RGB rendering of a hypso.Product (nm).
RGB_WAVELENGTHS = (640, 550, 460)


def camera_gsd(geo: xr.Dataset) -> float:
    """Median distance (m) between neighbouring camera pixel centres."""
    lat, lon = geo["latitude"].values, geo["longitude"].values
    crs = utm_crs(lat[::50, ::50], lon[::50, ::50])
    x, y = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        lon[::20, ::20].astype(float), lat[::20, ::20].astype(float))
    return float(np.median(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))) / 20)


def regrid_camera(geo: xr.Dataset, target="utm", resolution: float = None, method: str = "auto",
                  max_gb: float = 2.0) -> xr.Dataset:
    """A georeferenced camera image on a map grid.

    Parameters
    ----------
    geo : xarray.Dataset
        From :func:`hypso.rgb.georeference_camera` (R, G, B, latitude, longitude).
    target : str, int, pyproj.CRS or pyresample.geometry.AreaDefinition, default "utm"
        ``"utm"`` (zone of the image centre), any CRS, or a grid - e.g. the HSI
        product's after regridding, ``l1d.regrid(...).area_definition()``, to
        put both on the same grid.
    resolution : float, optional
        Pixel size in the CRS's units; default the camera's own (~30 m). Not
        with an AreaDefinition.
    method : {"auto", "average", "nearest"}, default "auto"
        Cell means of the camera pixels in each cell, or the nearest pixel;
        "auto" averages when cells are at least twice the camera's pixel
        size and takes the nearest pixel otherwise (averaging onto cells as
        small as the pixels leaves empty cells).
    max_gb : float, default 2
        Refuse grids larger than this.

    Returns
    -------
    xarray.Dataset
        ``R``, ``G``, ``B`` on ``(y, x)`` with CF projection coordinates and a
        ``crs`` grid mapping (NaN where the camera has no data), and the
        camera's attributes plus ``grid``.
    """
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from hypso.grid.resample import Averager, Resampler
    lat, lon = geo["latitude"].values, geo["longitude"].values
    if isinstance(target, AreaDefinition):
        if resolution is not None:
            raise ValueError("resolution is set by the AreaDefinition; do not pass it as well")
        area = target
    else:
        crs = utm_crs(lat, lon) if target == "utm" else pyproj.CRS.from_user_input(target)
        if resolution is None:
            metres = camera_gsd(geo)
            resolution = round(metres / 111_320.0, 5) if crs.is_geographic else float(max(1, round(metres)))
        area = grid_area(crs, resolution, lat, lon, area_id="camera_grid")
    size_gb = area.shape[0] * area.shape[1] * 3 * 4 / 1e9
    if size_gb > max_gb:
        raise ValueError(f"the {area.shape[0]} x {area.shape[1]} grid would need {size_gb:.1f} GB (max_gb={max_gb})")
    cube = np.stack([geo[c].values for c in CHANNELS], axis=-1).astype(np.float32)
    if method == "auto":
        cell = abs(area.pixel_size_x) * (111_320.0 if area.crs.is_geographic else 1)
        method = "average" if cell >= 2 * camera_gsd(geo) else "nearest"
    if method == "average":
        data = Averager(lat, lon, area).mean(cube)
    elif method == "nearest":
        radius = 1.5 * max(camera_gsd(geo), abs(area.pixel_size_x) * (111_320.0 if area.crs.is_geographic else 1))
        data = Resampler(SwathDefinition(lons=lon.astype(np.float64), lats=lat.astype(np.float64)), area, radius,
                         bilinear=False).nearest(cube)
    else:
        raise ValueError(f"method must be 'average' or 'nearest', got {method!r}")
    x, y = area.get_proj_vectors()
    x_attrs, y_attrs = coordinate_attrs(area.crs)
    size_x, size_y = abs(float(area.pixel_size_x)), abs(float(area.pixel_size_y))
    out = xr.Dataset({c: (("y", "x"), data[..., i].astype(np.float32), dict(geo[c].attrs))
                      for i, c in enumerate(CHANNELS)},
                     coords={"x": ("x", x, x_attrs), "y": ("y", y, y_attrs)},
                     attrs={**{k: v for k, v in geo.attrs.items()},
                            "grid": grid_name(area.crs, size_x, size_y), "grid_crs": area.crs.to_string(),
                            "grid_resolution_x": size_x, "grid_resolution_y": size_y, "grid_method": method})
    out["crs"] = ((), np.int32(0), area.crs.to_cf())
    for c in CHANNELS:
        out[c].attrs["grid_mapping"] = "crs"
    return out


def _rgb_and_grid(source, stretch):
    """(H x W x 3 image in 0-1 with NaN outside, x, y in metres, CRS)."""
    if isinstance(source, xr.Dataset):
        rgb = np.stack([source[c].values for c in CHANNELS], axis=-1).astype(float)
        crs = pyproj.CRS.from_cf(dict(source["crs"].attrs))
        x, y = source["x"].values, source["y"].values
    else:                                             # a regridded hypso.Product
        if source.grid is None:
            raise ValueError("plot_rgb_map needs a gridded product: regrid() it first")
        bands = [int(np.argmin(np.abs(source.wavelengths - w))) for w in RGB_WAVELENGTHS]
        rgb = source.cube.values[..., bands].astype(float)
        area = source.area_definition()
        crs = area.crs
        x, y = area.get_proj_vectors()
    low, high = np.nanpercentile(rgb, stretch, axis=(0, 1))
    image = np.clip((rgb - low) / (high - low), 0, 1)
    return image, np.asarray(x), np.asarray(y), crs


def _nice_length(span):
    exponent = 10 ** np.floor(np.log10(span))
    return max(m * exponent for m in (1, 2, 5) if m * exponent <= span)


def plot_rgb_map(source, ax=None, graticule: bool = True, scale_bar: bool = True, north_arrow: bool = True,
                 coastlines: bool = False, title: str = None, stretch=(2, 98), graticule_step: float = None):
    """Plot a gridded RGB image as a map.

    Parameters
    ----------
    source : xarray.Dataset or hypso.Product
        A gridded camera image (:func:`regrid_camera`) or a regridded product
        (its bands nearest 640, 550 and 460 nm).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; default a new figure.
    graticule : bool, default True
        Latitude/longitude lines with labels.
    scale_bar : bool, default True
        A scale bar (km) in the lower left.
    north_arrow : bool, default True
        An arrow to true north (in a UTM grid, true north is not straight up).
    coastlines : bool, default False
        Natural Earth 10 m coastlines (needs cartopy and its data; skipped
        with a warning otherwise).
    title : str, optional
    stretch : tuple of float, default (2, 98)
        Percentiles mapped to black and white per channel.
    graticule_step : float, optional
        Degrees between graticule lines; default chosen from the extent.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patheffects
    except ImportError as error:
        raise ImportError("plot_rgb_map needs matplotlib (pip install matplotlib)") from error
    outline = [patheffects.withStroke(linewidth=2.5, foreground="black")]      # readable on any background
    image, x, y, crs = _rgb_and_grid(source, stretch)
    image = np.where(np.isnan(image), 1.0, image)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    dx, dy = abs(x[1] - x[0]) if x.size > 1 else 1.0, abs(y[1] - y[0]) if y.size > 1 else 1.0
    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]
    ax.imshow(image, extent=extent, origin="upper" if y[0] > y[-1] else "lower", interpolation="nearest")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    to_lonlat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_map = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xx, yy = np.meshgrid(np.linspace(extent[0], extent[1], 50), np.linspace(extent[2], extent[3], 50))
    lons, lats = to_lonlat.transform(xx, yy)

    if graticule:
        span = max(np.ptp(lons), np.ptp(lats))
        step = graticule_step or _nice_length(span / 3)
        for value in np.arange(np.ceil(lats.min() / step) * step, lats.max(), step):
            lx, ly = to_map.transform(np.linspace(lons.min(), lons.max(), 200), np.full(200, value))
            ax.plot(lx, ly, color="white", lw=0.6, alpha=0.8, path_effects=[patheffects.withStroke(linewidth=1.4, foreground="black", alpha=0.5)])
            ax.annotate(f"{value:.6g}°N", (extent[0], ly[np.argmin(np.abs(lx - extent[0]))]), color="white",
                        fontsize=8, xytext=(3, 2), textcoords="offset points", path_effects=outline)
        for value in np.arange(np.ceil(lons.min() / step) * step, lons.max(), step):
            lx, ly = to_map.transform(np.full(200, value), np.linspace(lats.min(), lats.max(), 200))
            ax.plot(lx, ly, color="white", lw=0.6, alpha=0.8, path_effects=[patheffects.withStroke(linewidth=1.4, foreground="black", alpha=0.5)])
            ax.annotate(f"{value:.6g}°E", (lx[np.argmin(np.abs(ly - extent[2]))], extent[2]), color="white",
                        fontsize=8, xytext=(2, 3), textcoords="offset points", rotation=90, path_effects=outline)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if coastlines:
        try:
            from cartopy.io import shapereader
            path = shapereader.natural_earth(resolution="10m", category="physical", name="coastline")
            box = (lons.min() - 0.5, lats.min() - 0.5, lons.max() + 0.5, lats.max() + 0.5)
            for geometry in shapereader.Reader(path).geometries():
                if not geometry.intersects(__import__("shapely.geometry", fromlist=["box"]).box(*box)):
                    continue
                for line in getattr(geometry, "geoms", [geometry]):
                    lon_line, lat_line = np.asarray(line.coords).T
                    cx, cy = to_map.transform(lon_line, lat_line)
                    ax.plot(cx, cy, color="yellow", lw=0.7)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        except Exception as error:                    # cartopy or its data missing
            warnings.warn(f"coastlines skipped: {error}", UserWarning, stacklevel=2)

    width = extent[1] - extent[0]
    if scale_bar:
        length = _nice_length(width / 4)
        x0, y0 = extent[0] + 0.05 * width, extent[2] + 0.05 * (extent[3] - extent[2])
        ax.plot([x0, x0 + length], [y0, y0], color="white", lw=3, solid_capstyle="butt")
        ax.plot([x0, x0 + length], [y0, y0], color="black", lw=1, solid_capstyle="butt")
        ax.annotate(f"{length / 1000:g} km", (x0 + length / 2, y0), color="white", ha="center", fontsize=9,
                    xytext=(0, 5), textcoords="offset points", path_effects=outline)

    if north_arrow:
        cx, cy = extent[1] - 0.08 * width, extent[3] - 0.15 * (extent[3] - extent[2])
        lon_c, lat_c = to_lonlat.transform(cx, cy)
        nx, ny = to_map.transform(lon_c, lat_c + 0.01)
        direction = np.array([nx - cx, ny - cy]) / np.hypot(nx - cx, ny - cy)
        tip = np.array([cx, cy]) + direction * 0.06 * width
        ax.annotate("N", tip, xytext=(cx, cy), color="white", ha="center", fontsize=11, fontweight="bold",
                    path_effects=outline,
                    arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2, "path_effects": outline})

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
