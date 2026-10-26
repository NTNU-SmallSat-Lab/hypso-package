"""One reader and one writer for every HYPSO product level, built on
xarray.DataTree (REFACTOR_PROGRESS.md x37 "plain Dataset core", step 1).

In-memory model (what read_product returns and write_product takes):

    /                     root Dataset
        <prefix>          product cube, dims (lines, samples, bands) -
                          prefix "dn" (L1A), "Lt" (L1B/L1C), "rhot" (L1D),
                          or the AC variable for L2A (e.g. "Rrs", "rhos")
        wavelength        coord (bands): nominal band centre, 0.1 nm rounded
        radiation_wavelength  coord (bands): precise as-calibrated centre
        fwhm              coord (bands)
        latitude, longitude, crs_wgs84, <angles>   only when georeferenced
        pixel_flags       optional CF flag variable (land/cloud/custom masks,
                          hypso.masks.flags); the product itself is unmasked
        <other>           any other (lines, samples) variable - derived
                          products such as a chlorophyll map - passed through
                          with its attributes
        attrs             global attributes; processing_level is required
    /metadata/<group>     one child node per metadata group, passed through

On disk (write_product): the product is split into one 2D variable per band
(Lt_378, ...) - the per-band layout ESA SNAP reads - with geometry at the
root and metadata/* nested. All CF attributes come from io/cf.py, so a file
read in any supported layout is written back in the current one:

- current flat files (this package's io/writer.py output),
- legacy grouped files (/products + /geometry, pre-refactor and production),
- ground-station L1A (/products/dn as one uint16 (lines, samples, bands) cube).

Converting a ground-station L1A is read_product + write_product; the
original file stays the source of truth.
"""
import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr

from . import cf
from .naming import parse_product_name, product_name as make_product_name
from .provenance import PRODUCT_FORMAT_VERSION, add_step, utc_now
from .schema import get_schema

logger = logging.getLogger(__name__)

PRODUCT_DIMS = ("lines", "samples", "bands")
SPATIAL_DIMS = PRODUCT_DIMS[:2]
BAND_COORDS = ("wavelength", "radiation_wavelength", "fwhm")

ENCODING = {"zlib": True, "complevel": 4, "shuffle": True}

# Root geometry variable -> (cf attribute builder, long_name). Same table as
# io/writer.py's _INDIRECT/_DIRECT_ANGLE_FIELDS ("_direct" means indirect
# georeferencing - pre-existing naming, kept for readers that key off it).
_GEOMETRY_ATTRS = {
    "latitude": lambda: cf.latitude_attrs(),
    "longitude": lambda: cf.longitude_attrs(),
    "latitude_direct": lambda: {**cf.latitude_attrs(), "long_name": "Latitude (Indirect)"},
    "longitude_direct": lambda: {**cf.longitude_attrs(), "long_name": "Longitude (Indirect)"},
    "sensor_zenith": lambda: cf.zenith_angle_attrs("Sensor Zenith Angle"),
    "sensor_azimuth": lambda: cf.azimuth_angle_attrs("Sensor Azimuth Angle"),
    "solar_zenith": lambda: cf.zenith_angle_attrs("Solar Zenith Angle"),
    "solar_azimuth": lambda: cf.azimuth_angle_attrs("Solar Azimuth Angle"),
    "relative_azimuth": lambda: cf.azimuth_angle_attrs("Relative Azimuth Angle"),
    "sensor_zenith_direct": lambda: cf.zenith_angle_attrs("Sensor Zenith Angle (Indirect)"),
    "sensor_azimuth_direct": lambda: cf.azimuth_angle_attrs("Sensor Azimuth Angle (Indirect)"),
    "solar_zenith_direct": lambda: cf.zenith_angle_attrs("Solar Zenith Angle (Indirect)"),
    "solar_azimuth_direct": lambda: cf.azimuth_angle_attrs("Solar Azimuth Angle (Indirect)"),
    "relative_azimuth_direct": lambda: cf.azimuth_angle_attrs("Relative Azimuth Angle (Indirect)"),
}

# Legacy metadata/gcp names collided with root latitude/longitude in SNAP
# (b5ff9508); current files use the gcp_ prefix.
_GCP_RENAMES = {"latitude": "gcp_latitude", "longitude": "gcp_longitude",
                "sourceX": "gcp_sourceX", "sourceY": "gcp_sourceY"}

_LEVEL_ALIASES = {"L2": "L2A"}


def _level_key(processing_level: str) -> str:
    level = str(processing_level).upper()
    return _LEVEL_ALIASES.get(level, level)


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------

def _scalar(value) -> float:
    # radiation_wavelength was a 1-element tuple in pre-refactor files
    # (trailing-comma bug) - accept either form.
    return float(np.ravel(value)[0])


def _band_variables(ds: xr.Dataset) -> list:
    """Per-band 2D product variables (carry both `band` and `wavelength`
    attrs), sorted by their `band` index - never by name or file order."""
    names = [n for n, v in ds.data_vars.items()
             if v.dims == SPATIAL_DIMS and "band" in v.attrs and "wavelength" in v.attrs]
    return sorted(names, key=lambda n: int(ds[n].attrs["band"]))


def _flag_variables(ds: xr.Dataset) -> list:
    """CF flag variables (masks, see hypso.masks.flags), by their
    flag_meanings attribute."""
    return [n for n, v in ds.data_vars.items() if v.dims == SPATIAL_DIMS and "flag_meanings" in v.attrs]


def _extra_variables(ds: xr.Dataset, exclude) -> list:
    """Other (lines, samples) variables - derived products, including an AC
    tool's own flags - excluding band, geometry, grid-mapping and
    pixel_flags variables."""
    from hypso.masks.flags import FLAGS_NAME
    skip = set(exclude) | set(_GEOMETRY_ATTRS) | {"crs_wgs84", FLAGS_NAME}
    return [n for n, v in ds.data_vars.items() if v.dims == SPATIAL_DIMS and n not in skip]


def list_band_variables(path) -> list:
    """Per-band metadata of a product file without reading pixel data:
    [{"name", "band", "wavelength", "radiation_wavelength", "fwhm", "units",
    "long_name"}] sorted by band index, or one entry with band None for a
    single-cube (datacube) file. Used by the Satpy reader plugin."""
    import netCDF4
    with netCDF4.Dataset(path) as f:
        group = f["products"] if "products" in f.groups else f
        bands = [(n, v) for n, v in group.variables.items()
                 if v.dimensions == SPATIAL_DIMS and "band" in v.ncattrs() and "wavelength" in v.ncattrs()]
        if not bands:
            cubes = [(n, v) for n, v in group.variables.items() if len(v.dimensions) == 3]
            return [{"name": n, "band": None, "wavelength": None, "radiation_wavelength": None, "fwhm": None,
                     "units": getattr(v, "units", ""), "long_name": getattr(v, "long_name", "")} for n, v in cubes]
        bands.sort(key=lambda item: int(item[1].band))
        return [{"name": n, "band": int(v.band), "wavelength": float(v.wavelength),
                 "radiation_wavelength": _scalar(getattr(v, "radiation_wavelength", v.wavelength)),
                 "fwhm": float(v.fwhm), "units": getattr(v, "units", ""), "long_name": getattr(v, "long_name", "")}
                for n, v in bands]


def _fwhm_from_profile(sat_id: str, wavelengths: np.ndarray) -> np.ndarray:
    # Same nearest-neighbour lookup as capture_types._get_fwhm, for files that
    # store no per-band fwhm (ground-station L1A).
    from hypso.sensors import get_sensor_profile
    profile = get_sensor_profile(sat_id)
    return np.array([profile.fwhm_lookup_fwhm[np.argmin(np.abs(w - profile.fwhm_lookup_wl))]
                     for w in wavelengths], dtype=float)


def _product_from_bands(ds: xr.Dataset, names: list):
    prefix = names[0].rsplit("_", 1)[0]
    cube = np.stack([ds[n].values for n in names], axis=-1)
    coords = {
        "wavelength": ("bands", [float(ds[n].attrs["wavelength"]) for n in names]),
        "radiation_wavelength": ("bands", [_scalar(ds[n].attrs.get("radiation_wavelength", ds[n].attrs["wavelength"]))
                                           for n in names]),
        "fwhm": ("bands", [float(ds[n].attrs["fwhm"]) for n in names]),
    }
    return prefix, xr.DataArray(cube, dims=PRODUCT_DIMS, coords=coords)


def _product_from_cube(name: str, var: xr.DataArray, wavelengths, sat_id: str):
    wavelengths = np.asarray(wavelengths, dtype=float)
    if "fwhm" in var.attrs and np.size(var.attrs["fwhm"]) == wavelengths.size:
        fwhm = np.asarray(var.attrs["fwhm"], dtype=float)
    else:
        fwhm = _fwhm_from_profile(sat_id, wavelengths)
    coords = {"wavelength": ("bands", np.around(wavelengths, 1)),
              "radiation_wavelength": ("bands", wavelengths),
              "fwhm": ("bands", fwhm)}
    return name, xr.DataArray(var.values, dims=PRODUCT_DIMS, coords=coords)


def source_layout(path) -> str:
    """"current" (flat), "legacy" (/products + /geometry groups) or
    "ground_station_l1a" (one 3D /products/dn cube)."""
    import netCDF4
    with netCDF4.Dataset(path) as f:
        if "products" not in f.groups:
            return "current"
        return "ground_station_l1a" if "dn" in f["products"].variables else "legacy"


def read_product(path) -> xr.DataTree:
    """Read any supported HYPSO product file into the in-memory model (see
    module docstring). Raises ValueError if no product cube is found."""
    raw = xr.open_datatree(path, engine="netcdf4", mask_and_scale=False, decode_times=False,
                           decode_coords=False, decode_timedelta=False)
    try:
        root = raw.to_dataset(inherit=False)
        legacy = "products" in raw.children
        products = raw["products"].to_dataset(inherit=False) if legacy else root
        geometry = (raw["geometry"].to_dataset(inherit=False)
                    if legacy and "geometry" in raw.children else root)

        sat_id = root.attrs.get("sat_id") or root.attrs.get("instrument")
        names = _band_variables(products)
        if names:
            prefix, cube = _product_from_bands(products, names)
        else:
            cubes = [n for n, v in products.data_vars.items() if v.ndim == 3]
            if len(cubes) != 1:
                raise ValueError(f"{path}: no per-band product variables and {len(cubes)} 3D cubes")
            var = products[cubes[0]]
            wavelengths = var.attrs.get("wavelengths")
            if wavelengths is None or np.size(wavelengths) != var.shape[-1]:
                wavelengths = raw["metadata/corrections"].to_dataset(inherit=False)["wavelengths"].values
            prefix, cube = _product_from_cube(cubes[0], var, wavelengths, sat_id)

        level = root.attrs.get("processing_level") or ("L1A" if prefix == "dn" else None)
        if level is None:
            raise ValueError(f"{path}: no processing_level attribute")

        out = xr.Dataset({prefix: cube}, attrs={**root.attrs, "processing_level": _level_key(level)})
        for name in _GEOMETRY_ATTRS:
            if name in geometry.data_vars:
                out[name] = (SPATIAL_DIMS, geometry[name].values)
        if "latitude" in out and "crs_wgs84" in geometry.data_vars:
            out["crs_wgs84"] = ((), geometry["crs_wgs84"].values)
        for name in dict.fromkeys(_flag_variables(products) + _extra_variables(products, names)):
            attrs = {k: v for k, v in products[name].attrs.items() if k not in ("coordinates", "grid_mapping")}
            out[name] = (SPATIAL_DIMS, products[name].values, attrs)

        tree = xr.DataTree(out)
        if "metadata" in raw.children:
            for node in raw["metadata"].subtree:
                ds = node.to_dataset(inherit=False).load()
                if node.path.endswith("/gcp"):
                    ds = ds.rename({k: v for k, v in _GCP_RENAMES.items() if k in ds.variables})
                tree[node.path] = xr.DataTree(ds)
        # Not written to files; processing steps record it as their input.
        tree.encoding["source"] = str(path)
        return tree
    finally:
        raw.close()


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------

def product_name(tree: xr.DataTree) -> str:
    cubes = [n for n, v in tree.dataset.data_vars.items() if v.dims == PRODUCT_DIMS]
    if len(cubes) != 1:
        raise ValueError(f"expected one {PRODUCT_DIMS} product cube at the root, found {cubes}")
    return cubes[0]


def _encoding(var: xr.DataArray, dtype=None) -> dict:
    enc = {"_FillValue": None}
    if var.ndim:
        enc.update(ENCODING, chunksizes=var.shape)
    if dtype is not None:
        enc["dtype"] = dtype
    return enc


def _root_dataset(tree: xr.DataTree, datacube: bool):
    """Root Dataset as written, plus its encoding."""
    ds = tree.dataset
    prefix = product_name(tree)
    schema = get_schema(_level_key(ds.attrs["processing_level"]))
    cube = ds[prefix]
    has_geometry = "latitude" in ds.data_vars
    units = schema.product_units
    long_name = schema.product_long_name
    standard_name = schema.product_standard_name
    if schema.processing_level in ("L2", "L2A") and prefix in cf.L2A_PRODUCTS:
        product = cf.L2A_PRODUCTS[prefix]
        units, long_name, standard_name = product["units"], product["long_name"], product.get("standard_name")
    dtype = "u2" if prefix == "dn" else "f4"

    out, encoding = {}, {}
    if datacube:
        attrs = {"units": units, "long_name": long_name, "wavelength_units": "nanometers",
                 **({"standard_name": standard_name} if standard_name else {}),
                 "fwhm": cube["fwhm"].values, "wavelengths": cube["wavelength"].values}
        if has_geometry:
            attrs.update(cf.geolocation_ref_attrs())
        out[prefix] = xr.Variable(PRODUCT_DIMS, cube.values, attrs)
        encoding[prefix] = _encoding(out[prefix], dtype)
    else:
        for band in range(cube.sizes["bands"]):
            wave = float(cube["wavelength"].values[band])
            wave_name = str(int(wave))
            name = f"{prefix}_{wave_name}"
            attrs = cf.band_attrs(
                long_name=f"{long_name} Band {band} ({wave_name} nm)",
                units=units,
                wavelength_nm=wave,
                radiation_wavelength_nm=float(cube["radiation_wavelength"].values[band]),
                fwhm=float(cube["fwhm"].values[band]),
                wave_name=wave_name,
                band_index=band,
                include_geolocation=has_geometry,
                var_name=name,
                standard_name=standard_name,
            )
            out[name] = xr.Variable(SPATIAL_DIMS, cube.values[:, :, band], attrs)
            encoding[name] = _encoding(out[name], dtype)

    for name, builder in _GEOMETRY_ATTRS.items():
        if name in ds.data_vars:
            out[name] = xr.Variable(SPATIAL_DIMS, ds[name].values, builder())
            encoding[name] = _encoding(out[name], "f4")
    for name in dict.fromkeys(_flag_variables(ds) + _extra_variables(ds, [prefix])):
        var_attrs = {k: v for k, v in ds[name].attrs.items() if k not in ("coordinates", "grid_mapping")}
        if has_geometry:
            var_attrs.update(cf.geolocation_ref_attrs())
        out[name] = xr.Variable(SPATIAL_DIMS, ds[name].values, var_attrs)
        encoding[name] = _encoding(out[name])
    if has_geometry:
        # The grid-mapping variable's value carries no meaning (CF reads only
        # its attrs); keep the source file's, else 0.
        value = ds["crs_wgs84"].values if "crs_wgs84" in ds.variables else np.int32(0)
        out["crs_wgs84"] = xr.Variable((), value, cf.crs_wgs84_attrs())
        encoding["crs_wgs84"] = _encoding(out["crs_wgs84"], "i4")

    # Writing is not a processing step: history lines come from
    # provenance.add_step, not from here.
    level = _level_key(schema.processing_level)
    attrs = dict(ds.attrs)
    attrs.update(cf.global_attrs(level, schema.title))
    attrs["Conventions"] = f"{cf.CONVENTIONS}, ACDD-1.3"
    attrs["processing_level"] = level   # "L2A", not the format-1 "L2"
    attrs["product_format_version"] = PRODUCT_FORMAT_VERSION
    attrs["date_created"] = utc_now()
    if has_geometry:
        lat, lon = ds["latitude"].values, ds["longitude"].values
        attrs.update(geospatial_lat_min=float(np.nanmin(lat)), geospatial_lat_max=float(np.nanmax(lat)),
                     geospatial_lon_min=float(np.nanmin(lon)), geospatial_lon_max=float(np.nanmax(lon)),
                     geospatial_lat_units="degrees_north", geospatial_lon_units="degrees_east")
    return xr.Dataset(out, attrs=attrs), encoding


def write_product(tree: xr.DataTree, path, datacube: bool = False) -> None:
    """Write the in-memory model to `path` in the current file format."""
    root, root_encoding = _root_dataset(tree, datacube)
    nodes = {"/": root}
    encoding = {"/": root_encoding}
    for node in tree.subtree:
        if node.is_root:
            continue
        ds = node.to_dataset(inherit=False)
        nodes[node.path] = ds
        encoding[node.path] = {name: _encoding(var) for name, var in ds.variables.items()}
    xr.DataTree.from_dict(nodes).to_netcdf(path, engine="netcdf4", encoding=encoding)


def convert_product(src, out_dir=None) -> Path:
    """Write `src` (any supported layout) as a current-format product named
    by io/naming.py, next to `src` unless `out_dir` is given; records a
    "convert" step. The source file is left untouched."""
    src = Path(src)
    tree = read_product(src)
    attrs = tree.attrs
    level = _level_key(attrs["processing_level"])
    sat_id = attrs.get("sat_id") or attrs.get("instrument")
    name = parse_product_name(src)
    ac = name.ac if level == "L2A" else None
    dst = Path(out_dir or src.parent) / make_product_name(sat_id, name.time, name.target, level,
                                                          ac=ac, label=name.label).filename()
    if dst.resolve() == src.resolve():
        raise ValueError(f"{src} is already a current-format product name")
    add_step(tree, "convert", input_file=src, parameters={"source_layout": source_layout(src)})
    write_product(tree, dst)
    return dst
