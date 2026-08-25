"""Generic, schema-driven NetCDF level reader - the read-side counterpart to
io/writer.py, replacing the previous one-loader-file-per-level design
(load/l1b_nc_loader.py, l1c_nc_loader.py, l1d_nc_loader.py, l2a_nc_loader.py -
confirmed near-duplicates of each other) with one function, load_level_nc.

load_l1a_nc (load/l1a_nc_loader.py) is intentionally NOT migrated here: L1A
files are raw ground-segment output, produced outside hypso-package entirely
(there is no write_l1a_nc_file) - they keep the original nested `products`
group layout regardless of anything this refactor changes about how
hypso-package writes its *own* L1B/L1C/L1D/L2A output, so there is nothing to
generalize there.

Reads the flattened root-group layout io/writer.py produces (products and
geometry variables at the root group, metadata/* nested - see its module
docstring). This reader cannot read files written by the pre-refactor
write/l1b_nc_writer.py etc (nested products/geometry groups) - see
REFACTOR_PROGRESS.md's format-change note.
"""
import logging
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import netCDF4 as nc
from tqdm import tqdm

from hypso.load.utils import (
    load_capture_config_from_nc_file,
    load_timing_from_nc_file,
    load_adcs_from_nc_file,
    load_dimensions_from_nc_file,
    load_database_from_nc_file,
    load_corrections_from_nc_file,
    load_logfiles_from_nc_file,
    load_temperature_from_nc_file,
    load_ncattrs_from_nc_file,
    load_gcp_from_nc_file,
    load_srf_from_nc_file,
)

from .schema import LevelSchema, get_schema

logger = logging.getLogger(__name__)

# Root-level variable names io/writer.py's _write_geometry_root may write -
# used to tell geometry variables apart from product variables now that both
# live in the same (root) group. See io/writer.py's _INDIRECT_ANGLE_FIELDS/
# _DIRECT_ANGLE_FIELDS for where these names come from.
_KNOWN_GEOMETRY_VARS = frozenset((
    'latitude', 'longitude', 'latitude_direct', 'longitude_direct', 'crs_wgs84',
    'sensor_zenith', 'sensor_azimuth', 'solar_zenith', 'solar_azimuth', 'relative_azimuth',
    'sensor_zenith_direct', 'sensor_azimuth_direct', 'solar_zenith_direct',
    'solar_azimuth_direct', 'relative_azimuth_direct',
))

_METADATA_LOADERS = (
    ('capture_config', load_capture_config_from_nc_file),
    ('timing', load_timing_from_nc_file),
    ('adcs', load_adcs_from_nc_file),
    ('database', load_database_from_nc_file),
    ('corrections', load_corrections_from_nc_file),
    ('logfiles', load_logfiles_from_nc_file),
    ('temperature', load_temperature_from_nc_file),
    ('srf', load_srf_from_nc_file),
)


def _load_metadata(nc_file_path: Path) -> Tuple[dict, dict]:
    """metadata/capture_config, timing, adcs, database, corrections, logfiles,
    temperature, srf - group locations unchanged by the flattening (only
    products/geometry moved to root), reuses load/utils.py's readers as-is."""
    metadata_vars, metadata_attrs = {}, {}
    for key, loader in _METADATA_LOADERS:
        vars_, attrs_ = loader(nc_file_path)
        metadata_vars[key] = vars_
        metadata_attrs[key] = attrs_
    return metadata_vars, metadata_attrs


def _load_global_metadata(nc_file_path: Path) -> dict:
    return {
        'dimensions': load_dimensions_from_nc_file(nc_file_path),
        'ncattrs': load_ncattrs_from_nc_file(nc_file_path),
    }


def _load_geometry_root(nc_file_path: Path, has_geometry: bool) -> Tuple[dict, dict]:
    """latitude/longitude/crs_wgs84/angle variables at the ROOT group (see
    io/writer.py's _write_geometry_root). L1B (has_geometry=False) has no
    geometry data at all - returns empty dicts, matching load/l1b_nc_loader.py's
    behavior, rather than erroring on a group that was never written.

    geometry_attrs is returned empty: HypsoBase only ever reads geometry_vars
    (see HypsoBase._set_hypso_attributes's "Geometry attributes" loop) - the
    original loader's geometry_attrs (the /geometry group's own global attrs)
    was already unused downstream.
    """
    if not has_geometry:
        return {}, {}

    geometry_vars = {}
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        for name in _KNOWN_GEOMETRY_VARS:
            if name in f.variables:
                geometry_vars[name] = np.array(f.variables[name][:])

    return geometry_vars, {}


def _discover_product_variables(available_names, hint: str) -> Tuple[str, list]:
    """Return (mode, names): mode is 'datacube' (one 3D variable) or 'per_band'
    (multiple 2D variables, names unsorted - callers sort by each variable's
    `band` attribute, not this function, since sorting requires opening the
    variables). `hint` is schema.product_prefix - matches directly for L1B/L1C/
    L1D, whose prefix is fixed and known. L2A's product name is AC-tool-specific
    (e.g. "chla" vs "Rrs", set via l2_variable_name - see io/writer.py's
    write_l2a_nc) so a written L2A file's variable(s) may not match `hint` at
    all; in that case this falls back to auto-detecting from whichever
    non-geometry variables are actually present.
    """
    candidates = [n for n in available_names if n not in _KNOWN_GEOMETRY_VARS]

    if hint in candidates:
        return 'datacube', [hint]
    band_names = [n for n in candidates if n.startswith(hint + "_")]
    if band_names:
        return 'per_band', band_names

    if len(candidates) == 1:
        return 'datacube', candidates
    prefixes = {re.sub(r'_\d+$', '', n) for n in candidates}
    if len(prefixes) == 1:
        prefix = prefixes.pop()
        return 'per_band', [n for n in candidates if n.startswith(prefix + "_")]

    raise ValueError(f"Could not determine the product variable(s) for {hint!r} among "
                     f"root variables {candidates}")


def _load_cube(nc_file_path: Path, product_prefix_hint: str) -> np.ndarray:
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        mode, names = _discover_product_variables(f.variables.keys(), product_prefix_hint)

        if mode == 'datacube':
            return np.array(f.variables[names[0]][:], dtype='double')

        # Sort by each variable's `band` attribute - fixes a latent bug in the
        # original per-level loaders, which iterated group.variables in
        # dict/insertion order instead (see REFACTOR_PROGRESS.md).
        names_sorted = sorted(names, key=lambda n: int(f.variables[n].band))
        height, width = f.variables[names_sorted[0]].shape
        cube = np.empty((height, width, len(names_sorted)))
        for idx, name in enumerate(tqdm(names_sorted, desc="Loading bands")):
            cube[:, :, idx] = np.array(f.variables[name][:], dtype='double')
        return cube


def _load_cube_attrs(nc_file_path: Path, product_prefix_hint: str) -> dict:
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        mode, names = _discover_product_variables(f.variables.keys(), product_prefix_hint)

        if mode == 'datacube':
            var = f.variables[names[0]]
            return {attrname: getattr(var, attrname) for attrname in var.ncattrs()}

        names_sorted = sorted(names, key=lambda n: int(f.variables[n].band))
        wavelengths = [float(getattr(f.variables[n], "wavelength")) for n in names_sorted]
        fwhm = [float(getattr(f.variables[n], "fwhm")) for n in names_sorted]

        first = f.variables[names_sorted[0]]
        return {
            "units": getattr(first, "units", ""),
            "long_name": getattr(first, "long_name", ""),
            "wavelength_units": getattr(first, "wavelength_units", "nanometers"),
            "fwhm": fwhm,
            "wavelengths": wavelengths,
        }


def load_level_nc(nc_file_path: Path, level: str, load_cube: bool = True):
    """Generic reader for L1B/L1C/L1D/L2A files written by io/writer.py's
    write_level_nc/write_l2a_nc. Returns the same 9-tuple contract
    HypsoBase._load_capture_file expects from the per-level load_* functions
    this replaces: (metadata_vars, metadata_attrs, geometry_vars,
    geometry_attrs, gcp_vars, gcp_attrs, global_metadata, cube_attrs, cube).
    """
    schema = get_schema(level)

    metadata_vars, metadata_attrs = _load_metadata(nc_file_path)
    geometry_vars, geometry_attrs = _load_geometry_root(nc_file_path, schema.has_geometry)
    gcp_vars, gcp_attrs = load_gcp_from_nc_file(nc_file_path)
    global_metadata = _load_global_metadata(nc_file_path)

    cube_attrs = _load_cube_attrs(nc_file_path, schema.product_prefix)
    cube = _load_cube(nc_file_path, schema.product_prefix) if load_cube else None

    return (metadata_vars, metadata_attrs, geometry_vars, geometry_attrs,
            gcp_vars, gcp_attrs, global_metadata, cube_attrs, cube)


def load_l1b_nc(nc_file_path: Path, load_cube: bool = True):
    return load_level_nc(nc_file_path, "L1B", load_cube=load_cube)


def load_l1c_nc(nc_file_path: Path, load_cube: bool = True):
    return load_level_nc(nc_file_path, "L1C", load_cube=load_cube)


def load_l1d_nc(nc_file_path: Path, load_cube: bool = True):
    return load_level_nc(nc_file_path, "L1D", load_cube=load_cube)


def load_l2a_nc(nc_file_path: Path, load_cube: bool = True):
    """L2A's product variable name is AC-tool-specific (l2_variable_name), not
    a fixed schema constant - see _discover_product_variables. A written L2A
    file holds exactly one correction's output, matching the original
    load_l2a_nc's per-file (not per-correction) contract."""
    return load_level_nc(nc_file_path, "L2A", load_cube=load_cube)
