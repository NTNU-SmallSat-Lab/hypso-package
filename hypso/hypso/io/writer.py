"""Generic, schema-driven NetCDF level writer, replacing the previous one-writer-
file-per-level design (write/l1b_nc_writer.py, l1c_nc_writer.py, l1d_nc_writer.py,
l2a_nc_writer.py - confirmed near-duplicates of each other differing mostly by
find/replace of the level name and product variable name, see
REFACTOR_PROGRESS.md).

Structural layout differs from those in one deliberate way: products (Lt_*/
rhot_*/rho_w_*) and geometry (latitude/longitude/crs_wgs84/angles) now live at
the ROOT group instead of nested products/ and geometry/ groups - see
io/cf.py's geolocation_ref_attrs() docstring for why (CF's coordinates/
grid_mapping group-relative resolution only walks up to ancestors, never
sideways between siblings, so /products and /geometry as siblings could never
resolve). metadata/* stays nested - it's provenance/bookkeeping, not spatial
data needing CF resolution.

write_products_nc_file/write/products_writer.py (the user-generated/AC-product
-facing writer, tied to the products/_products property) is intentionally
untouched by this refactor - not migrated here, per the user's explicit
instruction to leave that property alone.
"""
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import netCDF4 as nc

# hypso.write.* imports are deliberately deferred into the functions that use
# them (below), not done at module level: hypso/write/__init__.py imports
# write_l1b_nc_file/etc *from this module*, so an eager top-level import here
# would try to fully execute hypso.write's __init__ (importing any of its
# submodules does that) while hypso.io.writer is itself still mid-import,
# causing a circular ImportError. By call time this module is fully loaded,
# so the cycle doesn't bite.

from . import cf
from .schema import LevelSchema, get_schema

logger = logging.getLogger(__name__)

COMP_SCHEME = 'zlib'
COMP_LEVEL = 4
COMP_SHUFFLE = True

_ADCS_FIELDS = (
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "quaternion_s", "quaternion_x", "quaternion_y", "quaternion_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
)

# (source cube/coeff attr, dimension name, output variable name, attr holding the
# authoritative length) - each entry independently optional (not every capture has
# e.g. an unbinned spectral set), matching the original's per-block try/except.
_CORRECTIONS_ARRAYS = (
    ("spectral_coeffs", "bands", "wavelengths", "wavelengths"),
    ("spectral_coeffs_unbinned", "bands_unbinned", "wavelengths_unbinned", "wavelengths_unbinned"),
    ("spectral_coeffs", "specrows", "spec_coeffs", "spectral_coeffs"),
    ("spectral_coeffs_unbinned", "specrows_unbinned", "spec_coeffs_unbinned", "spectral_coeffs_unbinned"),
)


def _write_metadata_common(satobj, netfile: nc.Dataset) -> None:
    """metadata/capture_config, timing, adcs, corrections - the group/attr/
    dimension boilerplate that was previously duplicated verbatim across every
    per-level writer file."""
    from hypso.write.utils import set_or_create_attr

    netfile.createGroup('metadata')

    meta_capcon = netfile.createGroup('metadata/capture_config')
    for md, val in satobj.metadata.capture_config.attrs.items():
        set_or_create_attr(meta_capcon, md, val)

    meta_timing = netfile.createGroup('metadata/timing')
    for md, val in satobj.metadata.timing.attrs.items():
        set_or_create_attr(meta_timing, md, val)

    meta_adcs = netfile.createGroup('metadata/adcs')
    for md, val in satobj.metadata.adcs.attrs.items():
        set_or_create_attr(meta_adcs, md, val)

    meta_corrections = netfile.createGroup('metadata/corrections')
    for md, val in satobj.metadata.corrections.attrs.items():
        set_or_create_attr(meta_corrections, md, val)

    # ADCS sample variables - one shared code path instead of 13 duplicated blocks.
    len_timestamps = satobj.metadata.dimensions["adcssamples"]
    netfile.createDimension('adcssamples', len_timestamps)

    ts_var = netfile.createVariable('metadata/adcs/timestamps', 'f8', ('adcssamples',),
                                     compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
    ts_var[:] = satobj.metadata.adcs.vars["timestamps"][:]

    for field in _ADCS_FIELDS:
        var = netfile.createVariable(f'metadata/adcs/{field}', 'f8', ('adcssamples',),
                                      compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
        var[:] = satobj.metadata.adcs.vars[field][:]

    # Rad calibration matrix.
    len_radrows, len_radcols = satobj.rad_coeffs.shape
    netfile.createDimension('radrows', len_radrows)
    netfile.createDimension('radcols', len_radcols)
    rad_var = netfile.createVariable('metadata/corrections/rad_matrix', 'f4', ('radrows', 'radcols'),
                                      compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
    rad_var[:] = satobj.rad_coeffs

    # Wavelengths / spectral coeffs. `dim_name not in netfile.dimensions` guards
    # against a latent duplicate-dimension bug in the original (it unconditionally
    # re-created the already-existing top-level 'bands' dimension here) without
    # changing the written values - same dimension, same size, just reused instead
    # of re-declared.
    # Written as f8, not the f4 the original used: these are the band-center
    # wavelengths the lazy SpectralResponse rebuild (HypsoCapture.spectral_response)
    # regenerates SRFs from after loading a file - f4's ~3e-5 nm rounding was
    # enough to shift a few Gaussian grid-snap indices, making the rebuilt SRF
    # differ from the in-session one. Full precision makes the round trip exact.
    for attr_name, dim_name, var_name, length_attr in _CORRECTIONS_ARRAYS:
        try:
            values = getattr(satobj, attr_name)
            length = getattr(satobj, length_attr).shape[0]
            if dim_name not in netfile.dimensions:
                netfile.createDimension(dim_name, length)
            var = netfile.createVariable(f'metadata/corrections/{var_name}', 'f8', (dim_name,),
                                          compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
            var[:] = values
        except Exception:
            logger.debug("Skipping optional metadata/corrections/%s (not available for this capture)",
                         var_name, exc_info=True)

    # Timestamps - uncompressed in the original (no compression params were ever
    # passed for this one variable); kept as-is, not a confirmed bug.
    timestamps_var = netfile.createVariable('metadata/timing/timestamps', 'f8', ('lines',))
    timestamps_var[:] = satobj.metadata.timing.vars["timestamps"][:]


def _write_var(netfile, name, data, attrs, dtype='f4', dims=('lines', 'samples')):
    # dims defaults to swath (lines, samples) - callers writing a schema's
    # geometry/product variables should pass schema.spatial_dims explicitly so
    # a future gridded schema (e.g. ("lat", "lon")) is honored; see
    # schema.py's spatial_dims docstring.
    var = netfile.createVariable(name, dtype, dims,
                                  compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
    var[:] = data
    for k, v in attrs.items():
        setattr(var, k, v)
    return var


# (output name, satobj source attr, cf attribute builder, long_name)
_INDIRECT_ANGLE_FIELDS = (
    ("sensor_zenith", "sat_zenith_angles", cf.zenith_angle_attrs, "Sensor Zenith Angle"),
    ("sensor_azimuth", "sat_azimuth_angles", cf.azimuth_angle_attrs, "Sensor Azimuth Angle"),
    ("solar_zenith", "solar_zenith_angles", cf.zenith_angle_attrs, "Solar Zenith Angle"),
    ("solar_azimuth", "solar_azimuth_angles", cf.azimuth_angle_attrs, "Solar Azimuth Angle"),
    ("relative_azimuth", "relative_azimuth_angles", cf.azimuth_angle_attrs, "Relative Azimuth Angle"),
)
_DIRECT_ANGLE_FIELDS = (
    ("sensor_zenith_direct", "sat_zenith_angles_direct", cf.zenith_angle_attrs, "Sensor Zenith Angle (Indirect)"),
    ("sensor_azimuth_direct", "sat_azimuth_angles_direct", cf.azimuth_angle_attrs, "Sensor Azimuth Angle (Indirect)"),
    ("solar_zenith_direct", "solar_zenith_angles_direct", cf.zenith_angle_attrs, "Solar Zenith Angle (Indirect)"),
    ("solar_azimuth_direct", "solar_azimuth_angles_direct", cf.azimuth_angle_attrs, "Solar Azimuth Angle (Indirect)"),
    ("relative_azimuth_direct", "relative_azimuth_angles_direct", cf.azimuth_angle_attrs, "Relative Azimuth Angle (Indirect)"),
)


def _write_geometry_root(satobj, netfile: nc.Dataset, schema: LevelSchema) -> None:
    """latitude/longitude/crs_wgs84/angle variables at the ROOT group (see
    module docstring). Reimplements write/geometry_group_writer.py, fixing:
    units="degrees" -> degrees_north/degrees_east (CF requires the directional
    form for a variable to be recognized as a latitude/longitude coordinate),
    blanket valid_min/valid_max=[-180,180] on every variable including latitude
    and zenith angles (physically [-90,90] and [0,180] respectively) -> the
    differentiated ranges in io/cf.py, and actually applies the compression
    params (previously received but commented out, so geometry data was
    written uncompressed despite COMP_SCHEME/LEVEL/SHUFFLE being passed in).

    Variable *names* (including the "_direct" suffix meaning indirect
    georeferencing - a pre-existing naming inconsistency, not introduced here)
    are kept exactly as the original to avoid silently changing what
    downstream readers key off of.
    """
    has_indirect = (getattr(satobj, 'latitudes', None) is not None
                     and getattr(satobj, 'longitudes', None) is not None)
    has_direct = (getattr(satobj, 'latitudes_direct', None) is not None
                  and getattr(satobj, 'longitudes_direct', None) is not None)

    if has_indirect:
        try:
            _write_var(netfile, 'latitude', satobj.latitudes, cf.latitude_attrs(), dims=schema.spatial_dims)
            _write_var(netfile, 'longitude', satobj.longitudes, cf.longitude_attrs(), dims=schema.spatial_dims)
        except Exception as ex:
            logger.error("Unable to write latitude/longitude to NetCDF file. The file may be "
                         "incomplete. Please run direct or indirect georeferencing. (%s)", ex)

    if has_direct:
        try:
            _write_var(netfile, 'latitude_direct', satobj.latitudes_direct,
                       {**cf.latitude_attrs(), "long_name": "Latitude (Indirect)"}, dims=schema.spatial_dims)
            _write_var(netfile, 'longitude_direct', satobj.longitudes_direct,
                       {**cf.longitude_attrs(), "long_name": "Longitude (Indirect)"}, dims=schema.spatial_dims)
        except Exception as ex:
            logger.error("Unable to write indirect latitude/longitude to NetCDF file. (%s)", ex)

    if has_indirect:
        try:
            crs_var = netfile.createVariable('crs_wgs84', 'i4', ())
            for k, v in cf.crs_wgs84_attrs().items():
                setattr(crs_var, k, v)
        except Exception:
            logger.debug("Skipping crs_wgs84 variable", exc_info=True)

        for name, source_attr, builder, long_name in _INDIRECT_ANGLE_FIELDS:
            try:
                _write_var(netfile, name, getattr(satobj, source_attr), builder(long_name), dims=schema.spatial_dims)
            except Exception:
                logger.debug("Skipping optional geometry variable %s", name, exc_info=True)

    if has_direct:
        for name, source_attr, builder, long_name in _DIRECT_ANGLE_FIELDS:
            try:
                _write_var(netfile, name, getattr(satobj, source_attr), builder(long_name), dims=schema.spatial_dims)
            except Exception:
                logger.debug("Skipping optional geometry variable %s", name, exc_info=True)


def _write_products(satobj, netfile: nc.Dataset, schema: LevelSchema, cube: np.ndarray,
                     wavelengths: np.ndarray, fwhm: np.ndarray, datacube: bool) -> None:
    """Product variable(s) at the ROOT group, named `<prefix>` (single 3D cube)
    or `<prefix>_<wave>` (one 2D variable per band - the SNAP-BEAM convention,
    see REFACTOR_PROGRESS.md's SNAP/CF research section for why this layout is
    kept rather than switched to a stacked (band,y,x) array)."""
    rounded_wavelengths = np.around(wavelengths, 1)

    if datacube:
        var = netfile.createVariable(
            schema.product_prefix, 'f4', schema.spatial_dims + ('bands',),
            compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
        var.units = schema.product_units
        var.long_name = schema.product_long_name
        var.wavelength_units = "nanometers"
        var.fwhm = fwhm
        var.wavelengths = rounded_wavelengths
        if schema.has_geometry:
            for k, v in cf.geolocation_ref_attrs().items():
                setattr(var, k, v)
        var[:] = cube
        return

    for band in range(cube.shape[-1]):
        wave = float(rounded_wavelengths[band])
        wave_name = str(int(wave))
        name = f"{schema.product_prefix}_{wave_name}"

        attrs = cf.band_attrs(
            long_name=f"{schema.product_long_name} Band {band} ({wave_name} nm)",
            units=schema.product_units,
            wavelength_nm=wave,
            radiation_wavelength_nm=float(wavelengths[band]),
            fwhm=float(fwhm[band]),
            wave_name=wave_name,
            band_index=band,
            include_geolocation=schema.has_geometry,
        )
        var = netfile.createVariable(
            name, 'f4', schema.spatial_dims,
            compression=COMP_SCHEME, complevel=COMP_LEVEL, shuffle=COMP_SHUFFLE)
        for k, v in attrs.items():
            setattr(var, k, v)
        var[:] = cube[:, :, band]


def _write_level_nc(satobj, schema: LevelSchema, dst_nc: str, cube: np.ndarray,
                     wavelengths: np.ndarray, fwhm: np.ndarray, datacube: bool = False,
                     write_srf: bool = False) -> None:
    """Shared implementation behind write_level_nc/write_l2a_nc. `cube` is
    already resolved by the caller (masking / AC-correction selection is a
    caller concern, not this function's)."""
    from hypso.write.utils import set_or_create_attr
    from hypso.write.calibration_filenames_writer import calibration_filenames_writer
    from hypso.write.metadata_gcp_group_writer import metadata_gcp_group_writer
    from hypso.write.metadata_srf_group_writer import metadata_srf_group_writer

    with nc.Dataset(dst_nc, 'w', format='NETCDF4') as netfile:
        lines = satobj.metadata.capture_config.attrs["frame_count"]
        samples = satobj.image_height
        bands = cube.shape[-1]

        for md, val in satobj.metadata.global_attrs.items():
            set_or_create_attr(netfile, md, val)
        for k, v in cf.global_attrs(schema.processing_level, schema.title).items():
            set_or_create_attr(netfile, k, v)
        set_or_create_attr(netfile, "processing_level", schema.processing_level)

        # CF wants "history" to be an append-only provenance trail, not a
        # value overwritten on every write - carry forward whatever the
        # source capture already had (e.g. an L1A file's own ground-station
        # history) and add one line for this write, rather than letting
        # cf.global_attrs()'s dict (applied above) clobber it.
        existing_history = satobj.metadata.global_attrs.get("history", "")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_entry = f"{timestamp}: {schema.processing_level} product generated by hypso-package"
        history = f"{existing_history}\n{new_entry}" if existing_history else new_entry
        set_or_create_attr(netfile, "history", history)

        calibration_filenames_writer(satobj=satobj, netfile=netfile)

        # Dimension *names* come from schema.spatial_dims (every level implemented
        # today keeps the ("lines", "samples") default - see its docstring for why
        # this indirection exists). Dimension *sizes* are still swath-specific
        # (satobj's frame_count/image_height); a future gridded schema would need
        # its own size source here too, not just different names.
        netfile.createDimension(schema.spatial_dims[0], lines)
        netfile.createDimension(schema.spatial_dims[1], samples)
        netfile.createDimension('bands', bands)

        _write_metadata_common(satobj, netfile)

        _write_products(satobj, netfile, schema, cube, wavelengths, fwhm, datacube=datacube)

        if schema.has_geometry:
            _write_geometry_root(satobj, netfile, schema)

        metadata_gcp_group_writer(satobj, netfile, COMP_SCHEME=COMP_SCHEME, COMP_LEVEL=COMP_LEVEL,
                                   COMP_SHUFFLE=COMP_SHUFFLE)

        if write_srf:
            metadata_srf_group_writer(satobj, netfile, COMP_SCHEME=COMP_SCHEME, COMP_LEVEL=COMP_LEVEL,
                                       COMP_SHUFFLE=COMP_SHUFFLE)

    return None


def write_level_nc(satobj, level: str, dst_nc: str, datacube: bool = False, masked: bool = False) -> None:
    """Write an L1A/L1B/L1C/L1D file for `satobj` at `dst_nc`. Use write_l2a_nc
    for L2 (AC-correction) output instead - its source cube is keyed by which
    AC tool produced it, not a single fixed satobj attribute."""
    schema = get_schema(level)
    cube_obj = getattr(satobj, f"masked_{schema.source_cube_attr}") if masked else getattr(satobj, schema.source_cube_attr)
    _write_level_nc(satobj, schema, dst_nc, cube=cube_obj.to_numpy(),
                     wavelengths=satobj.wavelengths, fwhm=satobj.fwhm, datacube=datacube,
                     write_srf=(level.upper() == "L1D"))


def _write_level_nc_file(satobj, level: str, file_attr: str, overwrite: bool, masked: bool,
                          skip_message: str, **kwargs) -> None:
    dst_nc = getattr(satobj, file_attr)
    if Path(dst_nc).is_file() and not overwrite:
        if satobj.VERBOSE:
            logger.info(skip_message)
        return None
    write_level_nc(satobj, level, dst_nc, masked=masked, **kwargs)
    return None


def write_l1b_nc_file(satobj, overwrite: bool = False, masked: bool = False, **kwargs) -> None:
    """Thin backward-compat wrapper over write_level_nc - name/signature
    confirmed imported directly by hypso-processing-pipeline, kept stable."""
    return _write_level_nc_file(satobj, "L1B", "l1b_nc_file", overwrite, masked,
                                "L1b NetCDF file has already been generated. Skipping.", **kwargs)


def write_l1c_nc_file(satobj, overwrite: bool = False, masked: bool = False, **kwargs) -> None:
    """Thin backward-compat wrapper over write_level_nc - name/signature
    confirmed imported directly by hypso-processing-pipeline, kept stable."""
    return _write_level_nc_file(satobj, "L1C", "l1c_nc_file", overwrite, masked,
                                "L1c NetCDF file has already been generated. Skipping.", **kwargs)


def write_l1d_nc_file(satobj, label: str = None, overwrite: bool = False, masked: bool = False, **kwargs) -> None:
    """Thin backward-compat wrapper over write_level_nc - name/signature
    confirmed imported directly by hypso-processing-pipeline, kept stable.
    `label` is accepted for signature compatibility but unused, matching the
    original write/l1d_nc_writer.py (it never used it to affect the output
    path either)."""
    return _write_level_nc_file(satobj, "L1D", "l1d_nc_file", overwrite, masked,
                                "L1d NetCDF file has already been generated. Skipping.", **kwargs)


def write_l2a_nc_file(satobj, correction: str = None, overwrite: bool = False, **kwargs):
    """Thin backward-compat wrapper over write_l2a_nc - name/signature confirmed
    imported directly by hypso-processing-pipeline, kept stable. `correction=None`
    writes every currently-populated correction in satobj.l2a_cube, matching the
    original write/l2a_nc_writer.py."""
    corrections = [correction] if correction is not None else list(satobj.l2a_cube.keys())

    l2a_nc_file = None
    for one_correction in corrections:
        l2a_nc_file = Path(satobj.parent_dir, satobj.l2a_name(atmospheric_correction=one_correction))

        if Path(l2a_nc_file).is_file() and not overwrite:
            if satobj.VERBOSE:
                logger.info("L2 NetCDF file has already been generated. Skipping.")
            continue

        write_l2a_nc(satobj, one_correction, str(l2a_nc_file), **kwargs)

    return l2a_nc_file


def write_l2a_nc(satobj, correction: str, dst_nc: str, datacube: bool = False) -> None:
    """Write an L2 (AC-correction) output file. `correction` selects which
    AC tool's result to write (e.g. "polymer", "acolite_l2r", "ocsmart",
    "dps") from satobj.l2a_cube[correction] - same lookup the original
    l2a_nc_writer.py used.

    The product variable name isn't a fixed schema constant: each AC adapter
    sets a distinct `l2_variable_name` cube attribute (e.g. "chla" vs "Rrs"),
    matching the original writer's per-correction variable naming - so it's
    read here and overridden onto L2A_SCHEMA per call.
    """
    schema = get_schema("L2A")
    cube_obj = satobj.l2a_cube[correction]
    try:
        variable_name = cube_obj.attrs['l2_variable_name']
    except Exception:
        logger.warning("No 'l2_variable_name' attribute found for correction=%r. Defaulting to %r.",
                       correction, schema.product_prefix)
        variable_name = schema.product_prefix
    schema = dataclasses.replace(schema, product_prefix=variable_name)

    _write_level_nc(satobj, schema, dst_nc, cube=cube_obj.to_numpy(),
                     wavelengths=satobj.wavelengths, fwhm=satobj.fwhm, datacube=datacube,
                     write_srf=True)
