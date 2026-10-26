"""Build the in-memory product tree (io/product.py) from a capture object
(HypsoCapture and the capture_types classes), so captures are written by
write_product like everything else (REFACTOR_PROGRESS.md x45).

Mirrors io/writer.py's _write_level_nc group by group, so the written file
matches the old writer's output apart from the format-2 additions
(provenance, product_format_version, geospatial bounds, pixel_flags) and
one fix: L1A gets no calibration-file global attributes (it is not
calibrated).

Transitional: once processing steps are Dataset -> Dataset functions
(step 2) they build and annotate the tree themselves, and this module goes
away with HypsoCapture.
"""
import re
from pathlib import Path

import numpy as np
import xarray as xr

from hypso.masks.flags import FLAGS_NAME, encode_flags

from .product import PRODUCT_DIMS, SPATIAL_DIMS
from .provenance import ac_attrs, add_step, calibration_attrs
from .schema import get_schema

_ADCS_FIELDS = (
    "timestamps",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "quaternion_s", "quaternion_x", "quaternion_y", "quaternion_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
)

# (satobj attr, dimension, variable, attr giving the length) - io/writer.py's
# _CORRECTIONS_ARRAYS; each optional.
_CORRECTIONS_ARRAYS = (
    ("spectral_coeffs", "bands", "wavelengths", "wavelengths"),
    ("spectral_coeffs_unbinned", "bands_unbinned", "wavelengths_unbinned", "wavelengths_unbinned"),
    ("spectral_coeffs", "specrows", "spec_coeffs", "spectral_coeffs"),
    ("spectral_coeffs_unbinned", "specrows_unbinned", "spec_coeffs_unbinned", "spectral_coeffs_unbinned"),
)

# Root geometry variable -> satobj attribute (io/writer.py's _INDIRECT/
# _DIRECT_ANGLE_FIELDS plus lat/lon).
_INDIRECT_GEOMETRY = {
    "latitude": "latitudes", "longitude": "longitudes",
    "sensor_zenith": "sat_zenith_angles", "sensor_azimuth": "sat_azimuth_angles",
    "solar_zenith": "solar_zenith_angles", "solar_azimuth": "solar_azimuth_angles",
    "relative_azimuth": "relative_azimuth_angles",
}
_DIRECT_GEOMETRY = {
    "latitude_direct": "latitudes_direct", "longitude_direct": "longitudes_direct",
    "sensor_zenith_direct": "sat_zenith_angles_direct", "sensor_azimuth_direct": "sat_azimuth_angles_direct",
    "solar_zenith_direct": "solar_zenith_angles_direct", "solar_azimuth_direct": "solar_azimuth_angles_direct",
    "relative_azimuth_direct": "relative_azimuth_angles_direct",
}

_CALIBRATION_FILES = (("radiometric", "rad_coeff_file"), ("smile", "smile_coeff_file"),
                      ("destriping", "destriping_coeff_file"), ("spectral", "spectral_coeff_file"))


def _attr(value):
    # hypso.write.utils.set_or_create_attr's conversions, which the old writer
    # applied to every metadata/global attribute.
    if isinstance(value, bool):
        return 1 if value else 0
    return "" if value is None else value


def _attrs(attrs: dict) -> dict:
    return {k: _attr(v) for k, v in (attrs or {}).items()}


def _calibration_filename_attrs(satobj) -> dict:
    # hypso.write.calibration_filenames_writer: the coefficient file's name,
    # overridden by the loaded file's own attribute, str() of either (so
    # "None" when neither exists).
    out = {}
    for kind, attr in _CALIBRATION_FILES:
        value = None
        try:
            value = Path(getattr(satobj, attr)).name
        except (AttributeError, TypeError):
            pass
        value = satobj.metadata.global_attrs.get(f"{kind}_file", value)
        out[f"{kind}_file"] = str(value)
    return out


def _metadata_nodes(satobj) -> dict:
    md = satobj.metadata
    nodes = {
        "metadata/capture_config": xr.Dataset(attrs=_attrs(md.capture_config.attrs)),
        "metadata/timing": xr.Dataset(
            {"timestamps": ("lines", np.asarray(md.timing.vars["timestamps"][:], dtype="f8"))},
            attrs=_attrs(md.timing.attrs)),
        "metadata/adcs": xr.Dataset(
            {f: ("adcssamples", np.asarray(md.adcs.vars[f][:], dtype="f8")) for f in _ADCS_FIELDS},
            attrs=_attrs(md.adcs.attrs)),
    }

    corrections = {"rad_matrix": (("radrows", "radcols"), np.asarray(satobj.rad_coeffs, dtype="f4"))}
    for attr, dim, name, length_attr in _CORRECTIONS_ARRAYS:
        try:
            values = np.asarray(getattr(satobj, attr), dtype="f8")
            if values.shape[0] != getattr(satobj, length_attr).shape[0]:
                continue
            corrections[name] = (dim, values)
        except Exception:
            continue   # not available for this capture
    nodes["metadata/corrections"] = xr.Dataset(corrections, attrs=_attrs(md.corrections.attrs))

    gcp_vars = {f"gcp_{k}": ("gcps", np.asarray(v, dtype="f4")) for k, v in (md.gcp.vars or {}).items()}
    nodes["metadata/gcp"] = xr.Dataset(gcp_vars, attrs=_attrs(md.gcp.attrs or {}))
    return nodes


def _srf_node(satobj) -> xr.Dataset:
    # hypso.write.metadata_srf_group_writer, variable for variable.
    from hypso.reflectance import earth_sun_distance

    data, attrs = {}, {}
    for attr, name in (("esun", "esun"), ("esun_wl", "esun_wavelengths"),
                       ("effective_fwhm", "effective_fwhm"),
                       ("effective_fwhm_unbinned", "effective_fwhm_unbinned")):
        value = getattr(satobj, attr, None)
        if value is not None:
            data[name] = xr.Variable((name,), np.asarray(value, dtype="f8"))
    if "esun" in data:
        data["esun"].attrs = {
            "units": "mW m-2 nm-1",
            "description": ("Band-averaged (SRF-weighted) solar irradiance F0 at 1 AU, from the "
                            "TSIS-1 HSRS v2 (2022-11-30); the F0 used for this file's TOA reflectance")}
        if getattr(satobj, "iso_time", None) is not None:
            attrs["earth_sun_distance_au"] = earth_sun_distance(satobj.iso_time)

    if getattr(satobj, "srf", None) is not None and getattr(satobj, "srf_ssi_wl", None) is not None:
        grid = np.asarray(satobj.srf_ssi_wl)
        for index, wavelength in enumerate(satobj.wavelengths):
            band = f"Band_{index}"
            response = satobj.srf[index, :].toarray().flatten()
            nonzero = response > 0
            values, wl = (response[nonzero], grid[nonzero]) if np.any(nonzero) else (response, grid)
            dim = f"wav_{band}"
            data[dim] = xr.Variable((dim,), wl, {"units": "nm"})
            data[band] = xr.Variable((dim,), values, {
                "band_info": band, "band_wavelength": float(wavelength), "index": index,
                "effective_fwhm": float(satobj.effective_fwhm[index]),
                "center_fwhm": float(satobj.fwhm[index])})
    return xr.Dataset(data, attrs=attrs)


def _flags(satobj):
    masks, descriptions = {}, {}
    for name, mask in (("land", getattr(satobj, "_land_mask", None)),
                       ("cloud", getattr(satobj, "_cloud_mask", None))):
        if mask is not None:
            masks[name] = mask
    for name, mask in (getattr(satobj, "_custom_masks", None) or {}).items():
        masks[re.sub(r"[^A-Za-z0-9_]", "_", str(name))] = mask
    if not masks:
        return None
    for name, mask in masks.items():
        description = getattr(mask, "attrs", {}).get("description")
        if description:
            descriptions[name] = description
        masks[name] = np.asarray(mask, dtype=bool)
    return encode_flags(masks, descriptions)


def capture_to_tree(satobj, level: str, correction: str = None, masked: bool = False) -> xr.DataTree:
    """The product tree for `level` ("L1A" ... "L1D", or "L2A" with
    `correction`, e.g. "polymer"). masked=True writes the NaN-masked cube
    (legacy behaviour, until AC staging applies pixel_flags itself); the
    flags are added either way."""
    level = level.upper()
    schema = get_schema(level)
    if level == "L2A":
        cube = satobj.l2a_cube[correction]
        prefix = cube.attrs.get("l2_variable_name", schema.product_prefix)
    else:
        source = f"masked_{schema.source_cube_attr}" if masked else schema.source_cube_attr
        cube = getattr(satobj, source)
        prefix = schema.product_prefix
    wavelengths = np.asarray(satobj.wavelengths, dtype=float)
    root = xr.Dataset(
        {prefix: (PRODUCT_DIMS, np.asarray(cube))},
        coords={"wavelength": ("bands", np.around(wavelengths, 1)),
                "radiation_wavelength": ("bands", wavelengths),
                "fwhm": ("bands", np.asarray(satobj.fwhm, dtype=float))})

    if schema.has_geometry:
        for table, lat, lon in ((_INDIRECT_GEOMETRY, "latitudes", "longitudes"),
                                (_DIRECT_GEOMETRY, "latitudes_direct", "longitudes_direct")):
            # io/writer.py writes each set only when its lat and lon both exist.
            if getattr(satobj, lat, None) is None or getattr(satobj, lon, None) is None:
                continue
            for name, attr in table.items():
                value = getattr(satobj, attr, None)
                if value is not None:
                    root[name] = (SPATIAL_DIMS, np.asarray(value, dtype="f4"))

    flags = _flags(satobj)
    if flags is not None:
        root[FLAGS_NAME] = flags

    attrs = _attrs(satobj.metadata.global_attrs)
    if level != "L1A":
        attrs.update(_calibration_filename_attrs(satobj))
    attrs["processing_level"] = schema.processing_level
    root.attrs = attrs

    tree = xr.DataTree(root)
    for path, ds in _metadata_nodes(satobj).items():
        tree[path] = xr.DataTree(ds)
    if level in ("L1C", "L1D", "L2A"):
        tree["metadata/srf"] = xr.DataTree(_srf_node(satobj))

    _record_step(tree, satobj, level, correction, masked)
    return tree


def _record_step(tree, satobj, level, correction, masked) -> None:
    parameters = {"masked": masked}
    for name in ("coeff_type",):
        if getattr(satobj, name, None) is not None:
            parameters[name] = getattr(satobj, name)
    if level in ("L1C", "L1D", "L2A"):
        parameters["georeferencing"] = ("indirect" if getattr(satobj, "latitudes", None) is not None
                                        else "direct" if getattr(satobj, "latitudes_direct", None) is not None
                                        else "none")
    sr = getattr(satobj, "_spectral_response", None)
    if level == "L1D" and sr is not None:
        parameters.update(ssi_source=sr.ssi_source, srf_grid=sr.grid, bin_factor=int(sr.bin_factor))

    calibration = None
    if level != "L1A" and getattr(satobj, "sat_id", None):
        files = {kind: getattr(satobj, attr, None) for kind, attr in _CALIBRATION_FILES}
        if any(files.values()):
            package = "hypso{}_calibration".format(str(satobj.sat_id).split("-")[-1])
            calibration = calibration_attrs(package, files)

    step = level.lower() if level != "L2A" else f"l2a_{correction}"
    ac = ac_attrs(correction) if level == "L2A" else None
    add_step(tree, step, input_file=getattr(satobj, "path", None), parameters=parameters,
             calibration=calibration, ac=ac)
