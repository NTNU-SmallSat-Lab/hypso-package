"""Consolidates the 23 scattered nc_* attributes HypsoCapture used to carry
(nc_adcs_vars, nc_adcs_attrs, nc_capture_config_vars, ... nc_dimensions,
nc_attrs, nc_cube_attrs) into one satobj.metadata: CaptureMetadata instance,
built once during load (hypso.io.dispatch). Confirmed zero external readers
of any of the 23 original names in hypso-processing-pipeline - safe to
restructure freely, no compatibility properties kept.

Each named group (adcs/capture_config/corrections/database/logfiles/
temperature/timing/srf/geometry/gcp) keeps its vars/attrs dicts verbatim in
a MetadataGroup rather than exploding every key into its own dataclass
field - these are netCDF variable/attribute blobs of varying shape per
sensor/capture, not a small fixed record, so a dict is what the internal
readers (calibration.pipeline, georeferencing.geo, hypso.write.*,
hypso.io.writer) actually need: e.g. satobj.metadata.corrections.attrs
['radiometric_coefficients_version'] where satobj.nc_corrections_attrs
['radiometric_coefficients_version'] used to be.
"""
from dataclasses import dataclass, field


@dataclass
class MetadataGroup:
    vars: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


@dataclass
class CaptureMetadata:
    adcs: MetadataGroup
    capture_config: MetadataGroup
    corrections: MetadataGroup
    database: MetadataGroup
    logfiles: MetadataGroup
    temperature: MetadataGroup
    timing: MetadataGroup
    srf: MetadataGroup
    geometry: MetadataGroup
    gcp: MetadataGroup
    dimensions: dict = field(default_factory=dict)
    global_attrs: dict = field(default_factory=dict)
    cube_attrs: dict = field(default_factory=dict)

    @classmethod
    def from_load_result(cls, nc_metadata_vars: dict, nc_metadata_attrs: dict,
                         nc_geometry_vars: dict, nc_geometry_attrs: dict,
                         nc_gcp_vars: dict, nc_gcp_attrs: dict,
                         nc_global_metadata: dict, nc_cube_attrs: dict) -> "CaptureMetadata":
        """Builds one CaptureMetadata from exactly what each load_* function
        (hypso.load) returns - see hypso.io.dispatch.load_capture_file, the
        one caller."""
        groups = {
            name: MetadataGroup(vars=nc_metadata_vars[name], attrs=nc_metadata_attrs[name])
            for name in ("adcs", "capture_config", "corrections", "database",
                        "logfiles", "temperature", "timing", "srf")
        }
        return cls(
            **groups,
            geometry=MetadataGroup(vars=nc_geometry_vars, attrs=nc_geometry_attrs),
            gcp=MetadataGroup(vars=nc_gcp_vars, attrs=nc_gcp_attrs),
            dimensions=nc_global_metadata["dimensions"],
            global_attrs=nc_global_metadata["ncattrs"],
            cube_attrs=nc_cube_attrs,
        )
