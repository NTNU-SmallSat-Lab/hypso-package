"""Tests for hypso.io.metadata.CaptureMetadata - consolidates the 23
scattered nc_* attributes (nc_adcs_vars, nc_capture_config_attrs, ...
nc_dimensions, nc_attrs, nc_cube_attrs) into one satobj.metadata, built once
during load (hypso.io.dispatch.load_capture_file). Confirmed zero external
readers of any of the 23 original flat names in hypso-processing-pipeline."""
import pytest

from conftest import requires_real_capture

pytestmark = requires_real_capture


def test_metadata_populated_after_load(satobj):
    from hypso.io.metadata import CaptureMetadata, MetadataGroup

    assert isinstance(satobj.metadata, CaptureMetadata)

    for group_name in ("adcs", "capture_config", "corrections", "database",
                       "logfiles", "temperature", "timing", "srf", "geometry", "gcp"):
        group = getattr(satobj.metadata, group_name)
        assert isinstance(group, MetadataGroup), group_name

    assert isinstance(satobj.metadata.dimensions, dict)
    assert isinstance(satobj.metadata.global_attrs, dict)
    assert isinstance(satobj.metadata.cube_attrs, dict)


def test_metadata_capture_config_matches_derived_attributes(satobj):
    # set_hypso_attributes derives satobj.frame_count/row_count/etc. from
    # satobj.metadata.capture_config.attrs - spot check they still agree
    # post-migration from the old flat nc_capture_config_attrs.
    attrs = satobj.metadata.capture_config.attrs
    assert satobj.frame_count == attrs["frame_count"]
    assert satobj.row_count == attrs["row_count"]
    assert satobj.column_count == attrs["column_count"]
    assert satobj.bin_factor == attrs["bin_factor"]


def test_metadata_adcs_vars_used_by_georeferencing(satobj):
    # geo.run_frame_interpolation reads satobj.metadata.adcs.vars /
    # satobj.metadata.timing.vars / satobj.metadata.capture_config.attrs -
    # confirm they're non-empty and shaped as that code expects.
    assert satobj.metadata.adcs.vars
    assert "framerate" in satobj.metadata.capture_config.attrs
    assert "exposure" in satobj.metadata.capture_config.attrs
