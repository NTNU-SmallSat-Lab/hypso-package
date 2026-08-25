"""Satpy FileHandler for HYPSO L1C/L1D NetCDF files (as written by
hypso.io.writer.write_l1c_nc_file/write_l1d_nc_file - the flat root-group
layout: per-band Lt_<wave>/rhot_<wave> variables and latitude/longitude/
crs_wgs84 all as root-level siblings, see hypso.io.writer's module docstring).

Registered via the [project.entry-points."satpy.readers"] entry in
hypso/pyproject.toml pointing at the hypso.satpy module - Satpy looks for
etc/readers/*.yaml next to that module (see etc/readers/hypso_l1c.yaml and
hypso_l1d.yaml, which reference HypsoL1FileHandler by file_reader).

Only L1C and L1D are covered this pass (both have geometry - the case
visualization/compositing needs; L1B is pre-georeferencing and isn't a
sensible standalone visualization target). See REFACTOR_PROGRESS.md for the
approved plan and why L1B/L2A are left as a follow-on.
"""
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.core.netcdf import NetCDF4FileHandler
from satpy.dataset.dataid import WavelengthRange

from hypso.io.reader import list_band_datasets

# file_type (from the matching reader YAML) -> product variable prefix. Both
# levels currently write in per-band mode ("Lt_378" etc, not a single
# datacube variable) when produced via write_l1c_nc_file/write_l1d_nc_file's
# default datacube=False - see io/writer.py's write_level_nc.
_PRODUCT_PREFIX_BY_FILE_TYPE = {
    "hypso_l1c_nc": "Lt",
    "hypso_l1d_nc": "rhot",
}


class HypsoL1FileHandler(NetCDF4FileHandler):
    """One handler class shared by the hypso_l1c/hypso_l1d readers - which
    level is being read is implied by self.filetype_info['file_type']
    (set by Satpy from whichever reader YAML matched this file), used to look
    up the right product variable prefix (_PRODUCT_PREFIX_BY_FILE_TYPE).
    """

    def __init__(self, filename, filename_info, filetype_info):
        super().__init__(filename, filename_info, filetype_info)

        sat_id = self["/attr/sat_id"]
        # Matches hypso.sensors.hypso1.HYPSO1_PROFILE.key / hypso2's .key -
        # NOT SensorProfile.sensor ("hypso1_hsi"/"hypso2_hsi", a different,
        # already-used string) - this is what composites/hypso1.yaml's
        # `sensor_name: hypso1` is matched against.
        self.sensor = "hypso1" if sat_id == "HYPSO-1" else "hypso2"
        self.platform_name = sat_id

        self._product_prefix = _PRODUCT_PREFIX_BY_FILE_TYPE[filetype_info["file_type"]]

    @property
    def start_time(self) -> datetime:
        return self.filename_info["start_time"]

    @property
    def end_time(self) -> datetime:
        return self.filename_info.get("end_time", self.start_time)

    def available_datasets(self, configured_datasets=None):
        """Pass through the statically-configured lat/lon entries (from the
        reader YAML) unchanged, then dynamically yield one dataset per band
        found in this file (see list_band_datasets - the exact band set
        varies by binning/calibration config, so it can't be enumerated in
        the YAML ahead of time, unlike a fixed-band instrument)."""
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info["file_type"]), ds_info

        file_type = self.filetype_info["file_type"]
        for band in list_band_datasets(self.filename, self._product_prefix):
            if band["band"] is None:
                # Single-datacube-mode file (datacube=True at write time) -
                # not handled dynamically here; the whole cube would need a
                # 3D dataset entry, which this pass doesn't build. Skip
                # rather than yield something get_dataset() can't serve.
                continue
            fwhm = band["fwhm"]
            center_um = band["wavelength"] / 1000.0
            half_fwhm_um = (fwhm / 2.0) / 1000.0
            yield True, {
                "name": band["name"],
                "file_type": file_type,
                "wavelength": WavelengthRange(
                    min=center_um - half_fwhm_um,
                    central=center_um,
                    max=center_um + half_fwhm_um,
                    unit="um",
                ),
                "units": band["units"],
                "standard_name": band["long_name"],
                "coordinates": ("longitude", "latitude"),
                "band_index": band["band"],
                "radiation_wavelength_nm": band["radiation_wavelength"],
            }

    def get_dataset(self, dataset_id, ds_info):
        name = ds_info["name"]
        if name not in self:
            return None

        data = self[name]
        if "lines" in data.dims or "samples" in data.dims:
            data = data.rename({d: r for d, r in (("lines", "y"), ("samples", "x")) if d in data.dims})

        data.attrs.update(ds_info)
        data.attrs["sensor"] = self.sensor
        data.attrs["platform_name"] = self.platform_name

        return data
