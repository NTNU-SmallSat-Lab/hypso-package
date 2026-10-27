"""Satpy FileHandler for HYPSO L1C/L1D NetCDF files (as written by
hypso.io.product.write_product (Product.write) - the flat root-group
layout: per-band Lt_<wave>/rhot_<wave> variables and latitude/longitude/
crs_wgs84 all as root-level siblings, see hypso.io.writer's module docstring).

Registered via the [project.entry-points."satpy.readers"] entry in
hypso/pyproject.toml pointing at the hypso.satpy module - Satpy looks for
etc/readers/*.yaml next to that module (see etc/readers/hypso_l1c.yaml and
hypso_l1d.yaml, which reference HypsoL1FileHandler by file_reader).

Only L1C and L1D are covered (both have geometry, which visualization and
compositing need; L1B is pre-georeferencing). L2A and gridded products are
not yet covered; Product.to_satpy() works for any georeferenced product.
"""
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.core.netcdf import NetCDF4FileHandler
from satpy.dataset.dataid import WavelengthRange

from hypso.io.product import list_band_variables

class HypsoL1FileHandler(NetCDF4FileHandler):
    """One handler class shared by the hypso_l1c, hypso_l1d and hypso_l2a
    readers: the bands are discovered from the file itself, whatever the
    cube variable (Lt, rhot, or an L2A method's Rrs, rho_w, rhos)."""

    def __init__(self, filename, filename_info, filetype_info):
        super().__init__(filename, filename_info, filetype_info)

        sat_id = self["/attr/sat_id"]
        # Matches hypso.sensors.hypso1.HYPSO1_PROFILE.key / hypso2's .key -
        # NOT SensorProfile.sensor ("hypso1_hsi"/"hypso2_hsi", a different,
        # already-used string). This is also the name Satpy would match a
        # `sensor_name:` against if per-sensor composite/enhancement YAMLs are
        # ever added.
        self.sensor = "hypso1" if sat_id == "HYPSO-1" else "hypso2"
        self.platform_name = sat_id

        # A regridded product (hypso.grid) is on a map grid: x/y coordinates
        # and a CF `crs` variable. Its bands get get_area_def()'s
        # AreaDefinition instead of a swath from latitude/longitude.
        self.gridded = "crs" in self and "x" in self and "y" in self

    @property
    def start_time(self) -> datetime:
        return self.filename_info["start_time"]

    @property
    def end_time(self) -> datetime:
        return self.filename_info.get("end_time", self.start_time)

    def available_datasets(self, configured_datasets=None):
        """Pass through the statically-configured lat/lon entries (from the
        reader YAML) unchanged, then dynamically yield one dataset per band
        found in this file (see list_band_variables - the exact band set
        varies by binning/calibration config, so it can't be enumerated in
        the YAML ahead of time, unlike a fixed-band instrument)."""
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info["file_type"]), ds_info

        file_type = self.filetype_info["file_type"]
        for band in list_band_variables(self.filename):
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
                **({} if self.gridded else {"coordinates": ("longitude", "latitude")}),
                "band_index": band["band"],
                "radiation_wavelength_nm": band["radiation_wavelength"],
            }

    def get_area_def(self, dsid):
        """The map grid of a regridded product (Satpy calls this for datasets
        without coordinates)."""
        if not self.gridded:
            raise NotImplementedError("swath product: geolocated through latitude/longitude")
        from hypso.grid.areas import area_from_grid
        crs_attrs = {key.rsplit("/", 1)[-1]: value for key, value in self.file_content.items()
                     if key.startswith("crs/attr/")}
        return area_from_grid(crs_attrs, self["x"].values, self["y"].values,
                              float(self["/attr/grid_resolution_x"]), float(self["/attr/grid_resolution_y"]),
                              str(self["/attr/grid"]))

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
