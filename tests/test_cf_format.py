"""Plan §Verification item 2: CF/format assertions against real written output.
Each test maps to one confirmed pre-refactor bug or deliberate format decision
(flat root layout, group-relative coordinates, fixed lat/lon/angle attrs,
Conventions present, radiation_wavelength scalar, `wave` gone, L1B free of
dangling geometry references) - so a change that reintroduces one fails loudly.

Skipped automatically when the reference capture isn't present."""
import numpy as np
import netCDF4 as nc
import pytest

from conftest import requires_real_capture

pytestmark = requires_real_capture


def band_variables(ds, prefix):
    return {name: var for name, var in ds.variables.items()
            if name.startswith(prefix + "_")}


@pytest.fixture()
def open_nc():
    handles = []

    def _open(path):
        ds = nc.Dataset(path, "r")
        handles.append(ds)
        return ds

    yield _open
    for ds in handles:
        ds.close()


@pytest.mark.parametrize("level", ["l1a", "l1b", "l1c", "l1d", "l2a"])
def test_flat_root_layout(written_nc_files, open_nc, level):
    # The core format decision: products + geometry flattened to root, only
    # metadata/* stays nested (CF coordinates can't resolve between sibling
    # groups, and SNAP's CF reader has NetCDF-3.5 heritage: no groups).
    ds = open_nc(written_nc_files[level])
    assert "products" not in ds.groups
    assert "geometry" not in ds.groups
    assert "metadata" in ds.groups


@pytest.mark.parametrize("level", ["l1a", "l1b", "l1c", "l1d", "l2a"])
def test_conventions_global_attr(written_nc_files, open_nc, level):
    ds = open_nc(written_nc_files[level])
    assert getattr(ds, "Conventions").startswith("CF-")


@pytest.mark.parametrize("level", ["l1b", "l1c", "l1d", "l2a"])
def test_history_and_references_global_attrs(written_nc_files, open_nc, level):
    ds = open_nc(written_nc_files[level])
    assert getattr(ds, "references") == "https://hypso.space"
    # one line per processing step (hypso.io.provenance)
    steps = [line.split(": ", 1)[1].split(" ")[0] for line in ds.history.splitlines()]
    expected = {"l1b": ["l1b"], "l1c": ["l1b", "l1c"], "l1d": ["l1b", "l1c", "l1d"],
                "l2a": ["l1b", "l1c", "l1d", "l2a_dps"]}[level]
    assert steps == expected
    assert ds.product_format_version == "2.0"


@pytest.mark.parametrize("level,prefix", [("l1c", "Lt"), ("l1d", "rhot")])
def test_geolocated_levels(written_nc_files, open_nc, level, prefix):
    ds = open_nc(written_nc_files[level])

    # geometry variables at root
    assert "latitude" in ds.variables and "longitude" in ds.variables
    assert "crs_wgs84" in ds.variables

    # fixed latitude attrs (bug: units="degrees", valid range copy-pasted
    # from longitude)
    lat = ds.variables["latitude"]
    assert lat.units == "degrees_north"
    assert float(lat.valid_min) == -90.0 and float(lat.valid_max) == 90.0
    lon = ds.variables["longitude"]
    assert lon.units == "degrees_east"
    assert float(lon.valid_min) == -180.0 and float(lon.valid_max) == 180.0

    # zenith vs azimuth ranges differentiated (bug: blanket [-180,180] on both)
    for name, var in ds.variables.items():
        if "zenith" in name:
            assert float(var.valid_min) == 0.0 and float(var.valid_max) == 180.0, name
        elif "azimuth" in name:
            assert float(var.valid_min) == -180.0 and float(var.valid_max) == 180.0, name

    # every band variable's coordinates/grid_mapping resolves group-relative
    bands = band_variables(ds, prefix)
    assert bands, f"no {prefix}_* band variables found at root"
    for name, var in bands.items():
        assert var.coordinates == "latitude longitude", name
        assert var.grid_mapping == "crs_wgs84", name


def test_l1b_has_no_dangling_geometry_references(written_nc_files, open_nc):
    # THE original motivating bug: L1B files carried coordinates/grid_mapping
    # attributes pointing at a geometry group L1B files don't have.
    ds = open_nc(written_nc_files["l1b"])
    assert "latitude" not in ds.variables
    assert "longitude" not in ds.variables
    bands = band_variables(ds, "Lt")
    assert bands
    for name, var in bands.items():
        assert "coordinates" not in var.ncattrs(), name
        assert "grid_mapping" not in var.ncattrs(), name


def test_band_wavelength_attrs(written_nc_files, open_nc):
    ds = open_nc(written_nc_files["l1c"])
    for name, var in band_variables(ds, "Lt").items():
        # bug: radiation_wavelength was written as a 1-element tuple
        assert np.ndim(var.radiation_wavelength) == 0, name
        # `wave` (redundant duplicate of `wavelength`) is gone; both real
        # wavelength attrs survive and differ in precision, not meaning
        assert "wave" not in var.ncattrs(), name
        assert "wavelength" in var.ncattrs(), name
        assert "band" in var.ncattrs(), name


def test_band_variables_sorted_by_band_attr(written_nc_files, open_nc):
    # The reader reconstructs cubes sorted by the `band` attribute (the old
    # loaders relied on insertion order - the confirmed latent band-order
    # bug); the writer must emit a usable, complete band index.
    ds = open_nc(written_nc_files["l1c"])
    indices = sorted(int(v.band) for v in band_variables(ds, "Lt").values())
    assert indices == list(range(len(indices)))


def test_effective_fwhm_unbinned_persisted(written_nc_files, open_nc):
    ds = open_nc(written_nc_files["l1d"])
    srf = ds["metadata/srf"]
    effective_fwhm = np.asarray(srf["effective_fwhm"][:])
    effective_fwhm_unbinned = np.asarray(srf["effective_fwhm_unbinned"][:])
    unbinned = ds["metadata/corrections"]["wavelengths_unbinned"][:]
    # one value per unbinned band, more than the binned bands
    assert effective_fwhm_unbinned.shape[0] == unbinned.shape[0] > effective_fwhm.shape[0]
    # binning widens the effective band
    assert effective_fwhm.mean() > effective_fwhm_unbinned.mean()


def test_l1d_carries_srfs_and_earth_sun_distance(written_nc_files, open_nc):
    srf = open_nc(written_nc_files["l1d"])["metadata/srf"]
    assert len([n for n in srf.variables if n.startswith("Band_")]) == 120
    assert 0.98 < srf.earth_sun_distance_au < 1.02
    assert srf["esun"].units == "mW m-2 nm-1"


def test_l2a_product_variable(written_nc_files, open_nc):
    ds = open_nc(written_nc_files["l2a"])
    assert band_variables(ds, "Rrs") and not band_variables(ds, "rhot")
    assert ds.atmospheric_correction == "dps" and ds.processing_level == "L2A"


def test_l1a_is_uint16_without_calibration_attrs(written_nc_files, open_nc):
    ds = open_nc(written_nc_files["l1a"])
    bands = band_variables(ds, "dn")
    assert len(bands) == 120 and all(v.dtype == np.uint16 for v in bands.values())
    assert not [k for k in ds.ncattrs() if k.endswith("_file")]


@pytest.mark.parametrize("level, prefix, name", [
    ("l1b", "Lt", "toa_outgoing_radiance_per_unit_wavelength"),
    ("l1c", "Lt", "toa_outgoing_radiance_per_unit_wavelength"),
    ("l1d", "rhot", "toa_bidirectional_reflectance"),
])
def test_band_standard_names(written_nc_files, open_nc, level, prefix, name):
    ds = open_nc(written_nc_files[level])
    assert all(var.standard_name == name for var in band_variables(ds, prefix).values())
