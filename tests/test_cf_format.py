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


@pytest.mark.parametrize("level", ["l1b", "l1c", "l1d", "l2a"])
def test_flat_root_layout(written_nc_files, open_nc, level):
    # The core format decision: products + geometry flattened to root, only
    # metadata/* stays nested (CF coordinates can't resolve between sibling
    # groups, and SNAP's CF reader has NetCDF-3.5 heritage: no groups).
    ds = open_nc(written_nc_files[level])
    assert "products" not in ds.groups
    assert "geometry" not in ds.groups
    assert "metadata" in ds.groups


@pytest.mark.parametrize("level", ["l1b", "l1c", "l1d", "l2a"])
def test_conventions_global_attr(written_nc_files, open_nc, level):
    ds = open_nc(written_nc_files[level])
    assert getattr(ds, "Conventions").startswith("CF-")


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


def test_l2a_dynamic_product_name(written_nc_files, open_nc):
    # The fabricated correction's l2_variable_name is "chla" - the writer must
    # use it (the old writer hardcoded per-level prefixes; the old loader only
    # ever tried ['rrs','Rrs','rho_w'] and could never find "chla").
    ds = open_nc(written_nc_files["l2a"])
    assert band_variables(ds, "chla")
    assert not band_variables(ds, "Rrs")


def test_written_file_round_trips_through_reader(written_nc_files, satobj):
    from hypso.io.reader import load_level_nc

    (_, _, nc_geometry_vars, _, _, _, _, _, cube) = load_level_nc(
        str(written_nc_files["l1c"]), level="l1c", load_cube=True)
    assert np.allclose(np.asarray(cube), satobj.l1b_cube.to_numpy(), equal_nan=True)
    assert np.allclose(np.asarray(nc_geometry_vars["latitude"]),
                       np.asarray(satobj.latitudes), equal_nan=True)


def test_spectral_response_lazy_rebuild_from_file(written_nc_files, satobj, tmp_path):
    # A capture LOADED from a written L1D file must rebuild an identical
    # SpectralResponse lazily (the SRF matrix is not persisted; the builder is
    # deterministic given the file's full-precision wavelengths/fwhm inputs).
    import shutil
    from hypso import Hypso

    named = tmp_path / "aeronetvenice_2025-03-04T10-38-05Z-l1d.nc"
    shutil.copy2(written_nc_files["l1d"], named)

    loaded = Hypso(str(named), load_cube=False)
    assert loaded._spectral_response is None
    sr = loaded.spectral_response  # triggers the rebuild

    ref = satobj.spectral_response
    assert np.array_equal(sr.srf.toarray(), ref.srf.toarray())
    assert np.array_equal(sr.esun, ref.esun)
    assert np.array_equal(np.asarray(loaded.wavelengths, dtype=float),
                          np.asarray(satobj.wavelengths, dtype=float))
    assert np.array_equal(np.asarray(loaded.fwhm, dtype=float),
                          np.asarray(satobj.fwhm, dtype=float))
    # legacy attrs backfilled for the Polymer connector
    assert loaded.srf is sr.srf
