"""hypso.maps: footprints, shapefiles, RGB images and maps (TODO 4u), on the
reference capture's L1D."""
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

shapely = pytest.importorskip("shapely")

from hypso import maps  # noqa: E402
from hypso.api import Product  # noqa: E402


def _corners(product):
    lat, lon = product.geometry["latitude"].values, product.geometry["longitude"].values
    return [(lon[i, j], lat[i, j]) for i, j in [(0, 0), (0, -1), (-1, -1), (-1, 0)]]


def test_swath_footprint(l1d):
    polygon = maps.footprint(l1d)
    assert polygon.geom_type == "Polygon" and polygon.is_valid
    lat, lon = l1d.geometry["latitude"].values, l1d.geometry["longitude"].values
    centre = shapely.geometry.Point(lon[lat.shape[0] // 2, lat.shape[1] // 2], lat[lat.shape[0] // 2, lat.shape[1] // 2])
    assert polygon.contains(centre)
    for corner in _corners(l1d):                               # corners are vertices
        assert polygon.exterior.distance(shapely.geometry.Point(corner)) < 1e-9
    assert l1d.footprint().equals(polygon)
    assert maps.georeferencing(l1d) == "indirect"


def test_masked_footprint(l1d):
    whole = maps._area_km2(maps.footprint(l1d))
    half = np.zeros(l1d.shape, bool)
    half[:, : l1d.shape[1] // 2] = True
    part = maps.footprint(l1d, mask=half)
    assert part.is_valid and 0.4 < maps._area_km2(part) / whole < 0.6
    data = maps.footprint(l1d, mask="data")                   # every L1D pixel has data
    assert 0.97 < maps._area_km2(data) / whole < 1.03
    with pytest.raises(ValueError, match="no pixels"):
        maps.footprint(l1d, mask=np.zeros(l1d.shape, bool))


def test_footprint_across_the_antimeridian(l1d):
    tree = l1d.tree.copy()
    root = tree.to_dataset(inherit=False)
    lon = root["longitude"].values
    shifted = lon - np.nanmean(lon) + 180.0                    # centred on the antimeridian
    root["longitude"] = (root["longitude"].dims, np.where(shifted > 180, shifted - 360, shifted).astype(lon.dtype),
                         root["longitude"].attrs)
    tree.dataset = root
    polygon = maps.footprint(Product(tree))
    minx, _, maxx, _ = polygon.bounds
    assert polygon.is_valid and 170 < minx < 180 < maxx < 190
    assert maps._area_km2(polygon) == pytest.approx(maps._area_km2(maps.footprint(l1d)), rel=0.02)


def test_write_shapefile_and_geojson(l1d, tmp_path):
    shapefile = pytest.importorskip("shapefile")
    path = l1d.write_footprint(tmp_path / "fp.shp")
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        assert path.with_suffix(suffix).is_file()
    assert "WGS_1984" in path.with_suffix(".prj").read_text()
    assert all(len(name) <= 10 for name, *_ in maps.FOOTPRINT_FIELDS)
    with shapefile.Reader(str(path)) as reader:
        assert len(reader) == 1 and reader.shapeType == shapefile.POLYGON
        record = reader.record(0).as_dict()
        ring = np.asarray(reader.shape(0).points)
    assert record["satellite"] == "HYPSO-2" and record["level"] == "L1D" and record["georef"] == "indirect"
    assert record["capture"] == "aeronetvenice_2025-03-04T10-38-05Z" and record["product"] == "L1D"
    assert record["area_km2"] > 0 and (record["lines"], record["samples"]) == l1d.shape
    x, y = ring[:, 0], ring[:, 1]
    assert np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0       # outer ring clockwise
    gj = json.loads(maps.write_footprints([l1d, l1d], tmp_path / "fp.geojson").read_text())
    assert len(gj["features"]) == 2 and gj["features"][0]["properties"]["target"] == "aeronetvenice"
    with pytest.raises(ValueError, match=".shp or .geojson"):
        maps.write_footprints(l1d, tmp_path / "fp.kml")


def test_rgb_image_and_png(l1d, tmp_path):
    from PIL import Image
    image = maps.rgb_image(l1d)
    assert image.shape == l1d.shape + (3,) and np.nanmin(image) >= 0 and np.nanmax(image) <= 1
    lines, samples = l1d.shape
    png = Image.open(maps.save_rgb(l1d, tmp_path / "rgb.png"))
    assert png.mode == "RGBA" and png.size == (lines, samples)      # server orientation: lines across
    png = Image.open(maps.save_rgb(l1d, tmp_path / "rgb3.png", bin_samples=3))
    assert png.size == (lines, samples // 3)
    png = Image.open(maps.save_rgb(l1d, tmp_path / "rgbp.png", orientation="product"))
    assert png.size == (samples, lines)
    # NaN pixels (e.g. masked in an L2A) are transparent
    cube = l1d.cube.copy()
    cube[:10] = np.nan
    tree = l1d.tree.copy()
    root = tree.to_dataset(inherit=False)
    root[cube.name] = cube
    tree.dataset = root
    alpha = np.asarray(Image.open(maps.save_rgb(Product(tree), tmp_path / "nan.png", orientation="product")))[..., 3]
    assert (alpha[:10] == 0).all() and (alpha[10:] == 255).all()


def test_stretch():
    rgb = np.stack([np.linspace(0, 1, 100), np.linspace(0, 2, 100), np.linspace(0, 4, 100)], axis=-1)
    per = maps._stretch(rgb, (0, 100), per_channel=True, gamma=None)
    np.testing.assert_allclose(per[-1], 1)
    np.testing.assert_allclose(per[0], 0)
    joint = maps._stretch(rgb, (0, 100), per_channel=False, gamma=None)
    np.testing.assert_allclose(joint[-1], [0.25, 0.5, 1.0])         # one gain: colours kept


@pytest.fixture(scope="module")
def mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def test_maps(l1d, mpl, tmp_path):
    from matplotlib.colors import LogNorm
    small = l1d.select_bands(wavelengths=list(maps.RGB_WAVELENGTHS))
    ax = maps.plot_map(small, resolution=300, coastlines=False)
    assert "RGB" in ax.get_title() and "aeronetvenice_2025-03-04T10-38-05Z" in ax.get_title()
    ax.figure.savefig(tmp_path / "rgb.png")
    ax = maps.plot_map(small, "rhot", 550, resolution=300, coastlines=False)
    bar = ax.figure.axes[-1]
    assert "TOA" in bar.get_ylabel() or "Reflectance" in bar.get_ylabel()
    chl = xr.DataArray(np.full(l1d.shape, 2.0, np.float32), attrs={"units": "mg m-3", "long_name": "Chlorophyll-a"})
    chl[:, :50] = np.nan
    ax = maps.plot_map(small.with_variables(chlor_a=chl), "chlor_a", resolution=300, coastlines=False)
    image = ax.get_images()[0]
    assert isinstance(image.norm, LogNorm)
    assert ax.figure.axes[-1].get_ylabel() == "Chlorophyll-a [mg m-3]"
    with pytest.raises(KeyError, match="not a variable"):
        maps.plot_map(small, "tsm")
    with pytest.raises(ValueError, match="wavelength"):
        maps.plot_map(small, "rhot")


def test_overview(l1d, mpl):
    polygon = maps.footprint(l1d)
    moved = shapely.affinity.translate(polygon, xoff=0.3)
    ax = maps.plot_overview([l1d, moved], labels=["a", "b"], coastlines=False, title="two")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


MACHI_FILE = Path("/home/camerop/HYPSO_DATA_REFACTOR/erie_2025-08-14T16-34-20Z/"
                  "HYPSO2_HSI.20250814T163420.erie.L2A.RHOS.machi.nc")


@pytest.mark.skipif(not MACHI_FILE.is_file(), reason=f"no MACHI L2A at {MACHI_FILE}")
def test_machi_band_subset_on_read():
    """read_product with a band selection renames full-band metadata to
    bands_full; select_bands must still subset MACHI's metadata (it raised
    KeyError: 'bands')."""
    import hypso
    rgb = hypso.open(MACHI_FILE, wavelengths=list(maps.RGB_WAVELENGTHS))
    assert rgb.cube.sizes["bands"] == 3 and rgb.tree["metadata/machi"].dataset.sizes["bands"] == 3
