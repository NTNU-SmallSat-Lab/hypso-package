"""hypso.masks.flags - masks as one CF flag variable (REFACTOR_PROGRESS.md x42)."""
import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from hypso.io.product import PRODUCT_DIMS, read_product, write_product
from hypso.masks.flags import FLAGS_NAME, apply_flags, decode_flags, encode_flags

SHAPE = (4, 3)
LAND = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], bool)
CLOUD = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], bool)


def test_encode_decode_round_trip():
    flags = encode_flags({"land": LAND, "cloud": CLOUD})
    assert flags.name == FLAGS_NAME and flags.dims == ("lines", "samples")
    assert flags.dtype == np.uint8 and flags.attrs["flag_masks"].dtype == np.uint8   # CF: same type
    assert flags.attrs["flag_meanings"] == "land cloud"
    assert int(flags.values[1, 0]) == 3          # land | cloud
    decoded = decode_flags(flags)
    assert np.array_equal(decoded["land"], LAND) and np.array_equal(decoded["cloud"], CLOUD)


def test_more_than_eight_masks_widens_dtype():
    masks = {f"m{i}": np.zeros(SHAPE, bool) for i in range(9)}
    masks["m8"][0, 0] = True
    flags = encode_flags(masks)
    assert flags.dtype == np.uint16 and int(flags.values[0, 0]) == 256
    assert decode_flags(flags)["m8"].values[0, 0]


@pytest.mark.parametrize("masks", [{}, {"sea ice": LAND}, {"land": LAND, "cloud": np.zeros((2, 2), bool)}])
def test_invalid_masks_rejected(masks):
    with pytest.raises(ValueError):
        encode_flags(masks)


def test_apply_flags():
    cube = xr.DataArray(np.ones(SHAPE + (2,)), dims=PRODUCT_DIMS)
    flags = encode_flags({"land": LAND, "cloud": CLOUD})
    land_only = apply_flags(cube, flags, ["land"])
    assert np.isnan(land_only.values[LAND]).all() and not np.isnan(land_only.values[~LAND]).any()
    both = apply_flags(cube, flags)
    assert int(np.isnan(both.values[:, :, 0]).sum()) == int((LAND | CLOUD).sum())
    with pytest.raises(KeyError):
        apply_flags(cube, flags, ["snow"])


def test_flags_survive_write_and_read(tmp_path):
    root = xr.Dataset(
        {"rhot": (PRODUCT_DIMS, np.random.default_rng(0).random(SHAPE + (2,)).astype(np.float32)),
         "latitude": (("lines", "samples"), np.full(SHAPE, 45, np.float32)),
         "longitude": (("lines", "samples"), np.full(SHAPE, 12, np.float32))},
        coords={"wavelength": ("bands", [400.0, 500.0]), "radiation_wavelength": ("bands", [400.04, 500.27]),
                "fwhm": ("bands", [5.46, 5.46])},
        attrs={"processing_level": "L1D"})
    root[FLAGS_NAME] = encode_flags({"land": LAND, "cloud": CLOUD}, descriptions={"land": "Land mask"})
    path = tmp_path / "a.nc"
    write_product(xr.DataTree(root), path)
    with nc.Dataset(path) as f:
        var = f[FLAGS_NAME]
        assert var.dtype == np.uint8 and var.flag_meanings == "land cloud"
        assert list(var.flag_masks) == [1, 2] and var.flag_masks.dtype == np.uint8
        assert var.coordinates == "latitude longitude" and var.comment == "land: Land mask"
        assert "_FillValue" not in var.ncattrs()
        assert not np.isnan(f["rhot_400"][:]).any()   # product stays unmasked
    decoded = decode_flags(read_product(path).dataset[FLAGS_NAME])
    assert np.array_equal(decoded["land"], LAND) and np.array_equal(decoded["cloud"], CLOUD)


def test_sea_land_cloud_reader(tmp_path):
    from hypso.masks import LABELS, read_sea_land_cloud, sea_land_cloud_masks
    labels = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    path = tmp_path / "sea-land-cloud.labels"
    path.write_bytes(labels.tobytes())
    np.testing.assert_array_equal(read_sea_land_cloud(path, (2, 3)), labels)
    masks = sea_land_cloud_masks(path, (2, 3))
    assert set(masks) == set(LABELS)
    np.testing.assert_array_equal(masks["cloud"], labels == 2)
    with pytest.raises(ValueError, match="holds 6 labels"):
        read_sea_land_cloud(path, (3, 3))
