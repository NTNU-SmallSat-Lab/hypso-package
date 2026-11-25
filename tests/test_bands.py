"""Band subsets (Product.select_bands / hypso.processing.bands, ×79)."""
import numpy as np
import pytest

import hypso
from conftest import LATS_PATH, LONS_PATH, requires_real_capture
from hypso.processing.bands import select_bands

pytestmark = requires_real_capture


def test_select_then_process_equals_process_then_select(l1b, l1d):
    # Each band keeps its full-set index, so F0 and the SRFs are looked up right.
    sub_b = hypso.Product(select_bands(l1b.tree, wavelength_range=(450, 700)))
    sub_d = sub_b.georeference(latitudes=LATS_PATH, longitudes=LONS_PATH, direct=False).toa_reflectance()
    kept = sub_d.cube["band_index"].values
    assert sub_d.cube.sizes["bands"] == kept.size < l1d.cube.sizes["bands"]
    np.testing.assert_allclose(sub_d.cube.values, l1d.cube.values[..., kept], rtol=1e-6)
    srf = sub_d.tree["metadata/srf"].dataset
    assert sorted(int(n.split("_")[1]) for n in srf.data_vars if n.startswith("Band_")) == sorted(kept.tolist())
    np.testing.assert_allclose(srf["F0"].values, l1d.tree["metadata/srf"].dataset["F0"].values[kept])


def test_selectors_and_errors(l1a, l1d):
    picked = hypso.Product(select_bands(l1d.tree, wavelengths=[443, 560, 665]))
    assert picked.cube.sizes["bands"] == 3
    assert np.abs(picked.wavelengths - [443, 560, 665]).max() < 2.5
    again = hypso.Product(select_bands(picked.tree, indices=picked.cube["band_index"].values[:2]))
    np.testing.assert_array_equal(again.cube["band_index"].values, picked.cube["band_index"].values[:2])
    assert picked.provenance[-1]["step"] == "select_bands"
    with pytest.raises(ValueError, match="calibration needs every band"):
        select_bands(l1a.tree, wavelength_range=(400, 700))
    with pytest.raises(ValueError, match="exactly one"):
        select_bands(l1d.tree)
    with pytest.raises(ValueError, match="not in this product"):
        select_bands(picked.tree, indices=[0])


def test_subset_round_trips_through_a_file(l1d, tmp_path):
    sub = hypso.Product(select_bands(l1d.tree, wavelength_range=(500, 600)))
    back = hypso.open(sub.write(out_dir=tmp_path, significant_bits=None))
    np.testing.assert_array_equal(back.cube["band_index"].values, sub.cube["band_index"].values)
    np.testing.assert_array_equal(back.cube.values, sub.cube.values)
    # the full product read from a file also knows its band indices
    full = hypso.open(l1d.write(out_dir=tmp_path, label="full", significant_bits=None))
    np.testing.assert_array_equal(full.cube["band_index"].values, np.arange(l1d.cube.sizes["bands"]))


def test_open_reads_only_the_selected_bands(l1d, tmp_path):
    from hypso.io.product import read_product
    path = l1d.write(out_dir=tmp_path, significant_bits=None)
    sub = read_product(path, wavelength_range=(450, 700))
    full = read_product(path)
    kept = sub.dataset["rhot"]["band_index"].values
    np.testing.assert_array_equal(sub.dataset["rhot"].values, full.dataset["rhot"].values[..., kept])
    srf = sub["metadata/srf"].dataset
    assert sorted(int(n.split("_")[1]) for n in srf.data_vars if n.startswith("Band_")) == sorted(kept.tolist())
    np.testing.assert_array_equal(srf["F0"].values, full["metadata/srf"].dataset["F0"].values[kept])
    assert list(sub["metadata/processing"].children)[-1] == "select_bands"
    assert sub.encoding["source"] == str(path)
    # same as reading everything and selecting afterwards
    after = select_bands(full, wavelength_range=(450, 700))
    np.testing.assert_array_equal(after.dataset["rhot"].values, sub.dataset["rhot"].values)



def test_product_api(l1d, tmp_path):
    path = l1d.write(out_dir=tmp_path, significant_bits=None)
    sub = hypso.open(path, wavelengths=[640, 550, 460])
    assert sub.cube.sizes["bands"] == 3 and sub.level == "L1D"
    np.testing.assert_array_equal(l1d.select_bands(wavelengths=[640, 550, 460]).cube.values, sub.cube.values)
