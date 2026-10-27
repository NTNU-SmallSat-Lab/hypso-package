"""hypso.satpy: Satpy scenes from a Product, and the Satpy reader plugin.

Skipped automatically when the reference capture isn't present."""
import numpy as np
import pytest

from conftest import requires_real_capture

pytestmark = requires_real_capture


def test_l1a_wavelengths_are_real(l1a):
    # A freshly read L1A must carry the capture's real band wavelengths (from
    # metadata/corrections), not a synthetic 0..N index (bug fixed in the
    # capture-object loader, kept by read_product).
    wavelengths = l1a.wavelengths
    assert not np.array_equal(wavelengths, np.arange(len(wavelengths)))
    assert wavelengths.min() > 300 and wavelengths.max() < 1000


def test_product_scene_matches_bands(l1d):
    scene = l1d.to_satpy()
    names = list(scene.keys())
    assert len(names) == len(l1d.wavelengths) == l1d.cube.shape[-1]
    central = sorted(scene[name].attrs["wavelength"].central for name in names)
    assert np.allclose(central, sorted(l1d.wavelengths))


def test_reader_plugin_reads_written_files(written_nc_files):
    from satpy import Scene
    scene = Scene(filenames=[str(written_nc_files["l1c"])], reader="hypso_l1c")
    names = scene.available_dataset_names()
    assert "latitude" in names and len([n for n in names if n.startswith("Lt_")]) == 120
    scene.load(["Lt_413"])
    assert scene["Lt_413"].shape == (598, 1092)


@pytest.mark.parametrize("level, ac", [("L1C", None), ("L1D", None), ("L2A", "polymer")])
def test_hsi_readers_match_only_hsi_names(level, ac):
    # The first name field is satellite + instrument (hypso.io.naming). These
    # readers are for the hyperspectral imager and match "_HSI" literally, so
    # another instrument's files (e.g. an RGB camera's) are left to its own reader.
    import re
    from datetime import datetime
    from pathlib import Path
    import yaml
    from trollsift import globify
    import hypso.satpy
    from hypso.io.naming import product_name
    name = product_name("HYPSO-2", datetime(2025, 3, 4, 10, 38, 5), "aeronetvenice", level, ac=ac).filename()
    reader = Path(hypso.satpy.__file__).parent / "etc" / "readers" / f"hypso_{level.lower()}.yaml"
    config = yaml.load(reader.read_text(), Loader=yaml.BaseLoader)
    patterns = [p for ft in config["file_types"].values() for p in ft["file_patterns"] if p.startswith("{platform}")]

    def matches(filename):
        import fnmatch
        return any(fnmatch.fnmatch(filename, globify(p)) for p in patterns)

    assert matches(name)
    assert not matches(name.replace("_HSI.", "_RGB."))
