"""Plan §Fix 2 (Bug C): get_l1a_satpy_scene used to hardcode wavelengths =
range(0,120), inconsistent with get_l1b/l1c/l1d_satpy_scene's own
satobj.wavelengths - and would IndexError/silently drop bands for any
capture whose real band count differed from 120. Fixed to read
satobj.wavelengths like every other level.

Skipped automatically when the reference capture isn't present."""
import numpy as np

from conftest import requires_real_capture, L1A_PATH

pytestmark = requires_real_capture


def test_l1a_satpy_scene_uses_real_wavelengths_immediately_after_load():
    # Regression test for a second bug found while fixing the first: L1A's
    # own loader (hypso.load.l1a_nc_loader) always sets cube_attrs = {} (no
    # per-band variables to read a wavelengths array back from, unlike L1B+),
    # so io/dispatch.py's wavelengths fallback fell all the way through to a
    # synthetic 0..119 index for any L1A-only capture - even though the real
    # per-capture wavelengths already exist in the file's own
    # metadata/corrections group. This must be checked on a freshly-loaded
    # capture, before calibration ever runs (calibration's own spectral
    # correction step would independently overwrite wavelengths with the
    # same real values, masking this bug if checked only post-calibration).
    from hypso import Hypso

    obj = Hypso(path=L1A_PATH, load_cube=False, verbose=False)
    wavelengths = np.asarray(obj.wavelengths)
    assert not np.array_equal(wavelengths, np.arange(len(wavelengths))), (
        "wavelengths fell back to a synthetic 0..N index instead of the "
        "real per-capture values in metadata/corrections"
    )
    assert wavelengths.min() > 300 and wavelengths.max() < 1000  # real nm range


def test_l1a_satpy_scene_matches_real_band_count(satobj):
    from hypso.satpy.satpy import get_l1a_satpy_scene

    scene = get_l1a_satpy_scene(satobj)
    band_names = list(scene.keys())
    assert len(band_names) == len(satobj.wavelengths) == satobj.l1a_cube.shape[-1]

    by_wavelength = sorted(band_names, key=lambda name: scene[name].attrs["wavelength"].central)
    scene_wavelengths = [scene[name].attrs["wavelength"].central for name in by_wavelength]
    assert np.allclose(scene_wavelengths, sorted(satobj.wavelengths))
