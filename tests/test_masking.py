"""Tests for hypso.masks.pipeline - the mask container/orchestration code
extracted from HypsoCapture (land_mask/cloud_mask/custom_masks state,
set_custom_mask/clear_custom_masks/load_mask_from_file, unified_mask, and
the four masked_l1x_cube properties).

Uses capture_types.spawn_as(satobj, type(satobj)) to get an isolated
same-type capture per test - the real satobj fixture is session-scoped and
shared across the whole suite, so mutating its masks directly here would
leak into other tests. spawn_as is the general-purpose successor to
HypsoCapture's old _spawn_next_level (removed - superseded once to_l1b()/
to_l1c()/to_l1d()/to_l2a() moved onto it for cross-type spawning); calling
it with the same class as source and target reproduces the exact same
same-type copy semantics _spawn_next_level had."""
import numpy as np
import pytest

from conftest import requires_real_capture
from hypso import capture_types

pytestmark = requires_real_capture


@pytest.fixture
def isolated(satobj):
    """A capture that can have its masks mutated without affecting the
    shared session-scoped satobj fixture used elsewhere in the suite."""
    return capture_types.spawn_as(satobj, type(satobj))


def test_land_mask_cloud_mask_roundtrip(isolated):
    shape = isolated.spatial_dimensions
    land = np.zeros(shape, dtype=bool)
    land[0, 0] = True

    isolated.land_mask = land
    assert isolated.land_mask is not None
    assert bool(isolated.land_mask.to_numpy()[0, 0]) is True

    isolated.land_mask = None
    assert isolated.land_mask is None


def test_custom_mask_register_and_clear(isolated):
    shape = isolated.spatial_dimensions
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True

    isolated.set_custom_mask("test_mask", mask)
    assert "test_mask" in isolated.custom_masks

    isolated.clear_custom_masks()
    assert isolated.custom_masks == {}


def test_masked_l1a_cube_applies_unified_mask(isolated):
    shape = isolated.spatial_dimensions
    land = np.ones(shape, dtype=bool)  # mask out everything

    isolated.land_mask = land
    masked = isolated.masked_l1a_cube

    assert np.isnan(masked.to_numpy()).all()


def test_masked_l1a_cube_no_mask_returns_unmasked(isolated):
    assert isolated.land_mask is None
    assert isolated.cloud_mask is None
    assert dict(isolated.custom_masks) == {}

    masked = isolated.masked_l1a_cube
    assert np.array_equal(masked.to_numpy(), isolated.l1a_cube.to_numpy(), equal_nan=True)


def test_masked_l1c_cube_mirrors_l1c_cube_not_always_none(isolated):
    # Regression test for a real pre-existing bug fixed in this pass:
    # masked_l1c_cube used to read self._l1c_cube, which is never actually
    # populated (l1c_cube the property instead returns a deepcopy of
    # _l1b_cube) - so it always silently returned None. Now mirrors
    # l1c_cube's own getter.
    assert isolated.land_mask is None
    assert isolated.cloud_mask is None

    masked = isolated.masked_l1c_cube
    assert masked is not None
    assert np.array_equal(masked.to_numpy(), isolated.l1c_cube.to_numpy(), equal_nan=True)


def test_masked_l1c_cube_applies_unified_mask(isolated):
    shape = isolated.spatial_dimensions
    land = np.ones(shape, dtype=bool)

    isolated.land_mask = land
    masked = isolated.masked_l1c_cube

    assert np.isnan(masked.to_numpy()).all()
