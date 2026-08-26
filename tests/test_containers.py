"""Unit tests for hypso.containers.DatasetDict and as_dataarray - the
Dataset-backed container (and its underlying single-array normalizer) that
superseded DataArrayDict/DataArrayValidator (both now deleted) for every
keyed collection in this package: l2a cubes, custom masks, and products. Each
test maps to one of the defects that motivated the replacement (see the
module docstring of hypso/containers.py). No real data needed."""
import numpy as np
import pytest
import xarray as xr

from hypso.containers import DatasetDict, as_dataarray


@pytest.fixture()
def cubes():
    return DatasetDict(attributes={"level": "L2"}, num_dims=3,
                       dim_names=("y", "x", "band"), key_attribute="correction")


def test_ndarray_converted_and_stamped(cubes):
    cubes["Polymer"] = np.ones((4, 5, 3))
    entry = cubes["polymer"]
    assert isinstance(entry, xr.DataArray)
    assert entry.dims == ("y", "x", "band")
    assert entry.attrs["level"] == "L2"
    assert entry.attrs["correction"] == "polymer"


def test_keys_lowercased_everywhere(cubes):
    # DataArrayDict defect: only __getitem__/get lowercased, so membership and
    # deletion were case-sensitive while lookup wasn't.
    cubes["Polymer"] = np.ones((4, 5, 3))
    assert "polymer" in cubes
    assert "Polymer" in cubes
    assert cubes.get("POLYMER") is not None
    del cubes["POLYMER"]
    assert len(cubes) == 0


def test_validation_raises_instead_of_storing(cubes):
    # DataArrayDict defect: it printed the exception and stored the bad value.
    with pytest.raises(ValueError):
        cubes["bad"] = np.ones((4, 5))  # 2D into a 3D container
    with pytest.raises(TypeError):
        cubes["bad"] = [[1, 2], [3, 4]]  # not ndarray/DataArray
    assert len(cubes) == 0


def test_shape_enforced_when_set(cubes):
    cubes.dim_shape = (4, 5)
    cubes["ok"] = np.ones((4, 5, 3))
    with pytest.raises(ValueError):
        cubes["wrong"] = np.ones((5, 4, 3))
    assert list(cubes) == ["ok"]


def test_update_path_validates(cubes):
    # DataArrayDict defect: dict.update() bypassed the __setitem__ override.
    with pytest.raises(ValueError):
        cubes.update({"bad": np.ones((2, 2))})
    assert len(cubes) == 0
    cubes.update({"OK": np.ones((4, 5, 3))})
    assert cubes["ok"].attrs["correction"] == "ok"


def test_attrs_mutation_persists(cubes):
    # The contract AC adapters rely on: assign, then mutate attrs in place.
    cubes["acolite_l2r"] = np.ones((4, 5, 3))
    cubes["acolite_l2r"].attrs["l2_variable_name"] = "rhos"
    assert cubes["acolite_l2r"].attrs["l2_variable_name"] == "rhos"


def test_dataarray_dims_renamed(cubes):
    da = xr.DataArray(np.ones((4, 5, 3)), dims=("lines", "samples", "bands"))
    cubes["x"] = da
    assert cubes["x"].dims == ("y", "x", "band")


def test_dataset_backing_and_dim_consistency(cubes):
    cubes["a"] = np.ones((4, 5, 3))
    assert isinstance(cubes.dataset, xr.Dataset)
    # entries share the Dataset's dims - a conflicting size is rejected by
    # xarray itself, the "standard library enforces it" property that
    # motivated the Dataset backing
    with pytest.raises(Exception):
        cubes["b"] = np.ones((9, 9, 9))


def test_copy_isolates_registry_and_attrs_shares_data(cubes):
    cubes["a"] = np.ones((4, 5, 3))
    dup = cubes.copy()
    dup["b"] = np.zeros((4, 5, 3))
    dup["a"].attrs["l2_variable_name"] = "changed"
    assert "b" not in cubes
    assert cubes["a"].attrs["l2_variable_name"] == "rrs" if "l2_variable_name" in cubes["a"].attrs else True
    # underlying data is aliased (same memory), by design
    assert dup["a"].values.base is cubes["a"].values.base or np.shares_memory(dup["a"].values, cubes["a"].values)


def test_2d_mask_container():
    masks = DatasetDict(dim_names=("y", "x"), num_dims=2)
    masks["quadrant"] = np.zeros((4, 5), dtype=bool)
    assert masks["quadrant"].dims == ("y", "x")
    assert dict(masks).keys() == {"quadrant"}


# --- as_dataarray: the standalone single-array normalizer DatasetDict and
# HypsoCapture's cube/mask formatters (_format_cube_dataarray/
# _format_mask_dataarray) both call, replacing the deleted DataArrayValidator ---

def test_as_dataarray_wraps_ndarray():
    da = as_dataarray(np.ones((4, 5, 3)), ("y", "x", "band"), num_dims=3)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x", "band")
    assert da.shape == (4, 5, 3)


def test_as_dataarray_renames_dataarray_dims():
    da_in = xr.DataArray(np.ones((4, 5)), dims=("lines", "samples"))
    da_out = as_dataarray(da_in, ("y", "x"), num_dims=2)
    assert da_out.dims == ("y", "x")


def test_as_dataarray_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="3-dimensional"):
        as_dataarray(np.ones((4, 5)), ("y", "x", "band"), num_dims=3)


def test_as_dataarray_rejects_non_array():
    with pytest.raises(TypeError):
        as_dataarray([[1, 2], [3, 4]], ("y", "x"), num_dims=2)


def test_as_dataarray_enforces_dim_shape():
    as_dataarray(np.ones((4, 5)), ("y", "x"), num_dims=2, dim_shape=(4, 5))
    with pytest.raises(ValueError, match="spatial dimensions"):
        as_dataarray(np.ones((4, 5)), ("y", "x"), num_dims=2, dim_shape=(9, 9))
