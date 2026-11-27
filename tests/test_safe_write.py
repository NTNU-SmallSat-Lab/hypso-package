"""write_netcdf_safely: local write, verified copy, atomic rename, one retry (NFS HDF errors, 2026-11)."""
import numpy as np
import pytest
import xarray as xr

from hypso.io import product as product_io
from hypso.io.product import write_netcdf_safely


def _tree():
    return xr.DataTree(xr.Dataset({"x": (("a",), np.arange(5.0))}, attrs={"title": "t"}))


def test_writes_complete_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPSO_WRITE_TMPDIR", str(tmp_path / "local"))
    (tmp_path / "local").mkdir()
    (tmp_path / "out").mkdir()
    path = write_netcdf_safely(_tree(), tmp_path / "out" / "f.nc")
    assert xr.open_dataset(path)["x"].values.tolist() == [0, 1, 2, 3, 4]
    assert sorted(p.name for p in path.parent.iterdir()) == ["f.nc"]                  # no .part left
    assert list((tmp_path / "local").iterdir()) == []                                 # local copy removed


def test_retries_after_hdf_error(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = xr.DataTree.to_netcdf

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("NetCDF: HDF error")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(xr.DataTree, "to_netcdf", flaky)
    path = write_netcdf_safely(_tree(), tmp_path / "f.nc")
    assert calls["n"] == 2 and xr.open_dataset(path)["x"].size == 5


def test_bad_copy_is_caught(tmp_path, monkeypatch):
    import shutil
    calls = {"n": 0}
    original = shutil.copyfile

    def corrupting(src, dst, *args, **kwargs):
        calls["n"] += 1
        original(src, dst, *args, **kwargs)
        if calls["n"] == 1:                                  # flip a byte in the first copy only
            with open(dst, "r+b") as f:
                f.seek(100)
                byte = f.read(1)
                f.seek(100)
                f.write(bytes([byte[0] ^ 0xFF]))
    monkeypatch.setattr(shutil, "copyfile", corrupting)
    path = write_netcdf_safely(_tree(), tmp_path / "f.nc")
    assert calls["n"] == 2 and xr.open_dataset(path)["x"].size == 5


def test_persistent_failure_leaves_nothing(tmp_path, monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("NetCDF: HDF error")
    monkeypatch.setattr(xr.DataTree, "to_netcdf", broken)
    (tmp_path / "f.nc").write_text("previous version")
    with pytest.raises(RuntimeError, match="HDF error"):
        write_netcdf_safely(_tree(), tmp_path / "f.nc", retries=2)
    assert (tmp_path / "f.nc").read_text() == "previous version"                   # untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.nc"]


def test_write_product_uses_it():
    import inspect
    assert "write_netcdf_safely(" in inspect.getsource(product_io.write_product)
