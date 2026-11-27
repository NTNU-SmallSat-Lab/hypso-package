"""AC tool version, git commit and uncommitted-changes flag in the L2A file's provenance and global
attributes (user, 2026-11-27)."""
import subprocess

import pytest

from conftest import requires_real_capture
from hypso.io.provenance import ac_attrs, ac_global_attrs
from hypso.processing.atmospheric import _git_identity


def test_ac_attrs_dirty_and_components():
    a = ac_attrs("polymer", version="v2", git_commit="abc", git_dirty=True,
                 components={"eoread": ("def", False)}, install_path="/x", settings_effective={"k": 1})
    assert (a["ac_git_commit"], a["ac_git_dirty"]) == ("abc", "true")
    assert (a["ac_eoread_git_commit"], a["ac_eoread_git_dirty"]) == ("def", "false")
    assert ac_attrs("x", git_commit="abc")["ac_git_dirty"] == "unknown"          # commit known, state not
    assert ac_attrs("x", version="v2.6.3")["ac_git_dirty"] == ""                  # a release install, no git
    g = ac_global_attrs(a)
    assert set(g) == {"ac_tool", "ac_version", "ac_git_commit", "ac_git_dirty", "ac_eoread_git_commit",
                      "ac_eoread_git_dirty"}                                     # no settings or paths
    assert ac_global_attrs(None) == {}


def test_git_identity(tmp_path):
    assert _git_identity(tmp_path) == ("", None)                                 # not a checkout
    assert _git_identity(None) == ("", None)
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / "f.txt").write_text("1")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    commit, dirty = _git_identity(tmp_path)
    assert len(commit) == 40 and dirty is False
    (tmp_path / "f.txt").write_text("2")
    assert _git_identity(tmp_path) == (commit, True)
    (tmp_path / "f.txt").write_text("1")
    (tmp_path / "untracked.txt").write_text("u")                                 # untracked files do not count
    assert _git_identity(tmp_path) == (commit, False)


@requires_real_capture
def test_l2a_global_attributes(l1d, tmp_path):
    import xarray as xr
    from hypso.versioning import package_info
    info = package_info("hypso")
    l2a = l1d.correct_atmosphere("dps")
    assert l2a.tree.attrs["ac_tool"] == "dps" and l2a.tree.attrs["ac_version"] == f"hypso {info['version']}"
    assert l2a.tree.attrs["ac_git_commit"] == (info["git_commit"] or "")
    path = l2a.write(out_dir=tmp_path)
    g = xr.open_dataset(path).attrs
    assert g["ac_tool"] == "dps" and g["ac_git_commit"] == (info["git_commit"] or "")
    assert g["ac_git_dirty"] in ("true", "false", "unknown", "")
    assert xr.open_dataset(path, group="metadata/processing/l2a_dps").attrs["ac_git_commit"] == g["ac_git_commit"]
    l2b_path = l2a.derive("BGC", "ndci").write(out_dir=tmp_path)
    assert xr.open_dataset(l2b_path).attrs["ac_tool"] == "dps"                    # the derived L2B keeps it
