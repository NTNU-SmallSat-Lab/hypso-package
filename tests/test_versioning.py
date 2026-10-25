"""Versioning (hypso.versioning) and the calibration coefficient manifests
(hypso.calibration.manifest) - see docs/versioning.rst."""
import re

import pytest
from packaging.version import Version

from hypso.calibration import manifest
from hypso.versioning import UNKNOWN_VERSION, package_info, version_from_describe

PACKAGES = ("hypso", "hypso1_calibration", "hypso2_calibration")


# --- calibration coefficient manifests ---

@pytest.mark.parametrize("package", manifest.CALIBRATION_PACKAGES)
def test_coefficients_match_manifest(package):
    # Fails when a coefficient file changes (or is added/removed) without
    # `python -m hypso.calibration.manifest update` in the same commit - that
    # commit should then be tagged as a new release.
    problems = {k: v for k, v in manifest.check(package).items() if v}
    assert not problems, (f"{package} data/ differs from {manifest.MANIFEST_NAME}: {problems}. "
                          f"Run `python -m hypso.calibration.manifest update {package}` and tag a release.")


def test_manifest_check_detects_changed_file(monkeypatch):
    package = "hypso2_calibration"
    listed = manifest.load(package)
    name = sorted(listed)[0]
    monkeypatch.setattr(manifest, "load", lambda p: {**listed, name: "0" * 64})
    assert manifest.check(package)["changed"] == [name]


# --- version derivation ---

@pytest.mark.parametrize("describe, dirty, date, expected", [
    ("v26.02.0-0-gc3c7c8c6", False, None, "26.2.0"),
    ("v26.02.0-0-gc3c7c8c6", True, "20261025", "26.2.0+d20261025"),
    ("v26.02.0-169-gc3c7c8c6", False, None, "26.2.1.dev169+gc3c7c8c6"),
    ("v26.02.0-169-gc3c7c8c6", True, "20261025", "26.2.1.dev169+gc3c7c8c6.d20261025"),
    ("v26.10.2-3-gabcdef12", True, None, "26.10.3.dev3+gabcdef12.dirty"),
])
def test_version_from_describe(describe, dirty, date, expected):
    assert version_from_describe(describe, dirty, date) == expected
    Version(expected)  # valid PEP 440


@pytest.mark.parametrize("package", PACKAGES)
def test_package_info_from_this_checkout(package):
    info = package_info(package)
    assert info["source"] == "git"
    assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"])
    assert isinstance(info["git_dirty"], bool)
    assert info["version"] != UNKNOWN_VERSION
    Version(info["version"])


def test_packages_share_one_version():
    # One vYY.M.N tag versions all three packages.
    assert len({package_info(p)["version"] for p in PACKAGES}) == 1


def test_hypso_dunder_version():
    import hypso
    assert hypso.__version__ == package_info("hypso")["version"]


def test_installed_metadata_of_another_copy_is_not_trusted():
    # In the `ac` env, production's hypso2_calibration is installed while
    # this checkout's copy is the one imported (sys.path, see conftest.py).
    # Its metadata version must not be reported for this copy.
    import importlib.metadata
    from pathlib import Path
    import hypso2_calibration
    from hypso.versioning import _matching_metadata_version
    try:
        dist = importlib.metadata.distribution("hypso2_calibration")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("hypso2_calibration not installed in this environment")
    imported = Path(hypso2_calibration.__file__).resolve()
    if any(Path(dist.locate_file(f)).resolve() == imported for f in dist.files or ()):
        pytest.skip("the installed hypso2_calibration is the imported copy")
    assert _matching_metadata_version(hypso2_calibration) is None
