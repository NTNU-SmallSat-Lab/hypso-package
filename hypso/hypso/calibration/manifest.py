"""SHA-256 manifest of each hypsoN_calibration package's coefficient files.

Each calibration package ships `coefficients_manifest.yaml` (next to its
__init__.py) listing the checksum of every file under its data/ directory.
tests/test_versioning.py fails when a coefficient file no longer matches,
so coefficients cannot change without the manifest changing in the same
commit - and that commit is the one to tag as a new release (version bumps
are not otherwise enforced: both packages sat at 26.02.0 while `04cd011f`
replaced the HYPSO-2 smile matrix).

The same sha256_file() is what product provenance records per loaded
coefficient file, so a product's coefficients can be matched to a manifest.

    python -m hypso.calibration.manifest check  [package ...]
    python -m hypso.calibration.manifest update [package ...]
"""
import hashlib
import importlib
import sys
from pathlib import Path

import yaml

MANIFEST_NAME = "coefficients_manifest.yaml"
CALIBRATION_PACKAGES = ("hypso1_calibration", "hypso2_calibration")

_HEADER = """\
# SHA-256 of every file under data/, maintained by
#   python -m hypso.calibration.manifest update {package}
# tests/test_versioning.py fails if a file changes without this manifest
# changing too. Commit the updated manifest together with the coefficient
# change, and tag that commit as a new release (vYY.M.N) - see
# docs/versioning.rst.
"""


def sha256_file(path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_dir(package: str) -> Path:
    return Path(importlib.import_module(package).__file__).resolve().parent


def compute(package: str) -> dict:
    """{path relative to the package dir (posix): sha256} for data/**."""
    root = _package_dir(package)
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted((root / "data").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def manifest_path(package: str) -> Path:
    """Path of a calibration package's checksum manifest."""
    return _package_dir(package) / MANIFEST_NAME


def load(package: str) -> dict:
    """The manifest's ``{path: sha256}``."""
    return yaml.safe_load(manifest_path(package).read_text())["files"]


def write(package: str) -> Path:
    """Recompute the package's checksums and write its manifest."""
    path = manifest_path(package)
    body = yaml.safe_dump({"package": package, "files": compute(package)}, sort_keys=True)
    path.write_text(_HEADER.format(package=package) + body)
    return path


def check(package: str) -> dict:
    """{"missing": [...], "unlisted": [...], "changed": [...]} - all empty
    when the package's data/ matches its manifest."""
    listed, actual = load(package), compute(package)
    return {
        "missing": sorted(set(listed) - set(actual)),
        "unlisted": sorted(set(actual) - set(listed)),
        "changed": sorted(k for k in set(listed) & set(actual) if listed[k] != actual[k]),
    }


def main(argv) -> int:
    """Command line: ``check`` or ``update`` the given (default: all) calibration packages."""
    if not argv or argv[0] not in ("check", "update"):
        print(__doc__)
        return 2
    packages = argv[1:] or CALIBRATION_PACKAGES
    status = 0
    for package in packages:
        if argv[0] == "update":
            print(f"wrote {write(package)}")
            continue
        problems = {k: v for k, v in check(package).items() if v}
        print(f"{package}: {'OK' if not problems else problems}")
        status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
