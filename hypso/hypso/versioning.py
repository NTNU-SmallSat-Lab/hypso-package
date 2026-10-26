"""Version and source identity of hypso and the hypsoN_calibration packages,
for product provenance (REFACTOR_PROGRESS.md x39/x40).

All three packages share one version, derived from git tags (vYY.M.N) by
setuptools_scm at build time - see docs/versioning.rst. At runtime the
version a product records must describe the code that actually ran, which
the build-time version does not always do:

- An editable install (`pip install -e`) writes _version.py once, at install
  time; later commits don't change it. So when a package's own directory is
  inside a git work tree, git is asked directly.
- importlib.metadata reports whatever distribution of that name is installed,
  which is not necessarily the copy being imported (e.g. a refactor checkout
  on sys.path while production's hypso2_calibration 26.5.1 is installed in
  the same env). Metadata is only trusted when its files include the
  imported module.

Lookup order: git -> _version.py next to the module -> matching installed
metadata -> "0.0.0+unknown".
"""
import functools
import importlib
import importlib.metadata
import subprocess
from pathlib import Path

UNKNOWN_VERSION = "0.0.0+unknown"

# Distribution name for each importable top-level package.
_DISTRIBUTIONS = {
    "hypso": "hypso",
    "hypso1_calibration": "hypso1_calibration",
    "hypso2_calibration": "hypso2_calibration",
}


def _git(args, cwd: Path):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def version_from_describe(describe: str, dirty: bool, date: str = None) -> str:
    """PEP 440 version from `git describe --tags --long` output, matching
    setuptools_scm's default scheme: on a tag -> the tag's version; past a
    tag -> next micro .dev<distance>+g<sha>; dirty -> local part gets
    .d<YYYYMMDD> (date) or .dirty."""
    from packaging.version import Version

    tag, distance, sha = describe.rsplit("-", 2)
    base = Version(tag.lstrip("v"))
    distance = int(distance)
    if distance == 0 and not dirty:
        return str(base)
    if distance == 0:
        version = str(base)
        local = []
    else:
        # Past a pre-release tag setuptools_scm bumps the pre-release number
        # (v26.10.0a1 + 2 commits -> 26.10.0a2.dev2), otherwise the micro.
        if base.pre:
            kind, number = base.pre
            next_version = f"{base.base_version}{kind}{number + 1}"
        else:
            next_version = f"{base.major}.{base.minor}.{base.micro + 1}"
        version = f"{next_version}.dev{distance}"
        local = [sha]
    if dirty:
        local.append(f"d{date}" if date else "dirty")
    return f"{version}+{'.'.join(local)}"


def _git_identity(package_dir: Path):
    top = _git(["rev-parse", "--show-toplevel"], package_dir)
    if top is None:
        return None
    commit = _git(["rev-parse", "HEAD"], package_dir)
    status = _git(["status", "--porcelain", "--untracked-files=no"], package_dir)
    dirty = bool(status)
    # --abbrev=9: the short-hash length setuptools_scm's versions show.
    describe = _git(["describe", "--tags", "--long", "--abbrev=9", "--match", "v[0-9]*"], package_dir)
    version = UNKNOWN_VERSION
    if describe:
        try:
            date = None
            if dirty:
                from datetime import datetime, timezone
                date = datetime.now(timezone.utc).strftime("%Y%m%d")
            version = version_from_describe(describe, dirty, date)
        except Exception:
            version = UNKNOWN_VERSION
    return {"version": version, "git_commit": commit, "git_dirty": dirty,
            "git_describe": describe, "source": "git"}


def _version_file(module) -> str:
    try:
        return importlib.import_module(f"{module.__name__}._version").__version__
    except Exception:
        return None


def _matching_metadata_version(module) -> str:
    dist_name = _DISTRIBUTIONS.get(module.__name__, module.__name__)
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    module_file = Path(module.__file__).resolve()
    for f in dist.files or ():
        try:
            if Path(dist.locate_file(f)).resolve() == module_file:
                return dist.version
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=None)
def package_info(package_name: str) -> dict:
    """Version and source identity of an importable package ("hypso",
    "hypso1_calibration", "hypso2_calibration"). Keys: name, version,
    git_commit, git_dirty, git_describe, source ("git" | "version_file" |
    "metadata" | "unknown"), path. Cached per process; git is only queried
    when this is called, never at import time."""
    module = importlib.import_module(package_name)
    package_dir = Path(module.__file__).resolve().parent
    info = {"name": package_name, "version": UNKNOWN_VERSION, "git_commit": None,
            "git_dirty": None, "git_describe": None, "source": "unknown",
            "path": str(package_dir)}

    git = _git_identity(package_dir)
    if git is not None:
        info.update(git)
        return info

    version = _version_file(module)
    if version:
        info.update(version=version, source="version_file")
        return info

    version = _matching_metadata_version(module)
    if version:
        info.update(version=version, source="metadata")
    return info
