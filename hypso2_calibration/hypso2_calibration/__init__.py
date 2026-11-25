from .main import *

# Written at build time by setuptools_scm (shared vYY.M.N tag with hypso).
# hypso.versioning.package_info("hypso2_calibration") gives the git-accurate
# identity for provenance; this is the plain build-time value.
try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0+unknown"
