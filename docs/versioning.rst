Versioning
==========

``hypso``, ``hypso1_calibration`` and ``hypso2_calibration`` share one version
number. A product file records it (and the calibration coefficient checksums)
so it can be traced to the exact code and coefficients that produced it.

Scheme
------

Calendar versions ``YY.M.N``: year, month, and a release counter within that
month, e.g. ``26.10.0``. Python packaging normalises away leading zeros, so the
existing tag ``v26.02.0`` is version ``26.2.0``.

The version comes from git tags (``setuptools_scm``), not from a number typed
into ``pyproject.toml``:

========================================  ====================================
State of the checkout                     Version
========================================  ====================================
On tag ``v26.10.0``                       ``26.10.0``
3 commits after that tag                  ``26.10.1.dev3+g25f4c3bf``
Same, with uncommitted changes            ``26.10.1.dev3+g25f4c3bf.d20261025``
========================================  ====================================

Making a release
----------------

1. If calibration coefficients changed, update the manifests (below) and
   commit them with the coefficient files.
2. Tag the release commit: ``git tag v26.10.0``. One tag versions all three
   packages.
3. Build or reinstall the packages. ``setuptools_scm`` writes each package's
   ``_version.py`` at build time (these files are git-ignored).

When to release:

* calibration coefficients changed (the manifest test makes this visible);
* the output file format changed (also bump ``product_format_version``, the
  file-layout version written into each product);
* any other change that should be traceable in products.

Version at runtime
------------------

``hypso.__version__`` and ``hypso.versioning.package_info(name)`` describe the
code that is actually imported:

1. If the package directory is inside a git checkout, git is asked directly
   (commit, uncommitted changes, ``git describe``). An editable install's
   ``_version.py`` is written once at install time and goes stale as you
   commit, so it is not used in this case.
2. Otherwise the package's ``_version.py``.
3. Otherwise the installed distribution's metadata, but only if that
   installation contains the imported module. Another copy of the same
   package may be installed in the environment (for example production's
   ``hypso2_calibration`` while a development checkout is on ``sys.path``),
   and its version must not be reported for this copy.
4. Otherwise ``0.0.0+unknown``.

The calibration packages' own ``__version__`` is the plain build-time value;
use ``package_info`` for provenance.

Calibration coefficient manifests
---------------------------------

Each calibration package contains ``coefficients_manifest.yaml``: the SHA-256
of every file under its ``data/`` directory. ``tests/test_versioning.py``
fails if a coefficient file is changed, added or removed without the manifest
being updated, so coefficient changes cannot slip in without a version bump
(as happened in ``04cd011f``, which replaced the HYPSO-2 smile matrix while
the version stayed ``26.02.0``).

.. code-block:: bash

   python -m hypso.calibration.manifest check            # both packages
   python -m hypso.calibration.manifest update hypso2_calibration

Product files record the same SHA-256 for each coefficient file actually
loaded, so a product's coefficients can be compared with any release's
manifest.
