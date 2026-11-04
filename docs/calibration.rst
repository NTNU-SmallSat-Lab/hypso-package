Calibration coefficients
========================

L1A to L1B calibration applies radiometric, smile, destriping and spectral
corrections, each from a coefficient file shipped in the sensor's calibration
package. The files are versioned, and you choose them with two arguments:

``coeff_type``
    The radiometric variant: ``abs``, ``moved`` or ``lab``. Default ``abs``.

``version``
    The coefficient release - ``V3.1`` (default), ``V3.1v2spec`` (for
    OC-SMART), ``V2``, ``V1``, for both satellites - or a dictionary that sets
    the version of individual corrections. Production's names ``v31`` and
    ``v31v2spec`` work too; ``V3`` maps to ``V3.1`` with a deprecation warning.

.. code-block:: python

   import hypso

   l1a = hypso.open("....L1A.nc")

   l1b = l1a.calibrate()                                # abs, V3.1
   l1b = l1a.calibrate(version="V3.1v2spec")            # V3.1 with the pre-V3 band names, for OC-SMART
   l1b = l1a.calibrate("moved", version="V2")           # the pre-V3 "moved" set

``hypso.processing.l1a_to_l1b(tree, ...)`` takes the same arguments for a
product tree.

Releases
--------

==============  ==========================  ===========  ========  ==============
Release         Radiometric variants        Smile        Spectral  FWHM
==============  ==========================  ===========  ========  ==============
V3.1 (default)  ``abs``, ``moved``          V3 (Sobel)   V3        V3.1 (in-orbit)
V3.1v2spec      ``abs``, ``moved``          V3 (Sobel)   **V2**    V3.1 (in-orbit)
V2              ``abs``, ``moved``          V2           V2        sensor profile
V1              ``lab``                     V2           V1        sensor profile
==============  ==========================  ===========  ========  ==============

* **V3.1** (2026-10-29, production's default ``v31``): ``abs`` is V3 ``abs``
  rescaled, about 3.7% lower radiance for full-frame captures and 2.9% for
  wide ones (the wide matrix is used as is); ``moved`` is V3 ``moved``. V3
  ``abs`` was V3 ``moved`` scaled to the absolute level of HYPSO-1 relative
  to PACE over Libya-4, about 11% lower.
* **V3.1v2spec** is V3.1 with the V2 spectral calibration, for OC-SMART. The
  V3 spectral calibration moves band centres by +0.3 to +1.7 nm, which changes
  the integer wavelengths in band names (``Lt_<nm>``, ``rhot_<nm>``), and
  OC-SMART's HYPSO band table picks bands by those names. POLYMER needs band
  settings for the V3 names with V3.1.
* **V2** is the set used before V3 (``moved``, and ``abs``, which used to be
  called ``adjusted``).
* **V1** is the lab calibration (previously ``original``).
* **V3** is superseded by V3.1: ``version="V3"`` uses V3.1 with a deprecation
  warning (the ``abs`` gains differ by about 3%).

The old names ``adjusted`` and ``original`` still work and mean ``abs``/V2 and
``lab``/V1, with a deprecation warning.

FWHM and SRFs
-------------

Each band's FWHM comes from a table of lines: the release's ``fwhm`` version
(V3.1: in-orbit FWHM at 8 lines from 430 to 764 nm) or, for releases without
one, the sensor profile's built-in table. It is interpolated linearly between
the lines and constant outside them; before 2026-11 each band took the nearest
line's value. The table is kept in ``metadata/corrections``
(``fwhm_table_wavelength``, ``fwhm_table_fwhm``), so TOA reflectance and the
SRFs use the same FWHM. The SRFs are Gaussians over +/-4 sigma (+/-3 before),
as in production hypso-package 11756523 and h2-onboard-calibration.

HYPSO-1
-------

HYPSO-1 uses the same scheme (``hypso1_calibration``'s
``data/coefficients.yaml``) with its own releases and a different default:

==============  ==========================  ===========  ==========  ========  ===============
Release         Radiometric variants        Smile        Destriping  Spectral  FWHM
==============  ==========================  ===========  ==========  ========  ===============
V3.1 (default)  ``abs``                     V3 (Sobel)   V1          V3        V3.1 (in-orbit)
V3.1v2spec      ``abs``                     V3 (Sobel)   V1          **V2**    V3.1 (in-orbit)
V2              ``moved``, ``abs``          V1           V1          V2        sensor profile
V1              ``lab``                     V1           V1          V1        sensor profile
==============  ==========================  ===========  ==========  ========  ===============

* The default is ``abs``/V3.1, as in production (``v31``); before 2026-11 it
  was ``moved``/V2. V3.1's gains are 0.972 times aria's HYPSO-1 V3; its smile
  is aria's full-frame V3 (Sobel) matrix, cropped for every capture type, and
  its spectral calibration aria's V3.
* ``abs`` is the set previously called ``adjusted``, ``lab`` the one called
  ``original``; the old names still work, with a deprecation warning. The
  coefficient files themselves are unchanged.
* Smile and destriping depend on the capture type (nominal, wide; custom and
  moon captures use the full-frame smile matrix and no destriping), and moon
  captures use the moon radiometric matrix whatever the variant.

What the packages contain
-------------------------

Each calibration package (``hypso1_calibration``, ``hypso2_calibration``)
keeps its files in ``data/``, one folder per correction, and lists which file
each correction uses, by release, in ``data/coefficients.yaml``:

=================  ==========================================================================
Folder             Contents
=================  ==========================================================================
``radiometric/``   Gains: V3.1 ``abs`` (full frame, and a pre-binned wide-capture matrix used
                   as is), ``moved`` (V3.1 and V2), V2 ``abs``, V1 ``lab``; HYPSO-1 also a
                   moon matrix, used for moon captures whatever the variant.
``smile/``         V3 (Sobel, full frame, cropped per capture) for V3.1; V2 (HYPSO-2) or the
                   V1 nominal, wide and full-frame matrices (HYPSO-1) for the older releases.
``destriping/``    HYPSO-1 only: the V1 matrices for nominal and wide captures, used by every
                   HYPSO-1 release including V3.1, as in production. There is no V3 or V3.1
                   destriping, and HYPSO-2 has none in any release.
``spectral/``      Band centres: V3 (V3.1), V2 (V3.1v2spec, V2), the lab centre row (V1).
``fwhm/``          The V3.1 in-orbit FWHM tables; older releases use the sensor profile's
                   built-in table.
``optional/``      HYPSO-2's keystone/distortion model (``h2_distortion_model_v1.npz``), outside
                   the releases and off by default: ``calibrate(keystone=True)`` applies it,
                   with a warning, until it is reviewed.
``reference/``     Files kept for reference that no release selects and calibration never
                   reads (below).
=================  ==========================================================================

The catalog's ``reference_files`` section describes each reference file; in
short:

* HYPSO-2: the lab smile matrix as in production, with its rows reversed (a
  known error, never applied: V1 uses the V2 smile); lab-era full-frame
  spectral calibration matrices (plain and smoothed) and polynomial
  coefficients; engineering-model and pre-launch flight-model spectral
  coefficients; a moon radiometric matrix (HYPSO-2 moon captures use the
  release's gains, as in production); pre-binned 2025 nominal and wide gains;
  a band-centre table.
* HYPSO-1: pre-binned V1 gains for nominal and wide captures (the releases
  crop the full-frame gains instead); a band-centre table.

A test (``tests/test_coefficients.py``) checks that every file in ``data/``
is either selected by the catalog or listed as a reference file, and that
every path the catalog names exists. File names, contents and checksums are
unchanged by the folder layout, so products' provenance (file name and
SHA-256) stays comparable across versions.

Mixing versions
---------------

Pass ``version`` as a dictionary with any of ``radiometric``, ``smile``,
``spectral``, ``spectral_full_frame`` and ``fwhm``; corrections you leave out come from
the V3.1 release:

.. code-block:: python

   # V3.1 radiometric (abs) with V2 smile
   l1a_to_l1b(l1a, version={"radiometric": "V3.1", "smile": "V2"})

What gets recorded
------------------

Each L1B (and every later level) records in ``metadata/processing/l1b``:
``coeff_type``, ``coefficient_release`` (``custom`` when mixed), the version
of each correction, the coefficient file names, and their SHA-256 checksums.

When calibration runs, the package also logs a summary at INFO level through
Python's ``logging`` module:

.. code-block:: text

   Calibration summary: HYPSO-2, capture type wide, coeff_type abs, release V3.1
     radiometric  applied V3.1/abs  h2_radiometric_calibration_matrix_wide_v31_abs.npz
     keystone     off (opt-in: keystone=True)
     smile        applied V3  smile_correction_matrix_HYPSO-2_full_sobel_v3.npz
     destriping   skipped (no coefficients)
     spectral     applied V3  spectral_array_calibrated_poly_full_v3.npz
     fwhm         table V3.1  h2_srf_fwhm_v31.npz

The package does not configure logging itself. To see INFO messages, configure
it in your script, for example ``logging.basicConfig(level=logging.INFO)``.

Adding coefficients
-------------------

The HYPSO-2 catalog is ``hypso2_calibration/data/coefficients.yaml``. Adding a
coefficient file or a release is an edit to that file, plus the new file in
the same directory. Then update the checksum manifest and tag a release (see
:doc:`versioning`)::

   python -m hypso.calibration.manifest update hypso2_calibration

Keystone (optional, unreviewed)
-------------------------------

``calibrate(keystone=True)`` also corrects keystone (spatial misregistration
across the spectrum) with HYPSO-2's distortion model, after radiometric
calibration and before smile. The model is shipped for review and marked not
enabled, so applying it warns; it is off by default, and HYPSO-1 has no model.
What it is and where it came from: :doc:`coefficient_history`.

Where the coefficient files came from is recorded in :doc:`coefficient_history`.
