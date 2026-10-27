Calibration coefficients
========================

L1A to L1B calibration applies radiometric, smile, destriping and spectral
corrections, each from a coefficient file shipped in the sensor's calibration
package. The files are versioned, and you choose them with two arguments:

``coeff_type``
    The radiometric variant: ``abs``, ``moved`` or ``lab``. Default ``abs`` for
    HYPSO-2, ``moved`` for HYPSO-1.

``version``
    The coefficient release - HYPSO-2: ``V3`` (default), ``V2``, ``V1``;
    HYPSO-1: ``V2`` (default), ``V1`` - or a dictionary that sets the version
    of individual corrections.

.. code-block:: python

   import hypso

   l1a = hypso.open("....L1A.nc")

   l1b = l1a.calibrate()                                # abs, V3
   l1b = l1a.calibrate("moved", version="V2")           # the pre-V3 "moved" set
   l1b = l1a.calibrate(version={"spectral": "V3"})      # V3, but V3 spectral too

``hypso.processing.l1a_to_l1b(tree, ...)`` takes the same arguments for a
product tree.

Releases
--------

=========  ==========================  ===========  ==============
Release    Radiometric variants        Smile        Spectral
=========  ==========================  ===========  ==============
V3         ``abs``, ``moved``          V3 (Sobel)   **V2**
V2         ``abs``, ``moved``          V2           V2
V1         ``lab``                     V2           V1
=========  ==========================  ===========  ==============

* **V3 uses the V2 spectral calibration on purpose**, for OC-SMART
  compatibility. The V3 spectral calibration moves band centres by +0.3 to
  +1.7 nm, which changes the integer wavelengths in band names (``Lt_<nm>``,
  ``rhot_<nm>``). OC-SMART's HYPSO band table and POLYMER's band settings pick
  bands by those names. You can still select V3 spectral explicitly; the
  package warns when you do.
* **V3** ``abs`` is V3 ``moved`` with adjusted absolute radiometric gains:
  about 11% lower radiance.
* **V2** is the set used before V3 (``moved``, and ``abs``, which used to be
  called ``adjusted``).
* **V1** is the lab calibration (previously ``original``).

The old names ``adjusted`` and ``original`` still work and mean ``abs``/V2 and
``lab``/V1, with a deprecation warning.

HYPSO-1
-------

HYPSO-1 uses the same scheme (``hypso1_calibration``'s
``data/coefficients.yaml``) with its own releases and a different default:

=========  ==========================  ===========  ==============  ==============
Release    Radiometric variants        Smile        Destriping      Spectral
=========  ==========================  ===========  ==============  ==============
V2         ``moved`` (default),        V1           V1              V2
           ``abs``
V1         ``lab``                     V1           V1              V1
=========  ==========================  ===========  ==============  ==============

* The default is ``moved``/V2, HYPSO-1's long-standing default (HYPSO-2's is
  ``abs``/V3).
* ``abs`` is the set previously called ``adjusted``, ``lab`` the one called
  ``original``; the old names still work, with a deprecation warning. The
  coefficient files themselves are unchanged.
* Smile and destriping depend on the capture type (nominal, wide; custom and
  moon captures use the full-frame smile matrix and no destriping), and moon
  captures use the moon radiometric matrix whatever the variant.

Mixing versions
---------------

Pass ``version`` as a dictionary with any of ``radiometric``, ``smile``,
``spectral`` and ``spectral_full_frame``; corrections you leave out come from
the V3 release:

.. code-block:: python

   # V3 radiometric (abs) with V2 smile
   l1a_to_l1b(l1a, version={"radiometric": "V3", "smile": "V2"})

What gets recorded
------------------

Each L1B (and every later level) records in ``metadata/processing/l1b``:
``coeff_type``, ``coefficient_release`` (``custom`` when mixed), the version
of each correction, the coefficient file names, and their SHA-256 checksums.

When calibration runs, the package also logs a summary at INFO level through
Python's ``logging`` module:

.. code-block:: text

   Calibration summary: HYPSO-2, capture type wide, coeff_type abs, release V3
     radiometric  applied V3/abs  h2_radiometric_calibration_matrix_full_v3_abs.npz
     smile        applied V3  smile_correction_matrix_HYPSO-2_full_sobel_v3.npz
     destriping   skipped (no coefficients)
     spectral     applied V2  spectral_array_calibrated_poly_full_v2.npz

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
