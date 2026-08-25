Architecture (post-refactor)
=============================

This page documents the packages introduced by the sensor-generalization / CF-NetCDF
refactor. It is a narrative complement to the auto-generated API reference (built
from docstrings via ``autoapi``, see :doc:`index`) - for exact signatures, follow the
API reference links; this page explains *why* things are shaped the way they are and
how to extend them.

.. note::
   This refactor is in progress. Sections below are added as each piece lands, not
   held back until everything is finished - see ``REFACTOR_PROGRESS.md`` at the repo
   root for the full plan and current status if a section you expect is still
   missing.

Sensor profiles (``hypso.sensors``)
------------------------------------

Instrument-specific data (FWHM, spectral response function reference arrays, and how
to resolve calibration coefficient files) used to be hardcoded in each sensor's
``Hypso1``/``Hypso2`` subclass. It now lives in a ``SensorProfile`` - a plain
dataclass registered once, at import time, in ``hypso.sensors.hypso1`` /
``hypso.sensors.hypso2``::

    from hypso.sensors import get_sensor_profile, registered_sensors

    profile = get_sensor_profile("HYPSO-1")
    profile.fwhm        # per-band FWHM array
    profile.srf_wl       # SRF reference wavelengths
    profile.calibration_files(capture_type, coeff_type="moved")  # -> dict of paths

``Hypso1``/``Hypso2`` are still the documented entry points (``Hypso1(path=...)``)
and stay as thin subclasses over ``HypsoBase`` for ``isinstance()`` compatibility -
they just pass their profile to ``HypsoBase.__init__`` instead of setting instrument
constants by hand.

**Adding a future sensor** means adding one new profile module (following
``hypso/sensors/hypso1.py`` as a template) and calling ``register_sensor(...)`` at
import time - no new subclass file is required unless you also want a named,
importable class for it.

Custom calibration coefficients (``hypso.calibration.registry``)
-------------------------------------------------------------------

A sensor's built-in ``coeff_type`` presets (``"moved"``, ``"adjusted"``,
``"original"`` for HYPSO-1/-2) are resolved through the sensor's
``SensorProfile.calibration_files``. To use your **own** radiometric/smile/
destriping/spectral coefficient files - without editing the bundled
``hypso1_calibration``/``hypso2_calibration`` packages - there are two options:

Register a named, reusable set
   .. code-block:: python

      from hypso.calibration import register_calibration_coeffs

      register_calibration_coeffs(
          sat_id="HYPSO-1",
          name="my_lab_calibration",
          files={
              "radiometric": "/path/to/my_radiometric.npz",
              "spectral": "/path/to/my_spectral.npz",
              # "smile" / "destriping" / "spectral_full_frame" default to None
              # (that correction stage is skipped) if omitted
          },
      )

      satobj.generate_l1b_cube(coeff_type="my_lab_calibration")

   Registration is process-global and by (``sat_id``, ``name``) - register once
   (e.g. at the top of a processing script) and refer to the set by name anywhere a
   built-in ``coeff_type`` is accepted.

Pass an explicit one-off set
   For a set you only need for a single run, skip registration entirely:

   .. code-block:: python

      satobj.generate_l1b_cube(coeff_files={
          "radiometric": "/path/to/my_radiometric.npz",
          "spectral": "/path/to/my_spectral.npz",
      })

Resolution order (checked in ``HypsoBase._set_calibration_coeff_files``): an explicit
``coeff_files=`` dict wins if given; otherwise a ``coeff_type`` name is looked up in
the custom registry first, and only falls back to the sensor's built-in presets if
no custom set is registered under that name. Existing calls that only ever passed
``coeff_type="moved"``/``"adjusted"``/``"original"`` are unaffected.

Custom masks (``set_custom_mask`` / ``load_mask_from_file``)
----------------------------------------------------------------

``land_mask`` and ``cloud_mask`` are the two built-in mask slots that
``masked_l1a_cube``/``masked_l1b_cube``/``masked_l1c_cube``/``masked_l1d_cube``
apply (OR'd together, masked pixels set to NaN). To apply an arbitrary
additional mask - e.g. an externally-produced sea/land/cloud classification -
without it having to fit into one of those two slots::

    import numpy as np

    # from an in-memory array
    satobj.set_custom_mask("sea_land_cloud", my_mask_array)

    # or loaded directly from a file (.nc requires variable=, .npy and .dat/.bin
    # are also supported - .dat/.bin uses the same raw-binary + reshape
    # convention as HYPSO's own indirect-georeferencing lat/lon files)
    satobj.load_mask_from_file("slc_mask.nc", variable="mask", name="sea_land_cloud")

    satobj.masked_l1d_cube  # reflects every registered mask automatically

Any number of custom masks may be registered at once (``satobj.custom_masks``
lists them by name); all of them, plus ``land_mask``/``cloud_mask`` if set, are
combined by ``_unified_mask()`` - nothing else needs to change to pick up a
newly-registered mask. ``set_custom_mask(name, None)`` or
``clear_custom_masks()`` removes them.

NetCDF I/O (``hypso.io``)
---------------------------

Replaces the previous one-writer-file-per-product-level design (``write/l1b_nc_writer.py``,
``l1c_nc_writer.py``, ``l1d_nc_writer.py``, ``l2a_nc_writer.py`` - near-duplicates of
each other) with a schema-driven writer:

- ``hypso.io.schema.LevelSchema`` - one instance per product level
  (``L1A_SCHEMA``/``L1B_SCHEMA``/``L1C_SCHEMA``/``L1D_SCHEMA``/``L2A_SCHEMA``),
  captures the product variable prefix/units/long name and, crucially, whether that
  level has geometry at all (``has_geometry``). L1A/L1B are pre-georeferencing and
  structurally cannot get a dangling ``coordinates``/``grid_mapping`` reference as a
  result - the writer has no branch that would let it happen, rather than relying on
  every writer file remembering to check.
- ``hypso.io.cf`` - shared CF (Climate and Forecast convention) attribute builders
  used by the writer for every geolocation and per-band product variable.
- ``hypso.io.writer.write_level_nc(satobj, level, dst_nc, ...)`` /
  ``write_l2a_nc(satobj, correction, dst_nc, ...)`` - the writer itself.

**NetCDF group layout** changed from nested ``products/*`` + ``geometry/*`` groups
to a flat layout: product variables (``Lt_378``, ``rhot_378``, ``rho_w_378``, ...)
and geometry variables (``latitude``, ``longitude``, ``crs_wgs84``, angle variables)
now live at the file's **root** group; only ``metadata/*`` (capture config, timing,
ADCS, corrections, GCPs, SRF) stays nested. This is a deliberate fix, not a stylistic
change: CF's ``coordinates``/``grid_mapping`` attribute resolution only walks
*upward* to ancestor groups, never sideways between siblings - with ``products`` and
``geometry`` as sibling groups, a product variable's ``coordinates`` reference could
never resolve. Flattening to root also matches SNAP's NetCDF/CF reader, which traces
its compatibility back to NetCDF-3.5 (no group support at all), making a flat layout
the safest target for SNAP regardless of the CF resolution issue.

Per-band **named variables** (``Lt_378``, ``Lt_382``, ...) are kept, not switched to
a single stacked ``(band, y, x)`` array - this already matches SNAP's NetCDF4-BEAM
convention. A single-cube mode (one 3D variable, ``datacube=True``) is also still
available for callers that want it.

.. important::
   Because the group layout changed, readers that hard-code the old ``/geometry/*``
   and ``/products/*`` paths (``eoread``'s Polymer HYPSO reader, ACOLITE's HYPSO
   ``l1_convert.py``) will need a matching update before they can read output written
   by the new writer. This is expected, tracked breakage - not a regression - and is
   deferred to a separate pass; see ``REFACTOR_PROGRESS.md``.
