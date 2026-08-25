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

Cube memory: one object per capture, not one per level
-----------------------------------------------------------

A capture's cubes (``l1a_cube``/``l1b_cube``/``l1d_cube``/``l2a_cube[correction]``)
can each be large, and a single ``Hypso1``/``Hypso2`` object can end up holding
several of them at once if you generate multiple levels in sequence. ``HypsoBase``
stays one object per *capture* rather than splitting into one class per *level*:
the levels share state that isn't level-specific (geometry, calibration
coefficients, capture metadata, sensor identity - all of which already round-trip
through a level's NetCDF file, see the next section), so a per-level split would
either duplicate that shared state or have each level object hold a reference back
to something shared anyway - which just becomes an orchestrating container by
another name, for comparatively little benefit. Two ways to opt out of holding
onto cubes you no longer need, depending on your style:

**Mutate in place, discard explicitly** - matches how
``generate_l1b_cube()``/``generate_l1c_cube()``/``generate_l1d_cube()`` already
work (in place, no return value - this is what `hypso-processing-pipeline` uses
today, unchanged)::

    satobj.generate_l1b_cube(coeff_type="moved")
    # ... use satobj.l1b_cube ...
    satobj.generate_l1d_cube()
    satobj.discard_cube("l1b")   # or: del satobj.l1b_cube
    # satobj.l1b_cube is now None; l1d_cube, geometry, calibration state, etc.
    # are untouched.

``discard_cube(level, correction=None)`` (equivalently, ``del satobj.l1a_cube`` /
``l1b_cube`` / ``l1c_cube`` / ``l1d_cube`` for those four) frees one level's cube.
``"l1b"`` and ``"l1c"`` free the *same* underlying array - ``l1c_cube`` has no
independent storage, it's a georeferenced view over the L1B data (see its property
getter). For L2A, pass ``correction=`` to discard one AC tool's result
(``discard_cube("l2a", correction="polymer")``) or omit it to discard every
registered correction at once.

**One object per cube, produced fresh each time** - ``to_l1b()``/``to_l1c()``/
``to_l1d()`` are the non-mutating counterparts to ``generate_l1b_cube()``/etc.:
each always returns a *new* object holding only that level's cube (plus the small
shared state - calibration coefficients, geometry, capture metadata - carried over
cheaply, not duplicated), and leaves the object you called it on completely
untouched::

    satobj.run_georeferencing(latitudes=lats, longitudes=lons)
    l1b_obj = satobj.to_l1b(coeff_type="moved")
    # satobj.l1a_cube is still there; l1b_obj.l1a_cube is None (cleared once
    # l1b_cube exists), l1b_obj.l1b_cube is populated.
    l1d_obj = l1b_obj.to_l1d()
    # l1b_obj.l1b_cube is still there; l1d_obj holds only l1d_cube.
    del l1b_obj  # or just let it go out of scope

This is a deliberately unconditional rule - call ``to_l1b()``, always get a new
object back - rather than e.g. a `mutate-vs-return` flag on ``generate_l1b_cube()``
itself: a flag would make the call site ambiguous about which behavior you're
getting without checking it or the docs, whereas a distinctly-named method makes it
obvious. ``generate_l1b_cube()``/``generate_l1c_cube()``/``generate_l1d_cube()`` are
unchanged and not being replaced - existing callers (``hypso-processing-pipeline``)
need no changes; migrating to the ``to_l1*()`` family is optional and can happen
later, at whatever pace makes sense for code that's live in production.

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

Satpy reader plugin (``hypso.satpy``)
-----------------------------------------

``hypso-package`` already had *ad hoc* Satpy support - ``hypso.satpy.satpy``'s
``get_l1c_satpy_scene()``/etc. hand-build a ``satpy.Scene`` from an already-loaded
``Hypso1``/``Hypso2`` object. This adds a real, registered Satpy **reader plugin**
(Satpy's standard third-party extension point) so HYPSO L1C/L1D files work through
the same ``Scene(reader=..., filenames=[...])`` interface as any other
Satpy-supported instrument, without loading a HYPSO object first - this is what lets
HYPSO participate in Satpy's cross-sensor tooling (composites, resampling,
multi-instrument scenes) on equal footing::

    from satpy import Scene

    scn = Scene(reader="hypso_l1c", filenames=["capture-l1c.nc"])
    scn.load(["Lt_550", "latitude", "longitude"])

Two reader names, ``hypso_l1c`` and ``hypso_l1d`` (L1B/L2A intentionally not covered
yet - both L1C and L1D have geometry, the case visualization/compositing needs; a
follow-on can add the other levels using the same pattern). Both share one
``HypsoL1FileHandler`` (``hypso/hypso/satpy/hypso_handler.py``, subclassing Satpy's
``NetCDF4FileHandler``) - which level is being read comes from which reader YAML's
``file_patterns`` matched, not a separate code path.

Registered via a ``satpy.readers`` entry point in ``hypso/pyproject.toml`` pointing
at the ``hypso.satpy`` module - Satpy's entry-point discovery looks for an ``etc/``
directory next to that module, which is where the reader YAMLs live
(``hypso/hypso/satpy/etc/readers/hypso_l1c.yaml``/``hypso_l1d.yaml``). **Requires an
installed (not just importable-via-sys.path) ``hypso`` package** - entry points are
read from installed package metadata; an editable install (``pip install -e
hypso/``) is enough and keeps working as source files change.

Per-band datasets (``Lt_378``, ``rhot_378``, ...) aren't statically enumerated in the
reader YAML the way a fixed-band instrument's would be - HYPSO's exact band set
varies by binning/calibration configuration - so ``HypsoL1FileHandler.
available_datasets()`` discovers them dynamically per file, reusing
``hypso.io.reader.list_band_datasets()`` (the same band-discovery-and-sort-by-
``band``-attribute logic ``io.reader``'s own cube loader already uses, promoted to a
public function rather than reimplemented a second time).

.. note::
   The existing ``composites/hypso1.yaml`` RGB recipe references band names from the
   *old* hand-built-Scene converter (``band_89``/``band_70``/``band_59``, i.e.
   position-based names), which this reader plugin's dynamic ``Lt_<wavelength>``
   naming doesn't match - the recipe needs to be re-pointed at specific wavelengths
   (Satpy composites can reference prerequisites by wavelength via ``DataQuery``,
   which would also make the recipe portable across sensors instead of hardcoded to
   HYPSO-1's specific band indices) before ``scn.load(["rgb"])`` works against this
   new reader. Not yet done - needs the real HYPSO-1 band-to-wavelength mapping to
   pick correct values, flagged here rather than guessed.
