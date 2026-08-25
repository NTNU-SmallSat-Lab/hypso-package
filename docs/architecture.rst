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

Resolution order (checked in ``hypso.calibration.pipeline.set_calibration_coeff_files``,
which ``generate_l1b_cube()``/``to_l1b()`` reach through
``hypso.calibration.pipeline.run_calibration`` - see the composition section below):
an explicit
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

``HypsoBase`` composition: where the god-object methods went
----------------------------------------------------------------

``HypsoBase`` started this refactor at 2,113 lines / 64 methods spanning five
unrelated concerns. It is now a coordinator of about 1,000 lines; each coherent
slice of what it used to implement inline lives in its own module, as free
functions or adapter methods taking the capture object (``satobj``) as an explicit
first parameter (they read/write many capture attributes, so passing the object in
beats either duplicating that state or holding a back-reference):

- **Georeferencing orchestration** → ``hypso.geo``.
  ``run_georeferencing()``/``run_direct_georeferencing()`` remain as thin
  delegating *methods* because external code calls those names on the object; the
  private ``_run_*`` helpers moved outright.
- **Calibration orchestration** → ``hypso.calibration.pipeline``
  (``set_calibration_coeff_files``/``load_calibration_coeff_files``/
  ``run_calibration``). No wrappers kept - the originals were private with no
  external callers.
- **Capture-file load dispatch** → ``hypso.io.dispatch``
  (``load_capture_file``/``set_hypso_attributes``/``check_capture_type``/
  ``parse_filename``/``compose_capture_name``). No wrappers kept, same reasoning.
- **Atmospheric-correction orchestration** → ``hypso.ac.adapters`` - see the next
  section.

The rule of thumb applied throughout: a name moves *without* a wrapper only if it
was private and a search over this repo and ``hypso-processing-pipeline`` (the
known external consumer) found zero callers; anything public keeps its exact
name/signature on ``HypsoBase`` as a one-line delegation.

Atmospheric-correction adapters (``hypso.ac.adapters``)
-----------------------------------------------------------

One adapter class per external AC tool - ``PolymerAdapter``, ``ACOLITEAdapter``,
``OCSMARTAdapter`` - behind a shared two-method interface
(``run_correction(satobj, **kwargs)`` / ``open_output(satobj, **kwargs)``), plus
per-tool extras (Polymer's SRF/SSI/ESUN NetCDF generators, OC-SMART's
``stage_input``). Registered like sensors: ``get_ac_adapter("polymer")``,
``registered_ac_adapters()``, and the ``AC_ADAPTERS`` namespace exposed as
``satobj.ac`` (so ``satobj.ac.polymer.run_correction(satobj, ...)`` is the
underlying call every ``satobj.ac_polymer_*``-style wrapper method makes).

This pass is **organizational, not a rewrite**: each adapter method body is the
corresponding old ``HypsoBase`` method body relocated verbatim - the same
subprocess/``sys.path``/external-tool-parsing logic, including its ``print``-based
logging. The point is the seam: a future rewrite of, say, the Polymer integration
now has one isolated file to work in (``adapters/polymer.py``) that touches neither
the other tools nor ``HypsoBase``, and a new AC tool means one new adapter module
plus a registry entry rather than another batch of methods on the coordinator.

Two things deliberately *not* adapters: dark-pixel subtraction (no external tool to
run or output file to open - it computes in-memory from the L1D cube; it was
already a free function in ``hypso/ac/ac_dark_pixel_subtraction.py``, still bound
as ``HypsoBase.ac_dark_pixel_subtraction``), and
``hypso.ac.ac_polymer_srf_getter`` (resolved *by dotted-string name* inside
Polymer itself via the ``srf_getter=`` argument, so its import path is frozen
regardless of how the adapters are organized).

**Subprocess isolation (Polymer).** Unlike the other adapters, ``PolymerAdapter.
run_correction`` is not purely organizational - it runs Polymer in a fresh
subprocess rather than importing it in-process. This is a real behavior change,
made for a confirmed reason rather than caution for its own sake: Polymer's v1
(HYPSO-SRF-patched) and v2 (stock) builds ship different, incompatible versions
of the same-named top-level packages (``core`` at least - v2's needs
``core.process.blockwise``, absent from v1's checkout), and Python's
``sys.modules`` import cache pins whichever version is imported first for the
rest of the process's lifetime. A long-lived process that runs a v1 correction
and a v2 correction without restarting (exactly what
``hypso-processing-pipeline``'s ``polymer_version="v1"/"v2"`` selection makes
possible) would get a silent mix of stale and fresh modules from the second
call onward - reproduced directly while building this: importing v1, then v2
after clearing only the ``polymer.*`` cache entries (a natural but incomplete
attempt), raised ``ModuleNotFoundError`` from the still-cached v1 ``core``.

The mechanism (``hypso/ac/adapters/base.py``): ``run_subprocess_driver(python_path,
driver_module, config, tool_name)`` writes ``config`` as JSON into a per-call
``TemporaryDirectory``, invokes ``<python_path> -m <driver_module> config.json
result.json``, and reads back the JSON result - raising ``ACRunError`` (carrying
the subprocess's stdout/stderr and, if available, the tool's own structured
error) on any failure. The per-call temp directory is deliberate: a shared/fixed
staging path is exactly the class of bug ``hypso-processing-pipeline`` had to
work around by hand for OC-SMART (a real concurrent-run collision was observed
there). ``hypso/ac/adapters/_polymer_driver.py`` is the part of
``run_correction`` that actually needs Polymer imported - version-specific
output selection (v1 needs ``polymer.main_v5.default_output_datasets``) and the
``run_polymer()`` call; path resolution and output-file renaming stay in the
parent process, in ``PolymerAdapter.run_correction`` itself.

``run_correction`` gained an optional ``python_path`` parameter (default
``sys.executable``) for this. An initial assumption that Polymer might need a
genuinely separate Python environment (its ``environment.yml`` pins Python
3.12) was checked directly and disproven: the real v1 checkout's
``eoread``/``polymer``/``core``/``eotools`` all import cleanly under whatever
interpreter this package itself runs under - no separate environment is
required by default. A caller that does need one may still pass a different
``python_path``; that environment must have ``hypso`` importable too, since
Polymer resolves ``srf_getter`` by dotted string name back into this package.

ACOLITE and OC-SMART are not yet subprocess-isolated - see
``REFACTOR_PROGRESS.md``'s AC-connector design notes for the fuller proposal
(config dataclasses, a uniform result/exception contract, moving output-path
computation out of ``hypso.io.dispatch``) this is the first slice of.

Keyed cube/mask containers (``hypso.containers.DatasetDict``)
-----------------------------------------------------------------

``HypsoBase._l2a_cubes`` (one entry per AC correction) and ``._custom_masks``
are held in a ``DatasetDict``: dict-style access backed by a real
``xarray.Dataset``. It supersedes the hand-rolled ``DataArrayDict``, which had
three defects: validation printed errors and stored the unvalidated value
anyway; subclassing ``dict`` directly let ``update()``/``setdefault()`` bypass
validation and made membership checks case-sensitive while lookups weren't;
and its ``DataArrayValidator`` inheritance was never actually used.
``DatasetDict`` implements ``collections.abc.MutableMapping`` (every mutation
path funnels through one validated, key-lowercasing core), raises on bad
shapes/dims, and exposes the backing dataset as ``.dataset`` for serialization
or cross-entry xarray operations. One non-obvious guard it adds: xarray's
``Dataset.__setitem__`` silently *reindexes* (truncates) an incoming array
whose dims disagree with existing entries - ``DatasetDict`` checks sizes
explicitly and raises instead. ``DataArrayDict`` itself remains only behind
the intentionally-untouched ``products`` surface.

Spectral response (``hypso.reflectance.spectral_response``)
---------------------------------------------------------------

``SpectralResponse`` is one frozen dataclass holding everything derived from
"Gaussian band responses sampled on an SSI wavelength grid": band centers,
FWHM, binned + unbinned sparse SRF matrices, the grid, the SSI, per-band esun
and effective FWHM - with ``bin_factor``/``ssi_source``/``grid`` as explicit
fields. One builder, ``compute_spectral_response()``, replaces the two
near-duplicate computation paths that existed before (the inline SRF block in
``compute_toa_reflectance`` and the CSIRO-variant ``compute_csiro_srfs``);
those two entry points remain as thin wrappers with unchanged signatures and
outputs (verified bit-identical against pre-change reference outputs).

The capture object's canonical spectral response is ``satobj.spectral_
response`` (and ``.spectral_response_csiro`` for the uniform-grid variant).
The legacy attribute families (``srf``/``srf_ssi``/``srf_ssi_wl``/``esun``/
``esun_wl``/``effective_fwhm`` and ``csiro_*``) are still populated with
identical values because the Polymer connector and the L1D metadata writer
read them - they go away when the AC connectors are migrated to consume
``SpectralResponse`` directly (the planned later AC-connector pass, together
with the eoread/ACOLITE reader updates for the new file layout). Until then
the generated Polymer SRF NetCDF format is frozen: Polymer resolves it via
eotools' ``get_SRF`` (``Band_<n>`` variables on ``wav_Band_<n>`` coords), so
its structure is external API. The SSI/ESUN NetCDFs the connector also writes
are informational only - nothing in the Polymer tree reads them (eoread takes
F0 from its own LISIRD auxdata).

Related renames: ``SensorProfile.srf_wl``/``srf_fwhm`` became
``fwhm_lookup_wl``/``fwhm_lookup_fwhm`` - they were never SRFs, but a
FWHM-vs-wavelength nearest-neighbor lookup table.

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
   No RGB composite recipe ships with the reader - the old ``plot/composites/
   hypso1.yaml`` (written for the hand-built-Scene converter's position-based
   ``band_NN`` names, incompatible with this reader's ``Lt_<wavelength>`` naming)
   was removed as unused rather than ported, per the decision that the Satpy
   integration's job is making HYPSO data loadable as a ``Scene``, not shipping
   visualization recipes. If a composite is ever wanted, Satpy's ``DataQuery``
   wavelength-based prerequisites are the right mechanism (portable across band
   configurations, no hardcoded band indices).
