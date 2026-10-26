# Refactor plan, capture-object era (archived)

> **Archived (2026-10-27).** Moved here verbatim from REFACTOR_PROGRESS.md: the AC-connector design notes, the
> approved-plan summary, the "next steps" list and the full approved plan from 2026-08. All of it describes the
> capture-object design (`HypsoBase` composition, AC adapter classes, `io/writer.py`/`reader.py`/`dispatch.py`)
> that the plain-Dataset rewrite (REFACTOR_PROGRESS.md ×37) replaced; every item was either completed or made
> moot by it. The current design is `docs/architecture.rst`. Commit hashes and entry numbers (×N) refer to
> REFACTOR_PROGRESS.md's status log and this repository's history.

## AC-connector pass — design notes (2026-08-25, not yet implemented)

User asked how the AC connectors could be improved. Grounded in reading the adapters + how
`hypso-processing-pipeline` actually drives them. The decisive evidence: **the pipeline already routes around
the connectors in three places** — treat `hypso_pipeline/ac_runners_hypso.py` as the field-tested spec for what
the connectors should have been:

1. **OC-SMART**: pipeline replaced `stage_input`/`run_correction` entirely (`run_ocsmart_correction`) because
   the connector (a) hardcodes bare `python3` — OC-SMART needs its own pinned Python 3.11 conda env
   (`python_path` param), (b) stages input into OC-SMART's *shared install dir* — a real concurrent-run
   collision was observed on the PACE side; pipeline stages into `capture_dir/ocsmart_staging/`, (c) leaves
   output in the install dir; pipeline redirects it to `capture_dir` and save/restores OC-SMART's global
   `OCSMART_Input.txt` (fixed-path config, no CLI args) even on failure. Only `open_output` is still used.
2. **Polymer**: pipeline bypasses `ac_polymer_open_output` for chla (`_read_polymer_chla`) because the v1
   loader hardcodes a literal "chla" variable; v2 only exposes log-scale "logchl" (pipeline reconstructs
   `10**logchl`). That version handling belongs in the adapter.
3. **ACOLITE**: pipeline mutates `satobj.acolite_dir` from config before calling; EARTHDATA credentials pass
   as plaintext kwargs.

**Future wishlist item, Polymer's side (not this repo — flagging for eventual collaboration with the Polymer
maintainers, per the standing "do not edit the AC processors" instruction):** add a native HYPSO entry to
Polymer's own `eotools/srf.py` SRF resolution (the `srf.csv`-based sensor/platform auto-lookup `get_SRF()`
already does for sensors it recognizes out of the box, e.g. EUMETSAT ones) so Polymer can resolve HYPSO's SRF
itself, the same way it does for those sensors — without hypso-package needing to supply an `srf_getter`
callback (`ac_polymer.ac_polymer_srf_getter`/`SRF_GETTER_PATH`) or generate a per-capture SRF NetCDF
(`generate_srf_nc`) at all. This would remove hypso-package's whole SRF-handoff mechanism as dead weight once
Polymer supports HYPSO natively — the right direction (Polymer owns sensor-specific knowledge for sensors it
supports, not every calling package supplying a getter), just not something achievable from this side alone.

Proposal (changes in hypso-package only; external tools untouched, per standing instruction):
- ~~Fold the pipeline's runner design back into the adapters~~ — **done for all three tools** (Polymer
  `8f0c2f98`, ACOLITE `d13127f2`, OC-SMART `b22388fe` — see the ×18/×19/×20 status entries).
- **Config dataclass per tool** (paths/version/interpreter) instead of 5-kwarg plumbing and satobj.*_dir
  mutation; credentials via env/netrc-style provider, not kwargs.
- ~~Process isolation for Polymer/ACOLITE~~ — **both done**: Polymer `8f0c2f98` (see ×18 status entry: real
  reason confirmed by direct reproduction — v1/v2 builds' same-named `core` package version conflict, not a
  Python-version mismatch as first guessed; that guess was checked and disproven), ACOLITE `d13127f2` (see
  ×19 status entry — no demonstrated version-conflict bug like Polymer's, justified instead by crash
  containment/parallelism/consistency; also moved EARTHDATA credentials off disk onto subprocess env vars
  as a security improvement made along the way). **OC-SMART not migrated onto this shared mechanism** — it
  already runs as a subprocess via `hypso-processing-pipeline`'s own `ac_runners_hypso.
  run_ocsmart_correction`, which this package's `OCSMARTAdapter` doesn't yet replicate (see item below).
- **Uniform contract**: `run_correction -> ACRunResult` (output paths/log); `open_output` registers into
  l2a_cube + returns one consistent shape; typed exceptions instead of print+None (pipeline currently
  defensive-checks a different return shape per tool). Fold `_read_polymer_chla`'s chla/logchl handling in.
- **Output-path ownership moves out of `io.dispatch.load_capture_file`** (it computes acolite/ocsmart output
  paths at *load* time — a coupling smell) into the adapters.
- **Adapters consume `SpectralResponse` explicitly** (generate_* take `sr` + band labels; resolve the
  satobj.wavelengths/fwhm-vs-sr.band_centers subtlety — the SRF nc's `band_wavelength` labels come from
  satobj.wavelengths, which is NOT sr.band_centers). Add a lazy rebuild so a file-loaded satobj can generate
  the Polymer SRF nc (today satobj.srf only exists after an in-session L1D run — the SRF matrix is not
  persisted in our files).
- Known bugs to fix in the same pass: polymer `open_output` has no `case _` (unknown input_product_level →
  NameError); `run_correction`'s dead `run_polymer_kwargs`/`if True:`/`srf_nc_path, srf_nc_path =` typo.
- ~~Open question for the user: the csiro_* path is computed+persisted but consumed by nothing — keep as
  provenance, make it the Polymer SRF source, or drop?~~ **Resolved 2026-08-25: drop.** User confirmed after
  investigation (see the ×17 status entry below) — `satobj.compute_csiro_srfs()` call removed from
  `hypso-processing-pipeline`.

## Approved plan summary

Full plan is in the "Approved Plan" block below (copied verbatim from plan-mode approval). Key decisions, most
important first:

1. **CF/SNAP compliance prioritized over the current AC readers working.** `eoread/hypso.py` (Polymer) and
   `acolite/hypso/l1_convert.py` (ACOLITE) both hard-code `/geometry/*`, `/products/*` group paths and **will
   break** once the NetCDF format changes here. This is expected and accepted by the user — those get updated in
   a separate, later pass. User will keep using the existing unmodified `hypso-package` install for real
   processing until then.
2. **NetCDF format: flatten `products` + `geometry` groups to root.** Keep `metadata/*` nested. Reasoning: CF's
   `coordinates` attribute only resolves group-relative names by walking *up* to ancestors, never sideways to
   siblings — `/products` and `/geometry` are siblings today, so it structurally can't resolve either way. SNAP's
   CF reader traces back to NetCDF-3.5 compatibility (no group support at all), so flat is also the safest layout
   for SNAP. Per-band **named variables** (`Lt_378`, `Lt_382`, …) are kept as-is (already the correct SNAP-BEAM
   convention) — not switching to a stacked `(band,y,x)` array.
3. **Wavelength attributes:** keep both `wavelength` (nominal/rounded) and `radiation_wavelength` (precise,
   fix the confirmed trailing-comma tuple bug, add `standard_name="sensor_band_central_radiation_wavelength"`) —
   they're genuinely different values, not duplicates. Drop only `wave` (true duplicate).
4. **Sensor generalization:** new `hypso/sensors/` package — `SensorProfile` dataclass + registry
   (`get_sensor_profile(sat_id)`), replacing hardcoded per-subclass instrument constants. `Hypso1`/`Hypso2` stay
   as thin subclasses (public API, confirmed exported from `hypso/__init__.py`).
5. **`HypsoBase` gets a real breakup** via composition (not mixins): `self.io`, `self.calibration`, `self.geo`,
   `self.ac` — each wraps a coherent slice of what's currently 2,113 lines / 64 methods in one file. Every
   currently-public method (`.ac_polymer_run_correction()` etc. — confirmed read directly by
   `hypso-processing-pipeline`) keeps working via a thin delegating wrapper.
6. **AC (`hypso/ac/`) stays functionally unchanged** — moved into per-tool adapter classes
   (`PolymerAdapter`/`ACOLITEAdapter`/`OCSMARTAdapter`) behind a shared `run_correction`/`open_output` interface,
   but each adapter's body is today's method body relocated verbatim, not rewritten. This is prep for a *later*
   AC rewrite, not the rewrite itself.
7. **`products`/`_products` property: do NOT touch.** Confirmed zero call sites today, but the user says it's
   intentional forward-looking infrastructure (for user-generated/AC-processor products), not dead code.
8. **Naming:** free to rename anything except the confirmed external-dependent surface — `write_l1b_nc_file`/
   `write_l1c_nc_file`/`write_l1d_nc_file`/`write_l2a_nc_file`/`write_products_nc_file` (imported directly by
   `hypso-processing-pipeline`'s `process_capture.py`) and `Hypso`/`Hypso1`/`Hypso2` plus whatever attribute
   surface the (untouched) `ac_*` methods read.
9. **Add a pytest test suite** under `tests/` — this repo has none today. Golden-file regression against a real
   capture is the primary correctness check (no test infra existed before, so this doubles as the first
   real safety net for this codebase).

Full unabridged plan text: see the "Approved Plan" section pasted at the bottom of this file (added once, not
duplicated per update — if it's missing, check `/home/camerop/.claude/plans/rosy-frolicking-summit.md` instead).

## Next steps (in order)

1. ~~Capture pre-refactor baseline~~ — done, committed.
2. ~~Build `hypso/sensors/`~~ — done, committed.
3. ~~Build `hypso/io/cf.py`, `io/schema.py`, `calibration/registry.py`~~ — done, committed (`326aec45`).
4. ~~Build/commit `io/writer.py`, `hypso/write/__init__.py` rewiring, `spatial_dims`, `docs/architecture.rst`~~ —
   done, committed (`289521bd`).
5. ~~Build `hypso/io/reader.py`, wire `hypso/load/__init__.py`, fix the band-sort bug~~ — done, committed
   (`43183e49`). `_load_capture_file` needed no code change (already imports load_l1b_nc/etc. from `hypso.load`,
   which now transparently resolves to the new reader).
6. ~~Add `HypsoBase.discard_cube(level)` / cube property deleters~~ — done, committed (`0f79edf5`).
6b. ~~Add `to_l1b()`/`to_l1c()`/`to_l1d()`~~ — done, committed (`602acb88`). ~~Deprecate `generate_l1b/l1c/
   l1d_cube()`~~ — done, committed (`d09c272e`).
6c. ~~Register a real Satpy reader plugin (`hypso_l1c`/`hypso_l1d`)~~ — separate plan-mode task, not part of the
   original plan below but done and committed (`3c3354e7`) at the user's request; see the status entry above.
7. ~~`self.io`/`self.geo` composition on `HypsoBase`~~ — **done** (all four slices landed; `HypsoBase.py` is
   down from 2,113 to 1,638 lines):
   - ~~Extract georeferencing orchestration into `hypso/geo.py`~~ — done, committed (`29df4546`).
     `run_direct_georeferencing()`/`run_georeferencing()` stay as thin delegating wrapper methods on
     `HypsoBase` (external callers depend on them); the private `_run_*` helpers moved outright.
   - ~~`self.io` (wrapping `_load_capture_file`'s dispatch through `hypso.io.reader`/`hypso.io.writer`)~~ —
     done, committed (`98a38c85`): `_load_capture_file`/`_set_hypso_attributes`/`_check_capture_type`/
     `_parse_filename`/`_compose_capture_name` extracted verbatim into `hypso/io/dispatch.py`, no wrappers kept
     (zero external callers). Also fixed the pre-existing `hypso.load` ↔ `hypso.io` circular import this
     surfaced — see the status entry above; `hypso/load/__init__.py` now resolves the four reader-backed names
     lazily via PEP 562 `__getattr__`.
   - ~~`self.calibration` (wrapping `_run_calibration`/`_load_calibration_coeff_files`)~~ — done, committed
     (`74990178`): extracted verbatim into `hypso/calibration/pipeline.py` module-level functions
     (`set_calibration_coeff_files`/`load_calibration_coeff_files`/`run_calibration`), no wrapper methods kept
     on `HypsoBase` (confirmed zero external callers of the three private methods).
   - The `self.label` uninitialized-attribute trap is **already fixed** (sensor_profile wiring, `884b8d5f`).
   - ~~Consolidate `run_georeferencing`/`_run_custom_georeferencing`~~ — done, committed: `_run_custom_
     georeferencing` deleted outright (confirmed dead code, zero callers anywhere - see status entry above),
     `run_georeferencing`'s existing optional `latitudes=None, longitudes=None` signature already covers what
     a merge would have produced.
8. ~~Extract `hypso/ac/` adapters (`self.ac`), moving `ac_*` method bodies verbatim~~ — done, committed
   (`42be73d2`): `hypso/ac/adapters/` (base interface + registry + one verbatim adapter per tool), every public
   `ac_*` name kept as a delegating wrapper on `HypsoBase`, `HypsoBase.ac = AC_ADAPTERS`. See status entry ×12
   for what deliberately stayed out of the adapter pattern and why.
9. ~~Cleanup: delete `.bak` files, delete `ac_6sv1_luts_OLD.py`/`_deepthought.py`~~ — done (see status entries
   above for what was deleted and, just as importantly, what was deliberately left alone and why -
   `geometry_definition/` has a real external consumer, `geometry`/`geometry_definition`/`georeferencing` were
   not renamed despite similar names). Also removed the orphaned `hypso/aeronet_oc/` submodule and
   `write/aeronet_oc_writer.py` (`50217513`) after confirming with the user that its only real consumer,
   `hypso-ac-processing`, is superseded by `hypso-processing-pipeline`.
10. ~~Build `tests/` (golden-file regression + CF/format assertions + unit tests, per plan §Verification)~~ —
    done, committed (`f98f8549`): 62 tests (40 unit + 22 real-data, auto-skipping without the reference
    capture), all passing. Run with `python -m pytest tests/` in the `ac` conda env.
11. ~~Run full verification against real data; update this file with results~~ — done continuously: every step
    was verified against the real capture as it landed, and the final suite (item 10) passes in full. Plan
    verification item 5 confirmed: `eoread/hypso.py` and `acolite/hypso/l1_convert.py` still need their
    separate-pass update before they can read the new flat-layout output (expected, accepted breakage).

**Commit frequently** (this repo tracks `origin` = `NTNU-SmallSat-Lab/hypso-package`, but this is the user's own
writable clone — commit locally as each numbered step above completes, don't wait for the whole refactor to
finish, so an interruption doesn't lose more than one step's worth of work).

---

## Approved Plan (verbatim, for reference)

# hypso-package refactor: sensor generalization + CF/SNAP NetCDF format + I/O deduplication

## Context

`ARCHITECTURE_PROPOSAL.md` (already committed to this workspace) identified a 2,113-line/64-method god object
(`HypsoBase`), per-sensor subclasses that hardcode instrument constants inline, and ~5 near-duplicate ~480-line
NetCDF writer files (one per product level) that caused a real, verified bug: L1B files write `coordinates`/
`grid_mapping` attributes pointing at a `geometry` group that L1B files don't have, because that attribute block
was copy-pasted from the L1C/L1D writer without checking it applies. Two Explore agents then fully read
`HypsoBase.py` and the `write`/`load` modules; findings below are grounded in specific file:line references from
those passes, not guesses. The driving motivation for this refactor, per the user, is **future-proofing and
generalization** — this isn't just a bug-fix pass, it's meant to make adding a future sensor, a future product
level, or a future AC tool integration cheap instead of requiring a new hand-copied file each time.

**Decisions revised after initial plan feedback** (both from the user directly, superseding my earlier, more
conservative defaults):
1. **CF/SNAP compliance is prioritized over keeping the current AC readers (`eoread/hypso.py`,
   `acolite/hypso/l1_convert.py`) working.** Those will be updated later, in a separate pass; the user will keep
   using the existing unmodified `hypso-package` install for real processing until then.
2. **AC (`hypso/ac/`, the `ac_*` orchestration methods) stays functionally unchanged this pass**, but should be
   *extracted and organized* so a future rewrite has a clean, isolated target — not left flattened into
   `HypsoBase`.
3. **`HypsoBase` gets a real breakup**, not just light cleanup — it's confirmed too long (2,113 lines).
4. **Submodules may be reorganized and renamed freely** for consistency, beyond just the I/O layer.
5. **`products`/`_products` (currently zero call sites) is intentional forward-looking infrastructure** — for
   user-generated and AC-processor products — not dead code. Leave it exactly as-is.

## Research: SNAP/CF wavelength convention (resolves a real ambiguity, not a guess)

Checked before finalizing the per-band attribute design, since the user specifically flagged uncertainty here:

- SNAP's NetCDF/CF reader documentation (step.esa.int) states it "supports any image-like NetCDF/CF file
  structure **up to NetCDF version 3.5**" — netCDF-3 classic has **no group support at all** (groups are a
  netCDF-4/HDF5 feature). This is a strong, concrete signal that flat (ungrouped) files are the safest target for
  SNAP, independent of the CF-vs-BEAM flavor question below.
- SNAP supports two distinct netCDF conventions: **NetCDF4-BEAM** (practical, per-band-named-variable +
  `wavelength` attribute — exactly HYPSO's current pattern) and **NetCDF4-CF** (strict CF). HYPSO's existing
  per-band-named-variable structure (`Lt_378`, `Lt_382`, …) already targets the BEAM convention correctly — this
  matches what `ARCHITECTURE_PROPOSAL.md` already recommended keeping (not switching to a stacked `(band,y,x)`
  array), and nothing here changes that recommendation.
- The *strict* CF convention for spectral wavelength (Unidata's `EC-netCDF-CF/swath/swath.adoc`) wants wavelength
  as a **coordinate variable** (`float band(band); band:standard_name = "sensor_band_central_radiation_wavelength";
  band:units = "um";`) tied to a shared `band` dimension — which structurally requires the stacked-array layout
  HYPSO deliberately isn't using. Going fully CF-strict here would mean abandoning the per-band-named-variable
  pattern, which is a much larger, different change than this pass is scoped for and contradicts the
  already-approved recommendation to keep it. **Not doing that.**
- What *does* change from my original plan: `radiation_wavelength` is not a redundant duplicate of `wavelength`
  after all — it maps to a real CF standard name (`sensor_band_central_radiation_wavelength`) and the two
  attributes carry genuinely different values in real files (`wavelength=378.5` — nominal/rounded label,
  `radiation_wavelength=378.54673723` — precise as-calibrated value). **Keep both**, fix the confirmed
  trailing-comma tuple bug (`l1c_nc_writer.py:164`) so `radiation_wavelength` is a proper scalar, and add
  `standard_name="sensor_band_central_radiation_wavelength"` to it. Drop only `wave` (confirmed true duplicate of
  `wavelength`, no distinct value).

## Design

### 1. NetCDF group structure: flatten `products` + `geometry` to root

Given priority #1 above, the fix for the sibling-group CF-resolution problem (`/products` and `/geometry` are
siblings; CF's group-relative `coordinates` resolution only walks up to ancestors, never sideways — confirmed in
the original proposal) is now the full fix, not the conservative one: **merge `products/*` and `geometry/*` into
the root group.** `latitude`/`longitude`/`crs_wgs84`/angle variables and every `<param>_<wave>` band variable all
become root-level siblings. `coordinates="latitude longitude"` (relative, CF-standard syntax, no more absolute
`/geometry/...` paths) then resolves correctly for every reader that respects CF group-relative rules, and root-
level variables are the safest possible layout for SNAP given the NetCDF-3.5-heritage finding above.

`metadata/*` (capture_config, corrections, adcs, gcp, timing, srf) **stays nested** — none of it needs CF
`coordinates`/`grid_mapping` resolution (it's provenance/bookkeeping, not spatial data), so nesting it doesn't
hurt compatibility, and keeping it out of the root namespace avoids clutter.

This is the change that breaks `eoread/hypso.py` and `acolite/hypso/l1_convert.py` (both read `/geometry/*`,
`/products/*` explicitly) — acceptable per the user's explicit call, deferred to a later coordinated update.

### 2. Sensor generalization (`hypso/hypso/sensors/`)

Unchanged from the original plan — this part wasn't affected by the feedback. New package:
`SensorProfile` (frozen dataclass) carrying everything `HypsoBase` currently reads from a subclass (confirmed
complete list from the Explore pass): `key`, `sat_id`, `sensor`, `platform`, `fwhm` (unbinned per-band array),
`srf_wl`/`srf_fwhm` (reference arrays for the nearest-neighbor FWHM lookup `_get_fwhm`/`_get_fwhm_unbinned`
already do), and `calibration_files: Callable[[capture_type, coeff_type], dict]` — replacing today's
subclass-defined `_set_calibration_coeff_files`. `hypso/sensors/hypso1.py`/`hypso2.py` hold the actual profile
instances (hardcoded fwhm/srf arrays move here verbatim); `hypso/sensors/__init__.py` holds the registry
(`get_sensor_profile(sat_id)`). Adding a future sensor means adding one profile file — no new subclass file
required, though `Hypso1`/`Hypso2`-style thin subclasses stay available for anything that imports them by name or
`isinstance()`-checks them (confirmed public exports, `hypso/__init__.py`).

`HypsoBase.__init__` gains a `sensor_profile: SensorProfile` parameter and does what subclasses currently do by
hand: sets `platform`/`sensor`/`sat_id`/`fwhm`/`srf_wl`/`srf_fwhm` from the profile, sets `self.label = label`
unconditionally (**fixes the confirmed uninitialized-attribute trap** — `_load_capture_file` does
`if self.label is not None` and base `__init__` never sets a default today), then calls `_load_capture_file`
itself. `_set_calibration_coeff_files` becomes a real base-class method calling
`self.sensor_profile.calibration_files(...)`.

### 3. Break up `HypsoBase` into composed managers, not mixins

Confirmed too long (2,113 lines / 64 methods spanning 5 unrelated concerns). Replace inheritance-based sprawl
with **composition**: `HypsoBase` (or a clearer new name — see naming freedom) becomes a coordinator holding:

- `self.io` — wraps the new schema-driven reader/writer (§4). Owns `_load_capture_file`'s dispatch logic.
- `self.calibration` — wraps `_run_calibration`/`_load_calibration_coeff_files`, now driven by
  `sensor_profile.calibration_files` instead of a subclass override.
- `self.geo` — wraps georeferencing orchestration. Consolidates the confirmed near-duplicate
  `run_georeferencing`/`_run_custom_georeferencing` into one method with an optional lat/lon override, keeps
  `run_direct_georeferencing`/`_run_frame_interpolation`/`_run_track_geometry`/`_run_angles_geometry` as they are
  (already thin wrappers around `hypso/geometry`/`hypso/georeferencing`, no near-duplication found there).
- `self.ac` — **the AC extraction requested in feedback.** Moves every `ac_*` method (Polymer/ACOLITE/OC-SMART/
  dark-pixel-subtraction orchestration — `ac_polymer_run_correction`, `ac_polymer_open_output`,
  `ac_acolite_run_correction`, `ac_acolite_open_output`, `ac_ocsmart_*`, `ac_dark_pixel_subtraction`, plus the
  Polymer SRF/SSI/ESUN netCDF generation helpers) out of `HypsoBase` verbatim — **same subprocess/sys.path/
  external-tool-parsing logic, just relocated**, not rewritten. Structured as one adapter class per AC tool
  (`PolymerAdapter`, `ACOLITEAdapter`, `OCSMARTAdapter`) behind a small shared interface
  (`run_correction(satobj, **kwargs)` / `open_output(satobj, **kwargs)`), registered the same way sensors are
  (§2) — this is what "prepare the ac functions to be refactored" concretely means: today every adapter's
  `run_correction`/`open_output` body is just today's method body moved as-is, but the seam for swapping one
  adapter's internals later (without touching the other two, or `HypsoBase`) now exists.
- Cube generation (`generate_l1b_cube`/`c`/`d`) and masking stay as `HypsoBase`'s own responsibility — this is
  the actual "what is a capture" concern, not an infrastructure concern, and calls into `self.calibration`/
  `self.io` rather than doing calibration/IO inline.

Every currently-public method name callers rely on (`.ac_polymer_run_correction()`, `.l1c_nc_file`, etc. — read
directly by `hypso-processing-pipeline`, confirmed) gets a **thin delegating wrapper** on the coordinator class
(`def ac_polymer_run_correction(self, ...): return self.ac.polymer.run_correction(self, ...)`) so nothing external
needs to change to keep working, even though the AC code isn't being touched internally this pass.

### 4. Schema-driven NetCDF I/O (`hypso/hypso/io/`)

Same core design as the original plan, replacing `write/l1b_nc_writer.py`, `l1c_nc_writer.py`, `l1d_nc_writer.py`,
`products_writer.py`, `l2a_nc_writer.py` and the per-level loaders — confirmed near-duplicates of each other
(Explore agent) — with one schema + one writer + one reader, now targeting the flattened root-group layout from
§1:

- `io/schema.py` — `VariableSpec`/`GroupSpec`/`LevelSchema` dataclasses; one concrete schema per level. L1A/L1B's
  schema has no geometry variables at all (structurally prevents the L1B dangling-reference bug from recurring,
  rather than relying on every writer remembering not to write it).
- `io/cf.py` — shared CF attribute builders: `latitude_attrs()` (`units="degrees_north"`, `standard_name`,
  `valid_min=-90/valid_max=90` — fixes the confirmed copy-paste bug), `longitude_attrs()` (`degrees_east`, verify
  ±180 is actually already correct), zenith vs. azimuth angle attrs with genuinely differentiated valid ranges
  (today both get a blanket `-180/180`), `wavelength_attrs()`/`radiation_wavelength_attrs()` per the research
  above, `global_attrs()` adding `Conventions="CF-1.10"`/`title`/`source`/`history`/`references` (absent today).
- `io/writer.py` — one `write_level_nc(satobj, level, dst_nc, ...)`. Reuses the already-generic shared helpers
  (`geometry_group_writer.py`, `metadata_gcp_group_writer.py`, `metadata_srf_group_writer.py`,
  `calibration_filenames_writer.py`, confirmed not sensor/level-specific) while fixing them: actually apply the
  `COMP_SCHEME`/`COMP_LEVEL`/`COMP_SHUFFLE` params `geometry_group_writer` already receives but ignores today
  (geometry data is currently written uncompressed), fix the `radiation_wavelength` tuple bug, replace the
  duplicated-3x ADCS variable block and group/dimension boilerplate with one shared code path.
- `io/reader.py` — one generic loader. Consolidates `load/utils.py`'s 8 near-identical `load_*_from_nc_file`
  functions into one `load_group(nc_path, group_path) -> (vars, attrs)`; **sorts band variables by their `band`
  attribute** when reconstructing a cube instead of relying on dict/variable insertion order (confirmed latent
  bug in `load_l1c_nc_cube`'s fallback path).
- `write_l1b_nc_file`/etc. (the 5 names confirmed imported directly by `hypso-processing-pipeline`) stay as thin
  wrappers around `write_level_nc` — see naming freedom below for what's *not* pinned.

### 5. Naming and submodule reorganization

- **Stays stable (confirmed external dependent):** `write_l1b_nc_file`, `write_l1c_nc_file`, `write_l1d_nc_file`,
  `write_l2a_nc_file`, `write_products_nc_file` (imported directly by `hypso-processing-pipeline`'s
  `process_capture.py`), and `Hypso`/`Hypso1`/`Hypso2` plus the `HypsoBase` public method/attribute surface
  `ac_polymer_run_correction` etc. depend on (`.l1c_nc_file`, `.parent_dir`, etc.).
- **Free to rename:** the `load_*` function variants (no confirmed external caller besides `HypsoBase` itself,
  which this refactor rewrites anyway), the internal writer-helper names (`*_group_writer` → consistent
  verb-first `write_geometry_group`/`write_gcp_group` alongside `write_level_nc`), `run_georeferencing`/
  `_run_custom_georeferencing` (collapsed to one method, §3), and all brand-new code (`sensors`, `io`, `ac`
  adapter packages) gets clean names with no legacy constraint.
- `geometry/`, `geometry_definition/`, `georeferencing/` currently read as three separate-but-related submodules
  (not deeply investigated this pass) — worth a naming/organization pass consolidating or clearly distinguishing
  them, done opportunistically while building `self.geo` (§3) rather than as a separate up-front investigation.
- Delete `.bak` files repo-wide (git history has them). Delete `hypso/ac/ac_6sv1_luts_OLD.py`/
  `ac_6sv1_luts_deepthought.py` after confirming (grep) nothing imports them.

## Verification and test suite

No test suite exists in this repo today (confirmed — no `pyproject.toml`/`setup.py` at the root, no `test*` files
besides stray `.bak`s). Add one under `hypso-package/tests/` (pytest), and use it as the actual verification
mechanism rather than a one-off manual pass:

1. **Golden-file regression, real data.** Before touching any code: run the current, unmodified L1A→L1D pipeline
   against a real capture (`aeronetvenice_2025-03-04T10-38-05Z` under `/home/camerop/HYPSO_DATA_AOC/`) and save
   cube values/shapes/key attrs as a baseline fixture. `tests/test_regression_real_capture.py` then runs the
   *new* code against the same capture and asserts cube values match exactly (calibration math isn't changing,
   only I/O/structure/class hierarchy, so this should be a strict equality check, not a tolerance-based one) —
   this is what actually catches a refactor-introduced regression, not just a "does it run" smoke test. Skipped
   automatically if `/home/camerop/HYPSO_DATA_AOC/` isn't present (so the suite still runs in a fresh clone).
2. **CF/format assertions**, against the same real-data output: root-level flat layout (no `products`/`geometry`
   groups), `coordinates="latitude longitude"` resolves, `latitude` has correct `units`/`valid_min`/`valid_max`,
   `Conventions` is present, `radiation_wavelength` is a scalar with the right `standard_name`, `wave` is gone,
   L1-equivalent-to-L1B level has no dangling geometry references. Each of these maps directly to one of the
   confirmed bugs from the research phase — one assertion per bug, so a future change that reintroduces one fails
   loudly.
3. **Unit tests, no real data required:** `hypso/sensors/` registry (profile lookup, required-field completeness
   for both HYPSO-1 and HYPSO-2); `hypso/io/cf.py`'s attribute builders (e.g. `latitude_attrs()` returns
   `units="degrees_north"`); `hypso/io/schema.py`'s per-level schemas (L1A/L1B have no geometry group, L1C/L1D/L2A
   do); the `ac.*` adapter registry exposes the expected `run_correction`/`open_output` interface on each adapter
   (structural check only — not exercising the actual subprocess/external-tool calls, since those aren't being
   rewritten and aren't the target of this refactor's correctness guarantees).
4. Confirm `Hypso(path)` still returns a working instance with every attribute the (untouched) `ac_*` adapter code
   reads intact, and every name in `write/__init__.py`/`load/__init__.py`/`hypso/__init__.py`'s current export
   list still imports.
5. Explicitly confirm (and note in the final summary) that `eoread/hypso.py` and `acolite/hypso/l1_convert.py`
   will need updating before they can read the new-format output — expected breakage, not a regression to chase
   down this pass.
