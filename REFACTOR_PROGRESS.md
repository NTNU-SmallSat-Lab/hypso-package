# hypso-package refactor — progress tracker

**If you're a future Claude session picking this up cold:** read this file first, then
`/home/camerop/.claude/plans/rosy-frolicking-summit.md` (the full approved plan — may not survive a session
restart, this file is the durable copy) and `ARCHITECTURE_PROPOSAL.md` (the original research/proposal this plan
implements). This workspace (`~/hypso-package-refactor/hypso-package`) is a git clone the user gave explicit
permission to modify — unlike `/home/camerop/AC/hypso-package`, which is **read-only**, do not touch it.

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

## Status (update this section as work progresses — most recent at top)

- **2026-08-25 (continued further, ×3):** User asked (across a few turns) whether product levels should live in
  separate objects (L1a object produces an L1b object, discard the old one) given real memory concern - large
  datacubes, several potentially resident at once. Discussed prior art (Satpy's one-Scene-per-capture model with
  GC-based disposal; ACOLITE/POLYMER's own disk-per-stage model with no in-memory chain at all) and recommended
  keeping the single coordinator object (levels share geometry/calibration/metadata state that a per-level split
  would have to either duplicate or reference back to a shared container anyway) with an explicit opt-out for
  memory-conscious workflows. Added `HypsoBase.discard_cube(level, correction=None)` plus property deleters
  (`del satobj.l1a_cube`/`l1b_cube`/`l1c_cube`/`l1d_cube`) - `"l1b"`/`"l1c"` free the same underlying array (l1c
  has no independent storage, see its property getter), and `discard_cube("l2a", correction=...)` removes one
  registered AC correction's cube (or all of them, if `correction` omitted) from the `l2a_cube` dict. Verified
  against the real capture: discarding l1a/l1b/l1d individually, discarding via `del satobj.l1c_cube` correctly
  also frees `l1b_cube`, per-correction and discard-all l2a behavior, and an unknown level raising `ValueError` -
  all pass; `tests/baseline/compare_to_baseline.py` still passes. Documented in `docs/architecture.rst`.
- **2026-08-25 (continued further, ×2):** Built `hypso/io/reader.py` — `load_level_nc(nc_file_path, level,
  load_cube=True)` + thin wrappers `load_l1b_nc`/`load_l1c_nc`/`load_l1d_nc`/`load_l2a_nc`, the read-side
  counterpart to `io/writer.py`, resolving the ⚠️ inconsistency noted below. Reuses `load/utils.py`'s group
  readers unchanged for `metadata/*` (unaffected by flattening) and `load_gcp_from_nc_file`; adds a new
  root-level geometry reader (matches a fixed set of known geometry variable names against whatever's at root,
  since geometry and product variables are now siblings there) and a new cube/cube_attrs reader that **sorts
  per-band variables by their `band` attribute** before reconstructing the cube - fixes the plan's flagged
  latent bug (the original per-level loaders used dict/insertion order instead). Also handles L2A's dynamic
  product variable name (AC-tool-specific `l2_variable_name`, e.g. "chla" - the *old* `load_l2a_nc_cube` only
  ever tried a hardcoded `['rrs','Rrs','rho_w']`, so it could never have found "chla" either): tries
  `schema.product_prefix` first, falls back to auto-detecting the actual non-geometry root variable(s) present.
  `load_l1a_nc` is untouched/not migrated - L1A is raw ground-segment input (no `write_l1a_nc_file` exists), not
  something this package's own writer format change affects. Rewired `hypso/load/__init__.py` to source
  `load_l1b_nc`/`load_l1c_nc`/`load_l1d_nc`/`load_l2a_nc` from `hypso.io.reader`.
  **Found and fixed a circular import** introduced by this wiring: `hypso.io.writer` imported from
  `hypso.write.*` at module level, but `hypso/write/__init__.py` imports `write_l1b_nc_file` etc. *from*
  `hypso.io.writer` - triggering `hypso.write`'s init while `hypso.io.writer` was still mid-import raised
  `ImportError: cannot import name 'write_l1b_nc_file' from partially initialized module`. Fixed by moving those
  four `hypso.write.*` imports from module level into the two functions that use them (`_write_metadata_common`,
  `_write_level_nc`) - deferred enough that `hypso.io.writer` is fully loaded by the time they run.
  Verified end-to-end against the real capture: `tests/baseline/compare_to_baseline.py` still passes, and a full
  write-then-read-back round trip (L1C and a fabricated `l2_variable_name="chla"` L2A correction, both
  datacube=True/False) matches the in-memory cube/latitude/wavelengths exactly. **Not yet committed** — do this
  first on resume.
- **2026-08-25 (continued further):** User asked for the ability to load and apply custom masks (e.g. a
  sea-land-cloud mask), beyond the existing hardcoded `land_mask`/`cloud_mask` slots. Added
  `HypsoBase.set_custom_mask(name, value)`/`clear_custom_masks()`/`custom_masks` property (a
  `dict[str, xr.DataArray]`, same validation path as land/cloud via a new shared `_format_mask_dataarray`
  helper — `_format_land_mask_dataarray`/`_format_cloud_mask_dataarray` are now thin wrappers over it) and
  `load_mask_from_file(path, name=..., variable=..., dtype=..., invert=...)` (`.nc`/`.npy`/`.dat`/`.bin`
  supported). `_unified_mask()` now ORs land_mask + cloud_mask + every registered custom mask together, so
  `masked_l1a/b/c/d_cube` pick up custom masks automatically with no other changes. Verified against the real
  capture: no-mask baseline unchanged, a fabricated quadrant mask correctly NaNs only that region of
  `masked_l1a_cube` and leaves the rest untouched, `clear_custom_masks()` reverts to unmasked, and
  `load_mask_from_file(.npy)` round-trips correctly. `tests/baseline/compare_to_baseline.py` still passes
  (confirms this is additive, not a change to the existing calibration/georeferencing path). Documented in
  `docs/architecture.rst`. **Not yet committed** — do this first on resume, then continue to `io/reader.py`
  (was in progress when this mask request interrupted it — see the ⚠️ note right below, still the current state
  of this repo).

  ✅ **Resolved** by the `io/reader.py` entry above (was: `hypso.write` and `hypso.load` disagreed on file
  layout between `289521bd` and the reader landing). `hypso.write`/`hypso.load` are consistent again as of this
  point.
- **2026-08-25 (continued):** Built `hypso/io/writer.py` — `write_level_nc(satobj, level, dst_nc, ...)` and
  `write_l2a_nc(satobj, correction, dst_nc, ...)`, the schema-driven replacement for
  `write/l1b_nc_writer.py`/`l1c_nc_writer.py`/`l1d_nc_writer.py`/`l2a_nc_writer.py`. Implements the flattened
  root-group layout (products + geometry at root, `metadata/*` stays nested), fixes the `radiation_wavelength`
  tuple bug and lat/lon/angle CF attribute bugs via `io/cf.py`, and actually applies compression to geometry
  variables (previously received but ignored). Reuses `calibration_filenames_writer`/`metadata_gcp_group_writer`/
  `metadata_srf_group_writer` from `write/` unchanged (already generic, no bugs found there). Found and fixed
  along the way: `write/l2a_nc_writer.py`'s product variable name/units/long_name are NOT fixed per level —
  they come from `satobj.l2a_cube[correction].attrs['l2_variable_name']` (e.g. "chla" vs "Rrs", set per AC
  adapter) — `write_l2a_nc` now reads this dynamically instead of hardcoding a prefix (`L2A_SCHEMA`'s
  `product_prefix="Rrs"` is only the same-as-original fallback for a missing attribute). Also fixed a latent bug
  in the original `write_l2a_nc_file` wrapper: when writing multiple corrections and only the *first* file
  already existed, it `return`ed and silently skipped writing the rest too (should `continue`) — fixed.
  Added thin backward-compat wrappers `write_l1b_nc_file`/`write_l1c_nc_file`/`write_l1d_nc_file`/
  `write_l2a_nc_file` (exact original names/signatures) and rewired `hypso/write/__init__.py` to import these 4
  from `hypso.io.writer` instead of the old per-level modules — `write_products_nc_file` still comes from
  `write/products_writer.py`, untouched. Verified: all 5 confirmed-external names still importable from
  `hypso.write` with the right `__module__`; every level × `datacube={True,False}` combination (L1B/L1C/L1D/L2A,
  L2A using a fabricated `l2_variable_name="chla"` correction) writes successfully with `Conventions=CF-1.10` and
  correct root-level variable naming; `tests/baseline/compare_to_baseline.py` **still passes** after wiring
  `hypso.write` through the new writer (confirms `HypsoBase`'s calibration/georeferencing/cube-generation path is
  untouched by this I/O change).
  Mid-session, user asked to leave the door open for a future gridded Level 3 product (no L3 generation code
  exists yet - not implemented). Added `LevelSchema.spatial_dims` (default `("lines", "samples")`, every current
  schema keeps the default) and threaded it through `io/writer.py` in place of hardcoded `('lines','samples')`
  dimension-name tuples, so a future L3 schema could declare `spatial_dims=("lat","lon")` and reuse the same
  product/geometry-writing code - dimension *sizing* (currently swath-specific: frame_count/image_height) would
  still need a real L3 branch when L3 is actually implemented; this only keeps that door open, not open it.
  User also asked for docs "as I go along" rather than at the end - added `docs/architecture.rst` (Sphinx,
  wired into `docs/index.rst`'s toctree) covering `hypso.sensors`, the calibration coefficient registry, and
  `hypso.io`'s layout/CF rationale; will keep extending this page alongside each future piece rather than
  writing it in one pass at the end. Not yet committed — next action is to commit `io/writer.py`,
  `hypso/write/__init__.py`, the `io/schema.py` `spatial_dims` addition, and `docs/`.
- **2026-08-25:** Built `hypso/io/cf.py` (CF attribute builders: `global_attrs`, `latitude_attrs`,
  `longitude_attrs`, `zenith_angle_attrs`, `azimuth_angle_attrs`, `crs_wgs84_attrs`, `geolocation_ref_attrs`,
  `band_attrs`) and `hypso/io/schema.py` (`LevelSchema` + `L1A_SCHEMA`/`L1B_SCHEMA`/`L1C_SCHEMA`/`L1D_SCHEMA`/
  `L2A_SCHEMA` + `get_schema(level)`). Both smoke-tested importing/calling cleanly. Grounded in a full read of
  `write/l1c_nc_writer.py`, `write/geometry_group_writer.py`, `write/calibration_filenames_writer.py`,
  `write/metadata_gcp_group_writer.py`, `write/metadata_srf_group_writer.py`, `write/utils.py` — confirmed the
  `radiation_wavelength` trailing-comma tuple bug, the lat/lon `units="degrees"`+blanket `valid_min/max=[-180,180]`
  bug (shared incorrectly with azimuth/zenith), and that geometry compression params are received but never
  applied (commented out). Also confirmed `write/products_writer.py` (backs the `products` property — **do not
  touch**, user's explicit instruction) and `write/l2a_nc_writer.py` (the actual `write_l2a_nc_file` used by
  `hypso-processing-pipeline`, distinct from `products_writer.py`) are separate code paths.
  Mid-session, user asked for an easy way to **plug in custom calibration coefficient sets** — added
  `hypso/calibration/registry.py` (`register_calibration_coeffs(sat_id, name, files)` /
  `get_custom_calibration_coeffs(sat_id, name)`, dict shape matches the bundled `get_hypsoN_calibration_files`
  resolvers: `radiometric`/`smile`/`destriping`/`spectral`/`spectral_full_frame`) and wired it into
  `HypsoBase._set_calibration_coeff_files` with three tiers, checked in order: (1) explicit `coeff_files=...`
  dict for a true one-off set, (2) a `coeff_type` name previously registered via `register_calibration_coeffs`,
  (3) fallback to `self.sensor_profile.calibration_files(...)` (today's built-in "moved"/"adjusted"/"original"
  presets) — unchanged for existing callers. `generate_l1b_cube`/`_run_calibration` already forward `**kwargs`
  through to `_set_calibration_coeff_files`, so `generate_l1b_cube(coeff_files={...})` works with no further
  wiring. Verified: registry smoke test + a full real-capture `generate_l1b_cube(coeff_type='moved')` run still
  gives l1b_cube mean=42.92898... matching baseline exactly. Committed (`326aec45`).
- **2026-08-25, earlier:** `hypso/sensors/` (SensorProfile + registry + hypso1/hypso2 profiles) built, `HypsoBase`
  wired to `sensor_profile`, `Hypso1`/`Hypso2` shrunk to thin subclasses, pre-refactor baseline captured and
  re-verified passing against this point (all committed: `9dfe2952`, `f080a61e`, `884b8d5f`).
- **2026-08-25, session start:** Plan approved. Verified the repo imports cleanly with
  `hypso1_calibration`/`hypso2_calibration` siblings on `sys.path` (no pip install needed).

## Next steps (in order)

1. ~~Capture pre-refactor baseline~~ — done, committed.
2. ~~Build `hypso/sensors/`~~ — done, committed.
3. ~~Build `hypso/io/cf.py`, `io/schema.py`, `calibration/registry.py`~~ — done, committed (`326aec45`).
4. ~~Build/commit `io/writer.py`, `hypso/write/__init__.py` rewiring, `spatial_dims`, `docs/architecture.rst`~~ —
   done, committed (`289521bd`).
5. ~~Build `hypso/io/reader.py`, wire `hypso/load/__init__.py`, fix the band-sort bug~~ — done, committed
   (`43183e49`). `_load_capture_file` needed no code change (already imports load_l1b_nc/etc. from `hypso.load`,
   which now transparently resolves to the new reader).
6. ~~Add `HypsoBase.discard_cube(level)` / cube property deleters~~ — done, verified against real data. Commit
   this (this session's uncommitted work — do this first on resume).
7. Update `HypsoBase`: `self.io`/`self.geo` composition, fix the `self.label` uninitialized-attribute trap
   (**already fixed** as part of the sensor_profile wiring, `884b8d5f`), consolidate
   `run_georeferencing`/`_run_custom_georeferencing`.
8. Extract `hypso/ac/` adapters (`self.ac`), moving `ac_*` method bodies verbatim.
9. Cleanup: delete `.bak` files, delete `ac_6sv1_luts_OLD.py`/`_deepthought.py` (after grep-confirming unused).
10. Build `tests/` (golden-file regression + CF/format assertions + unit tests, per plan §Verification).
11. Run full verification against real data; update this file with results.

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
