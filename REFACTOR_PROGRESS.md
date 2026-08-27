# hypso-package refactor — progress tracker

**If you're a future Claude session picking this up cold:** read this file first, then
`/home/camerop/.claude/plans/rosy-frolicking-summit.md` (the full approved plan — may not survive a session
restart, this file is the durable copy) and `ARCHITECTURE_PROPOSAL.md` (the original research/proposal this plan
implements). This workspace (`~/hypso-package-refactor/hypso-package`) is a git clone the user gave explicit
permission to modify — unlike `/home/camerop/AC/hypso-package`, which is **read-only**, do not touch it.

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

## Status (update this section as work progresses — most recent at top)

- **2026-08-26 (×28): print() → logging conversion.** User request, independent of the HypsoCapture
  rearchitecture. 174 live `print()` calls across 20 files converted to `logger.info`/`warning`/`error`/
  `exception` (commit `03b604be`) - each file gets its own `logger = logging.getLogger(__name__)`, matching
  the pattern already established in `HypsoCapture.py`/`io/dispatch.py`/`georeferencing/geo.py`.

  Deliberately left as `print()`, not missed: `ac/adapters/ocsmart.py`'s `print(line, end="")` relays an
  external subprocess's raw stdout back through `sys.stdout` in real time (documented in the surrounding
  comment) - a logger call would add timestamp/formatting noise to output meant to pass through verbatim.
  Also excluded: `reflectance/write_ssi_npz.py` and `utils/ncdebug.py` (developer scripts/inspection tools
  whose whole purpose is direct console output) and `ac/loading_acolite_output.py` (a standalone personal
  dev script, zero importers).

  Small adjacent fixes made while touching this code (not a separate pass): several bare/unused-variable
  `except:` narrowed to `except Exception:`; one re-raise in `calibration/correction.py` now chains with
  `from ex`; a mismatched-variable bug in `load/ocsmart_h5_loader.py`'s error message (reported the wrong
  loop variable) fixed; five `ac_*_open_output` call sites across the AC adapters switched from the
  deprecated `l2a_cube` alias to `l2a_cubes`, avoiding a spurious `DeprecationWarning` on every AC correction
  load.

  138 tests pass (full suite, unchanged - no new tests needed, this is a pure diagnostics-output change with
  no behavior difference for anything the test suite observes).

- **2026-08-26 (×27): L1ACapture — the typed hierarchy now spans L1A through L2A.** User asked when
  `HypsoCapture` splitting into per-level classes could begin; answered directly that additive `L1ACapture`
  (mirroring `L1BCapture`/etc.) had no blocker and could start now, while retiring `HypsoCapture` itself needs
  Phase C (pipeline migration, still blocked) first. User chose to add `L1ACapture` now (commit `db30f2c3`).

  Reached via new `HypsoCapture.to_l1a()` — `Hypso(path)` is completely unchanged, still the load entry point;
  `.to_l1a()` retypes an already-loaded object rather than adding a second loader. No generation step —
  `spawn_l1a` just renames cube storage from the old class's `_l1a_cube` to the uniform `_cube` every class
  other than `L2ACapture` uses. `spawn_l1b` generalized to accept either source shape (`_l1a_cube` from
  `HypsoCapture`, or `_cube` from a new `L1ACapture`), since both are now real entry points into `to_l1b()`.
  `L1ACapture` also gets `to_l1c()`/`to_l1d()` convenience one-hops, matching `HypsoCapture`'s own ergonomics.

  138 tests pass (full suite; 3 new in `tests/test_capture_types.py`). `docs/architecture.rst`'s "Tracking
  which level an object represents" section updated to mention `L1ACapture`.

- **2026-08-26 (×26): georeferencing angle/track consolidation — the "HypsoBase → HypsoCapture" plan's last
  remaining piece, closed out.** Re-scoped fresh (Explore + Plan agent, re-verified rather than executing the
  stale original plan) since `capture_types.py` (×25) didn't exist when this was first drafted and turned out
  to duplicate one of the two bugs fixed here.

  New `hypso/georeferencing/geo_state.py`: `GeoAngles` (5 fields)/`TrackGeometry` (3 fields), frozen
  dataclasses, held as `self.angles`/`angles_direct`/`track`/`track_direct` — frozen because
  `capture_types.spawn_as`'s shallow `__dict__` copy would otherwise let a mutable instance's in-place field
  mutation silently corrupt both the spawned object and its source. `framepose` (no direct variant) stays a
  flat attribute, untouched. `HypsoCapture` gets 10 read-only compatibility properties (all 10 angle names,
  not just the 3 confirmed externally-load-bearing ones) — the reason `io/writer.py`'s existing lookup table
  and `write/geometry_group_writer.py`'s hardcoded reads needed **zero changes**, honoring the earlier
  instruction to leave that specific writer alone. `georeferencing/geo.py`'s two writers and
  `io/dispatch.py`'s load-path writer (`set_hypso_attributes`) updated to construct whole instances instead of
  17 individual `setattr`s.

  **Two real, independently-duplicated bugs fixed**: `HypsoCapture._generate_l1d_cube_impl` and
  `capture_types._spawn_l1d` both had `hasattr(satobj, 'solar_zenith_angles_direct')` checks that would have
  silently and permanently broken (always `True`) once that name became a property — fixed to
  `angles_direct.solar_zenith is not None`. The `capture_types.py` copy was also a latent, already-real crash:
  `L1BCapture` never had that property/attribute at all, so `use_direct_georef=True` would have raised
  `AttributeError` the first time anyone exercised it (untested until this pass added coverage).

  129 existing + 6 new tests (`tests/test_geo_state.py`) pass, verified in two separate runs. Commit
  `bfc0d722`. This closes out the entire "HypsoBase → HypsoCapture" plan from ×24/×25 — nothing left pending
  from it except Phase C/D (pipeline migration, blocked - see ×25's entry).

- **New TODO (user request, ×29): Zarr support.** Not scoped or implemented - recorded with full architectural
  thoughts in `ARCHITECTURE_PROPOSAL.md`'s new section 7. Short version: bigger than "add another writer" -
  the current writer is netCDF4-API-native (not xarray-native), Zarr stores are directories not files (touches
  every `*_nc_file`/`Path`/`.is_file()` assumption), the per-band-named-variable layout was chosen specifically
  for SNAP (which doesn't read Zarr at all), and it would be additive alongside NetCDF, never a replacement
  (SNAP/Polymer/ACOLITE only read NetCDF). Recommendation: don't take this on opportunistically - scope it on
  its own, ideally after the eoread/ACOLITE NetCDF coordination below has landed, and only once there's a
  concrete driving use case (vs. speculative future-proofing).

- **2026-08-26 (×30): `compute_csiro_srfs`/`ac_dark_pixel_subtraction` cleanup — the ×24 TODO acted on.**
  Before removing anything, widened the "zero external callers" check beyond `hypso-processing-pipeline` (the
  only repo the ×24 TODO had actually verified) — found `compute_csiro_srfs()` still had a real, live caller:
  `/home/camerop/AC/hypso-ac-processing/csiro.py`, `2b_process_capture.py`, and `aoc_pipeline/03_process_capture.py`
  all called `satobj.compute_csiro_srfs()` and read the `csiro_*` attributes directly, and that repo had commits
  as recent as 2026-08-02 — not obviously dead. Checked in with the user before removing anything (its own
  `config.py` docstring says `hypso-processing-pipeline` "consolidates" `hypso-ac-processing`'s config system,
  suggesting predecessor/superseded, but that alone wasn't proof it had actually stopped being run). **User
  confirmed `hypso-ac-processing` is fully superseded** — with that, `compute_csiro_srfs` genuinely has zero
  remaining callers anywhere.

  Removed outright (not just re-deprecated): `compute_csiro_srfs()` itself (`reflectance/toa_reflectance.py`),
  its export from `reflectance/__init__.py`, its `HypsoCapture` binding, the `csiro_*` attribute-restore loop in
  `io/dispatch.py`'s `set_hypso_attributes` (the "missing comma" one — see the ×17 entry below for that bug's
  history), and the five `csiro_*` NetCDF-variable write blocks in `write/metadata_srf_group_writer.py`.
  `compute_spectral_response(grid="uniform-1000")` (already existed, verified equivalent) is the direct
  replacement for anyone who still needs that exact grid/SSI combination. `spectral_response.py`'s module
  docstring updated to stop claiming `compute_csiro_srfs` is "still a thin wrapper" (no longer true) and record
  the removal + its replacement.

  `ac_dark_pixel_subtraction`: still actively used (`hypso-processing-pipeline/stage2_ac/process_capture.py:421`,
  `ac_runners_pace.py`'s direct import of `dark_pixel_subtraction_per_band`) — not removable, and the *move*
  question (does it belong in the pipeline's own `ac_runners_hypso.py` instead) needs `hypso-processing-pipeline`
  edits, off-limits right now. Did the packaging-consistency piece that's actionable within this repo alone:
  `hypso/ac/adapters/__init__.py`'s `AC_ADAPTERS` namespace now also exposes `dark_pixel_subtraction` (a bare
  function, not an `ACAdapter` instance — its own docstring already explained why DPS doesn't fit
  run_correction/open_output: no external tool, no output file, pure in-memory computation from the L1D cube —
  that reasoning stands, not overridden), so `self.ac.dark_pixel_subtraction(satobj, ...)` now reads consistently
  alongside `self.ac.polymer/.acolite/.ocsmart`. `HypsoCapture.ac_dark_pixel_subtraction` changed from the
  obscure `from X import fn; fn = fn` class-body binding to a real thin delegating method
  (`return self.ac.dark_pixel_subtraction(self, ...)`), matching the `ac_polymer_run_correction ->
  self.ac.polymer.run_correction(self, ...)` pattern already used for the three real adapters. Zero external
  contract change — `satobj.ac_dark_pixel_subtraction()` still works exactly as before (confirmed by
  `tests/test_public_api.py`'s existing callable-surface check, which already expected this exact name).

  Full suite run after both changes to confirm no regressions.

- **2026-08-26 (×31): capture-dimensions audit — the ×25/×26 TODO acted on, uncovered a live calibration
  bug.** Audited every hardcoded dimension/capture-type assumption in `hypso-package`
  (`spatial_dimensions`/`frame_count`/`row_count`/`column_count`/`bin_factor`/`check_capture_type`). Most of
  these (`spatial_dimensions` etc.) are correctly *derived per-capture* from the loaded file's own metadata,
  not hardcoded — not fragile. The one genuinely hardcoded, sensor-agnostic piece: `io/dispatch.py`'s
  `check_capture_type()` classified every capture into `"nominal"`/`"moon"`/`"wide"`/`"custom"` via one flat
  `if/elif` chain (`frame_count == 956`/`== 106`, `image_height == 1092`) with **no reference to
  `satobj.sensor_profile`** — a design gap against `hypso/sensors/`'s own registry (built earlier this
  session specifically so per-sensor differences don't need central dispatch code).

  **Confirmed live bug, not just future-sensor risk**: `capture_type` feeds `sensor_profile.calibration_files(
  capture_type, ...)`, which delegates to the installed `hypso1_calibration`/`hypso2_calibration` packages.
  Reading their source: `hypso1_calibration.get_hypso1_calibration_files`'s `match capture_type:` had cases
  for `"custom"`/`"nominal"`/`"wide"` but **none for `"moon"`** — fell through to the catch-all, returning
  radiometric/smile/destriping/spectral coefficient files all as `None`. `calibration/pipeline.py`'s
  `load_calibration_coeff_files` wraps each load in a bare `try/except: satobj.X = None`, and `run_calibration`
  gates each step on `if satobj.X_coeffs is not None`, so **any HYPSO-1 capture with `frame_count == 106`
  silently skipped all four calibration steps** — no error, no warning, uncalibrated L1B data. Found
  `radiometric_calibration_matrix_HYPSO-1_moon.npz` already shipped, unused, in `hypso1_calibration`'s
  `data/` — strong evidence a "moon" case was intended but never wired up.
  (`hypso2_calibration.get_hypso2_calibration_files` has the opposite problem — it ignores `capture_type`
  entirely, so the classification has zero effect on HYPSO-2's calibration file selection either way; left
  as-is, no evidence this is wrong for HYPSO-2 specifically.)

  Grepped `hypso-processing-pipeline`/`hypso-ac-processing` for "moon" — no hits, so this may be latent (moon/
  lunar-calibration captures possibly processed out-of-band) rather than actively firing today; couldn't
  confirm from code alone.

  User directed both fixes:
  1. **`hypso/sensors.SensorProfile`** gained a new `capture_type_thresholds: tuple[tuple[str, str, int], ...]`
     field — ordered `(capture_type, satobj_attr, expected_value)` rules, first match wins, no match →
     `"custom"`. `hypso1.py`/`hypso2.py` each now declare their own (currently identical — no evidence HYPSO-2's
     frame geometry differs, so this preserves today's actual behavior exactly) `CAPTURE_TYPE_THRESHOLDS`.
     `check_capture_type()` rewritten to iterate `satobj.sensor_profile.capture_type_thresholds` generically
     instead of hardcoding — a future sensor with different dimensions now just declares its own thresholds at
     registration, no central-dispatch-code change needed, matching the registry's existing design intent.
  2. **`hypso1_calibration/hypso1_calibration/main.py`** (confirmed part of this same editable git clone,
     `hypso-package-refactor/hypso-package/hypso1_calibration/` — not an external/off-limits package) gained a
     `case "moon":` branch: radiometric uses the pre-existing unused moon file; smile falls back to the same
     full-frame matrix the `"custom"` case already uses (`read_coeffs_from_file`'s `'smile'` branch in
     `calibration/correction.py` crops full-frame matrices to the capture's actual AOI/bin_factor unless the
     path contains `'wide'`/`'nominal'` — the existing, designed fallback path for non-standard geometries, not
     a new mechanism); destriping skipped, same as `"custom"`. This is an inference by analogy to how `"custom"`
     already handles unrecognized geometries, not verified against calibration domain documentation — flagged
     to the user as a judgment call, not asserted as definitively correct science.

  Found and worked around a real environment gotcha while verifying: `pip install -e .` for
  `hypso1_calibration` switched it to setuptools' PEP 660 finder-based editable install, which broke resolution
  when Python is invoked from `hypso-package`'s own repo root — that directory contains a bare
  `hypso1_calibration/` subdirectory (the calibration package's own project root), which `PathFinder` resolves
  as an empty namespace package *before* the custom editable finder (appended later in `sys.meta_path`) ever
  gets consulted, since the prior real install (a plain, non-editable site-packages copy) no longer exists to
  short-circuit that. Reverted to a normal non-editable reinstall (`pip install <path>`, no `-e`) to restore
  the working configuration while keeping the fix.

  New tests in `tests/test_sensors.py`: `capture_type_thresholds` well-formedness for both profiles,
  `check_capture_type`'s classification against a fake capture (all four outcomes), and a regression test
  confirming `hypso1_calibration`'s `"moon"` case now returns real (non-`None`) calibration files.

- **2026-08-26 (×32): eoread/ACOLITE/OC-SMART reader coordination — the deferred format-migration TODO
  acted on.** User set up sibling checkouts of Polymer/eoread/eotools/core, ACOLITE, and an OC-SMART
  installation zip under `/home/camerop/hypso-package-refactor/` (all confirmed already on a `refactor`
  branch/tag where applicable) and asked for the three readers' HYPSO-specific code to transparently support
  both this package's current flat NetCDF layout (products/geometry at file root, since the ×20-era CF/SNAP
  refactor) and the original grouped layout (`products`/`geometry` HDF5 groups) — `hypso-processing-pipeline`
  hasn't migrated onto the new writer yet, so both are genuinely live (confirmed: a real pipeline output
  dated 2026-08-25 is still grouped-layout).

  Generated real reference files first (`hypso/io/writer.py`'s `write_level_nc`/`write_l2a_nc` against the
  same reference capture the test suite uses, both `datacube=False` — the actual pipeline default — and
  `datacube=True`) and inspected them with `h5py`/`xarray` directly, exactly as each external reader does,
  rather than reasoning about netCDF4/HDF5 attribute serialization from writer-side code alone. Caught two
  things this way that pure code-reading would have missed: scalar attributes round-trip through h5py as
  1-element arrays in *both* layouts (so `att['wavelength'][0]`-style indexing in existing reader code needed
  no change), and `xarray.open_dataset` auto-promotes `latitude`/`longitude` to coordinate variables once
  they're resolvable in the same (root) group as the `coordinates=` attribute referencing them — new in the
  flat layout, and the direct cause of a `MergeError` hit and fixed below.

  - **`eoread/hypso.py`** (`Level1_HYPSO`): added `_hypso_format()` (cheap `h5py` open, checks for a
    top-level `products` group) and `_read_hypso_l1c_products_and_geometry()`, which returns
    `(ds_root, ds_nav, ltoa, wave_names, wavelengths)` for either layout — the only extraction step that
    actually differs; F0/Rtoa computation and attribute assembly downstream are completely unchanged, since
    they only depend on this unified result. Also handles a single stacked `(lines,samples,bands)` `Lt`
    datacube variable (only possible in the flat layout, `write_level_nc(datacube=True)`), and sorts band
    variables by each one's own `band` attribute rather than name/insertion order (matches the same
    band-order-bug-avoidance convention this package's own reader already uses). Removed the speculative
    `Level1_HYPSO_future` (targeted a `navigation`-group + stacked-`Lt` layout that never materialized — the
    real new layout is flat-root, now handled by `Level1_HYPSO` directly, so keeping it would misrepresent
    what layouts actually exist). Fixed a real bug found while verifying: reading `latitude`/`longitude` off
    the flat root dataset raised `MergeError` (xarray couldn't tell if the assignment target should be a
    coordinate) — `reset_coords()` demotes them back to plain data variables first. Verified end-to-end
    (not just import) against three real files: the 2026-08-25 grouped-layout pipeline output, and both
    per-band/datacube variants of a freshly-generated flat-layout file — sensible `Ltoa`/`Rtoa` values in all
    three, non-zero away from a genuinely-low-response band confirmed identical in the real production file
    too. Added `tests/test_hypso.py` coverage (skipped if the grouped-layout sample path isn't present).
  - **`acolite/hypso/l1_convert.py`**: added `_hypso_products_and_geometry_groups(f)`, returning
    `(products, geometry)` h5py group-like objects pointing at the real groups (old layout) or the file root
    itself (new layout). Every `f['/products']`/`f['/geometry/']` reference replaced with these — the rest of
    the function (band info, geolocation, the actual radiance→reflectance conversion, the per-band write
    loop) is untouched. Also made per-band ordering explicit (sorted by each variable's `band` attribute,
    same reasoning as eoread) instead of relying on h5py's iteration order happening to match — needed one
    numpy-version fix along the way (`int()` on a genuinely 1-element array raises on this numpy; unwrap via
    `.reshape(-1)[0]` first). Verified with real `l1_convert()` runs (not just import) against all three file
    shapes for both L1C and L1D — real ACOLITE output files with sensible `rhot_*` reflectance values.
  - **OC-SMART** (`OCSMART_Linux_v2.6.3/src/L1B.py`, not a git repo — edited in place, no commit possible):
    found this file already had a `_2025`/`_2026`-dated fallback chain for HYPSO geometry lookups
    (`/navigation/*_indirect` → `/navigation/*` → `/geometry/*`, with "keep these options here for backwards
    compatibility" comments) — evidently NTNU's own prior attempt at exactly this kind of format-migration
    resilience, just not yet extended to the flat layout. Added one more fallback tier (`latitude`/
    `longitude`/`solar_zenith`/`solar_azimuth`/`sensor_zenith`/`sensor_azimuth` directly at file root) to each
    of the six lookups, following the file's own established style. Also fixed the `rhot_<wavelength>`
    product-variable lookup (`readl1b`'s HYPSO_HSI branch): was `np.array(f['/products'])`, unconditionally
    assuming a `/products` group; now checks for it and falls back to filtering root keys by `rhot_` prefix
    (the flat layout's root holds many other unrelated names a raw `/products` group never did). Since
    OC-SMART needs its own pinned conda environment (basemap/GDAL/etc., confirmed incompatible with this
    session's environment per the adapter's own docstring) full end-to-end execution wasn't feasible here —
    verified by reproducing the exact modified lookup logic standalone (h5py/numpy/re only, no OC-SMART
    class instantiation needed) against real old- and new-layout L1D files, confirming correct geometry
    shapes and sensible reflectance-input values in both.

  hypso-package itself: no changes — all edits landed in the three sibling tool checkouts.

- **2026-08-26 (×33): fixed a real bug found opening hypso-package's own output in ESA SNAP — `metadata/
  gcp/latitude`/`longitude` name-collided with the real per-pixel geolocation.** User reported SNAP couldn't
  open a current-format L1D file and shared its log — `java.lang.ArrayIndexOutOfBoundsException: Index 321589
  out of bounds for length 29` inside `org.esa.snap.core.dataio.geocoding.util.RasterUtils.
  computeResolutionInKm`, called from `CfGeocodingPart.readPixelBasedGeoCoding`.

  Diagnosed by elimination with real files rather than guessing: generated a minimal reproducer (just
  `latitude`/`longitude` + one dummy CF-referencing band, real lat/lon values copied from an actual capture,
  no groups/metadata at all) — opened fine, ruling out the earlier ×32-era worry that SNAP's CF reader
  rejects netCDF4 group structure outright. Added back all 120 real bands with real per-band attributes —
  still opened fine, ruling out band count. Then added just the `metadata/gcp` group (ground-control-point
  tie points from HYPSO's raw georeferencing, carried through from L1A) — **crashed identically**. Its
  `latitude`/`longitude` variables are exactly length 29 — the same 29 from the stack trace. Root cause:
  those two variables carry no `standard_name`/`units` of their own to distinguish them from real coordinate
  variables, so SNAP's CF/geocoding scanner finds them by bare name alone, treats them as a second candidate
  geolocation source alongside the real per-pixel root `latitude`/`longitude` (598×1092 = 653,016 pixels),
  and ends up indexing the 29-element GCP array with a full-raster pixel offset.

  Confirmed zero consumers of the exact netCDF variable names anywhere — `load_gcp_from_nc_file` (load/
  utils.py) reads whatever keys exist under `metadata/gcp` generically (`for key in group.variables.keys()`),
  no code anywhere reads `satobj.metadata.gcp.vars["latitude"]` by that literal key, and grepping
  `hypso-processing-pipeline`/eoread/ACOLITE/OC-SMART turned up nothing either. Fix: `write/
  metadata_gcp_group_writer.py` now writes `metadata/gcp/gcp_<key>` instead of `metadata/gcp/<key>`
  (`gcp_latitude`/`gcp_longitude`/`gcp_sourceX`/`gcp_sourceY`) — a one-line rename, no reader-side change
  needed. Verified end-to-end, not just reasoned about: user confirmed a freshly-generated real L1D file (full
  `metadata` group, `esun`/`effective_fwhm`, everything - not a stripped reproducer) now opens in their SNAP
  install. Full test suite re-run after the change, no regressions.

  Also corrected an unrelated false alarm from the same investigation: a reproducer script of mine displayed
  string attributes as literal `b'...'` text in SNAP - that was `str()` called on a raw `numpy.bytes_` value
  in *my own test script*, not a bug in hypso-package's actual writer (verified directly: the real file's
  `long_name` etc. round-trip as clean Python `str` via both `netCDF4` and raw `h5py`).

- **2026-08-26 (×34): added `effective_fwhm_unbinned`.** User asked to confirm `metadata/srf/effective_fwhm`
  is post-binning FWHM (yes — confirmed from source: `spectral_response.py` computes it via
  `compute_effective_fwhm(srfs_csr=binned_srfs_csr, ...)`, an empirical half-max-width measurement on the
  *binned* SRF, distinct from the plain per-band `fwhm` product attribute, which is just the nominal
  lookup-table value at binned band centers). User then asked whether "effective_fwhm" is a good name
  (yes — it's standard hyperspectral-calibration terminology for "measured from the actual SRF," as opposed
  to a nominal spec value) and pointed out there should be a pre-binning counterpart too — correct: no
  unbinned effective FWHM existed anywhere; `fwhm_unbinned` was only ever the *nominal* value fed in to build
  the unbinned Gaussian SRFs, never something empirically measured back out of them.

  Added `SpectralResponse.effective_fwhm_unbinned` (`reflectance/spectral_response.py`) — the same
  `compute_effective_fwhm` function, called against the pre-binning `srfs_csr` instead of `binned_srfs_csr`
  (already in scope right before `bin_srf` runs, one extra line). Threaded through everywhere
  `sr.effective_fwhm` already flowed: `HypsoCapture.spectral_response`'s lazy rebuild-from-file path,
  `_generate_l1d_cube_impl`, and `capture_types._spawn_l1d`. Persisted to
  `metadata/srf/effective_fwhm_unbinned` (`write/metadata_srf_group_writer.py`, mirrors the existing
  `effective_fwhm` block exactly) and restored on load (`io/dispatch.py`, mirrors the existing
  `effective_fwhm` restore exactly) — matches this codebase's established `_unbinned`-suffix convention
  (`wavelengths`/`wavelengths_unbinned`, `spec_coeffs`/`spec_coeffs_unbinned`). Deliberately did NOT touch
  `compute_toa_reflectance`'s legacy 7-tuple return (its docstring explicitly commits to "unchanged" for
  existing callers) or `ac/adapters/polymer.py`'s per-band `effective_fwhm` lookup (no identified need for an
  unbinned variant there).

  Verified against real capture data, not just unit-level: `effective_fwhm_unbinned` (~5.46 nm) closely
  recovers the nominal `fwhm_unbinned` value the Gaussian SRFs were built from (small discretization error,
  as expected), while `effective_fwhm` (binned, ~6.0 nm) is measurably larger — correct physics, since binning
  sums neighboring bands' SRFs together and widens the combined response. Confirmed the full write→read round
  trip on a real L1D file. New test `test_effective_fwhm_unbinned_persisted` in `tests/test_cf_format.py`
  asserts both the persistence and this binned-vs-unbinned relationship. Full suite re-run, no regressions.

- **2026-08-27 (×35): capture-dimensions plan Fix 1 implemented + a second fwhm-staleness bug found while
  verifying it; two review flags recorded.** Implemented the approved plan's Fix 1 (`HypsoCapture.__init__`
  no longer pre-seeds `self.fwhm` from the fixed-length sensor-profile default; `io/dispatch.py`'s
  `set_hypso_attributes` now unconditionally derives `fwhm`/`fwhm_unbinned` from this capture's own
  wavelengths via `capture_types._get_fwhm`/`_get_fwhm_unbinned`, replacing the dead `UNBINNED_BAND_COUNT`
  fallback). Verified against real capture data (not just unit-level): confirmed 88 of 120 bands genuinely
  differ from the old sensor-default values (not a coincidental no-op) immediately after load.

  **Writing a real L1B file to disk as part of that verification caught a second, related bug the plan
  hadn't anticipated**: `calibration/pipeline.py`'s `run_calibration` has a "spectral correction" step
  (lines ~178-190) that overwrites `satobj.wavelengths`/`wavelengths_unbinned` with calibration-refined
  values — but nothing resynced `fwhm`/`fwhm_unbinned` afterward, so bands whose refined wavelength crossed
  a `fwhm_lookup_wl` boundary kept the *load-time* fwhm value in the written L1B file (confirmed directly:
  bands 60/100 wrote `5.46` instead of the correct `3.34`/`3.42`, while bands 0/32 happened to still match).
  Same root cause as Fix 1 (fwhm going stale relative to wavelengths), different trigger. Fixed the same way:
  both spectral-correction branches now call `_get_fwhm`/`_get_fwhm_unbinned` again immediately after
  updating wavelengths. Re-verified: all four checked bands now write correctly. No circular-import issue
  from adding `from hypso.capture_types import _get_fwhm, _get_fwhm_unbinned` to `calibration/pipeline.py`
  (checked: `capture_types.py` only imports `calibration.pipeline` deferred, inside a function body).

  **Four review flags recorded per user request, not acted on yet**:
  - `hypso.containers`'s `DatasetDict`/`as_dataarray` (the ×21-era replacement for the deleted
    `DataArrayDict`/`DataArrayValidator`) — user wants to review whether a less-custom (e.g. more directly
    xarray-native) solution exists, despite this already being described as "the standard/generalizable
    choice" in its own docstring. Not investigated yet - flagged for a future pass.
  - `hypso/georeferencing/georeferencing.py` (593 lines: `GCPList`, `GCP`, `PointsCSV`, `Georeferencer`) —
    user's instinct: should only need simplified code for applying GCPs. Checked before recording: **zero
    callers found anywhere** for any of its four classes - not in this package (`geo.py`, the module that
    actually does georeferencing work, never references them), not in `hypso-processing-pipeline`
    (read-only grep). `GCP`/`GCPList`/`Georeferencer` (not `PointsCSV`) are re-exported as public API from
    `georeferencing/__init__.py` despite this. Strong candidate for simplification or removal, but not
    deleted - only public-API classes, and grep can't rule out external notebook/script usage outside this
    workspace. Flagged for a future pass, not acted on yet.
  - Combining `hypso/io/` and `hypso/load/` into one submodule — user noticed the overlap. Checked before
    recording: `io/` (this session's schema-driven `dispatch.py`/`reader.py`/`writer.py`/`cf.py`/`schema.py`)
    already reaches INTO `load/` today (`io/dispatch.py` imports `load_l1a_nc`/etc. from `hypso.load`;
    `io/reader.py` imports from `hypso.load.utils`) and `io/writer.py` similarly reaches into a THIRD
    directory, `hypso/write/` (`calibration_filenames_writer`/`metadata_gcp_group_writer`/
    `metadata_srf_group_writer`). So the real shape is `io/` orchestrating two older, still-load-bearing
    per-format/per-group helper directories (`load/`, `write/`), not two independent, equally-scoped
    modules that happen to overlap - a genuine merge would need to reconcile three directories, not two,
    and `load/`'s AC-tool-specific loaders (`acolite_l2_nc_loader.py`/`ocsmart_h5_loader.py`/
    `polymer_l2_nc_loader.py`) look like a different concern (parsing OUTPUT from external AC tools) than
    `io/`'s own per-level NetCDF schema. Flagged for a future architecture pass to actually decide the right
    shape, not acted on yet.
  - Removing `hypso/classification/` — user's instinct, checked before recording: this is already an
    explicit, documented compatibility shim (`from hypso.masks import *`, one line), not accidentally-still-
    there dead code - its own docstring says the real code moved to `hypso.masks`, kept solely because
    `hypso-processing-pipeline/hypso_pipeline/stage2_ac/process_capture.py` still does `from hypso.
    classification import decode_jon_cnn_*` (confirmed the only caller anywhere, read-only grep). Removing
    it outright would break that import - needs either confirming the pipeline has migrated to `hypso.masks`
    first, or a deprecation-warning period, before this repo can safely delete it. Flagged, not acted on yet.
  - Integrating `hypso/resample/` more closely into `hypso-package` - user's instinct. Checked before
    recording: `resample_cube`/`resample_products` (`datacube_resamplers.py`/`resamplers.py`, generalized
    and real-data-tested in the ×21 pass) have **zero callers anywhere** - not re-exported from `hypso/
    __init__.py`, no `HypsoCapture` method/property delegates to them (unlike calibration/georeferencing/
    masks/AC, all composed onto the capture object as `self.X`), and zero external callers in
    `hypso-processing-pipeline` either. Working, tested code that exists as orphaned free functions rather
    than a first-class capability of the capture object. Flagged for a future pass to wire in properly
    (e.g. a `satobj.resample.cube(...)`-style composition, matching the pattern already used elsewhere in
    this package), not acted on yet.
  - Two possibly-missing SNAP-specific NetCDF attributes for reflectance products - user's instinct.
    Checked before recording: `units="1"` for L1D `rhot_<wave>` variables is already a deliberate, correct
    CF choice (dimensionless ratio - `L1D_SCHEMA.product_units` in `io/schema.py`), not a gap. But the
    reflectance variable's `standard_name` was flagged in this session's original SNAP research
    (`ARCHITECTURE_PROPOSAL.md` §4.4) as "candidate: `toa_bidirectional_reflectance` - verify" and never
    actually resolved - `cf.band_attrs()` still sets no `standard_name` on any band variable today. Separately,
    no attribute for spectral bandwidth *distinct from* the already-written `fwhm` has ever been investigated -
    worth checking given the precedent already found this session: SNAP's Spectral Viewer specifically needed
    `radiation_wavelength`/`radiation_wavelength_unit` (not the generic `wavelength`) to auto-recognize
    per-band wavelengths, confirmed against a real ACOLITE output file - a parallel SNAP-specific bandwidth
    attribute name (distinct from CF's own `fwhm` convention) is plausible but unverified. Both need a real
    SNAP install to check (same caveat the original research already carried), not something to guess at
    from documentation alone. Flagged, not acted on yet.

  **New TODO (user request): complete documentation/docstrings/comments across the package.** Quantified
  before recording, not just asserted: of 85 `.py` files under `hypso/hypso/`, 54 have no module-level
  docstring at all; of 373 functions/methods, 226 have no docstring (measured via a one-off `ast`-based
  scan, not a permanent tool in this repo). This session's own new/heavily-touched modules (`capture_types.py`,
  `io/`, `georeferencing/geo_state.py`, `containers.py`, etc.) are generally well-documented as part of the
  work that created/modified them; the gap is concentrated in older, untouched parts of the package. Not
  scoped or started - a real pass would need to decide priority (public API surface first vs. blanket
  coverage) rather than mechanically stub docstrings everywhere.

- **2026-08-27 (×36): capture-dimensions plan Fix 3 implemented — imaging-mode calibration schema, in
  YAML, replacing filename-substring inference.** During Plan Mode review, user pushed back twice on the
  originally-scoped Fix 3 (a bolted-on shape guard inferring "is this file pre-baked" from a `'wide' in
  str(coeff_path)` filename check): first, that "nominal"/"wide"/"moon"/"custom" are HYPSO *imaging modes*
  that the satellite may grow more of, so adding one should mean adding config/data, not new code branches;
  second, that this schema should live outside `.py` files entirely if possible. Both landed: the plan's
  Fix 3 became a declarative YAML schema, not a Python-dict one.

  **`hypso/hypso/sensors/hypso1_modes.yaml`/`hypso2_modes.yaml`** (new, package data): per capture_type,
  the classification rule (same `classify_attr`/`classify_value` `SensorProfile.capture_type_thresholds`
  already used, now sourced from YAML) plus `crop_modes` (per coefficient type, `"as_is"` pre-baked vs the
  default `"crop_and_bin"`). Loaded once at import time in `sensors/hypso1.py`/`hypso2.py` via
  `importlib.resources` — the same mechanism this codebase already uses for SatPy's own reader/composite
  YAML configs (`hypso/satpy/etc/*.yaml`), so this isn't a new pattern for the package. `SensorProfile`
  gained the new frozen field `capture_mode_crop_modes: dict[str, dict[str, str]]`.
  **Caught and fixed my own mistake before it shipped**: initially wrote `hypso2_modes.yaml` as an empty
  document (matching HYPSO-2 having no crop-mode variation), which would have silently dropped HYPSO-2's
  classification thresholds entirely (every capture reclassified as "custom") - fixed to carry the same
  `nominal`/`moon`/`wide` thresholds HYPSO-1 has, just with empty `crop_modes`. Also preserved the exact
  original threshold *order* (`nominal, moon, wide`) rather than the alphabetical order first written,
  since first-match-wins ordering could matter for a hypothetical capture satisfying two rules from
  different attrs (`frame_count` vs `image_height`) at once - documented inline in both YAML files so a
  future edit doesn't silently reorder it.

  **`hypso1_calibration/hypso1_calibration/data/capture_modes.yaml`** (new, co-located with the `.npz`
  files it describes): per capture_type, which coefficient files this mode uses - replaces the `match
  capture_type:` block in `hypso1_calibration/main.py` (including the `"moon"` case added earlier this
  session) with a YAML lookup. Verified mechanical: captured golden filename selections for every
  capture_type from the pre-refactor code, confirmed the YAML-driven version matches exactly (by filename -
  absolute paths differ because the package physically reinstalls to a different location, expected).
  `hypso2_calibration` left untouched (no per-capture_type file variation exists there to schema-ify).

  **`calibration/correction.py`**: new `CalibrationShapeMismatchError`; `read_coeffs_from_file` gained an
  explicit `crop_mode` parameter, replacing the filename-substring check entirely (`destriping` also wired
  through symmetrically with `smile`, though no file shipped today exercises its `"crop_and_bin"` branch -
  future-proofing, not new active behavior). Both `"as_is"` branches now check the loaded array's shape
  against what this capture's own AOI/bin_factor implies, raising the new exception on mismatch.
  **Found a real bug in the plan itself while re-verifying this pass**: `read_coeffs_from_file`'s own outer
  `except BaseException: raise ValueError(...)` would have silently converted the new exception into a
  generic `ValueError` *inside this function*, before it ever reached `pipeline.py`'s carve-out - needed a
  `except CalibrationShapeMismatchError: raise` here too, not just at the caller.

  **`calibration/pipeline.py`**: `set_calibration_coeff_files` resolves `smile_coeff_crop_mode`/
  `destriping_coeff_crop_mode` from the new schema (defaulting to `"crop_and_bin"` for `coeff_files=`/
  registered custom sets, preserving their existing behavior exactly - neither ever supported `"as_is"`).
  `load_calibration_coeff_files` passes `crop_mode` through and re-raises the new exception past its bare
  `except Exception` for smile/destriping only. **Found another real edge case while implementing**:
  `run_calibration(set_coeffs=False)` skips `set_calibration_coeff_files` (the only place crop_mode gets
  set) but still calls `load_calibration_coeff_files` unconditionally - used `getattr(..., 'crop_and_bin')`
  defaults rather than a plain attribute read, so a second calibration call reusing previously-set file
  paths doesn't `AttributeError`.

  Verified end-to-end: a deliberately-mismatched fake capture (nominal's real row_count is 684, tested
  against 500) raises `CalibrationShapeMismatchError` correctly; the matching case still succeeds with the
  correct `(684, 120)` shape and no exception. New tests in `tests/test_sensors.py`: crop-mode structure
  well-formedness for both profiles, the exact expected HYPSO-1 crop-mode values, the golden-filename
  regression for `hypso1_calibration`'s YAML migration, and both the mismatch-raises and matching-succeeds
  cases. 152 tests pass (full suite).

  Packaging: both `MANIFEST.in`s gained `global-include *.yaml`; both `pyproject.toml`s gained an explicit
  `pyyaml` dependency (already present transitively via `satpy` for hypso-package itself, but now imported
  directly rather than only by SatPy internals; genuinely new for `hypso1_calibration`, which previously
  depended on `numpy` only).

- **2026-08-26 (×25): type-per-level capture objects.** Follow-on from ×24. User asked: "1 HypsoCapture = 1
  cube, so why level-specific cube/mask accessor names, and why no validation preventing e.g. AC from L1B or
  jumping L1A→L1D directly? Some way of tracking the processing level represented by the object?" First
  drafted (in Plan Mode) as an incremental fix - make `product_level`/`product_symbol`/`cube_name` authoritative
  via `_advance_product_level`/`_reset_product_level` helpers - but the user pushed back, asking whether that
  was coherent given hypso-package's goal of being the foundation for `hypso-pipeline` (orchestrating AC +
  downstream products, retrospective and possibly NRT). Correct answer: no - a driftable runtime string kept
  in sync by convention is the same fragile pattern that produced the bugs being patched. Replanned as a real
  architectural change instead: **type-per-level objects** (`hypso/capture_types.py`, new module) -
  `L1BCapture`/`L1CCapture`/`L1DCapture`/`L2ACapture`, entered only via `to_l1b()`/`to_l1c()`/`to_l1d()`/
  `to_l2a()` (added earlier this session, zero external callers before this - safe to change what they
  return). "What level is this" becomes a fact of the object's *type* - an `L1BCapture` has no `l1a_cube`
  attribute at all, `AttributeError` on wrong-level access instead of a silent stale `None`.

  Deliberately does **not** touch `HypsoCapture`'s deprecated in-place `generate_l1b_cube()`/
  `generate_l1c_cube()`/`generate_l1d_cube()` family (still what `hypso-processing-pipeline` calls today) -
  fundamentally incompatible with strict type-per-level, since it mutates `self` and lets one object hold
  `l1a_cube`/`l1b_cube`/`l1d_cube` simultaneously. Migrating the pipeline onto the typed family is a separate,
  future effort (recorded in the plan file, not attempted - that repo is off-limits right now).

  Technical wrinkles solved during implementation (both flagged in the plan up front, not discovered by
  surprise): `copy.copy` can't change an object's type, so `_spawn_next_level` (removed - dead code once
  nothing called it) is replaced by `capture_types.spawn_as` (`object.__new__` + `__dict__` update - the
  standard mechanism, same aliasing semantics); and business logic reading a hardcoded old-class attribute
  name (`calibration_pipeline.run_calibration`'s `satobj.l1a_cube`) needed a temporary plain-attribute alias
  during the one call that needs it, removed immediately after, to keep the "no l1a_cube attribute afterward"
  guarantee real.

  **Two real bugs found and fixed only because the real-capture test suite caught them** (both would have
  shipped silently otherwise): `_spawn_l1d` set `.cube` from `compute_reflectance`'s raw return value without
  wrapping it through `_format_cube_dataarray` (the original `l1d_cube` property setter did this implicitly) -
  `.cube` was a bare `numpy.ndarray`, not an `xr.DataArray`, until a test calling `.to_numpy()` on it caught
  the `AttributeError`. `to_l2a`'s "did this succeed" gate checked "is `l2a_cubes` non-empty" instead of "did
  THIS call add something new" - misfired (incorrectly cleared L1D-derived state on an actual no-op) whenever
  a capture already had an unrelated correction registered from elsewhere - caught by a test exercising a
  shared session-scoped fixture that legitimately has other tests' corrections already on it
  (`conftest.py`'s `written_nc_files` fixture adds its own "testac" directly onto the shared `satobj`).

  Also found and fixed: `masks.pipeline.format_mask_dataarray` called `satobj._update_dataarray_attrs(...)` -
  a `HypsoCapture`-only method that doesn't exist on the new classes, which would have made `land_mask`/
  `cloud_mask`/`.masked_cube` unusable on them. Inlined (the logic never needed `satobj` in the first place).

  `docs/architecture.rst`'s "Cube memory" section explicitly revisited and reconciled, not silently
  contradicted - added a new "Tracking which level an object represents" subsection explaining why the
  original per-level-split rejection doesn't apply once spawning is by-value.

  129 tests pass (full suite, including new `tests/test_capture_types.py`). Commits `f631a42a` (implementation),
  `f1c741be` (unrelated: recorded a user-supplied ESA SNAP compatibility note in `ARCHITECTURE_PROPOSAL.md`
  while working - `radiation_wavelength`/`radiation_wavelength_unit` must be kept for SNAP's Spectral Viewer,
  not dropped as originally proposed there).

- **2026-08-26 (×24): HypsoCapture rearchitecture.** User: "too many [attributes], really
  chaotic," questioned the `api`/`_impl` split, asked to simplify AC interfaces and rename `HypsoBase`. Plan
  drafted via Plan Mode and saved to `/home/camerop/.claude/plans/flickering-jumping-chipmunk.md` (durable copy
  in case of disconnect, per user request), then expanded conversationally with several follow-on findings.
  Done so far:
  - `30f286e6`: renamed `HypsoBase` → `HypsoCapture` (zero external references, clean rename); removed 5
    zero-external-caller AC wrapper methods (`ac_ocsmart_run_correction` + its only caller, a permanently-dead
    `TOGGLE_OCSMART=False` branch; the 4 `ac_polymer_get_*` accessors, superseded by
    `PolymerAdapter`/`SpectralResponse`); fixed a real bug where `l2a_cube` (singular property) returned the
    multi-correction `DatasetDict` while `io/dispatch.py` already referenced the nonexistent plural
    `l2a_cubes` — renamed the property, kept `l2a_cube` as a deprecated alias (confirmed external readers in
    the pipeline's `extraction.py`/`2d_hypso_noise_snr.py`). 104 tests pass.
  - `afd6df3c`: extracted masking (`land_mask`/`cloud_mask`/`custom_masks` state, the formatters,
    `set_custom_mask`/`clear_custom_masks`/`load_mask_from_file`, `unified_mask`, and the four
    `masked_l1x_cube` properties, collapsed into one `get_masked_cube`) into `hypso/masks/pipeline.py`,
    alongside the existing `jon_cnn_classifier.py`/`jonas_svm_classifier.py` (the actual mask *algorithms* -
    this new module is the *container* that applies masks regardless of which algorithm produced them, not a
    duplicate of the already-removed dead `hypso/mask/` directory). **Found while generalizing the four
    `masked_l1x_cube` bodies, preserved as-is rather than silently fixed**: `masked_l1c_cube` reads
    `self._l1c_cube`, which is never actually populated — `_generate_l1c_cube_impl` never sets it; the
    `l1c_cube` *property* instead returns a deepcopy of `_l1b_cube` relabeled. So `masked_l1c_cube` always
    silently returns `None` today (or crashes if a mask is set). Needs a decision: should
    `get_masked_cube(satobj, "l1c")` instead mirror `l1c_cube`'s own getter (read `_l1b_cube`)? Also fixed
    `resample_cube(use_indirect_georef=True)` (read nonexistent `satobj.latitudes_indirect`/etc. - now reads
    the same `satobj.latitudes`/`longitudes`/`resolution` the default path uses; a real, separately-flagged
    finding is that the *default* `False` path might itself be semantically inverted - see the plan file's
    step 5 note, not yet acted on). Added `to_l2a(correction, **kwargs)`: mirrors `to_l1b`/`to_l1c`/`to_l1d`'s
    spawn-a-new-object pattern (cheap - same shallow-copy `_spawn_next_level` mechanism, no cube duplication),
    dispatching through the existing `hypso.ac.adapters.get_ac_adapter` registry rather than a hardcoded
    per-tool if/elif (caught by user review - the first draft hardcoded tool names redundantly with the
    registry). `ac_polymer_open_output`/`ac_acolite_open_output`/`ac_ocsmart_open_output` are now deprecated
    in favor of it (warn + delegate to a non-deprecated `_impl` sibling, matching `generate_l1b_cube`/`to_l1b`'s
    existing pattern) - kept working unchanged for the pipeline's current in-place callers. 112 tests pass.
  - `42532bcf`: **fixed** the `masked_l1c_cube` bug flagged above — `get_masked_cube(satobj, "l1c")` now mirrors
    `l1c_cube`'s own getter (deepcopy of `_l1b_cube`, relabeled) instead of reading the never-populated
    `_l1c_cube`, with a new regression test (`tests/test_masking.py`). Also consolidated the 23 scattered
    `nc_*` attributes (`nc_adcs_vars`, `nc_capture_config_attrs`, ... `nc_dimensions`, `nc_attrs`,
    `nc_cube_attrs`) into one `satobj.metadata: CaptureMetadata` instance (`hypso/io/metadata.py`, new) built
    once during load — confirmed zero external readers of any of the 23 names, so no compatibility properties
    needed; every internal reader updated (`calibration/pipeline.py`, `georeferencing/geo.py`, three
    `write/*.py` files, `io/writer.py`, and `io/dispatch.py`'s own `set_hypso_attributes`/`check_capture_type`,
    the bulk of the migration). 121 tests pass (golden-file regression against the pre-refactor baseline
    included — the strongest signal this didn't change any numeric output).

  Remaining: georeferencing angle/track consolidation (`GeoAngles`/`TrackGeometry` dataclasses, the `bbox`/
  `gsd`/`framepose` attributes, `io/writer.py`/`write/geometry_group_writer.py`'s angle lookup rewrite) — see
  the plan file's step 4 for full detail. Paused for a check-in before starting (largest/riskiest remaining
  step) — the user instead moved on to the type-per-level question, see ×25 above; this piece is still
  pending, not abandoned.

  **`compute_csiro_srfs`/`ac_dark_pixel_subtraction` cleanup — resolved, see ×30 below.**


- **2026-08-25 (continued further, ×23): srf_getter maintainability fix + a future Polymer-side wishlist.**
  User asked for a plain-language explanation of the `srf_getter` dotted-string mechanism and why
  `ac_polymer.py` sits outside `adapters/`. Investigated precisely (read `eotools/srf.py`'s actual
  `srf_getter.rsplit(".", 1)` → `importlib.import_module` → `getattr` resolution) and corrected an
  overstatement from an earlier turn: the location isn't "frozen by an external constraint" — checked, and
  nothing outside this repo's own `_polymer_driver.py` hardcodes the string, so moving the function was always
  technically possible, just required updating the string in lockstep with no functional upside.

  User asked what would make this easiest to maintain: **derive the dotted path from the function object
  instead of hand-typing it** (`5d5be8db`) — `ac_polymer.py` now has `SRF_GETTER_PATH = f"{fn.__module__}.
  {fn.__qualname__}"`, imported by `_polymer_driver.py` instead of a retyped literal. Real finding while
  making the change: the derived value (`"hypso.ac.ac_polymer.ac_polymer_srf_getter"`, the function's true
  defining module) differs from the old hand-typed one (`"hypso.ac.ac_polymer_srf_getter"`, which only worked
  via `hypso/ac/__init__.py`'s re-export) — both resolve to the same function (verified), but a new test that
  first asserted the OLD literal failed immediately, which is exactly the drift class this fix exists to
  prevent. Fixed by asserting resolvability (reproducing Polymer's own `rsplit`/`import_module`/`getattr`
  mechanism) rather than any hardcoded string, in both a new test and a strengthened existing one. Verified
  end-to-end against the real Polymer installation too. Full suite: 104 tests pass.

  **Recorded per user request**: a wishlist item for Polymer's own side (not this repo) — add a native HYPSO
  entry to Polymer's `eotools/srf.py` sensor/platform auto-lookup (the `srf.csv`-based mechanism `get_SRF()`
  already uses for sensors it recognizes out of the box), so Polymer resolves HYPSO's SRF itself and
  hypso-package's whole `srf_getter`/`generate_srf_nc` handoff mechanism becomes unnecessary. See the
  AC-connector design-notes section above for the full note.

- **2026-08-25 (continued further, ×22): submodule-layout audit acted on.** User asked "is everything in the
  correct submodules?" — did a fresh survey (not from memory) of the current layout. Found: (1) a stray empty
  `utils/data/` directory left over from the HSI2RGB move (×16) — removed; (2) `ac/loading_acolite_output.py`'s
  default-on `TOGGLE_6SV1` branch had a confirmed-broken import (`hypso.ac.__init__.py`'s `ac_6sv1` import was
  already commented out) — pre-existing, not introduced this session; (3) `ac/` had seven loose `ac_6sv1*.py`
  files plus `ac_srem.py`/`ac_srem_oyam.py` sitting flat, unlike Polymer/ACOLITE/OC-SMART's clean `adapters/`
  treatment; (4) `geo.py` sat at the top level while its siblings in the same "extracted orchestration"
  pattern (`calibration/pipeline.py`, `io/dispatch.py`) live inside their subject subpackage.

  User's answers: remove 6S and unused AC methods and loose files; move `geo.py`. Executed (`47ce75f3`,
  `a9095d4f`):
  - Deleted all 7 `ac_6sv1*.py` files and `ac_srem.py`/`ac_srem_oyam.py` — confirmed zero callers anywhere
    (this repo or the pipeline) before removing; SREM/SREM-OYAM were only ever imported into `hypso.ac`'s own
    namespace and never called.
  - Fixed `loading_acolite_output.py`: removed the broken `TOGGLE_6SV1` branch and the dead, never-branched-on
    `TOGGLE_SREM` declaration. While in there, also fixed two OTHER pre-existing stale references found along
    the way (not part of what was asked, but left broken would have undermined the fix): `from hypso.write
    import ... write_l2_nc_file` (that name never existed — real one is `write_l2a_nc_file`) and its three call
    sites — this script's default config path would have raised `ImportError` at module load regardless of
    which AC toggle was active, before and after the 6SV1/SREM removal.
  - Moved `geo.py` → `georeferencing/geo.py` (one internal import site, confirmed via grep; no naming
    collisions with `georeferencing.py`'s own classes) and updated every reference (`HypsoBase.py`,
    `calibration/pipeline.py`, `io/dispatch.py` comments, `tests/test_public_api.py`'s
    `test_composition_modules_import` — which would have started failing with `ModuleNotFoundError` on the
    stale path if left unfixed, docs/architecture.rst).

  Verified: `hypso.ac` no longer exposes any 6SV1/SREM names; `loading_acolite_output.py` parses cleanly; full
  suite (103 tests) passes both after the AC-method removal and after the `geo.py` move.

  **Still open, not acted on (judgment calls, not asked about yet)**: `ac_polymer.py`'s `ac_polymer_srf_getter`
  correctly stays outside `adapters/` (Polymer resolves it by dotted string — frozen path regardless of
  organization); `geometry/`/`geometry_definition/`/`georeferencing/`'s naming similarity was already
  investigated and deliberately left alone earlier this session (real distinct responsibilities, one has a
  real external consumer blocking a rename) — still the right call, not revisited.

- **2026-08-25 (continued further, ×21): DataArrayDict/DataArrayValidator fully retired; resample
  generalized.** Started from a user question ("why are DataArrayDict/DataArrayValidator still present?").
  Answer had three parts: (1) `_products` — the standing "do not touch" exception; (2) `DataArrayValidator`
  also had a second, independent live use — HypsoBase's five `_format_l1a/b/c/d_dataarray`/
  `_format_mask_dataarray` single-array formatters, a different role than the dict-container problem
  `DatasetDict` fixed, never in scope of that earlier work; (3) a bonus finding — `resample/
  datacube_resamplers.py`'s `resample_products` referenced `DataArrayDict` but its import was commented out
  — turned out to be a false alarm on first read (the reference is itself inside a commented-out block; the
  function's real body was just `print("not yet implemented")` + `return None` — corrected this to the user
  after initially calling it "broken").

  User then reconsidered `_products` itself ("we have not used them... should be dropped/produced as a
  separate object") — **checked before agreeing, and found the opposite**: `hypso-processing-pipeline`'s
  Polymer stage does `satobj.products['chla'] = ...` then `write_products_nc_file(..., file_name=
  "polymer_chl.nc")` on every Polymer run. Real, active, load-bearing usage — reported this back rather than
  going along with the premise. **`products/` DEBATE — REVISIT LATER**: given this is real, load-bearing
  infrastructure with a slightly awkward API shape (a property that raises `AttributeError` on direct
  assignment, requiring `products[key] = value`), it may be worth a cleaner redesign at some point — e.g. a
  more explicit "AC-tool-produced auxiliary product" object/API rather than a dict-property bolted onto
  `HypsoBase`. Not scoped, not urgent — flagged here per user request so the discussion isn't lost.

  Once `products`' real usage was established, user asked to reimplement it (not drop it) on `DatasetDict`
  anyway — closing the loop `DatasetDict`'s own docstring had left open. Did all three pieces the user asked
  for (`06abd02c`, `cb0519ba`):
  1. **`_format_l1x_dataarray` → one method.** Collapsed the four near-identical formatters into
     `_format_cube_dataarray(data, level)` + a `_CUBE_DATAARRAY_ATTRS` table; extracted the single-array
     shape/dims logic `DataArrayValidator` provided into `hypso.containers.as_dataarray()`, now shared by
     both `DatasetDict._as_dataarray` and the HypsoBase formatters.
  2. **`products` → `DatasetDict`.** `HypsoBase._products` now matches `_custom_masks`'s construction;
     validation raises instead of silently swallowing (a real fix, not just a container swap).
     `write_products_nc_file` needed zero changes (only ever called `.items()`/`.to_numpy()`). With every
     consumer moved, `DataArrayDict.py`/`DataArrayValidator.py` are **deleted outright** — confirmed zero
     remaining references anywhere in the package.
  3. **Resamplers generalized** (user: "resamplers could be generalized"). `resample_l1a/b/c/d_cube` collapsed
     into one `resample_cube(cube, satobj, area_def, ...)` + thin per-level wrappers (zero external callers of
     any of the four, confirmed, so free to collapse). `resample_products()` — previously the dead stub found
     above — is now a real, working implementation reusing the same `resample_cube()`, returning a plain
     `xr.Dataset`.

  **All three verified against real data, not just structurally**: real-capture `products` round-trip through
  `write_products_nc_file`; real-capture + real `pyresample.AreaDefinition` resampling for all four cube
  levels and for `resample_products()` (confirmed non-trivial finite fraction in the output, i.e. it actually
  resampled real data, not a shape-only check). 12 new tests total (`test_containers.py` `as_dataarray` tests,
  a `test_cf_format.py` products test, `test_resample.py`). Full suite: **103 tests pass**. Docs
  (architecture.rst) updated for containers and the new resample section.

- **2026-08-25 (continued further, ×20): AC-connector pass, part 3 — OC-SMART, the last of the three.**
  Resumed after a detour into `hypso-processing-pipeline` credential hygiene (hardcoded EARTHDATA secret
  found in `hypso/ac/loading_acolite_output.py` while checking OC-SMART callers → led to a template-config
  system + netrc/env-var credential resolution across both repos — see that repo's own history; paused
  mid-way when told a different Claude session was concurrently working there, so **all of that work was
  left uncommitted in that repo** and not touched further here). Folded `hypso-processing-pipeline`'s own
  field-tested `ac_runners_hypso.run_ocsmart_correction` design into `OCSMARTAdapter` verbatim
  (capture-local staging, `OCSMART_Input.txt` save/restore even on failure, output routed into
  `capture_dir`, streamed console output, new `python_path`/`skip_existing`/`l2_prod`/`solz_limit`/
  `senz_limit` params, `ACRunError` on failure — consistent exception type across all three adapters now).
  **Found and fixed a real, previously-undocumented bug**: both the old adapter and
  `io.dispatch.load_capture_file` staged/named OC-SMART files using `str(satobj.sensor).upper()` (e.g.
  `"HYPSO2_HSI"`) — but OC-SMART's own autodetection only recognizes the satellite-agnostic `"HYPSO_HSI"`
  prefix, confirmed independently by the pipeline's own prior debugging of the identical issue. Wrong
  prefix → OC-SMART silently produces NO output at all (exit 0, no exception). Fixed at the source
  (`OCSMARTAdapter.HYPSO_PREFIX = "HYPSO_HSI"`), and removed the two now-redundant path attributes from
  `io/dispatch.py` entirely — output-path ownership moves into the adapter (`output_path()`), closing the
  "coupling smell" the original AC-connector proposal flagged, confirmed zero other readers first. Merged
  `stage_input`+`run_correction` into one call (the config save/restore has to wrap both together to
  restore correctly on failure, so the two-call split was never actually independently safe) — confirmed
  zero external callers of the split anywhere before merging; updated the one internal caller
  (`loading_acolite_output.py`).

  **Verified against the real OC-SMART installation and a real written L1D file**: staged correctly, and
  OC-SMART's own console output showed `"Sensor : HYPSO HSI"` — direct proof the prefix fix actually works
  (pre-fix, this exact scenario produced silent nothing). The run then failed inside OC-SMART's *own*
  `src/L1B.py` (`AttributeError: 'L1B' object has no attribute 'latitude'`) reading geolocation from the
  new flattened NetCDF layout — **the same class of breakage already known/accepted for eoread/ACOLITE, now
  confirmed for a third external reader** (not a bug here — everything up to that point, staging through
  config-restore-on-failure, worked correctly, separately confirmed via a mocked-subprocess test forcing a
  non-zero exit). 4 new tests + 2 pre-existing tests updated for the removed `stage_input` API. Full suite:
  90 tests pass. Committed (`b22388fe`), docs extended.

  **All three AC connectors now share a consistent design**: subprocess-based (isolated for Polymer/ACOLITE,
  always-was for OC-SMART), `ACRunError` on failure, `python_path`-configurable interpreter. Remaining AC-
  connector proposal items: per-tool config dataclasses, a uniform `run_correction`/`open_output` *return*
  shape, folding `_read_polymer_chla`'s v1/v2 handling into the Polymer adapter.

  **New TODO (user request, ×20): variable capture dimensions/bands/sizes.** User wants `hypso-package` to
  be able to handle captures with differing spatial dimensions, band counts, etc. — not yet scoped. Likely
  touches: `hypso.sensors.SensorProfile` (currently assumes fixed `UNBINNED_BAND_COUNT`/fwhm arrays sized
  for one specific configuration per sensor), `io.schema.LevelSchema` (dimension naming/sizing is currently
  swath-specific and largely assumed-fixed per the `spatial_dims` seam left for L3), and anywhere `bin_factor`/
  `image_height`/`image_width` are treated as constant across a session. Needs a real scoping pass before
  implementation — flagging here so it isn't lost, not starting it yet.

- **2026-08-25 (continued further, ×19): AC-connector pass, part 2 — ACOLITE subprocess isolation.** Same
  `run_subprocess_driver`/`ACRunError` mechanism as Polymer (×18), applied to `ACOLITEAdapter.run_correction`
  via a new `_acolite_driver.py`. Checked first whether ACOLITE has a Polymer-like multi-build conflict: it
  doesn't — every config file in `hypso-processing-pipeline` (HYPSO-stage and PACE-stage alike) points
  `acolite_path` at the same single checkout — so this isolation's justification is narrower and honestly
  documented as such: crash containment (ACOLITE's gdal/pyresample/cartopy stack) + parallelism + consistency
  with Polymer's pattern, not a demonstrated bug. Confirmed real imports work in the active `ac` env exactly
  as Polymer's did (checked directly, not assumed) — no separate environment needed by default here either.
  Dropped a dead `import acolite as ac` found while in this code (used only in an already-commented-out line).

  **Security improvement made along the way**: `EARTHDATA_u`/`EARTHDATA_p` credentials no longer travel
  through the JSON config file `run_subprocess_driver` writes to disk (even briefly, even in a private
  per-call `TemporaryDirectory`) — `run_subprocess_driver` gained an `extra_env` parameter, and the driver
  reads `HYPSO_ACOLITE_EARTHDATA_USERNAME`/`PASSWORD` from the subprocess's own environment instead. Pinned
  by a test asserting `"EARTHDATA"` never appears in the written config.json.

  **Verified against the real ACOLITE installation** (skipped if absent, same pattern as Polymer's real-tool
  test): unlike Polymer's `Level1_HYPSO`, ACOLITE's own `acolite_run` does **not** raise on a missing input
  file — it logs `"Path ... does not exist."` and returns normally. So the proof of real execution here isn't
  an exception but a real ACOLITE-written run log file in the output directory (confirmed present, containing
  the exact expected message) — proving the real `acolite`/`acolite.acolite.settings`/`acolite.acolite.
  acolite_run` imports succeeded and real ACOLITE code actually ran. 6 new ACOLITE tests alongside the
  existing 6 Polymer ones (12 total in `tests/test_ac_subprocess.py`). Full suite: 86 tests pass. Committed
  (`d13127f2`), docs extended.

  ~~**Remaining from the AC-connector proposal**: OC-SMART not yet migrated...~~ — OC-SMART done, see the
  ×20 status entry below (`b22388fe`). Still open: per-tool config dataclasses; uniform `run_correction`/
  `open_output` result *return* shape (the exception type is now uniform — `ACRunError` — across all three);
  folding the pipeline's `_read_polymer_chla` v1/v2 chla-vs-logchl handling into the Polymer adapter.

- **2026-08-25 (continued further, ×18): AC-connector pass, part 1 — Polymer subprocess isolation.**
  Investigated two user questions before writing any code:
  1. *Does Polymer need a separate Python env like OC-SMART?* Checked directly (not assumed): the real v1
     (HYPSO-SRF) checkout's `environment.yml` pins Python 3.12, but the active `ac` env is 3.13. Tested the
     actual import chain (`eoread.hypso.Level1_HYPSO`, `polymer.main_v5.run_polymer`) against the real
     checkout in the `ac` env — **both import cleanly**. The environment.yml pin is aspirational, not a hard
     requirement. **Original concern disproven.**
  2. *Then why does subprocess isolation matter?* Reproduced the real problem live: imported v1's `polymer`,
     deleted only `sys.modules['polymer']`/`['polymer.main_v5']` (an easy, natural mistake), then imported
     v2 from a different checkout — got `ModuleNotFoundError: No module named 'core.process'` because v1's
     `core` was still cached under that name and v2's `polymer.main_v5` needs a different, incompatible
     `core` (with a `core.process.blockwise` submodule v1's lacks). **Confirmed, demonstrated justification**:
     v1/v2 builds ship different same-named packages; Python's `sys.modules` cache makes switching between
     them unsafe within one long-lived process — directly relevant since `hypso-processing-pipeline` supports
     `polymer_version="v1"/"v2"` per call.
  User approved proceeding on this basis. Built: `hypso/ac/adapters/base.py`'s `ACRunError`/
  `run_subprocess_driver` (per-call `TemporaryDirectory` for config/result JSON — directly avoids the
  concurrent-run collision class of bug already hand-worked-around for OC-SMART's staging dir) and
  `hypso/ac/adapters/_polymer_driver.py` (the part of `run_correction` needing Polymer imported — version
  selection + the `run_polymer()` call; path resolution/renaming stay in the parent). `run_correction` gained
  an additive `python_path=None` param (→ `sys.executable`); threaded through the frozen
  `ac_polymer_run_correction` wrapper. Also fixed, while in this code: `open_output`'s missing `case _`
  (unrecognized `input_product_level` → unhelpful `AttributeError`, now a clear `ValueError`), the dead
  `run_polymer_kwargs`/`srf_nc_path, srf_nc_path =` typo, and the unused `run_polymer_dataset` import.
  **Verified with real evidence, not just structural checks**: `tests/test_ac_subprocess.py` (6 tests) —
  driver-logic unit tests with stubbed Polymer, `ACRunError` propagation through a real subprocess spawn,
  and (the strongest one) a real-subprocess run against the **actual Polymer v1 checkout on this machine**
  with a deliberately-missing input file — confirmed it fails with a clean `FileNotFoundError`, proving
  `eoread`/`polymer`/`core`/`eotools` all import correctly through the isolated subprocess, not with any
  import error. Full suite: 80 tests pass. Committed (`8f0c2f98`), docs extended (architecture.rst).
  ~~ACOLITE not yet subprocess-isolated~~ — done, see the ×19 status entry above (`d13127f2`).

- **2026-08-25 (continued further, ×17): csiro path fully resolved — call removed from the pipeline (a
  cross-repo change, NOT in this workspace).** Investigated two user questions before acting:
  1. *Does `hypso-processing-pipeline` need to make the `compute_csiro_srfs()` call?* Traced the call site
     (`stage2_ac/process_capture.py:193`, right before the three `ac_polymer_generate_*_nc()` calls) and
     confirmed `_generate_l1d()` (called a few lines earlier) already populates `self.srf`/`self.spectral_
     response` — the family those three calls actually read. `compute_csiro_srfs()` populates a *different*
     attribute family that nothing downstream reads (confirmed by grepping the whole pipeline repo). **Dead
     work, safe to delete.**
  2. *Would the OLD, unmodified `hypso-package` (`/home/camerop/AC/hypso-package`, still the one actually
     used for production processing per this file's own note) still function if the call is dropped?*
     Checked that codebase directly (not this refactor workspace): `write/metadata_srf_group_writer.py`
     guards every `csiro_*` write with `hasattr(satobj, 'csiro_ssi') and ... is not None`; the old
     `HypsoBase.__init__` never defaults these attributes to `None`; and the only other `csiro`-named
     functions in that package (`aeronet_oc/aeronet_oc.py`'s `aeronet_oc_calculate_rrs_csiro`/
     `aeronet_oc_generate_csiro_gaussian_srfs`) are an unrelated, coincidentally-named Rrs helper family that
     never touches `satobj.csiro_*`. **Confirmed safe** — the only effect of dropping the call is that L1D
     files no longer carry the five `metadata/srf/csiro_*` variables, which nothing reads anyway.

  User confirmed: **removed `satobj.compute_csiro_srfs()`** from `/home/camerop/AC/hypso-processing-
  pipeline/hypso_pipeline/stage2_ac/process_capture.py` (a separate repo/git history from this workspace —
  left the surrounding unrelated uncommitted work there, config edits + a new `colocation/`/`resampling.py`,
  untouched and **not committed**, since that's the user's own in-progress work in a different repo). Replaced
  with an explanatory comment (why it was there, why it's gone, cross-references this file). Left
  `hypso.reflectance.compute_csiro_srfs()` itself as a `DeprecationWarning`-emitting function (×16 entry) rather
  than deleting it outright, in case anything else outside this one pipeline still calls it — actual removal
  from `hypso-package` is a separate future step once nothing calls it anywhere.

- **2026-08-25 (continued further, ×16):** Three user-directed items (`753d9c99`, `7e5d7897`):
  1. **Utils grab-bag split**: `utils/utils_file.py` deleted → `utils/ncdebug.py` (NetCDF inspection family,
     kept as one interdependent toolset), `utils/misc.py` (`is_integer_num`, the only function with a caller),
     `spectral_analysis/hsi2rgb.py` (**HSI2RGB moved per user question "better submodule?"** — rendering RGB
     from spectra is spectral analysis; kept for the RGB-camera TODO; its `D_illuminants.mat` moved to
     `spectral_analysis/data/`, MANIFEST updated). Running HSI2RGB for the first time ever revealed it was
     already broken under NumPy 2.x (`np.trapz` removed) — fixed to `np.trapezoid`, smoke-tested. Deleted with
     zero callers: `MyProgressBar`, `find_all_files`/`find_file`/`find_dir`, utils-local `haversine`,
     `find_closest_water_lat_lon_match` (was already fully commented out).
  2. **Lazy `spectral_response` rebuild for file-loaded captures** (`HypsoBase.spectral_response` is now a
     property): closes the "file-loaded satobj can't generate the Polymer SRF nc" gap. Made EXACT (all three
     Polymer ncs xr-identical to in-session references; pinned by
     `test_spectral_response_lazy_rebuild_from_file`) by fixing three precision leaks: corrections spectral
     arrays f4→f8 in `io/writer`, srf-group esun/effective_fwhm f4→f8 in `metadata_srf_group_writer`, and
     `io/reader` now reconstructing `satobj.wavelengths` from precise `radiation_wavelength` instead of the
     rounded `wavelength` label (fallback kept for old files). Rebuild mirrors the in-session `_get_fwhm()`
     step (profile-default fwhm differs at lookup boundary bands). Files written pre-f8-fix rebuild to f4
     precision only.
  3. **csiro deprecated, not removed**: user said drop unless used — it IS called, by
     `hypso-processing-pipeline/hypso_pipeline/stage2_ac/process_capture.py:193` (`satobj.compute_csiro_srfs()`),
     though its results are consumed by nothing (the Polymer generate_* calls right after it read the OTHER
     attribute family, and the persisted `csiro_*` file fields have no known readers). Added a
     `DeprecationWarning` (still fully functional, same pattern as `generate_l1*_cube`); actual removal happens
     when the pipeline drops that call in the AC-connector pass. **User also approved subprocess isolation**
     for Polymer/ACOLITE (see the AC-connector design notes section) — queued as the next major work item.
  Full suite: 73 tests pass.

- **2026-08-25 (continued further, ×15):** Four user-directed changes, each committed separately:
  1. **`classification/` → `masks/`** (`cabd8860`): the CNN/SVM classifiers exist to produce sea/land/cloud
     masks, so the mask-oriented name fits; `hypso.classification` stays as a one-line compat shim (confirmed
     external: pipeline imports `decode_jon_cnn_*` from it).
  2. **Wavelength-based `true_color` Satpy composite** (`cabd8860`, same commit): user reconsidered the earlier
     "composite not needed" — added `satpy/etc/composites/hypso1.yaml`+`hypso2.yaml` with WAVELENGTH
     prerequisites (0.640/0.550/0.460 µm) resolved by Satpy against each band's `WavelengthRange` (works across
     binning configs, unlike the removed position-based recipe), registered via a new `satpy.composites` entry
     point (verified in satpy's own `composites/config_loader.py`). Editable install re-run to refresh
     entry-point metadata. Verified `scn.load(["true_color"])` end-to-end against a real L1C file.
  3. **`DatasetDict`** (`4b3e56b4`): option (b) from the container assessment — `hypso/containers.py`, a
     `MutableMapping` over a real `xr.Dataset`, replacing `DataArrayDict` for `_l2a_cubes`/`_custom_masks`
     (validation now raises; `update()` can't bypass; keys lowercase everywhere; `.dataset` exposes the backing
     Dataset). **Trap found while testing**: xarray `Dataset.__setitem__` silently *reindexes/truncates* an
     incoming array whose dims disagree with existing entries — DatasetDict guards with an explicit size check.
     `DataArrayDict` kept only for the untouchable `_products`. 10 new unit tests (`tests/test_containers.py`).
  4. **`SpectralResponse` redesign** (`d0580af0`): `hypso/reflectance/spectral_response.py` — one frozen
     dataclass + one `compute_spectral_response()` builder superseding the two near-duplicate SRF paths and both
     loose attribute families (see the module docstring, written as the canonical explanation of what it
     supersedes). `compute_toa_reflectance`/`compute_csiro_srfs` stay as thin wrappers; HypsoBase populates
     `satobj.spectral_response`/`spectral_response_csiro` AND the legacy attrs with identical values (Polymer
     connector + srf metadata writer still read legacy names — migrating them is the later AC-connector pass;
     **user explicitly said do not edit the AC processors themselves**). Confirmed from Polymer's own source
     (`eotools/srf.py`, `main_v5.py`): only the SRF nc is load-bearing (`Band_<n>`/`wav_Band_<n>` format —
     FROZEN); the SSI/ESUN ncs are read by nothing (eoread uses its own LISIRD F0). Also: renamed
     `SensorProfile.srf_wl/srf_fwhm` → `fwhm_lookup_wl/fwhm_lookup_fwhm` (they're a FWHM lookup table, not
     SRFs; zero external users), fixed the `get_esun_nc_path` copy-paste bug and the `io/dispatch` csiro_list
     missing-comma bug, removed `compute_esun`'s two unused variants. **Verified bit-identical** against
     pre-change reference outputs (both attr families, L1D mean, all three Polymer ncs via `xr.identical`).
     Full suite: 72 tests pass.

- **2026-08-25 (continued further, ×14):** Second cleanup pass at the user's request ("clean up the submodules,
  some are un-used"), committed (`9b2f2903`). Deleted after function-level (not just module-level) zero-caller
  confirmation across this repo + `hypso-processing-pipeline` + the original repo's demo:
  `chlorophyll_estimation/` + `dimensionality_reduction/` (both user-confirmed mid-survey), `download/`,
  `mask/` (live mask generation is `classification/`'s CNN decoders, which the pipeline imports - `hypso.mask`'s
  land/cloud/water functions had zero callers anywhere), `plot/` including `composites/hypso1.yaml` (user:
  **RGB composite not needed** - the Satpy integration's job is Scene loading, already covered by the reader
  plugin + the `get_l1*_satpy_scene` converters, both kept; this also closes follow-ups (2) and (3) from the
  ×13 entry), the four shadowed per-level loader files and four shadowed per-level writer files that
  `hypso.io.reader`/`writer` replaced (previously "kept in case anything imports their internals" - now
  confirmed nothing does), and dead `reflectance/` data: `toa_reflectance_v1.py`, the **58MB** p005nm source
  `.nc` (only its derived 7.7MB `.npz` is read - by `toa_reflectance.py` AND `hypso-processing-pipeline`'s
  `get_f0`, so the `.npz` is load-bearing external API), the p1nm `.nc` (only v1 read it), the `.xls` duplicate
  of the used Thuillier `.csv`, and `f0.txt`. Kept deliberately: `classification/`, `resample/`,
  `spectral_analysis/`, `geometry_definition/` (each has a real external consumer), `write_ssi_npz.py`
  (provenance for the shipped `.npz`). Pruned MANIFEST.in entries pointing at directories that don't exist.
  `hypso/hypso/` is now 9.1MB. Verified: all imports, 40 unit tests, baseline exact match.

  **New TODO (user request, ×14): HYPSO-2 RGB camera support.** HYPSO-2 carries an RGB camera alongside the
  HSI; the package should support loading its imagery and provide a method for mapping/registering the RGB
  image onto the HSI image - both a fixed (calibrated/default) mapping and a custom (user-supplied) one.
  Natural fits with existing seams: `SensorProfile` (HYPSO-2's profile can declare the RGB camera + default
  mapping), `hypso.io` (loading), and the custom-mask-style registration API (user-supplied mappings). Not
  started - design TBD with the user.

- **2026-08-25 (continued further, ×13): ALL NUMBERED PLAN ITEMS COMPLETE.** Built the formal pytest suite (plan
  item 10, the last open item): `tests/conftest.py` (session-scoped real-capture fixture running the exact
  baseline pipeline; per-level written-NetCDF fixture; real-data tests auto-skip when `HYPSO_DATA_AOC` is absent
  so the suite runs in a fresh clone) plus six test modules — `test_regression_real_capture.py` (golden-file vs
  `baseline.json`), `test_cf_format.py` (one assertion per confirmed pre-refactor format bug + a full
  write→read round trip), `test_sensors.py`/`test_io_schema_cf.py`/`test_ac_adapters.py` (registry/builder unit
  tests, adapter checks structural-only per plan), `test_public_api.py` (frozen external surface incl. the
  string-resolved `hypso.ac.ac_polymer_srf_getter` path and import-order independence). **62 tests, all
  passing** (40 unit in ~4s; 22 real-data in ~2.5min). Installed `pytest` into the `ac` conda env (second
  env-modifying action this session, after the earlier approved `pip install -e`). Committed (`f98f8549`).
  Noted, not fixed (third-party): netCDF4-vs-NumPy-2.5 `DeprecationWarning`s from inside netCDF4's
  `__setitem__` on the writer's `var[:] = ...` assignments.

  **Remaining known follow-ups (all deliberate, none blocking):** (1) update `eoread/hypso.py` +
  `acolite/hypso/l1_convert.py` for the new flat NetCDF layout — the plan's explicitly-accepted breakage,
  separate later pass (plan verification item 5: confirmed still pending, expected); (2) re-point
  `composites/hypso1.yaml`'s RGB recipe at real wavelengths for the new Satpy reader (needs the real HYPSO-1
  band-to-wavelength mapping — flagged in `docs/architecture.rst`, not guessed); (3) `composites/hypso1.yaml`
  has no MANIFEST.in entry (pre-existing packaging gap, noted ×6); (4) optional later migration of
  `hypso-processing-pipeline` from `generate_l1*_cube()` to `to_l1*()` at the user's pace.

- **2026-08-25 (continued further, ×12):** Completed plan item 8, the largest remaining piece: extracted all 14
  `ac_polymer_*`/`ac_acolite_*`/`ac_ocsmart_*` methods plus the shared private `_get_inferred_wavelength_band_map`
  helper out of `HypsoBase` into a new `hypso/ac/adapters/` package - `base.py` (the `ACAdapter`
  `run_correction`/`open_output` interface + the wavelength-map helper, now public as
  `get_inferred_wavelength_band_map(satobj, ...)`), `polymer.py`/`acolite.py`/`ocsmart.py` (one stateless adapter
  class per tool, bodies **verbatim** per the plan's explicit decision #6 - prints deliberately NOT converted to
  logging here, unlike the geo extraction, since this code can't be regression-tested locally and the diff should
  stay purely mechanical), and a registry in `__init__.py` mirroring `hypso.sensors`
  (`get_ac_adapter`/`registered_ac_adapters`/`AC_ADAPTERS`). **Unlike the calibration/dispatch extractions, every
  public `ac_*` name stays on `HypsoBase` as a thin delegating wrapper** (all confirmed called by
  `hypso-processing-pipeline`); `HypsoBase.ac = AC_ADAPTERS` (class attribute, adapters are stateless singletons)
  gives the plan's `self.ac.polymer.run_correction(self, ...)` seam. Only the private band-map helper moved
  wrapper-less (zero external callers). Deliberately NOT adapters: `ac_dark_pixel_subtraction` (no external
  tool/output file, already a free function in `hypso/ac/`, still bound as a method) and `ac_polymer_srf_getter`
  (Polymer resolves it BY DOTTED-STRING NAME - `"hypso.ac.ac_polymer_srf_getter"` - so its import path is frozen;
  noted in the adapter docstrings). One dead commented-out block (ocsmart open_output's inline duplicate of the
  wavelength mapper) dropped rather than moved. `HypsoBase.py`: 1,638 → **1,000 lines** (2,113 at refactor
  start); `sys`, the five AC output-loader imports, and the never-used `find_file` import dropped from it.
  Verified: adapter-registry structural checks, all import orders, baseline exact match, plus real-capture
  functional checks *through the wrappers* (polymer id/path helpers set satobj attrs correctly; all three
  `open_output` missing-file branches return their original sentinels without raising). Also extended
  `docs/architecture.rst` (new "HypsoBase composition" + "Atmospheric-correction adapters" sections; fixed a
  stale `_set_calibration_coeff_files` reference). Committed (`42be73d2`, docs `1c8b7247`).
- **2026-08-25 (continued further, ×11):** Finished plan item 7 (`HypsoBase` composition) by extracting the
  load-dispatch slice into `hypso/io/dispatch.py` — `load_capture_file`/`set_hypso_attributes`/
  `check_capture_type`/`parse_filename`/`compose_capture_name`, moved verbatim, each taking `satobj` explicitly,
  same convention as `hypso.geo` and `hypso.calibration.pipeline`. All five had zero external callers (checked
  this repo and `/home/camerop/AC/hypso-processing-pipeline` — note the repo lives under `AC/`, not directly in
  `~`, which cost a wasted grep this session), so no wrappers were kept on `HypsoBase`; `__init__` now calls
  `io_dispatch.load_capture_file(self, path=..., load_cube=...)` directly. `HypsoBase.py` is down from 2,113 →
  **1,638 lines**. The now-unused `re` and `trollsift.Parser` imports were removed with it, as was a
  commented-out old `_parse_filename` variant that sat between the two moved methods (dead code, in git history).

  **Found and fixed a pre-existing circular import** surfaced (not caused) by this change: `hypso.io.reader` does
  `from hypso.load.utils import (...)` at module level, which initializes the `hypso.load` package, whose
  `__init__.py` in turn did an eager `from hypso.io.reader import load_l1b_nc, ...`. Whichever of the two was
  imported second failed with `partially initialized module`. This was masked before only because `HypsoBase.py`
  always imported `hypso.load` (line 24) *before* anything pulled in `hypso.io` — `import hypso.io` on its own
  was already broken at HEAD, confirmed by the traceback failing at `io/__init__.py`'s reader line, before the
  new dispatch line. Fixed by deferring those four names in `hypso/load/__init__.py` behind a module-level
  `__getattr__` (PEP 562) rather than reordering imports (which would have left the same trap for the next
  entry point — e.g. the Satpy handler, which imports `hypso.io.reader` directly). `from hypso.load import
  load_l1b_nc` still works unchanged for callers.

  Verified: all three import orders (`hypso.io` first, `hypso.load` first, plain `hypso` first) succeed;
  `satpy.available_readers()` still finds `hypso_l1c`/`hypso_l1d` and the file handler imports;
  `tests/baseline/compare_to_baseline.py` passes with exact matches (`l1b_cube` mean=42.928984 as in every prior
  run). Committed (`98a38c85`).
- **2026-08-25 (continued further, ×10):** Resumed after a session disconnect that left the `self.calibration`
  composition (plan item 7) mid-flight, uncommitted. Picked up the in-progress work as found: extracted
  `HypsoBase._set_calibration_coeff_files`/`_run_calibration`/`_load_calibration_coeff_files` verbatim into new
  module-level functions in `hypso/calibration/pipeline.py` (`set_calibration_coeff_files`/
  `load_calibration_coeff_files`/`run_calibration`, each taking `satobj` explicitly) — same convention as
  `hypso.geo`. Confirmed via grep the three original private methods had zero external callers, so (like the
  private `_run_*` georeferencing helpers, unlike `run_georeferencing`/`run_direct_georeferencing`) no wrapper
  methods were kept on `HypsoBase` — its `_generate_l1b_cube_impl` now calls
  `calibration_pipeline.run_calibration(self, ...)` directly. Also found (already staged from before the
  disconnect) the deletion of three scratch scripts — `calibration/calibration_pipeline.py`,
  `calibration_pipeline_functions.py`, `make_destriping_matrix.py` — confirmed genuinely dead: hardcoded local
  data paths (`../../../Data/HYPSO-1/frohavet/`), broken relative imports of modules that don't exist in this
  package (`utilities`, `read_images`, `show_bip`), never importable as part of the installed package (same
  pattern as the `georeferencing/example*.py` deletion earlier this session), and confirmed zero references
  anywhere in this repo or `hypso-processing-pipeline` before finalizing. Cleaned up a leftover multi-blank-line
  gap in `HypsoBase.py` left by the extraction. Verified: `from hypso import Hypso` imports cleanly (clean CWD),
  `hypso.calibration.pipeline`'s three functions import and resolve correctly, and
  `tests/baseline/compare_to_baseline.py` still passes with an exact match (`l1b_cube` mean=42.928984, matching
  every prior run). Committed (`74990178`).
- **2026-08-25 (continued further, ×9):** User asked (mid-cleanup, IDE had `dimensionality_reduction/__init__.py`
  open, likely incidental) whether `hypso.aeronet_oc` is used by `hypso-processing-pipeline` and could be
  removed if not. Confirmed via grep: NOT used by `hypso-processing-pipeline` (which reimplemented its own
  AERONET-OC matchup logic independently in `hypso_pipeline/aeronet.py`/`extraction.py`, per that module's own
  "Ported from hypso-matchup-processing" docstring) and NOT imported anywhere inside `hypso-package` itself
  (`hypso/__init__.py`, `HypsoBase.py`) - **but** initially found it IS imported by a third repo,
  `/home/camerop/AC/hypso-ac-processing/` (`2b_process_capture.py`/`2c_process_matchups.py`, last commit
  2026-08-02) that this session had no prior context on. Flagged this to the user rather than assuming it was
  safe to delete just because the *specifically-asked-about* repo didn't use it. User confirmed
  `hypso-processing-pipeline` supersedes `hypso-ac-processing` and explicitly said not to worry about breaking
  it (this is a refactor branch). Deleted `hypso/aeronet_oc/` (5 files: `aeronet_oc.py`, `aoc.py`,
  `aoc_hypso.py`, `plotting.py`, `utils.py`, ~190KB) and `hypso/write/aeronet_oc_writer.py`, and removed the
  latter's `write_aeronet_oc_matchup_nc_file` export from `hypso/write/__init__.py` (confirmed nothing else in
  `hypso-package` imported it first). Verified: `from hypso import Hypso; from hypso.write import
  write_l1c_nc_file` still imports cleanly, `tests/baseline/compare_to_baseline.py` still passes.

  **Note for future sessions**: confirming "is X used by hypso-processing-pipeline" is not the same question as
  "is X safe to delete" - there is at least one other repo (`hypso-ac-processing`) that historically depended on
  parts of this package outside `hypso-processing-pipeline`. It's now confirmed superseded/not a concern per the
  user, but the general lesson (check beyond the one repo you already have open) applies to any future
  similar-looking cleanup question.
- **2026-08-25 (continued further, ×8):** Continued the main refactor plan (composition work, item 7). Extracted
  georeferencing orchestration (`run_direct_georeferencing`, `run_georeferencing`, and the private
  `_run_frame_interpolation`/`_run_track_geometry`/`_run_angles_geometry` helpers) verbatim into a new
  `hypso/geo.py` module - matches the plan's `self.geo` composition item, and the AC-adapter convention already
  used elsewhere in this codebase (`hypso/ac/*.py`'s free functions taking `satobj` as an explicit first
  parameter, rather than a stored back-reference, since these read many `HypsoBase` attributes).
  `HypsoBase.run_direct_georeferencing()`/`run_georeferencing()` are now thin delegating wrappers (`return
  geo.run_direct_georeferencing(self)` etc.) - kept as methods, not moved, since `run_direct_georeferencing()`
  is called externally (`hypso/ac/loading_acolite_output.py`) and `run_georeferencing()` by
  `hypso-processing-pipeline`; both names/signatures are unchanged. The private `_run_*` helpers had zero
  external callers (confirmed by grep before moving) so they moved outright with no wrapper kept on
  `HypsoBase`. Also converted this code's `print(...)` calls to `logger.info(...)` while it was already being
  touched, consistent with the user's earlier logging-module request. Removed the now-unused `from
  hypso.geometry import (...)` block from `HypsoBase.py` (confirmed via grep those six names had zero remaining
  uses there) in favor of `from hypso import geo`.

  **Hit and resolved a testing artifact (not a real bug)**: running a bare `python3 -c "from hypso import
  Hypso; ..."` from CWD `hypso-package-refactor/hypso-package` (rather than with the `sys.path.insert(0,
  'hypso')` used throughout this whole session, since the package is now `pip install -e`'d) returned an EMPTY
  Python **namespace package** for `hypso` instead of the real package - because that CWD contains a
  subdirectory literally named `hypso` (the project root, one level above the real `hypso/hypso/` package dir)
  which Python's implicit-namespace-package resolution picks up via the CWD-on-sys.path entry, shadowing the
  real editable-installed package. Confirmed by re-running the identical test from `/tmp` (a CWD without a
  same-named subdirectory), which worked correctly. **Not a code bug** - purely an artifact of testing from that
  specific directory now that the old `sys.path.insert` workaround is gone; future sessions testing this repo
  post-`pip install -e` should `cd` somewhere without a `hypso/` subdirectory (or keep using
  `sys.path.insert`) to avoid hitting this again.

  Verified: `from hypso import Hypso` imports cleanly (from a clean CWD),
  `tests/baseline/compare_to_baseline.py` still passes (this is the highest-risk area touched so far given it's
  literally the geolocation math baseline compares latitude/longitude against), and separately verified
  `run_direct_georeferencing()` (the direct-georef code path, NOT exercised by `compare_to_baseline.py` which
  uses indirect georeferencing) still runs correctly and populates `latitudes_direct`/angle attributes.
- **2026-08-25 (continued further, ×7 - back to the main refactor plan):** User asked to continue the plan and
  clean up/reorganize submodules where it makes sense. Surveyed via an Explore agent (`.bak` files, the two
  `ac_6sv1_luts_*` files, and the `geometry`/`geometry_definition`/`georeferencing` submodules for overlap and
  dead code), confirmed each finding with a direct grep before acting rather than trusting the agent's report
  blindly:
  - Deleted `HypsoBase.py.bak` (confirmed `HypsoBase.py` itself is current/actively maintained; git history keeps
    the `.bak` content anyway).
  - Deleted `hypso/ac/ac_6sv1_luts_OLD.py` and `ac_6sv1_luts_deepthought.py` - confirmed zero references anywhere
    in `hypso/hypso/` **and** in `hypso-processing-pipeline` (the known external consumer) before removing, per
    the plan's explicit instruction.
  - Deleted `hypso/georeferencing/example.py` and `example_image_mode_change.py` (confirmed both have a genuinely
    broken bare `import georeferencing` - not `from . import`/`from hypso.georeferencing import` - meaning
    they've never been runnable as part of the installed package) plus their three orphaned data assets
    (`erie_2022_08_27T16_05_36-bin3.png`/`.points`, `transformation_settings.png`) that only those two scripts
    referenced.
  - **Did NOT touch `geometry_definition/`** despite it being unused anywhere inside `hypso/hypso/` (including
    `HypsoBase.py`) - checked the read-only original `/home/camerop/AC/hypso-package/demo/demo_processing.py`
    first (a real external consumer this refactor workspace doesn't otherwise touch) and found
    `from hypso.geometry_definition import generate_area_def` there. Renaming/merging it into `geometry/` for
    naming-clarity reasons alone would have broken that real, existing usage for a purely cosmetic gain - not
    worth it, left as-is.
  - **Did NOT rename `geometry`/`geometry_definition`/`georeferencing`** despite their similar names (flagged in
    the original plan as worth a naming pass) - the survey confirmed they have genuinely distinct, non-
    overlapping responsibilities (automated satellite-pose geolocation math / pyresample area-def builders /
    manual GCP-based calibration respectively), so the "confusion" is naming-similarity only, and per the
    `geometry_definition` finding above, at least one of these three has a real external import path that a
    rename would break. Decided the risk wasn't justified by a purely cosmetic improvement.
  - Consolidated `run_georeferencing`/`_run_custom_georeferencing` (the exact duplication flagged in the
    original plan) by **deleting `_run_custom_georeferencing`** entirely, rather than merging: confirmed via
    grep it had zero callers anywhere (internal or in `hypso-processing-pipeline`) and its body was a pure
    duplicate of `run_georeferencing`'s "explicit latitudes/longitudes passed in" branch -
    `run_georeferencing(latitudes=None, longitudes=None)` already covers both the override and
    use-self.latitudes-as-is cases in one method, so there was nothing left to actually merge.
  Verified after each deletion: `from hypso import Hypso` still imports cleanly and
  `tests/baseline/compare_to_baseline.py` still passes.
- **2026-08-25 (continued further, ×6 - separate plan-mode task):** User asked about Satpy compatibility, then
  clarified they want it "more tightly integrated... working with other sensors as well as with visualization" -
  a real registered Satpy reader plugin, not an extension of the existing hand-built-Scene converter functions
  in `hypso/hypso/satpy/satpy.py` (left untouched, still valid for anyone with an already-loaded HypsoBase
  object). This was planned separately in plan mode (approved plan written to
  `/home/camerop/.claude/plans/rosy-frolicking-summit.md`, overwriting the completed sensor-generalization plan
  that had lived there - that plan's content is preserved in this file's own approved-plan block further down,
  so nothing was lost) since it's a distinct, non-trivial feature, not a continuation of the CF/NetCDF refactor
  itself.

  **Built and verified working end-to-end against real data:**
  - `hypso/hypso/satpy/etc/readers/hypso_l1c.yaml` and `hypso_l1d.yaml` - reader configs, `file_patterns`
    matching HYPSO's existing filename convention with `{start_time:...}` so Satpy's default `start_time`/
    `end_time` (parsed straight from the filename, standard Satpy behavior) needed zero extra code.
  - `hypso/hypso/satpy/hypso_handler.py` - `HypsoL1FileHandler(NetCDF4FileHandler)`, one class shared by both
    readers (level implied by `self.filetype_info['file_type']`). `available_datasets()` dynamically discovers
    per-band datasets (band count/wavelengths vary by binning/calibration config, unlike a fixed-band
    instrument) by calling the newly-promoted-to-public `hypso.io.reader.list_band_datasets()` - reusing the
    exact band-discovery-and-sort-by-`band`-attribute logic `io.reader`'s cube loader already had, not
    reimplementing it. Sets `self.sensor = "hypso1"/"hypso2"` (confirmed this must match
    `hypso.sensors.hypso1.HYPSO1_PROFILE.key`, NOT `.sensor` which is the different, already-used
    `"hypso1_hsi"` string) so the existing `composites/hypso1.yaml`'s `sensor_name: hypso1` lookup finds it
    with zero changes needed to that file.
  - `hypso/pyproject.toml`: added `[project.entry-points."satpy.readers"]` (`hypso = "hypso.satpy"`) -
    confirmed via reading Satpy's own `_config.py`/`readers/core/config.py` source that this is exactly how
    third-party reader discovery works (resolves the entry point to a module, looks for `etc/` next to it).
    `hypso/MANIFEST.in`: added `recursive-include hypso/satpy/etc/ *` so the reader YAMLs are actually included
    in built distributions (noted along the way: the *existing* `composites/hypso1.yaml` doesn't have a
    MANIFEST.in entry either - a pre-existing gap, not introduced by this work, not fixed here since it's out
    of scope).
  - **Ran `pip install -e hypso/`** in the `ac` conda environment (confirmed with the user first - this is the
    first environment-modifying command run this session, everything before this used `sys.path.insert` for
    testing) - required because Satpy's entry-point discovery reads installed package metadata, not just
    sys.path importability. This also means `hypso1_calibration`/`hypso2_calibration` are now real installed
    dependencies going forward, not just sys.path hacks.
  - `hypso.io.reader`: renamed `_discover_product_variables` → `discover_product_variables` (now public, no
    behavior change, confirmed no other references to the old name anywhere in the repo) and added
    `list_band_datasets(nc_file_path, product_prefix_hint) -> list[dict]` (per-band `name`/`band`/`wavelength`/
    `radiation_wavelength`/`fwhm`/`units`/`long_name`, sorted by `band` - metadata-only, no pixel data) - the
    new public entry point `hypso_handler.py`'s `available_datasets()` needed, built as an addition alongside
    the existing tested `_load_cube`/`_load_cube_attrs`, not a modification to them.

  **Verified against a real capture** (generated L1C/L1D files via the already-tested `write_level_nc`, renamed
  to match the real filename convention `aeronetvenice_2025-03-04T10-38-05Z-l1c.nc`/`-l1d.nc`): both
  `hypso_l1c`/`hypso_l1d` appear in `satpy.available_readers()`; `Scene(reader="hypso_l1c", filenames=[...])`
  finds 122 datasets (120 bands + latitude + longitude); loading a mid-spectrum band
  (`Lt_588`) through the Satpy `Scene` gives a value matching a direct `netCDF4` read of the same variable
  exactly; `scn.start_time` correctly parses from the filename; `hypso_l1d`'s `rhot_*` datasets load
  correctly too. `tests/baseline/compare_to_baseline.py` still passes (confirms the `io.reader` rename/addition
  didn't touch anything the core pipeline depends on).

  **Known follow-up, not yet done** (documented in `docs/architecture.rst`, not silently left broken): the
  existing `composites/hypso1.yaml` RGB recipe references `band_89`/`band_70`/`band_59` (the *old* converter
  functions' position-based naming) which doesn't match this reader's `Lt_<wavelength>` dynamic naming - so
  `scn.load(["rgb"])` won't resolve against the new reader yet. Re-pointing the recipe at real wavelengths
  (ideally via Satpy's `DataQuery(wavelength=...)`, which would also make it portable across sensors) needs
  HYPSO-1's actual band-to-wavelength mapping to get right - flagged rather than guessed, since getting this
  wrong would produce a wrong-looking RGB composite silently.

- **2026-08-25 (continued further, ×5):** User confirmed the eventual-deprecation direction from the previous
  entry and asked to add the `DeprecationWarning` now (pipeline migration still deferred - the warning itself is
  invisible in production by default, Python suppresses `DeprecationWarning` unless a caller opts in to seeing
  it). Added `warnings.warn(..., DeprecationWarning, stacklevel=2)` to `generate_l1b_cube()`/`generate_l1c_cube()`/
  `generate_l1d_cube()`, each pointing at its `to_l1*()` counterpart. To avoid the warning firing from *internal*
  call sites (which would be noisy/wrong - it's meant for external mutating-API callers, not the library's own
  plumbing), extracted the actual bodies into private `_generate_l1b_cube_impl()`/`_generate_l1c_cube_impl()`/
  `_generate_l1d_cube_impl()`; the public `generate_*_cube()` methods now just warn then delegate. Updated every
  internal caller to use the `_impl` versions directly: `generate_l1c_cube()`'s/`generate_l1d_cube()`'s own
  internal L1B-fallback calls, and `to_l1b()`/`to_l1c()`/`to_l1d()` (added in the previous entry - these must
  never trigger the warning, they're the *non-deprecated* path). Confirmed `hypso/ac/loading_acolite_output.py`
  (the only other file calling `generate_l1b_cube()`/etc.) is a standalone personal script using the public API
  like any external caller, not internal library code - correctly left alone, appropriate for it to warn too.
  Verified against a real capture with `warnings.catch_warnings(record=True)`:
  `generate_l1b_cube()`/`generate_l1c_cube()` each emit exactly one `DeprecationWarning` (not a duplicate from
  the internal L1B fallback), `to_l1b()` emits zero, and cube values are unaffected (`l1b_cube` mean matches
  baseline exactly in both the warned and unwarned paths). `tests/baseline/compare_to_baseline.py` still passes.
- **2026-08-25 (continued further, ×4):** Continuation of the "separate objects per level" discussion below -
  user kept pushing on it across several turns (memory concern, then "is `discard_cube` intuitive - shouldn't 1
  object = 1 cube be simpler/more transparent?", then confirmed the operational `hypso-processing-pipeline`
  should NOT need to change now, migration can happen later). Landed on: add `to_l1b()`/`to_l1c()`/`to_l1d()` as
  new, *unconditionally non-mutating* counterparts to `generate_l1b_cube()`/`generate_l1c_cube()`/
  `generate_l1d_cube()` - always return a new `Hypso1`/`Hypso2` instance holding only that level's cube (clearing
  the levels it superseded on the new object - e.g. `to_l1b()`'s result has `l1a_cube=None`), leaving the object
  called on completely untouched. Deliberately a distinctly-named method rather than a `mutate-vs-return` flag on
  the existing methods (a flag makes the call site ambiguous; a new name doesn't). `generate_*_cube()` are
  UNCHANGED - not replaced - so `hypso-processing-pipeline` needs zero changes; migrating to `to_l1*()` is
  optional, later, at the user's pace.

  Implementation: `_spawn_next_level()` does `copy.copy(self)` (cheap - cube/geometry/calibration-coefficient
  arrays are only ever read, never mutated in place, anywhere in this class, so aliasing them between self and
  the new object is safe) then re-copies the *mutable container* attributes one level deep
  (`_custom_masks` dict, `_l2a_cubes` DataArrayDict) so a later mutation on one object's masks/l2a-corrections
  can't silently bleed into the other's. Verified against the real capture:
  `satobj.to_l1b(coeff_type='moved')` leaves `satobj.l1a_cube` populated and gives back a new object with
  `l1a_cube=None`/`l1b_cube` matching the baseline mean exactly; `l1b_obj.to_l1d()` same pattern one level
  further; `l1b_obj.set_custom_mask(...)` and `l1b_obj.l2a_cube['fake']=...` do NOT leak into `satobj` (confirms
  the container re-copy is doing its job); `tests/baseline/compare_to_baseline.py` still passes. Documented in
  `docs/architecture.rst` (rewrote the "Freeing cube memory" section to present both `discard_cube()`/`del` and
  `to_l1*()` as the two opt-outs, explaining when each applies). **Not yet committed** — do this first on
  resume.
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
