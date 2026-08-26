# hypso-package: Architecture & NetCDF Format Proposal

**Status: proposal only — nothing in this repo has been changed to produce this document.** It exists to brief a
future Claude session (or a human) before any implementation work starts. Every "current state" claim below was
verified against this repo's actual code and against real capture files under `/home/camerop/HYPSO_DATA_AOC/`
(HYPSO-2, both L1A–L1D and Polymer/ACOLITE-derived L2) as of 2026-08-25 — not inferred from documentation or
memory. Where a claim couldn't be verified this way, it's flagged explicitly under "Needs verification" rather
than stated as fact.

**Blast radius warning, read this first:** at least three *other* repos hard-code the current NetCDF group layout
this proposal wants to change:
- `eoread/eoread/hypso.py` (Polymer's HYPSO-1/2 reader) — reads `/products`, `/geometry`,
  `/metadata/corrections.radiometric_coefficients_version`.
- `acolite/acolite/hypso/l1_convert.py` (ACOLITE's HYPSO reader) — reads `/products`, `/geometry`,
  `gatts['processing_level']`, `gatts['instrument']`.
- `hypso-processing-pipeline` — reads L1D/L1C/L2A output paths and relies on `HypsoBase`'s public surface
  (`Hypso()`, `.l1c_nc_file`, `.ac_polymer_run_correction()`, etc.) extensively.

Any format change here is only safe to roll out alongside coordinated updates to all three. Section 6 spells out
exactly what would break and where.

---

## 1. Current state — package architecture (verified)

- **`HypsoBase.py` is a 2,113-line, 64-method god object.** It mixes five unrelated concerns in one class: NetCDF
  I/O orchestration, atmospheric-correction tool integration (14 `ac_*` methods — Polymer, ACOLITE, OC-SMART,
  6SV1, dark-pixel subtraction), cube generation for every level × masked/unmasked variant (8 methods:
  `l1a_cube`/`masked_l1a_cube`/`generate1a_cube` … through `l1d_cube`), land/cloud masking, and georeferencing
  orchestration. `Hypso1`/`Hypso2` (`hypso1.py`, `hypso2.py`, ~105 lines each) are thin subclasses that mostly set
  per-instrument constants (fwhm array, `platform`/`sensor`/`sat_id` strings) and otherwise inherit the entire
  surface unchanged.
- **The processing submodules underneath are actually reasonably factored** — `calibration/`, `geometry/`,
  `georeferencing/`, `classification/`, `mask/`, `resample/` are each scoped to one concern. The problem is
  specifically that `HypsoBase` doesn't *use* them as a clean composition; it calls into them and also does its
  own I/O, AC-tool subprocess/sys.path wrangling, and attribute bookkeeping inline.
- **`load/` and `write/` are one file per level, and the writers are near-duplicates of each other.**
  `write/l1b_nc_writer.py` (477 lines) and `write/l1c_nc_writer.py` (475 lines) diverge almost entirely by
  `l1b`→`l1c` / `L1B`→`L1C` string substitution — `l1d_nc_writer.py` (479 lines) is the same pattern again. This
  isn't just a maintenance annoyance: **it's the direct cause of a real bug found while researching this
  proposal** (see 2.2) — a per-band attribute block got copied from the L1C/L1D writer into the L1B writer without
  checking whether L1B actually has the geometry group it references. A single bug fix currently has to be applied
  correctly in 3+ near-identical files by hand, and evidently wasn't.
- **`ac/` has accumulated dead/experimental variants alongside the supported ones**:
  `ac_6sv1_luts_OLD.py`, `ac_6sv1_luts_deepthought.py` sit next to `ac_6sv1_lut.py` with no marker distinguishing
  "in use" from "abandoned experiment."
- **No test suite anywhere in the repo** (`find` for `test*` under the repo root returns nothing but stray
  `.bak` files). No `pyproject.toml`/`setup.py` at the repo root either.
- **`.bak` files are used as manual version control** — `HypsoBase.py.bak`, `HypsoBase.py.bak-before-log-reorg-20260805`,
  `ocsmart_h5_loader.py.bak`, `l2a_nc_writer.py.bak` all currently sit in the tree next to the files they're backups
  of. This is a process smell (git already does this job) but it's also a concrete sign that changes to this repo
  have historically felt risky enough to warrant manual safety copies — which is itself an argument for adding
  test coverage before restructuring anything.

## 2. Current state — NetCDF format (verified against real files)

Verified against a real HYPSO-2 capture's L1A/L1B/L1C/L1D files and a Polymer-derived L2A file
(`aeronetvenice_2025-03-04T10-38-05Z`).

### 2.1 What's already there and worth keeping

- **Per-band variables, not a stacked spectral dimension** — `Lt_378`, `Lt_382`, … (L1B/L1C) or `rhot_378`, … (L1D),
  each a 2D `(lines, samples)` array with its own `units`, `long_name`, `wavelength`, `fwhm`. This is a
  CF-legal pattern (used by plenty of real EO CF-NetCDF products) and it's the pattern SNAP's spectral-view
  grouping is built around (it groups bands by a per-variable wavelength attribute). **Keep this**, don't switch to
  a stacked `(bands, lines, samples)` array — that would be a strictly worse fit for SNAP and would break every
  downstream reader that already expects `<param>_<wave>` variable names (Polymer's `eoread/hypso.py`, ACOLITE's
  `l1_convert.py`, this repo's own `matching.py`/`extraction.py` callers in hypso-processing-pipeline all key off
  this naming).
- **`crs_wgs84` (L1C/L1D `geometry` group) is a correctly-formed CF grid mapping variable**: `grid_mapping_name`,
  `longitude_of_prime_meridian`, `semi_major_axis`, `inverse_flattening`, `geographic_crs_name` are all present and
  correct for `latitude_longitude`. This is genuinely reusable as-is.
- **Group-per-concern layout** (`products`, `geometry`, `metadata`, `metadata/corrections`, `metadata/adcs`,
  `metadata/gcp`, `metadata/timing`, `metadata/capture_config`) is a sensible logical separation and matches how
  the file is actually used (readers only ever touch one or two groups at a time).

### 2.2 Concrete bugs found (not style opinions — verified against real files)

1. **L1B's per-band variables carry `coordinates`/`grid_mapping` attributes pointing at a group that doesn't
   exist in the file.** `Lt_378.attrs['coordinates'] == '/geometry/longitude /geometry/latitude'` and
   `Lt_378.attrs['grid_mapping'] == '/geometry/crs_wgs84'`, but `f['geometry']` raises `KeyError` — **L1B has no
   `geometry` group at all** (it's pre-georeferencing, which is correct; the dangling attribute pointing at it
   isn't). Any CF-aware tool that tries to resolve those attributes on an L1B file will fail or silently ignore
   them. Root cause: this attribute block was copied into the L1B writer from the L1C/L1D writer (see 1) without
   removing the parts that don't apply yet at L1B.
2. **`latitude` (L1C/L1D `geometry` group) has the wrong `units` and the wrong `valid_range`.**
   `units == 'degrees'` (should be `degrees_north` — CF requires this exact string for a variable to be
   recognized as a latitude coordinate) and `valid_min`/`valid_max == [-180, 180]` (should be `[-90, 90]`; this
   looks like `longitude`'s valid range copy-pasted onto `latitude`). No `standard_name` is set on either
   `latitude` or `longitude` (CF's primary machine-readable identification mechanism, more authoritative than
   `units` for tool auto-detection — including SNAP's).
3. **`coordinates`/`grid_mapping` use absolute HDF5 paths (`/geometry/latitude`), which isn't standard CF syntax
   even where the target exists.** Worse: CF's own group-relative resolution rule (inherited from the netCDF-4
   grouped-data model) only walks *up* the group tree from the referencing variable's own group to its ancestors
   — it does not search sibling groups. `Lt_378` lives in `/products`; `latitude` lives in `/geometry`; those are
   **siblings**, not ancestor/descendant. That means even switching to a CF-standard *relative* name
   (`coordinates="latitude longitude"` instead of the absolute path) would **still not resolve** under the
   letter of the convention, because `/geometry` isn't an ancestor of `/products`. See 5.2 for the two ways to
   actually fix this.
4. **Redundant, inconsistent per-band wavelength metadata.** Each `Lt_<wave>` variable carries `wavelength`,
   `wave`, *and* `radiation_wavelength` — three representations of nearly the same quantity, with slightly
   different numeric values (e.g. `wavelength=378.5`, `radiation_wavelength=378.54673723` for the same band).
   It's not documented anywhere which one is authoritative, and a consumer has no principled way to pick.
5. **No `Conventions` global attribute at any level.** Without it, generic CF tooling (and SNAP's CF reader) has
   no signal to even attempt CF-aware interpretation of the file — it has to fall back to guessing from variable
   names.

## 3. Proposed package rearchitecture

Goal: keep the public API stable (`Hypso(path)` → `Hypso1`/`Hypso2` instance; `.ac_polymer_run_correction()` etc.
— hypso-processing-pipeline and others depend on this surface directly), fix the *internal* structure that makes
bugs like 2.2.1 possible, and make the codebase testable incrementally rather than requiring a rewrite.

### 3.1 Split I/O from a per-level *schema*, not per-level *code*

This is the highest-leverage change, because it directly addresses 1 (duplicated writers) and therefore 2.2.1 (the
bug that duplication caused). Replace `load/l1a_nc_loader.py` … `load/l2a_nc_loader.py` and
`write/l1b_nc_writer.py` … `write/l1d_nc_writer.py` with:

- **One `LevelSchema` per product level**, as data (a dataclass or a small YAML/JSON file per level, not a
  Python module with hand-written read/write logic) — declares: which groups exist at this level (e.g. L1A/L1B
  have no `geometry` group; L1C/L1D/L2A do), which variables live in each group, their dtype/dims, and their full
  CF attribute set (see section 5). Concretely:

  ```python
  # sketch, not final
  L1B_SCHEMA = LevelSchema(
      processing_level="L1B",
      groups=["products", "metadata"],   # no "geometry" — this is what would have caught bug 2.2.1
      products=PerBandVariable(
          name_template="Lt_{wave}",
          units="W m-2 um-1 sr-1",
          standard_name=None,  # see 5.3 - needs CF Standard Name Table verification
          long_name_template="Top-of-Atmosphere Radiance Band {index} ({wave} nm)",
          coordinates=None,     # correctly absent at L1B - structurally can't drift into bug 2.2.1 again
      ),
  )
  ```

- **One generic reader and one generic writer**, each driven by whichever `LevelSchema` is passed in. A fix to
  "how do we write a per-band CF attribute block" gets made once and applies to every level that uses it,
  instead of needing to be repeated correctly N times.
- This doesn't have to happen all at once — it can be introduced one level at a time, with the old per-level
  writer kept working until its replacement is verified against a real capture (this repo has no test suite, so
  "verified against a real capture" — the same empirical approach used throughout this session's other ports —
  is the practical substitute until one exists).

### 3.2 Give `HypsoBase` a composition seam instead of doing everything inline

Don't break the public `Hypso()`/`Hypso1`/`Hypso2` API. Do extract the things `HypsoBase` currently does inline
into objects it *delegates* to, mirroring the separation the submodules already imply:

- `hypso/io/` — the schema-driven reader/writer from 3.1. `HypsoBase` calls into this instead of importing 5
  separate loader/writer modules.
- `hypso/ac/` — keep as the atmospheric-correction adapters, but (a) move `ac_6sv1_luts_OLD.py` and
  `ac_6sv1_luts_deepthought.py` into an `ac/experimental/` subfolder (or delete them — git history already
  preserves them) so the supported surface is unambiguous, and (b) each adapter should own its own
  sys.path/subprocess wrangling rather than `HypsoBase` doing it inline per-AC-tool (currently
  `ac_polymer_run_correction` and `ac_acolite_run_correction` each hand-roll their own `sys.path.insert` calls
  directly in `HypsoBase`).
- `hypso/processing/` (or keep the existing `calibration/`, `geometry/`, `georeferencing/`, `classification/`,
  `mask/` as they are — they're already reasonably scoped) — `HypsoBase`'s `*_cube`/`masked_*_cube` methods
  become thin calls into these rather than each reimplementing masking logic inline.
- `HypsoBase` itself shrinks to: identity/metadata (sat_id, platform, paths), and orchestration (call io, call
  processing, call ac, in the right order) — no algorithmic logic of its own.

### 3.3 Process recommendations (not code)

- Stop keeping `.bak` files in the tree; rely on git. If uncommitted work needs a safety net, a branch or stash
  serves the same purpose without cluttering `import`-adjacent directories (a stray `.bak` file next to a `.py`
  file has, in this session's experience with a different repo, already caused confusion about which file is
  "real").
- Add a minimal test suite before doing any of the I/O rearchitecture in 3.1 — even just "read this real L1C file,
  assert the values that matter didn't change" per level, run against the real captures already sitting under
  `/home/camerop/HYPSO_DATA_AOC/`. Given there's no existing test infrastructure to match, a good first target is
  exactly the kind of golden-file regression test that would have caught 2.2.1 and 2.2.2 immediately.

## 4. Proposed CF-compliant, SNAP-compatible NetCDF format

### 4.1 Global attributes (every level)

Add, in addition to what's already there (`sat_id`, `institution`, `naming_authority`, etc. — keep those, they're
fine and not CF-conflicting):

| Attribute | Value | Why |
|---|---|---|
| `Conventions` | `"CF-1.10"` (verify current version at cfconventions.org before implementation) | Without this, CF/SNAP tooling has no signal to attempt CF-aware parsing at all — see 2.2.5. |
| `title` | e.g. `"HYPSO-2 L1C Top-of-Atmosphere Radiance"` | CF-recommended global attribute. |
| `source` | e.g. `"HYPSO-2 Hyperspectral Imager"` | CF-recommended; distinct from `instrument` (keep both — `source` is the CF-standard name for this concept, other tools look for it specifically). |
| `history` | Appended-to provenance string, e.g. `"{timestamp}: L1B generated from {L1A filename} by hypso-package {version}"` | CF-recommended; also solves "which radiometric_file/smile_file/etc. was used" more durably than scattered per-level attrs. |
| `references` | Link to HYPSO/NTNU SmallSat Lab documentation | CF-recommended. |

### 4.2 Group structure: keep it, but fix coordinate resolution

Given the blast radius in the warning at the top, **don't flatten the group structure** — `/products`,
`/geometry`, `/metadata/*` are already load-bearing for three other repos. Instead, fix the actual defect (2.2.3):
`/products` and `/geometry` are siblings, so CF's group-relative `coordinates` resolution can't reach across from
one to the other no matter how the attribute is spelled. Two real options, in order of preference:

1. **Move `latitude`/`longitude` (and ideally the geometry angles) to the root group**, so they become an
   *ancestor* of `/products` rather than a sibling, and `coordinates="latitude longitude"` (relative, CF-standard
   syntax) resolves correctly per the convention. This is the CF-correct fix, but it does move variables that
   `eoread/hypso.py`, `acolite/hypso/l1_convert.py`, and hypso-processing-pipeline currently read via
   `f['/geometry/latitude']` — all three would need their group path updated in the same coordinated change (see
   section 6). `crs_wgs84` should move with them for the same reason (`grid_mapping` has the identical
   sibling-group problem).
2. **If moving those variables isn't acceptable given the blast radius, keep the current sibling-group layout,
   but stop pretending `coordinates`/`grid_mapping` resolve under strict CF.** Remove the attribute from levels
   where it can't resolve (L1B, per bug 2.2.1) rather than pointing it somewhere wrong, and treat SNAP
   compatibility for the geolocation itself as a **separate, tool-specific concern** — verify empirically (open a
   real L1C file in an actual SNAP install) whether SNAP's own reader locates `/geometry/latitude`/`/geometry/longitude`
   through some non-CF-strict heuristic before assuming it needs the CF fix at all. This wasn't verified in this
   research pass — no SNAP install was available — and should be the first thing checked before deciding between
   options 1 and 2.

### 4.3 Coordinate variable fixes (regardless of which 4.2 option is chosen)

| Variable | Fix |
|---|---|
| `latitude` | `units="degrees_north"` (not `"degrees"`), `standard_name="latitude"`, `valid_min=-90.0`, `valid_max=90.0` (not `±180`). |
| `longitude` | `units="degrees_east"` (not `"degrees"`), `standard_name="longitude"`, `valid_min=-180.0`, `valid_max=180.0` (verify this one's current values are actually already correct — only `latitude`'s were confirmed wrong in this pass). |
| `sensor_zenith`/`solar_zenith` | `standard_name="sensor_zenith_angle"`/`"solar_zenith_angle"` (verify exact strings against the current CF Standard Name Table), `units="degree"`. |
| `sensor_azimuth`/`solar_azimuth` | `standard_name="sensor_azimuth_angle"`/`"solar_azimuth_angle"`, `units="degree"`. |

### 4.4 Per-band variable attributes (L1B/L1C/L1D products)

Collapse the redundant `wavelength`/`wave`/`radiation_wavelength` (2.2.4) into **one** authoritative value per
band. Recommendation: keep `wavelength` (it's what every downstream consumer — Polymer's reader, ACOLITE's
reader, the analysis pipeline — actually reads today) and drop `wave`/`radiation_wavelength`, or if the extra
precision `radiation_wavelength` carries is meaningful (it looks like a per-detector-measured value distinct from
the nominal band-center `wavelength`), keep it under a clearly-distinct name and document which one is nominal vs
measured — but don't carry three names for what consumers currently treat as one number.

**User correction (recorded before this was implemented): `radiation_wavelength` must be kept, not dropped —
it's required for ESA SNAP compatibility.** Confirmed authoritative requirement: SNAP's Spectral Viewer only
auto-recognizes per-band wavelengths from two specific attributes — `radiation_wavelength` (the central
wavelength value) and `radiation_wavelength_unit` (typically `"nm"`) — not from `wavelength` or `wave`. Example
from a real ACOLITE L2W output already in this convention (`/home/camerop/HYPSO_DATA_BALTICALGAE/
balticAlgae_2026-08-24T10-43-52Z/balticAlgae_2026-08-24T10-43-52Z-moved-l2a-acolite_l2w.nc`, `rho_w_378`
variable): `radiation_wavelength = 378.5`, `radiation_wavelength_unit = "nm"`, alongside `wavelength = 378.5`
and `wave = 378.5` (redundant duplicates of the same value under different names). Whoever picks up this
cleanup should: keep `wavelength` (what downstream Python consumers read) AND `radiation_wavelength` +
`radiation_wavelength_unit` (required by SNAP, not optional/nice-to-have) on every per-band variable; `wave`/
`wave_name` are the ones confirmed safe to drop as pure duplication. More broadly for SNAP's generic
CF-compliant NetCDF reader (not just the spectral viewer): standard_name/units on lat/lon
(`"latitude"`/`"degrees_north"`, `"longitude"`/`"degrees_east"`), UDUNITS-compatible `units` strings and a
`long_name`/`standard_name` on every data variable, and a consistent `_FillValue`/`missing_value` on all of
them — this section's existing recommendations already cover those; the wavelength-attribute point above was
the one gap.

| Attribute | L1B/L1C (`Lt_<wave>`) | L1D (`rhot_<wave>`) |
|---|---|---|
| `units` | `"W m-2 um-1 sr-1"` (already present, keep) | `"1"` (dimensionless reflectance — CF convention for unitless) |
| `standard_name` | Needs verification against the current CF Standard Name Table — no exact "at-sensor spectral radiance from an imaging spectrometer" term is confidently known to exist; closest candidates to check: `toa_outgoing_radiance_per_unit_wavelength` or leave unset with a clear `long_name` if nothing matches. | Candidate: `toa_bidirectional_reflectance` — verify. |
| `long_name` | Keep current pattern (`"Top-of-Atmosphere Radiance Band {i} ({wave} nm)"`) | Equivalent reflectance phrasing |
| `wavelength` | Single authoritative value, `units` attribute of its own (`"nm"`) | same |
| `fwhm` | Keep | Keep |
| `coordinates`/`grid_mapping` | Per 4.2 — only present where it actually resolves | same |
| `_FillValue` | Not currently confirmed present — verify and add if missing; required for correct handling of masked/invalid pixels by any CF or SNAP reader | same |

### 4.5 Per-level schema summary

Reusing the verified real-file structure from section 2, made explicit as the schema 3.1 would encode:

| Level | Groups | Product variables | Geometry present? |
|---|---|---|---|
| L1A | `products` (`dn`), `metadata/*` | Raw digital numbers, no radiometric calibration | No |
| L1B | `products` (`Lt_<wave>`), `metadata/*` | Calibrated TOA radiance | No (bug 2.2.1: currently claims otherwise) |
| L1C | `products` (`Lt_<wave>`), `geometry`, `metadata/*` | Calibrated TOA radiance | Yes |
| L1D | `products` (`rhot_<wave>`), `geometry`, `metadata/*` | TOA reflectance (radiance→reflectance already applied) | Yes |
| L2 (Polymer/ACOLITE-derived, repackaged by `write/l2a_nc_writer.py`) | `products` (`rho_w_<wave>`, `chla`, …), `geometry`, `metadata/*` (`metadata/srf` for per-band solar irradiance) | Water-leaving reflectance / derived products, AC-method-dependent | Yes |

## 5. Migration & compatibility notes — what actually breaks

If any of section 4's format changes ship, these need coordinated updates in the **same** change:

- **`eoread/eoread/hypso.py`** (`/home/camerop/AC/Polymer/Polymer/eoread`) — `Level1_HYPSO()` reads
  `ds_root['/geometry']`, `ds_root['/products']`, `ds_corrections.radiometric_coefficients_version`. A group-path
  change (4.2 option 1) breaks this reader directly.
- **`acolite/acolite/hypso/l1_convert.py`** (`/home/camerop/AC/ACOLITE/acolite`) — reads
  `f['/geometry/latitude']` etc. directly (this exact code was ported/verified in this same session). Same
  exposure as above.
- **hypso-processing-pipeline** — doesn't read group paths directly (it goes through `HypsoBase`'s public methods),
  so it's insulated from format changes as long as `HypsoBase`'s Python-level API (`.l1c_nc_file`,
  `.ac_polymer_run_correction()`, etc.) stays stable. This is a good argument for keeping section 3's
  rearchitecture API-compatible rather than a clean-slate rewrite.

## 6. Suggested rollout sequence

1. Verify the open items marked "needs verification" above (CF Standard Name Table lookups; SNAP's actual
   behavior on the current files, tested in a real SNAP install) — cheap, and changes which 4.2 option makes
   sense.
2. Add a minimal golden-file regression test harness (3.3) before touching any writer — otherwise there's no way
   to know a "fix" didn't introduce a new version of bug 2.2.1 elsewhere.
3. Fix the concrete bugs in 2.2 first, independent of any broader rearchitecture — they're real defects today
   regardless of whether 3.1/4.2 ever happen, and fixing them doesn't require deciding the harder group-structure
   question yet (2.2.1, 2.2.2, 2.2.4, 2.2.5 are all fixable without touching group layout at all).
4. Only then take on 3.1 (schema-driven I/O) and 4.2's group-structure decision together, since 3.1 is what makes
   4.2 safe to change in one place instead of five.
5. Coordinate the eoread/ACOLITE reader updates (section 5) in the same change as any group-path change, not
   after.

## 7. TODO: Zarr support (user request - not scoped/implemented, thoughts recorded below)

**Status as of 2026-08-26: sections 1-6 above are now almost entirely implemented** (`io/schema.py`, `io/cf.py`,
`io/writer.py`'s flat-root layout, the golden-file test suite - see `REFACTOR_PROGRESS.md`'s later status
entries). This section is a fresh, forward-looking addition, not part of the original research pass.

**Why this is a bigger fork than "add another writer", and why it interacts directly with the CF/SNAP work above**:

1. **The current writer is netCDF4-API-native, not xarray-native.** `io/writer.py` calls `netfile.createGroup`/
   `createVariable`/`createDimension` directly against a `netCDF4.Dataset` - it never builds an `xr.Dataset`
   in memory first. Zarr support via `xarray`'s own `Dataset.to_zarr()` needs exactly that in-memory `xr.Dataset`
   as its input. The clean way to add Zarr without duplicating `LevelSchema`'s logic a second time: refactor the
   writer to build one `xr.Dataset` from the schema + arrays + attrs, then serialize via `.to_netcdf()` *or*
   `.to_zarr()` as a final, pluggable step - not two independent writer implementations. That's a real rewrite of
   the writer's internals, not an additive change alongside it.
2. **Zarr stores are directories, not files** (a tree of chunk files + `.zarray`/`.zgroup`/`.zattrs` metadata).
   Essentially everything in this codebase assumes a single-file `Path` - `satobj.l1a_nc_file`/`l1b_nc_file`/etc.,
   every `nc.Dataset(path)` call, every `dest_file.is_file()` check in the AC adapters (would need `.exists()`
   for a directory store instead). Supporting Zarr means introducing a "store" concept that's sometimes a file,
   sometimes a directory, threaded through load/write/AC-adapter code that currently only knows "file path."
3. **Per-band named variables (`Lt_378`, `Lt_382`, ...) were deliberately kept, not stacked into one
   `(band,y,x)` array, specifically for SNAP/BEAM compatibility** (section 4.2's own decision). SNAP doesn't
   read Zarr at all, so that constraint doesn't apply there - a stacked array with a `wavelength` coordinate is
   the more natural, more chunking-efficient Zarr layout. If one schema needs to describe both output shapes,
   that's more design surface, not just a second backend for the same shape.
4. **Zarr would be additive, not a replacement.** SNAP, Polymer's `eoread/hypso.py`, and ACOLITE's
   `l1_convert.py` all only read NetCDF - none of that goes away if Zarr support is added, so this is a genuinely
   new export path alongside NetCDF, not a migration off it.
5. **Where Zarr's actual strengths line up with a real need**: cloud-native storage (S3/GCS-backed stores) and
   chunked/parallel access for large capture archives - it pairs naturally with dask, already used elsewhere in
   this codebase (`resample/resamplers.py`'s `KDTreeNearestXarrayResampler`). This reads more like a
   bulk-archival/cloud-pipeline concern than a per-capture, SNAP-viewing one - worth asking whether it belongs in
   `hypso-package` itself at all, versus a post-hoc archival step in `hypso-processing-pipeline` that consumes
   already-written NetCDF files and re-packs them into Zarr for cold storage/cloud analysis, leaving
   `hypso-package`'s own writer untouched.

**Recommendation, not a decision**: don't take this on opportunistically alongside other work. It's a genuinely
separate fork (schema-to-multiple-backends) best scoped on its own, and ideally after the eoread/ACOLITE
NetCDF-format coordination (section 5/6) has landed - running two format transitions at once multiplies risk for
comparatively little shared benefit. Also worth settling first: is there a concrete driving use case (a specific
cloud/archival need) or is this speculative future-proofing? That answer changes whether the right shape is "a
second writer backend in hypso-package" versus "a separate archival tool downstream of it."
