"""Type-per-level capture objects - L1BCapture/L1CCapture/L1DCapture/
L2ACapture. Entered only via the to_l1b()/to_l1c()/to_l1d()/to_l2a() spawn
family (HypsoCapture.py) - "what level is this object" is a fact of its
TYPE (an L1BCapture simply has no l1a_cube attribute at all - AttributeError
on wrong-level access, not a silent None), not a runtime string that has to
be kept in sync by every method that changes level. See
docs/architecture.rst's "Cube memory" section for the two coexisting APIs
this deliberately does NOT touch: HypsoCapture's deprecated in-place
generate_l1b_cube()/generate_l1c_cube()/generate_l1d_cube() family (still
externally used by hypso-processing-pipeline) is completely unchanged -
these classes are reached only through the non-mutating to_l1x()/to_l2a()
methods, which had zero external callers before this change and still do.

Deliberately excludes an L1A type: HypsoCapture/Hypso1/Hypso2 remain the
load-from-file entry point, unchanged.

Zero business logic is duplicated here - calibration_pipeline/geo/
AC_ADAPTERS/masking_pipeline/compute_reflectance/compute_spectral_response
are the exact same free functions HypsoCapture's own deprecated methods
call, taking satobj explicitly, with no isinstance(satobj, HypsoCapture)
checks anywhere - they work unmodified against any object shaped like a
capture, which is what makes this additive rather than a rewrite.
"""
import copy
from typing import ClassVar, Union

import numpy as np
import xarray as xr

from hypso.containers import as_dataarray
from hypso.masks import pipeline as masking_pipeline

# Per-level attrs stamped onto the uniform ._cube attribute - same table
# HypsoCapture.py uses for its own l1a_cube/l1b_cube/l1c_cube/l1d_cube
# (_CUBE_DATAARRAY_ATTRS there); duplicated here rather than imported to
# avoid a circular import (HypsoCapture.py imports this module to construct
# L1BCapture/etc. in to_l1b()/etc.). Keep in sync if the source table
# changes - see HypsoCapture.py's own copy for the canonical version.
_CUBE_DATAARRAY_ATTRS = {
    "l1b": {'level': "L1b", 'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
           'description': "Top-of-Atmosphere Radiance (Lt)"},
    "l1c": {'level': "L1c", 'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
           'description': "Top-of-Atmosphere Radiance (Lt)"},
    "l1d": {'level': "L1d", 'units': r"sr^{-1}", 'description': "Top-of-Atmosphere Reflectance (Rhot)",
           'correction': None},
}


def _update_dataarray_attrs(data: xr.DataArray, attrs: dict) -> xr.DataArray:
    for key, value in attrs.items():
        if key not in data.attrs:
            data.attrs[key] = value
    return data


def _format_cube_dataarray(satobj, data: Union[np.ndarray, xr.DataArray], level: str) -> xr.DataArray:
    data = as_dataarray(data, tuple(satobj.dim_names_3d), num_dims=3,
                        dim_shape=tuple(satobj.spatial_dimensions))
    return _update_dataarray_attrs(data, _CUBE_DATAARRAY_ATTRS[level])


def _get_fwhm(satobj) -> list:
    return [satobj.fwhm_lookup_fwhm[np.argmin(np.abs(band - satobj.fwhm_lookup_wl))]
            for band in satobj.wavelengths]


def _get_fwhm_unbinned(satobj) -> list:
    return [satobj.fwhm_lookup_fwhm[np.argmin(np.abs(band - satobj.fwhm_lookup_wl))]
            for band in satobj.wavelengths_unbinned]


def spawn_as(source, target_cls: type):
    """Construct a new instance of target_cls from source, aliasing
    source's __dict__ (shallow - big arrays shared, not duplicated) rather
    than deep-copying. copy.copy(source) (HypsoCapture._spawn_next_level's
    mechanism) can't be reused here since it always returns an instance of
    type(source) - it cannot produce a different class. This is the
    standard way to change an object's effective type while keeping
    shallow-copy semantics (object.__new__(cls) + state restore is what
    copy.copy itself does internally for the same-type case).

    Mutable *container* attributes still need their own independent copy
    (same rationale as _spawn_next_level's own docstring) so source and the
    new object can diverge independently after this point.
    """
    new_obj = target_cls.__new__(target_cls)
    new_obj.__dict__.update(source.__dict__)
    new_obj._custom_masks = source._custom_masks.copy()
    new_obj._l2a_cubes = source._l2a_cubes.copy()
    return new_obj


class _CaptureCommon:
    """Shared base for L1BCapture/L1CCapture/L1DCapture/L2ACapture. Never
    constructed directly - always produced by spawn_as, called from
    HypsoCapture.to_l1b()/to_l1c()/to_l1d()/to_l2a() or from another
    _CaptureCommon subclass's own to_* method."""

    LEVEL: ClassVar[str]

    @property
    def cube(self) -> xr.DataArray:
        return self._cube

    @property
    def masked_cube(self) -> xr.DataArray:
        return masking_pipeline.get_masked_cube_uniform(self)

    # Mask state (land_mask/cloud_mask/custom_masks) is capture-wide, not
    # level-specific - carried over via spawn_as's __dict__ update - so
    # these are the exact same thin wrappers HypsoCapture.py has over
    # hypso.masks.pipeline, needed here for .masked_cube to actually be
    # settable on a typed object.
    @property
    def land_mask(self):
        return self._land_mask

    @land_mask.setter
    def land_mask(self, value):
        self._land_mask = masking_pipeline.format_land_mask_dataarray(self, value) if value is not None else None

    @property
    def cloud_mask(self):
        return self._cloud_mask

    @cloud_mask.setter
    def cloud_mask(self, value):
        self._cloud_mask = masking_pipeline.format_cloud_mask_dataarray(self, value) if value is not None else None

    @property
    def custom_masks(self) -> dict:
        return dict(self._custom_masks)

    def set_custom_mask(self, name: str, value, description: str = None) -> None:
        return masking_pipeline.set_custom_mask(self, name, value, description)

    def clear_custom_masks(self) -> None:
        return masking_pipeline.clear_custom_masks(self)

    def load_mask_from_file(self, path, name: str = None, variable: str = None,
                            dtype=np.bool_, invert: bool = False) -> xr.DataArray:
        return masking_pipeline.load_mask_from_file(self, path, name=name, variable=variable,
                                                     dtype=dtype, invert=invert)

    def run_georeferencing(self, latitudes: np.ndarray = None, longitudes: np.ndarray = None) -> None:
        from hypso.georeferencing import geo
        return geo.run_georeferencing(self, latitudes=latitudes, longitudes=longitudes)

    def run_direct_georeferencing(self) -> None:
        from hypso.georeferencing import geo
        return geo.run_direct_georeferencing(self)


def _spawn_l1d(source, use_direct_georef=False, use_thuillier=False, use_unbinned=True,
              generate_figures=False) -> "L1DCapture":
    """Shared body for L1BCapture.to_l1d()/L1CCapture.to_l1d() - both read
    source.cube as their input radiance (l1c_cube is a relabeled view over
    the same l1b data, so the computation is identical either way), matching
    HypsoCapture._generate_l1d_cube_impl's original logic verbatim, just
    reading a uniform .cube instead of branching on which level-specific
    cube attribute happens to be populated."""
    from hypso.reflectance import compute_reflectance, compute_spectral_response

    toa_radiance = source.cube
    new_obj = spawn_as(source, L1DCapture)
    del new_obj._cube

    new_obj.fwhm = _get_fwhm(new_obj)
    new_obj.fwhm_unbinned = _get_fwhm_unbinned(new_obj)

    if use_direct_georef and hasattr(new_obj, 'solar_zenith_angles_direct'):
        if new_obj.VERBOSE:
            print('[WARNING] Computing TOA reflectance using DIRECT georeferencing geometry.')
        solar_zenith_angles = new_obj.solar_zenith_angles_direct
    else:
        solar_zenith_angles = new_obj.solar_zenith_angles

    if use_unbinned:
        sensor_wavelengths = new_obj.wavelengths_unbinned
        sensor_fwhm = new_obj.fwhm_unbinned
        sensor_bin_factor = new_obj.bin_factor
    else:
        sensor_wavelengths = new_obj.wavelengths
        sensor_fwhm = new_obj.fwhm
        sensor_bin_factor = 1

    sr = compute_spectral_response(
        band_centers_unbinned=sensor_wavelengths,
        fwhm_unbinned=sensor_fwhm,
        bin_factor=sensor_bin_factor,
        ssi_source="thuillier" if use_thuillier else "tsis",
        grid="native-truncated",
        generate_figures=generate_figures,
    )

    reflectance = compute_reflectance(toa_radiance=toa_radiance, sr=sr,
                                      iso_time=new_obj.iso_time,
                                      solar_zenith_angles=solar_zenith_angles)
    new_obj._cube = _format_cube_dataarray(new_obj, reflectance, "l1d")
    new_obj.spectral_response = sr

    # Legacy attribute family - see HypsoCapture._generate_l1d_cube_impl's
    # own comment for why these are still populated alongside
    # spectral_response.
    new_obj.srf = sr.srf
    new_obj.srf_ssi = sr.ssi
    new_obj.srf_ssi_wl = sr.grid_wl
    new_obj.esun = sr.esun
    new_obj.esun_wl = sr.band_centers
    new_obj.effective_fwhm = sr.effective_fwhm

    return new_obj


class L1BCapture(_CaptureCommon):
    LEVEL = "l1b"

    def to_l1c(self) -> "L1CCapture":
        """l1c has no independent cube storage - it's a deepcopy of the l1b
        data, relabeled (matching HypsoCapture.l1c_cube's own getter)."""
        new_obj = spawn_as(self, L1CCapture)
        new_obj.run_georeferencing()
        cube = copy.deepcopy(new_obj._cube)
        cube.attrs['level'] = 'L1c'
        new_obj._cube = cube
        return new_obj

    def to_l1d(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True,
               generate_figures=False) -> "L1DCapture":
        return _spawn_l1d(self, use_direct_georef=use_direct_georef, use_thuillier=use_thuillier,
                          use_unbinned=use_unbinned, generate_figures=generate_figures)


class L1CCapture(_CaptureCommon):
    LEVEL = "l1c"

    def to_l1d(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True,
               generate_figures=False) -> "L1DCapture":
        return _spawn_l1d(self, use_direct_georef=use_direct_georef, use_thuillier=use_thuillier,
                          use_unbinned=use_unbinned, generate_figures=generate_figures)


class L1DCapture(_CaptureCommon):
    LEVEL = "l1d"

    def to_l2a(self, correction: str, **kwargs) -> "L2ACapture":
        return spawn_l2a(self, correction, **kwargs)


class L2ACapture(_CaptureCommon):
    """L2A keeps the existing _l2a_cubes DatasetDict instead of a single
    _cube - one capture can legitimately hold multiple simultaneous AC
    corrections (polymer/acolite_l2r/acolite_l2w/ocsmart/dps) at once; that's
    a real multi-value structure, not level ambiguity, so .cube/.masked_cube
    are overridden here rather than inherited."""

    LEVEL = "l2a"

    @property
    def l2a_cubes(self):
        return self._l2a_cubes

    @property
    def cube(self) -> xr.DataArray:
        if len(self._l2a_cubes) == 1:
            return next(iter(self._l2a_cubes.values()))
        raise ValueError(
            f"This capture has {len(self._l2a_cubes)} L2A correction(s) "
            f"registered ({sorted(self._l2a_cubes)}) - .cube can't pick "
            f"one; use .l2a_cubes[correction] instead."
        )

    @property
    def masked_cube(self) -> xr.DataArray:
        raise ValueError(
            "masked_cube is not defined for L2A captures (no masked_l2a_cube "
            "precedent - see hypso.masks.pipeline). Use .l2a_cubes[correction] directly."
        )


def spawn_l1b(source, coeff_type: str = None, **kwargs) -> "L1BCapture":
    """Shared body for HypsoCapture.to_l1b()."""
    from hypso.calibration import pipeline as calibration_pipeline

    if getattr(source, "_l1a_cube", None) is None:
        raise ValueError(
            "Cannot generate L1b: this capture's l1a_cube is not populated "
            f"(current type={type(source).__name__}). L1b can only be "
            "generated from L1a."
        )

    new_obj = spawn_as(source, L1BCapture)

    # run_calibration reads satobj.l1a_cube (the property name on the OLD
    # multi-level class) - L1BCapture doesn't define that property (its own
    # cube isn't L1a data), so alias it as a plain attribute on the COPY
    # just for this one call (source must stay untouched), then remove it
    # once calibration has read it - this is what gives to_l1b()'s promise
    # that the returned object genuinely has no l1a_cube afterward.
    new_obj.l1a_cube = new_obj._l1a_cube
    calibrated = calibration_pipeline.run_calibration(new_obj, coeff_type=coeff_type, **kwargs)
    del new_obj.l1a_cube
    del new_obj._l1a_cube
    for stale in ("_l1c_cube", "_l1d_cube"):
        if hasattr(new_obj, stale):
            delattr(new_obj, stale)

    new_obj._cube = _format_cube_dataarray(new_obj, calibrated, "l1b")
    return new_obj


def spawn_l2a(source, correction: str, **kwargs) -> "L2ACapture":
    """Shared body for HypsoCapture.to_l2a() and L1DCapture.to_l2a() -
    dispatches through the same hypso.ac.adapters registry self.ac uses
    rather than a hardcoded per-tool if/elif (matching HypsoCapture.to_l2a's
    existing design). Gated on success (adapter.open_output can legitimately
    no-op per its own partial-success design - see ACOLITEAdapter's
    docstring) so a no-op call returns an untouched full copy, not a
    half-cleared broken one.

    "Success" is measured as "the correction dict gained at least one new
    key", not "the correction dict is non-empty" - source (an L1DCapture,
    or a HypsoCapture whose l2a_cubes already has unrelated corrections
    registered on it from elsewhere) may already have other corrections
    present before this call, and a no-op open_output for THIS correction
    must not be mistaken for success just because something else was
    already there.
    """
    from hypso.ac.adapters import get_ac_adapter

    adapter = get_ac_adapter(correction)
    new_obj = spawn_as(source, L2ACapture)
    keys_before = set(new_obj._l2a_cubes)
    adapter.open_output(new_obj, **kwargs)

    if set(new_obj._l2a_cubes) > keys_before:
        for stale in ("_l1a_cube", "_l1b_cube", "_l1c_cube", "_l1d_cube", "_cube", "l1a_cube"):
            if hasattr(new_obj, stale):
                delattr(new_obj, stale)

    return new_obj
