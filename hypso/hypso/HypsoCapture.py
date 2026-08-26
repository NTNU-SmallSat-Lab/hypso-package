import logging
from pathlib import Path
from typing import Union, Literal
import xarray as xr
import copy
import numpy as np
from datetime import datetime, timezone
import warnings

logger = logging.getLogger(__name__)


from hypso.calibration import pipeline as calibration_pipeline

from hypso.masks import pipeline as masking_pipeline

from hypso import capture_types

from hypso.georeferencing import geo

from hypso.io import dispatch as io_dispatch

from hypso.ac.adapters import AC_ADAPTERS

from hypso.georeferencing import Georeferencer, \
                                check_star_tracker_orientation, \
                                GeoAngles, TrackGeometry

from hypso.reflectance import compute_reflectance, compute_spectral_response

from hypso.containers import DatasetDict, as_dataarray

import netCDF4 as nc


# Per-level attrs stamped onto l1a_cube/l1b_cube/l1c_cube/l1d_cube by
# HypsoCapture._format_cube_dataarray. L1b/L1c share the same units/description
# (L1c is L1b data + georeferencing, no new physical quantity - see
# HypsoCapture.l1c_cube's own docstring).
_CUBE_DATAARRAY_ATTRS = {
    "l1a": {'level': "L1a", 'units': "counts", 'description': "Digital Number (DN)"},
    "l1b": {'level': "L1b", 'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
           'description': "Top-of-Atmosphere Radiance (Lt)"},
    "l1c": {'level': "L1c", 'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
           'description': "Top-of-Atmosphere Radiance (Lt)"},
    "l1d": {'level': "L1d", 'units': r"sr^{-1}", 'description': "Top-of-Atmosphere Reflectance (Rhot)",
           'correction': None},
}


class HypsoCapture:

    # Atmospheric-correction adapters (self.ac.polymer/.acolite/.ocsmart - see
    # hypso.ac.adapters). Class-level because the adapters are stateless
    # singletons shared between captures; the ac_* wrapper methods below
    # delegate through this.
    ac = AC_ADAPTERS

    def __init__(self, path: Union[str, Path] = None, sensor_profile: "SensorProfile" = None,
                 label: str = None, load_cube: bool = True, verbose: bool = False):

        """
        Initialization of HYPSO Class.

        :param path: Absolute path to NetCDF file
        :param sensor_profile: SensorProfile (see hypso.sensors) describing the sensor this
            capture is from - platform/sensor/sat_id names, fwhm/fwhm_lookup_wl/fwhm_lookup_fwhm arrays, and
            the calibration-coefficient-file resolver. Previously each sensor required its own
            HypsoCapture subclass (Hypso1/Hypso2) to hardcode these; now any sensor with a
            registered SensorProfile works via HypsoCapture directly - see hypso.hypso1/hypso.hypso2
            for the (now ~10-line) subclasses kept for named-class/isinstance() compatibility.
        :param label: Capture processing label (e.g. "moved", "moved_unmasked") - see
            hypso.io.dispatch's load_capture_file/parse_filename. Previously left unset by
            HypsoCapture.__init__ (only subclasses set it), which meant a subclass forgetting to set
            self.label before calling load_capture_file would hit an unexplained AttributeError;
            always set here instead.
        :param load_cube: Whether to load the capture's data cube immediately. Forwarded to
            hypso.io.dispatch.load_capture_file.
        :param verbose: Verbose logging.

        """

        self.path = Path(path).absolute()

        # Sensor identity/characteristics - previously hardcoded per-subclass in Hypso1/Hypso2's
        # own __init__; now supplied by the caller via a registered SensorProfile (hypso.sensors).
        self.sensor_profile = sensor_profile
        if sensor_profile is not None:
            self.platform = sensor_profile.platform
            self.sensor = sensor_profile.sensor
            self.sat_id = sensor_profile.sat_id
            self.fwhm = sensor_profile.fwhm.copy()
            self.fwhm_lookup_wl = sensor_profile.fwhm_lookup_wl
            self.fwhm_lookup_fwhm = sensor_profile.fwhm_lookup_fwhm
        else:
            # No profile given (e.g. a subclass that still wants to set these fields itself) -
            # match the previous defaults so downstream code that checks hasattr(self, 'fwhm')
            # etc. behaves the same either way.
            self.platform = None
            self.sensor = None

        self.label = label
        self.VERBOSE = verbose

        # Initialize capture name and target
        self.capture_name = None
        self.capture_target = None

        # Initialize directory and file info
        self.capture_dir = None
        self.parent_dir = None
        self.l1a_nc_file = None
        self.l1b_nc_file = None
        self.l1c_nc_file = None
        self.l1d_nc_file = None

        # Initialize datacubes
        self._l1a_cube = None
        self._l1b_cube = None
        self._l1c_cube = None
        self._l1d_cube = None

        # Canonical spectral response (see the spectral_response property).
        self._spectral_response = None


        # Initialize dimensions
        #self.capture_type = None
        #self.spatial_dimensions = (956, 684)  # 1092 x variable
        #self.standard_dimensions = {
        #    "nominal": 956,  # Along frame_count
        #    "wide": 1092  # Along image_height (row_count)
        #}

        # Initialize masks. Custom masks live in a DatasetDict (see
        # hypso.containers for what it supersedes and why) - dict-style access
        # is unchanged, but entries are validated (raising on bad shape/dims)
        # and backed by one xarray.Dataset.
        self._land_mask = None
        self._cloud_mask = None
        self._custom_masks = DatasetDict(dim_names=('y', 'x'), num_dims=2)

        # Initialize latitude and longitude
        # TODO: store latitude and longitude as xarray
        self.latitudes = None
        self.longitudes = None
        self.latitudes_direct = None
        self.longitudes_direct = None

        # Georeferencing angles/track geometry (see hypso.georeferencing.geo_state)
        # - populated by geo.run_georeferencing/run_direct_georeferencing, or
        # (angles only) by io.dispatch.set_hypso_attributes when loading an
        # existing NC file. Deliberately NOT including framepose here - see
        # geo_state.py's module docstring for why it stays a separate flat
        # attribute, never initialized in __init__ (see run_direct_georeferencing's
        # existence-probe callers in geo.py, which rely on it being genuinely
        # absent until computed).
        self.angles = GeoAngles()
        self.angles_direct = GeoAngles()
        self.track = TrackGeometry()
        self.track_direct = TrackGeometry()

        # Other
        self.dim_names_3d = ["y", "x", "band"]
        self.dim_names_2d = ["y", "x"]

        # Products dictionary - a DatasetDict (see hypso.containers for what
        # it supersedes and why), matching _custom_masks/_l2a_cubes'
        # construction. Confirmed live external consumer: hypso-processing-
        # pipeline's Polymer stage writes the chlorophyll product here
        # (satobj.products['chla'] = ...) and persists it via
        # write_products_nc_file - this is NOT unused/dead infrastructure.
        self._products = DatasetDict(dim_names=('y', 'x'), num_dims=2)

        # Constants
        self.UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here
        self.AVERAGE_FWHM = 3.33 #8.2 
        self.UNBINNED_BAND_COUNT = 1936
        
        # Atmospheric Correction
        self.ocsmart_dir = None
        self.acolite_dir = None

        # DEBUG
        self.DEBUG = False

        # Level-2 datacubes - a DatasetDict (see hypso.containers for what it
        # supersedes and why); one entry per AC correction, keyed by tool
        # ("polymer"/"acolite_l2r"/...), each entry's key stored in its
        # attrs['correction'].

        l2_attributes = {'level': "L2",
                    'units': r"sr^{-1}",
                    'description': "Bottom of Atmosphere Reflectance (Rrs)",
                    'l2_variable_name': "rrs"
                    }

        self._l2a_cubes = DatasetDict(attributes=l2_attributes, num_dims=3, key_attribute='correction')

        # Only load here when constructed with a sensor_profile - this is what lets HypsoCapture be
        # used directly (no per-sensor subclass required) for any sensor with a registered
        # profile. A subclass that omits sensor_profile keeps the old contract: it's responsible
        # for setting whatever it needs (platform/sensor/sat_id/fwhm/etc.) and calling
        # io_dispatch.load_capture_file(self, ...) itself, after its own __init__ body runs.
        if sensor_profile is not None:
            io_dispatch.load_capture_file(self, path=path, load_cube=load_cube)


    @property
    def l2a_cubes(self):
        """Every currently-populated L2A correction, keyed by tool name
        ("polymer"/"acolite_l2r"/...) - a DatasetDict, not a single cube,
        since one capture can hold multiple independent AC results at once.
        See l2a_cube for the deprecated singular-named alias."""

        self._l2a_cubes.dim_shape = tuple(self.spatial_dimensions)
        self._l2a_cubes.dim_names = tuple(self.dim_names_3d)
        self._l2a_cubes.num_dims = 3

        return self._l2a_cubes

    @l2a_cubes.setter
    def l2a_cubes(self, value):
        raise AttributeError("[ERROR] Use \"l2a_cubes[key] = value\" to set items.")

    @property
    def l2a_cube(self):
        """Deprecated alias for l2a_cubes - the singular name for what's
        actually a multi-entry container caused a real bug (a stray
        satobj.l2a_cubes[...] reference in io/dispatch.py silently had no
        matching property until this rename). Kept, unchanged, for existing
        callers (e.g. hypso-processing-pipeline's extraction.py/
        2d_hypso_noise_snr.py) - not removed until those have migrated."""
        warnings.warn(
            "l2a_cube is deprecated - use l2a_cubes (same object; the name "
            "now matches what it actually returns, a multi-correction "
            "container, not a single cube).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.l2a_cubes

    @l2a_cube.setter
    def l2a_cube(self, value):
        raise AttributeError("[ERROR] Use \"l2a_cubes[key] = value\" to set items.")

    def l2a_name(self, label: str = None, atmospheric_correction: str = None):

        if label:
            label = "-" + str(label)
        elif hasattr(self, "coeff_type"):
            label = "-" + str(getattr(self, "label"))
        else:
            label = ""


        if atmospheric_correction:
            atmospheric_correction = "-" + str(atmospheric_correction)
        elif hasattr(self, "atmospheric_correction"):
            atmospheric_correction = "-" + str(getattr(self, "atmospheric_correction"))
        else:
            atmospheric_correction = ""


        #aeronetvenice_2025-07-22T09-57-52Z-moved-l2a-polymer
        l2a_name = self.capture_name + label + "-l2a" + atmospheric_correction + ".nc" 


        return l2a_name
    

    def _update_dataarray_attrs(self, data: xr.DataArray, attrs: dict) -> xr.DataArray:

        for key, value in attrs.items():
            if key not in data.attrs:
                data.attrs[key] = value

        return data

    def _format_cube_dataarray(self, data: Union[np.ndarray, xr.DataArray], level: str) -> xr.DataArray:
        """Validate/wrap a 3D (lines, samples, band) cube array for the given
        product level, stamping that level's standard attrs. One shared
        implementation instead of a near-identical _format_l1a_dataarray/
        _format_l1b_dataarray/_format_l1c_dataarray/_format_l1d_dataarray
        method per level - each of those only ever differed in this attrs
        dict, called from exactly one place (the matching l1x_cube setter)."""
        data = as_dataarray(data, tuple(self.dim_names_3d), num_dims=3,
                            dim_shape=tuple(self.spatial_dimensions))
        data = self._update_dataarray_attrs(data, _CUBE_DATAARRAY_ATTRS[level])

        return data


    @property
    def l1a_cube(self):
        return self._l1a_cube


    @l1a_cube.setter
    def l1a_cube(self, value):
        self._l1a_cube = self._format_cube_dataarray(value, "l1a")


    @l1a_cube.deleter
    def l1a_cube(self):
        self._l1a_cube = None


    @property
    def l1b_cube(self):
        return self._l1b_cube


    @l1b_cube.setter
    def l1b_cube(self, value):
        self._l1b_cube = self._format_cube_dataarray(value, "l1b")


    @l1b_cube.deleter
    def l1b_cube(self):
        # Also frees l1c_cube's data - see its property getter, l1c has no
        # independent storage, it's a georeferenced view over this same array.
        self._l1b_cube = None


    @property
    def l1c_cube(self):
        # Return l1b cube since it is the same as the l1c cube
        cube = copy.deepcopy(self._l1b_cube)
        cube.attrs['level'] = 'L1c'
        return cube


    @l1c_cube.setter
    def l1c_cube(self, value):
        self._l1c_cube = self._format_cube_dataarray(value, "l1c")


    @l1c_cube.deleter
    def l1c_cube(self):
        # l1c has no independent storage (see the getter above) - deleting it
        # deletes the l1b data it's a view over.
        self._l1b_cube = None


    @property
    def l1d_cube(self):
        return self._l1d_cube

    @l1d_cube.setter
    def l1d_cube(self, value):
        self._l1d_cube = self._format_cube_dataarray(value, "l1d")


    @l1d_cube.deleter
    def l1d_cube(self):
        self._l1d_cube = None


    # Read-only compatibility properties over self.angles/self.angles_direct
    # (hypso.georeferencing.geo_state.GeoAngles) - the 10 flat names these
    # replace are set by geo.run_georeferencing/run_direct_georeferencing and
    # io.dispatch.set_hypso_attributes, and read by io/writer.py's
    # _INDIRECT_ANGLE_FIELDS/_DIRECT_ANGLE_FIELDS table and
    # write/geometry_group_writer.py's hardcoded blocks - both keep working
    # unmodified as long as these resolve via plain getattr()/attribute
    # access, which properties do. No setters (truly read-only - self.angles/
    # self.angles_direct are what's actually written to) and no
    # DeprecationWarning: unlike l2a_cube's deprecated alias, these aren't
    # being phased out - solar_zenith_angles/sat_zenith_angles/
    # relative_azimuth_angles are still directly read by
    # hypso-processing-pipeline, and all 10 are read on every L1B/L1C/L1D/L2A
    # write, so warning on every read would be pure noise.
    @property
    def sat_zenith_angles(self):
        return self.angles.sensor_zenith

    @property
    def sat_azimuth_angles(self):
        return self.angles.sensor_azimuth

    @property
    def solar_zenith_angles(self):
        return self.angles.solar_zenith

    @property
    def solar_azimuth_angles(self):
        return self.angles.solar_azimuth

    @property
    def relative_azimuth_angles(self):
        return self.angles.relative_azimuth

    @property
    def sat_zenith_angles_direct(self):
        return self.angles_direct.sensor_zenith

    @property
    def sat_azimuth_angles_direct(self):
        return self.angles_direct.sensor_azimuth

    @property
    def solar_zenith_angles_direct(self):
        return self.angles_direct.solar_zenith

    @property
    def solar_azimuth_angles_direct(self):
        return self.angles_direct.solar_azimuth

    @property
    def relative_azimuth_angles_direct(self):
        return self.angles_direct.relative_azimuth


    # Mask orchestration (formatters, land_mask/cloud_mask/custom_masks state,
    # set_custom_mask/clear_custom_masks/load_mask_from_file, unified_mask, and
    # the four masked_l1x_cube bodies - collapsed into one get_masked_cube) was
    # extracted verbatim into hypso.masks.pipeline - part of the HypsoCapture
    # breakup called for in the approved refactor plan. land_mask/cloud_mask
    # stay as thin delegating properties because they're called externally
    # (hypso-processing-pipeline sets satobj.land_mask/cloud_mask directly);
    # everything else here (custom_masks, set_custom_mask, clear_custom_masks,
    # load_mask_from_file, masked_l1x_cube) has zero confirmed external callers
    # but is public, documented API, so it's kept as thin wrappers rather than
    # moved without one - same treatment as the ac_polymer_*/ac_acolite_*
    # wrappers above.

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
        """Read-only view of every mask registered via set_custom_mask, keyed by
        name (e.g. "sea_land_cloud"). Combined into masked_l1a/b/c/d_cube the same
        way land_mask/cloud_mask are - see hypso.masks.pipeline.unified_mask."""
        return dict(self._custom_masks)


    def set_custom_mask(self, name: str, value: Union[np.ndarray, xr.DataArray, None],
                        description: str = None) -> None:
        return masking_pipeline.set_custom_mask(self, name, value, description)


    def clear_custom_masks(self) -> None:
        return masking_pipeline.clear_custom_masks(self)


    def load_mask_from_file(self, path: Union[str, Path], name: str = None, variable: str = None,
                            dtype=np.bool_, invert: bool = False) -> xr.DataArray:
        return masking_pipeline.load_mask_from_file(self, path, name=name, variable=variable,
                                                     dtype=dtype, invert=invert)


    @property
    def masked_l1a_cube(self) -> xr.DataArray:
        return masking_pipeline.get_masked_cube(self, "l1a")


    @property
    def masked_l1b_cube(self) -> xr.DataArray:
        return masking_pipeline.get_masked_cube(self, "l1b")


    @property
    def masked_l1c_cube(self) -> xr.DataArray:
        return masking_pipeline.get_masked_cube(self, "l1c")


    @property
    def masked_l1d_cube(self) -> xr.DataArray:
        return masking_pipeline.get_masked_cube(self, "l1d")


    def discard_cube(self, level: str, correction: str = None) -> None:
        """Free a generated cube's memory (sets it to None) without discarding
        the rest of this capture (geometry, calibration coefficients, metadata)
        - for a workflow that generates several levels in sequence and wants to
        release each one once done with it, rather than holding every level's
        full-size cube in memory for the lifetime of this object. Equivalent to
        `del satobj.l1a_cube` / `del satobj.l1b_cube` / `del satobj.l1d_cube`
        for those three levels; provided as one method so the level can be a
        variable/config value instead of a hardcoded attribute name, and
        because l2a needs an extra argument (correction) that a plain `del`
        can't take.

        :param level: one of "l1a", "l1b", "l1c", "l1d", "l2a" (case-insensitive).
            "l1b" and "l1c" both free the same underlying array - see
            l1c_cube's property getter, l1c has no independent storage, it's a
            georeferenced view over the l1b data.
        :param correction: for level="l2a" only - the correction key to
            discard (e.g. "polymer"). If None, every registered l2a correction
            is discarded.
        """
        level = level.lower()

        if level == 'l1a':
            self._l1a_cube = None
        elif level in ('l1b', 'l1c'):
            self._l1b_cube = None
        elif level == 'l2a':
            if correction is None:
                self._l2a_cubes.clear()
            else:
                self._l2a_cubes.pop(correction, None)
        elif level == 'l1d':
            self._l1d_cube = None
        else:
            raise ValueError(f"discard_cube: unknown level {level!r}, expected one of "
                             f"'l1a', 'l1b', 'l1c', 'l1d', 'l2a'")

        return None


    @property
    def products(self):

        self._products.dim_shape = tuple(self.spatial_dimensions)
        self._products.dim_names = tuple(self.dim_names_2d)
        self._products.num_dims = 2

        return self._products

    @products.setter
    def products(self, value):
        raise AttributeError("[ERROR] Use \"products[key] = value\" to set items.")
























    # Load dispatch (_load_capture_file/_set_hypso_attributes/_check_capture_type/
    # _parse_filename/_compose_capture_name) was extracted verbatim into
    # hypso.io.dispatch - part of the HypsoCapture breakup called for in the approved
    # refactor plan (self.io composition). No wrapper methods kept here: confirmed
    # via grep these five had zero external callers. HypsoCapture.__init__ now calls
    # io_dispatch.load_capture_file(self, ...) directly.

    # Calibration orchestration (_set_calibration_coeff_files/_run_calibration/
    # _load_calibration_coeff_files) was extracted verbatim into
    # hypso.calibration.pipeline - part of the HypsoCapture breakup called for in the
    # approved refactor plan (self.calibration composition). No wrapper methods kept
    # here: confirmed via grep these three had zero external callers, unlike
    # hypso.georeferencing.geo's run_georeferencing()/run_direct_georeferencing(), which stayed as
    # methods because those specific names are called externally. Internal callers
    # within this file now call calibration_pipeline.set_calibration_coeff_files/
    # load_calibration_coeff_files/run_calibration(self, ...) directly.


    # Georeferencing orchestration (run_direct_georeferencing/run_georeferencing and
    # their private _run_* helpers) was extracted verbatim into hypso.georeferencing.geo - part of
    # the HypsoCapture breakup called for in the approved refactor plan (self.geo
    # composition). These stay as thin delegating wrappers, not moved themselves,
    # because run_direct_georeferencing() is called externally
    # (hypso/ac/loading_acolite_output.py) and run_georeferencing() by
    # hypso-processing-pipeline - both names/signatures must keep working unchanged.
    # The private _run_frame_interpolation/_run_track_geometry/_run_angles_geometry
    # helpers had no external callers, so they moved to hypso.georeferencing.geo outright with no
    # wrapper kept here.

    def run_direct_georeferencing(self) -> None:
        return geo.run_direct_georeferencing(self)


    def run_georeferencing(self, latitudes: np.ndarray = None, longitudes: np.ndarray = None) -> None:
        return geo.run_georeferencing(self, latitudes=latitudes, longitudes=longitudes)



    def generate_l1b_cube(self, coeff_type: str = None, **kwargs) -> None:
        """Mutates this object in place (sets self.l1b_cube). Deprecated in favor
        of to_l1b(), which returns a new object instead - see
        docs/architecture.rst's "Cube memory" section for why. Kept, unchanged,
        for existing in-place callers (e.g. hypso-processing-pipeline) - not
        removed until those have migrated."""
        warnings.warn(
            "generate_l1b_cube() mutates this object in place and is deprecated "
            "in favor of to_l1b(), which returns a new object instead of "
            "mutating this one - see docs/architecture.rst's 'Cube memory' "
            "section. generate_l1b_cube() is not being removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._generate_l1b_cube_impl(coeff_type=coeff_type, **kwargs)


    def _generate_l1b_cube_impl(self, coeff_type: str = None, **kwargs) -> None:

        print("[INFO] Generating L1b cube")
        if self.l1a_cube is None:
            return None

        self.l1b_cube = calibration_pipeline.run_calibration(self, coeff_type=coeff_type, **kwargs)

        return None



    def generate_l1c_cube(self, coeff_type: str = None, **kwargs) -> None:
        """Mutates this object in place (sets self.l1b_cube, runs
        georeferencing). Deprecated in favor of to_l1c() - see
        generate_l1b_cube()'s docstring for why."""
        warnings.warn(
            "generate_l1c_cube() mutates this object in place and is deprecated "
            "in favor of to_l1c(), which returns a new object instead of "
            "mutating this one - see docs/architecture.rst's 'Cube memory' "
            "section. generate_l1c_cube() is not being removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._generate_l1c_cube_impl(coeff_type=coeff_type, **kwargs)


    def _generate_l1c_cube_impl(self, coeff_type: str = None, **kwargs) -> None:

        print("[INFO] Generating L1c cube")
        if self.l1b_cube is None:
            self._generate_l1b_cube_impl(coeff_type=coeff_type, **kwargs)

        self.run_georeferencing()

        return None



    def generate_l1d_cube(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True, generate_figures=False) -> None:
        """Mutates this object in place (sets self.l1d_cube). Deprecated in
        favor of to_l1d() - see generate_l1b_cube()'s docstring for why."""
        warnings.warn(
            "generate_l1d_cube() mutates this object in place and is deprecated "
            "in favor of to_l1d(), which returns a new object instead of "
            "mutating this one - see docs/architecture.rst's 'Cube memory' "
            "section. generate_l1d_cube() is not being removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._generate_l1d_cube_impl(use_direct_georef=use_direct_georef, use_thuillier=use_thuillier,
                                            use_unbinned=use_unbinned, generate_figures=generate_figures)


    @property
    def spectral_response(self) -> "SpectralResponse":
        """The capture's canonical SpectralResponse (see
        hypso.reflectance.spectral_response for what it supersedes).

        Set by L1D generation; for a capture LOADED from a file it is rebuilt
        lazily on first access - the SRF matrix itself is not persisted in the
        NetCDF files (only esun/effective_fwhm reach metadata/srf), and the
        builder is deterministic given wavelengths_unbinned/fwhm_unbinned/
        bin_factor, which the file does carry. The rebuild assumes the default
        L1D configuration (TSIS SSI, unbinned inputs, native-truncated grid) -
        a capture originally processed with use_thuillier=True or
        use_unbinned=False would need compute_spectral_response called
        explicitly with those settings instead.

        The rebuild also backfills any *missing* legacy SRF attributes
        (srf/srf_ssi/srf_ssi_wl, plus esun/esun_wl/effective_fwhm when the
        file lacked them) so the Polymer connector's generate_srf_nc/ssi/esun
        work on a file-loaded capture; values already loaded from the file are
        left untouched.
        """
        if self._spectral_response is None:
            if not hasattr(self, "fwhm_unbinned"):
                self._get_fwhm_unbinned()
            # Mirror the in-session L1D path (_generate_l1d_cube_impl), which
            # recomputes per-band fwhm from the (precise) wavelengths via the
            # sensor's lookup table - a file-loaded capture otherwise still
            # carries the static profile default, which differs at lookup
            # boundary bands.
            self._get_fwhm()

            sr = compute_spectral_response(
                band_centers_unbinned=self.wavelengths_unbinned,
                fwhm_unbinned=np.asarray(self.fwhm_unbinned),
                bin_factor=self.bin_factor,
                ssi_source="tsis",
                grid="native-truncated",
            )
            self._spectral_response = sr

            for attr, value in (("srf", sr.srf),
                                ("srf_ssi", sr.ssi),
                                ("srf_ssi_wl", sr.grid_wl),
                                ("esun", sr.esun),
                                ("esun_wl", sr.band_centers),
                                ("effective_fwhm", sr.effective_fwhm)):
                if getattr(self, attr, None) is None:
                    setattr(self, attr, value)

        return self._spectral_response

    @spectral_response.setter
    def spectral_response(self, value):
        self._spectral_response = value


    def _generate_l1d_cube_impl(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True, generate_figures=False) -> None:

        print("[INFO] Generating L1d cube")
        self._get_fwhm()
        self._get_fwhm_unbinned()


        if self.l1b_cube is not None:
            toa_radiance = self.l1b_cube
        elif self.l1c_cube is not None:
            toa_radiance = self.l1c_cube
        else:
            self._generate_l1b_cube_impl()
            toa_radiance = self.l1b_cube

        if use_direct_georef and self.angles_direct.solar_zenith is not None:

            if self.VERBOSE:
                print('[WARNING] Computing TOA reflectance using DIRECT georeferencing geometry.')

            solar_zenith_angles=self.solar_zenith_angles_direct
        else:
            solar_zenith_angles=self.solar_zenith_angles

        if use_unbinned:
            sensor_wavelengths = self.wavelengths_unbinned
            sensor_fwhm = self.fwhm_unbinned
            sensor_bin_factor = self.bin_factor
        else:
            sensor_wavelengths = self.wavelengths
            sensor_fwhm = self.fwhm
            sensor_bin_factor = 1


        # The spectral response (SRFs/SSI/esun/effective FWHM) is now one
        # value object - see hypso.reflectance.spectral_response's module
        # docstring for what this supersedes and the naming it fixes.
        sr = compute_spectral_response(
            band_centers_unbinned=sensor_wavelengths,
            fwhm_unbinned=sensor_fwhm,
            bin_factor=sensor_bin_factor,
            ssi_source="thuillier" if use_thuillier else "tsis",
            grid="native-truncated",
            generate_figures=generate_figures,
        )

        self.l1d_cube = compute_reflectance(toa_radiance=toa_radiance, sr=sr,
                                            iso_time=self.iso_time,
                                            solar_zenith_angles=solar_zenith_angles)

        self.spectral_response = sr

        # Legacy attribute family - SUPERSEDED by self.spectral_response but
        # still populated with the exact same values because the Polymer
        # connector (hypso.ac.adapters.polymer's generate_srf_nc/ssi/esun) and
        # write/metadata_srf_group_writer.py still read these names. To be
        # removed when the AC connectors are migrated to read
        # self.spectral_response directly (the later AC-connector pass, see
        # REFACTOR_PROGRESS.md). Note what the old names actually held:
        # srf = the BINNED SRF matrix, esun_wl = binned BAND CENTERS (not an
        # SSI wavelength grid, despite the srf_ssi_wl name symmetry).
        self.srf = sr.srf
        self.srf_ssi = sr.ssi
        self.srf_ssi_wl = sr.grid_wl
        self.esun = sr.esun
        self.esun_wl = sr.band_centers
        self.effective_fwhm = sr.effective_fwhm

        return None


    def to_l1a(self) -> "capture_types.L1ACapture":
        """Returns a new L1ACapture (see hypso.capture_types) - self is left
        completely untouched. Hypso(path) (this class's own load entry
        point, unchanged) still returns a plain HypsoCapture/Hypso1/Hypso2;
        calling .to_l1a() on that result is how to opt into the typed
        to_l1x() chain from the very start instead of only from to_l1b()
        onward. No generation step - L1A data is already fully loaded by
        the time this is called, this just retypes the object.
        """
        return capture_types.spawn_l1a(self)


    def to_l1b(self, coeff_type: str = None, **kwargs) -> "capture_types.L1BCapture":
        """Returns a new L1BCapture (see hypso.capture_types) - self
        (including its own l1a_cube) is left completely untouched. Unlike
        generate_l1b_cube() (kept unchanged - this method does not replace
        it - for hypso-processing-pipeline's existing in-place, mutating
        usage), the returned object's TYPE says it's at L1B: it has no
        l1a_cube attribute at all, not just a cleared one - see
        docs/architecture.rst's "Cube memory" section.
        """
        return capture_types.spawn_l1b(self, coeff_type=coeff_type, **kwargs)


    def to_l1c(self, coeff_type: str = None, **kwargs) -> "capture_types.L1CCapture":
        """Convenience one-hop from L1A straight to L1C - equivalent to
        self.to_l1b(coeff_type=coeff_type, **kwargs).to_l1c(). See
        to_l1b()'s docstring for why this returns a typed object instead of
        another HypsoCapture."""
        return self.to_l1b(coeff_type=coeff_type, **kwargs).to_l1c()


    def to_l1d(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True,
               generate_figures=False) -> "capture_types.L1DCapture":
        """Convenience one-hop from L1A straight to L1D - equivalent to
        self.to_l1b().to_l1d(...). See to_l1b()'s docstring for why this
        returns a typed object instead of another HypsoCapture."""
        return self.to_l1b().to_l1d(use_direct_georef=use_direct_georef, use_thuillier=use_thuillier,
                                    use_unbinned=use_unbinned, generate_figures=generate_figures)


    def to_l2a(self, correction: str, **kwargs) -> "capture_types.L2ACapture":
        """Returns a new L2ACapture (see hypso.capture_types) holding this AC
        correction's L2A cube(s), instead of mutating self's shared
        l2a_cubes container the way ac_polymer_open_output()/
        ac_acolite_open_output()/ac_ocsmart_open_output() do. self is left
        completely untouched, including its own l2a_cubes entries. Cheap for
        the same reason to_l1b()/to_l1c()/to_l1d() are - see
        capture_types.spawn_as's docstring: big arrays (including the L1D
        cube this is built from) are aliased, not duplicated.

        Dispatches through the same hypso.ac.adapters registry self.ac uses
        (get_ac_adapter) rather than a hardcoded per-tool if/elif - any AC
        tool registered there (see hypso/ac/adapters/__init__.py) works here
        with no changes needed.

        :param correction: which AC tool's output to open - one of
            get_ac_adapter's registered keys ("polymer"/"acolite"/"ocsmart").
            ACOLITE's own open_output opens acolite_l2r and/or acolite_l2w
            together depending on which of the
            acolite_l2r_output_nc_file/acolite_l2w_output_nc_file kwargs (or
            satobj.acolite_l2r_output_nc_file/acolite_l2w_output_nc_file
            attrs) resolve to an existing file - pass "acolite" for either or
            both; check the returned object's l2a_cubes keys for what
            actually landed.
        :param kwargs: forwarded to the matching adapter's open_output.
        """
        return capture_types.spawn_l2a(self, correction, **kwargs)




    # Atmospheric-correction orchestration (every ac_polymer_*/ac_acolite_*/
    # ac_ocsmart_* method, plus the shared _get_inferred_wavelength_band_map
    # helper) was extracted verbatim into hypso.ac.adapters - one adapter class
    # per external tool behind a shared run_correction/open_output interface
    # (self.ac composition; the plan's "prepare the AC functions to be
    # refactored" step - organizational only, bodies not rewritten). Unlike the
    # calibration/load-dispatch extractions, every public ac_* name below stays
    # as a thin delegating wrapper: these are confirmed external API
    # (hypso-processing-pipeline calls them as satobj methods). The private
    # _get_inferred_wavelength_band_map moved without a wrapper (zero external
    # callers; it now lives in hypso.ac.adapters.base), as did five other
    # zero-external-caller wrappers removed in the HypsoCapture cleanup:
    # ac_ocsmart_run_correction (only caller was a permanently-dead branch in
    # ac/loading_acolite_output.py, removed alongside it) and
    # ac_polymer_get_id_sensor/get_srf_nc_path/get_ssi_nc_path/get_esun_nc_path
    # (superseded by hypso.ac.adapters.PolymerAdapter + SpectralResponse).

    def ac_ocsmart_open_output(self, h5_file_path: Path = None):
        """Mutates this object in place (populates self.l2a_cubes["ocsmart"]).
        Deprecated in favor of to_l2a(correction="ocsmart", ...), which returns
        a new object instead - see docs/architecture.rst's "Cube memory"
        section for why. Kept, unchanged, for existing in-place callers (e.g.
        hypso-processing-pipeline) - not removed until those have migrated."""
        warnings.warn(
            "ac_ocsmart_open_output() mutates this object's shared l2a_cubes "
            "in place and is deprecated in favor of "
            "to_l2a(correction='ocsmart', ...), which returns a new object "
            "instead of mutating this one - see docs/architecture.rst's "
            "'Cube memory' section. ac_ocsmart_open_output() is not being "
            "removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ac_ocsmart_open_output_impl(h5_file_path=h5_file_path)


    def _ac_ocsmart_open_output_impl(self, h5_file_path: Path = None):
        return self.ac.ocsmart.open_output(self, h5_file_path=h5_file_path)


    def ac_acolite_run_correction(self, settings_file: Path = None,
                                  input_product_level: str = 'l1c',
                                  EARTHDATA_u: str = None,
                                  EARTHDATA_p: str = None,
                                  python_path: str = None
                                  ):
        return self.ac.acolite.run_correction(self, settings_file=settings_file,
                                              input_product_level=input_product_level,
                                              EARTHDATA_u=EARTHDATA_u,
                                              EARTHDATA_p=EARTHDATA_p,
                                              python_path=python_path)


    def ac_acolite_open_output(self, acolite_l2r_output_nc_file: Path = None, acolite_l2w_output_nc_file: Path = None):
        """Mutates this object in place (populates self.l2a_cubes["acolite_l2r"]
        and/or ["acolite_l2w"]). Deprecated in favor of
        to_l2a(correction="acolite", ...), which returns a new object instead
        - see docs/architecture.rst's "Cube memory" section for why. Kept,
        unchanged, for existing in-place callers (e.g.
        hypso-processing-pipeline) - not removed until those have migrated."""
        warnings.warn(
            "ac_acolite_open_output() mutates this object's shared l2a_cubes "
            "in place and is deprecated in favor of "
            "to_l2a(correction='acolite', ...), which returns a new object "
            "instead of mutating this one - see docs/architecture.rst's "
            "'Cube memory' section. ac_acolite_open_output() is not being "
            "removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ac_acolite_open_output_impl(
            acolite_l2r_output_nc_file=acolite_l2r_output_nc_file,
            acolite_l2w_output_nc_file=acolite_l2w_output_nc_file)


    def _ac_acolite_open_output_impl(self, acolite_l2r_output_nc_file: Path = None, acolite_l2w_output_nc_file: Path = None):
        return self.ac.acolite.open_output(self,
                                           acolite_l2r_output_nc_file=acolite_l2r_output_nc_file,
                                           acolite_l2w_output_nc_file=acolite_l2w_output_nc_file)


    def ac_polymer_generate_srf_nc(self):
        return self.ac.polymer.generate_srf_nc(self)


    def ac_polymer_generate_ssi_nc(self):
        return self.ac.polymer.generate_ssi_nc(self)


    def ac_polymer_generate_esun_nc(self):
        return self.ac.polymer.generate_esun_nc(self)


    def ac_polymer_run_correction(self,
                                  polymer_base_path: str,
                                  polymer_path: str = None,
                                  eoread_path: str = None,
                                  eotools_path: str = None,
                                  core_path: str = None,
                                  input_product_level: str = "l1c",
                                  optional_output_datasets: list = ["SPM"],
                                  if_exists: str = "overwrite",
                                  polymer_version: str = "v1",
                                  python_path: str = None):
        return self.ac.polymer.run_correction(self,
                                              polymer_base_path=polymer_base_path,
                                              polymer_path=polymer_path,
                                              eoread_path=eoread_path,
                                              eotools_path=eotools_path,
                                              core_path=core_path,
                                              input_product_level=input_product_level,
                                              optional_output_datasets=optional_output_datasets,
                                              if_exists=if_exists,
                                              polymer_version=polymer_version,
                                              python_path=python_path)


    def ac_polymer_open_output(self,
                               polymer_l2_output_nc_file: Path = None,
                               input_product_level="l1c",
                               version = "v1"
                               ):
        """Mutates this object in place (populates self.l2a_cubes["polymer"]).
        Deprecated in favor of to_l2a(correction="polymer", ...), which
        returns a new object instead - see docs/architecture.rst's "Cube
        memory" section for why. Kept, unchanged, for existing in-place
        callers (e.g. hypso-processing-pipeline) - not removed until those
        have migrated."""
        warnings.warn(
            "ac_polymer_open_output() mutates this object's shared l2a_cubes "
            "in place and is deprecated in favor of "
            "to_l2a(correction='polymer', ...), which returns a new object "
            "instead of mutating this one - see docs/architecture.rst's "
            "'Cube memory' section. ac_polymer_open_output() is not being "
            "removed yet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ac_polymer_open_output_impl(
            polymer_l2_output_nc_file=polymer_l2_output_nc_file,
            input_product_level=input_product_level,
            version=version)


    def _ac_polymer_open_output_impl(self,
                                     polymer_l2_output_nc_file: Path = None,
                                     input_product_level="l1c",
                                     version = "v1"
                                     ):
        return self.ac.polymer.open_output(self,
                                           polymer_l2_output_nc_file=polymer_l2_output_nc_file,
                                           input_product_level=input_product_level,
                                           version=version)


    def _get_fwhm(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths: 
            idx = np.argmin(np.abs(band - self.fwhm_lookup_wl))
            fwhm_per_band.append(self.fwhm_lookup_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm = fwhm
        
        return None


    def _get_fwhm_unbinned(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths_unbinned: 
            idx = np.argmin(np.abs(band - self.fwhm_lookup_wl))
            fwhm_per_band.append(self.fwhm_lookup_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm_unbinned = fwhm
        
        return None


    from hypso.reflectance import compute_csiro_srfs

    compute_csiro_srfs = compute_csiro_srfs

    from hypso.ac import ac_dark_pixel_subtraction

    ac_dark_pixel_subtraction = ac_dark_pixel_subtraction
