import logging
from pathlib import Path
from typing import Union, Literal
import xarray as xr
import copy
#from .DataArrayValidator import DataArrayValidator
import numpy as np
from datetime import datetime, timezone
from trollsift import Parser
import sys
import re
import warnings

logger = logging.getLogger(__name__)


from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              get_custom_calibration_coeffs


from hypso import geo

from hypso.georeferencing import Georeferencer, \
                                check_star_tracker_orientation

from hypso.load import load_l1a_nc, \
                        load_l1b_nc, \
                        load_l1c_nc, \
                        load_l1d_nc, \
                        load_l2a_nc, \
                        load_ocsmart_h5, \
                        load_acolite_l2r_nc, \
                        load_acolite_l2w_nc, \
                        load_polymer_l2_v1_nc, \
                        load_polymer_l2_v2_nc

from hypso.reflectance import compute_toa_reflectance

from hypso.utils import find_file

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict

import netCDF4 as nc


class HypsoBase:

    def __init__(self, path: Union[str, Path] = None, sensor_profile: "SensorProfile" = None,
                 label: str = None, load_cube: bool = True, verbose: bool = False):

        """
        Initialization of HYPSO Class.

        :param path: Absolute path to NetCDF file
        :param sensor_profile: SensorProfile (see hypso.sensors) describing the sensor this
            capture is from - platform/sensor/sat_id names, fwhm/srf_wl/srf_fwhm arrays, and
            the calibration-coefficient-file resolver. Previously each sensor required its own
            HypsoBase subclass (Hypso1/Hypso2) to hardcode these; now any sensor with a
            registered SensorProfile works via HypsoBase directly - see hypso.hypso1/hypso.hypso2
            for the (now ~10-line) subclasses kept for named-class/isinstance() compatibility.
        :param label: Capture processing label (e.g. "moved", "moved_unmasked") - see
            _load_capture_file/_parse_filename. Previously left unset by HypsoBase.__init__ (only
            subclasses set it), which meant a subclass forgetting to set self.label before calling
            _load_capture_file would hit an unexplained AttributeError; always set here instead.
        :param load_cube: Whether to load the capture's data cube immediately. Forwarded to
            _load_capture_file.
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
            self.srf_wl = sensor_profile.srf_wl
            self.srf_fwhm = sensor_profile.srf_fwhm
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


        # Initialize dimensions
        #self.capture_type = None
        #self.spatial_dimensions = (956, 684)  # 1092 x variable
        #self.standard_dimensions = {
        #    "nominal": 956,  # Along frame_count
        #    "wide": 1092  # Along image_height (row_count)
        #}

        # Initialize masks
        self._land_mask = None
        self._cloud_mask = None
        self._custom_masks: dict = {}

        # Initialize latitude and longitude
        # TODO: store latitude and longitude as xarray
        self.latitudes = None
        self.longitudes = None
        self.latitudes_direct = None
        self.longitudes_direct = None

        # Other
        self.dim_names_3d = ["y", "x", "band"]
        self.dim_names_2d = ["y", "x"]

        # Products dictionary
        self._products = DataArrayDict()

        # Constants
        self.UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here
        self.AVERAGE_FWHM = 3.33 #8.2 
        self.UNBINNED_BAND_COUNT = 1936
        
        # Atmospheric Correction
        self.ocsmart_dir = None
        self.acolite_dir = None

        # DEBUG
        self.DEBUG = False

        # Level-2 datacubes

        l2_attributes = {'level': "L2",
                    'units': r"sr^{-1}",
                    'description': "Bottom of Atmosphere Reflectance (Rrs)",
                    'l2_variable_name': "rrs"
                    }

        self._l2a_cubes = DataArrayDict(attributes=l2_attributes, num_dims=3, key_attribute='correction')

        # Only load here when constructed with a sensor_profile - this is what lets HypsoBase be
        # used directly (no per-sensor subclass required) for any sensor with a registered
        # profile. A subclass that omits sensor_profile keeps the old contract: it's responsible
        # for setting whatever it needs (platform/sensor/sat_id/fwhm/etc.) and calling
        # self._load_capture_file(...) itself, after its own __init__ body runs.
        if sensor_profile is not None:
            self._load_capture_file(path=path, load_cube=load_cube)


    @property
    def l2a_cube(self):

        self._l2a_cubes.dim_shape = self.spatial_dimensions
        self._l2a_cubes.dim_names = self.dim_names_3d
        self._l2a_cubes.num_dims = 3

        return self._l2a_cubes   

    @l2a_cube.setter
    def l2a_cubes(self, value):
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

    def _format_l1a_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1a",
                      'units': "counts",
                      'description': "Digital Number (DN)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data
    
    def _format_l1b_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1b",
                      'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
                      'description': "Top-of-Atmosphere Radiance (Lt)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data

    def _format_l1c_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1c",
                      'units': r'$mW\cdot  (m^{-2}  \cdot sr^{-1} nm^{-1})$',
                      'description': "Top-of-Atmosphere Radiance (Lt)"
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data

    def _format_l1d_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:

        attributes = {'level': "L1d",
                      'units': r"sr^{-1}",
                      'description': "Top-of-Atmosphere Reflectance (Rhot)",
                      'correction': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_3d)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray], description: str) -> xr.DataArray:
        """Validate/wrap a 2D (lines, samples) boolean-ish mask array. Shared by
        land_mask/cloud_mask and set_custom_mask - a mask is a mask regardless of
        what it represents, so there's one validation path, not one per name."""
        attributes = {
                      'description': description,
                      'method': None
                     }

        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dim_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=data)
        data = self._update_dataarray_attrs(data, attributes)

        return data


    def _format_land_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:
        return self._format_mask_dataarray(data, "Land mask")


    def _format_cloud_mask_dataarray(self, data: Union[np.ndarray, xr.DataArray]) -> xr.DataArray:
        return self._format_mask_dataarray(data, "Cloud mask")


    @property
    def l1a_cube(self):
        return self._l1a_cube


    @l1a_cube.setter
    def l1a_cube(self, value):
        self._l1a_cube = self._format_l1a_dataarray(value)


    @l1a_cube.deleter
    def l1a_cube(self):
        self._l1a_cube = None


    @property
    def l1b_cube(self):
        return self._l1b_cube


    @l1b_cube.setter
    def l1b_cube(self, value):
        self._l1b_cube = self._format_l1b_dataarray(value)


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
        self._l1c_cube = self._format_l1c_dataarray(value)


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
        self._l1d_cube = self._format_l1d_dataarray(value)


    @l1d_cube.deleter
    def l1d_cube(self):
        self._l1d_cube = None


    @property
    def land_mask(self):
        return self._land_mask 

    @land_mask.setter
    def land_mask(self, value):
        if value is not None:
            self._land_mask = self._format_land_mask_dataarray(value)
        else:
            self._land_mask = None


    @property
    def cloud_mask(self):
        return self._cloud_mask   

    @cloud_mask.setter
    def cloud_mask(self, value):
        if value is not None:
            self._cloud_mask = self._format_cloud_mask_dataarray(value)
        else:
            self._cloud_mask = None


    @property
    def custom_masks(self) -> dict:
        """Read-only view of every mask registered via set_custom_mask, keyed by
        name (e.g. "sea_land_cloud"). Combined into masked_l1a/b/c/d_cube the same
        way land_mask/cloud_mask are - see _unified_mask."""
        return dict(self._custom_masks)


    def set_custom_mask(self, name: str, value: Union[np.ndarray, xr.DataArray, None],
                        description: str = None) -> None:
        """Register (or clear, if value is None) a named custom mask - e.g. an
        externally-produced sea/land/cloud classification, not just the built-in
        land_mask/cloud_mask slots. Any number of custom masks may be registered
        at once; all of them (plus land_mask/cloud_mask, if set) are OR'd
        together by _unified_mask and applied by masked_l1a/b/c/d_cube - no
        further wiring needed once registered.

        :param name: key this mask is stored/removed under (also used in
            load_mask_from_file's `name=` argument).
        :param value: 2D (lines, samples) boolean-ish array/DataArray, True where
            a pixel should be masked out. None removes this mask.
        :param description: optional human-readable note, stored on the
            DataArray's `description` attribute (defaults to `name`).
        """
        if value is None:
            self._custom_masks.pop(name, None)
            return None

        self._custom_masks[name] = self._format_mask_dataarray(value, description or name)
        return None


    def clear_custom_masks(self) -> None:
        """Remove every registered custom mask (land_mask/cloud_mask are unaffected)."""
        self._custom_masks = {}
        return None


    def load_mask_from_file(self, path: Union[str, Path], name: str = None, variable: str = None,
                            dtype=np.bool_, invert: bool = False) -> xr.DataArray:
        """Load a 2D (lines, samples) mask from disk and, if `name` is given,
        register it via set_custom_mask in the same call.

        Supported formats, dispatched by file extension:
          - .nc: reads `variable` (required) from the file's root group via
            netCDF4 - for a mask produced by another tool (e.g. a sea/land/cloud
            classification saved as its own NetCDF product).
          - .npy: numpy .npy array.
          - .dat/.bin: raw binary, reshaped to self.spatial_dimensions using
            `dtype` (matches the convention HYPSO's own indirect-georeferencing
            lat/lon files use).

        :param path: path to the mask file.
        :param name: if given, also calls set_custom_mask(name, data) - the mask
            is registered and immediately reflected in masked_l1a/b/c/d_cube.
        :param variable: required for .nc input - the variable name to read.
        :param dtype: numpy dtype to interpret raw binary (.dat/.bin) data as.
        :param invert: if True, flip the mask (use when the source file marks
            *valid* pixels with True rather than masked-out pixels).
        :return: the loaded mask as a validated xr.DataArray (same object stored
            under `name`, if `name` was given).
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == '.nc':
            if variable is None:
                raise ValueError("load_mask_from_file requires variable=... for a .nc source file")
            with nc.Dataset(path, format="NETCDF4") as f:
                data = np.array(f.variables[variable][:])
        elif suffix == '.npy':
            data = np.load(path)
        elif suffix in ('.dat', '.bin'):
            data = np.fromfile(path, dtype=dtype).reshape(self.spatial_dimensions)
        else:
            raise ValueError(f"load_mask_from_file: unsupported file extension {suffix!r} "
                             f"(expected .nc, .npy, .dat, or .bin)")

        data = data.astype(bool)
        if invert:
            data = ~data

        if name is not None:
            self.set_custom_mask(name, data)
            return self._custom_masks[name]

        return self._format_mask_dataarray(data, path.name)


    @property
    def masked_l1a_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1a_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1a_cube   
        

    @property
    def masked_l1b_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1b_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1b_cube   


    @property
    def masked_l1c_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1c_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1c_cube   


    @property
    def masked_l1d_cube(self) -> xr.DataArray:

        unified_mask = self._unified_mask()

        if unified_mask is not None:

            return self._l1d_cube.where(~unified_mask, other=np.nan)

        else:
            return self._l1d_cube


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


    def _unified_mask(self) -> xr.DataArray:
        """OR land_mask, cloud_mask, and every registered custom mask (see
        set_custom_mask/load_mask_from_file) together - masked_l1a/b/c/d_cube all
        apply whatever this returns, so a custom mask needs no changes there."""
        masks = [m for m in (self._land_mask, self._cloud_mask) if m is not None]
        masks.extend(self._custom_masks.values())

        if not masks:
            return None

        unified_mask = masks[0]
        for mask in masks[1:]:
            unified_mask = unified_mask | mask

        return unified_mask


    @property
    def products(self):

        self._products.dim_shape = self.spatial_dimensions
        self._products.dim_names = self.dim_names_2d
        self._products.num_dims = 2

        return self._products   

    @products.setter
    def products(self, value):
        raise AttributeError("[ERROR] Use \"products[key] = value\" to set items.")
























    def _compose_capture_name(self, fields: dict) -> str:


        if hasattr(self, '_use_old_filename_format'):
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}") # Old filename format
        else:
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}") # New filename format

        capture_name = p.compose(fields)

        return capture_name


    def _parse_filename(self, path: str) -> dict:

        path = Path(path).absolute()
        filename = path.name

        pattern = re.compile(
            r"""
            (?P<capture_target>.+?)_
            (?P<capture_datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)
            -
            (?:(?P<coeff_type>[^-]+)-)?
            (?P<product_level>l\d[a-z])
            (?:-(?P<atmospheric_correction>[^.]+))?
            \.
            (?P<file_type>\w+)
            """,
            re.VERBOSE,
        )

        match = pattern.fullmatch(filename)

        if not match:
            raise ValueError(f"Could not parse filename: {filename}")

        fields = match.groupdict()

        fields["capture_datetime"] = datetime.strptime(
            fields["capture_datetime"],
            "%Y-%m-%dT%H-%M-%SZ",
        )

        return fields


    '''
    def _parse_filename(self, path: str) -> dict:

        path = Path(path).absolute()
        field = None

        try:
            # New filename format
            #aegean_2024-08-22T08-41-46Z-l1a.nc
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        except:
            # Old filename format
            setattr(self, '_use_old_filename_format', True)
            p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}-{product_level:3s}{atmospheric_correction:->}.{file_type}")
            fields = p.parse(str(path.name))
        
        return fields
    '''

    def _load_capture_file(self, path: Path, load_cube: bool = True) -> None:

        path = Path(path).absolute()

        fields = self._parse_filename(path=path)

        for key, value in fields.items():
            setattr(self, key, value)

        capture_name = self._compose_capture_name(fields=fields)

        self.capture_name = capture_name

        self.capture_dir = Path(path.parent.absolute())
        self.parent_dir = Path(path.parent.absolute())


        if self.label is not None:
            label = "-" + str(self.label)
        else:
            label = "" 


        self.l1a_name = capture_name + label + "-l1a"
        self.l1b_name = capture_name + label + "-l1b"
        self.l1c_name = capture_name + label + "-l1c"
        self.l1d_name = capture_name + label + "-l1d"
        #self.l2a_name = capture_name + label + "-l2a"

        self.l1a_nc_file = Path(path.parent, self.l1a_name + ".nc")
        self.l1b_nc_file = Path(path.parent, self.l1b_name + ".nc")
        self.l1c_nc_file = Path(path.parent, self.l1c_name + ".nc")
        self.l1d_nc_file = Path(path.parent, self.l1d_name + ".nc")

        product_level = fields['product_level']

        match product_level:
            case "l1a":
                if self.VERBOSE: print('[INFO] Loading L1a capture ' + self.capture_name)

                load_func = load_l1a_nc
                cube_name = "l1a_cube"
                setattr(self, "cube_name", cube_name)
                setattr(self, "product_level", "l1a")
                setattr(self, "product_symbol", "DN")
                
            case "l1b":
                if self.VERBOSE: print('[INFO] Loading L1b capture ' + self.capture_name)

                load_func = load_l1b_nc
                cube_name = "l1b_cube"
                setattr(self, "cube_name", cube_name)
                setattr(self, "product_level", "l1b")
                setattr(self, "product_symbol", "Lt")

            case "l1c":
                if self.VERBOSE: print('[INFO] Loading L1c capture ' + self.capture_name)

                load_func = load_l1c_nc
                cube_name = "l1b_cube" # L1c cube is the same as the L1b cube
                setattr(self, "cube_name", cube_name)
                setattr(self, "product_level", "l1c")
                setattr(self, "product_symbol", "lt")

            case "l1d":
                if self.VERBOSE: print('[INFO] Loading L1d capture ' + self.capture_name)

                load_func = load_l1d_nc
                cube_name = "l1d_cube"
                setattr(self, "cube_name", cube_name)
                setattr(self, "product_level", "l1d")
                setattr(self, "product_symbol", "rhot")

            case "l2a":
                if self.VERBOSE: print('[INFO] Loading L2a capture ' + self.capture_name)

                ac = getattr(self, 'atmospheric_correction', None)

                if ac is not None:
                    print("[INFO] L2a Detected atmospheric correction: " + str(ac))
                else:
                    print("[WARNING] No L2a atmospheric correction detected.")
                    setattr(self, "atmospheric_correction", "default")

                load_func = load_l2a_nc
                cube_name = "l2a_cube"
                setattr(self, "cube_name", cube_name)
                setattr(self, "product_level", "l2a")
                setattr(self, "product_symbol", "Rrs") # TODO: polymer and dps is rho_w
                
            case _:
                print("[ERROR] Unsupported product level:")
                print(product_level)
                return None

        # TODO: find a better method to pass all of this information
        nc_metadata_vars, \
        nc_metadata_attrs, \
        nc_geometry_vars, \
        nc_geometry_attrs, \
        nc_gcp_vars, \
        nc_gcp_attrs, \
        nc_global_metadata, \
        nc_cube_attrs, \
        nc_cube = load_func(nc_file_path=path, load_cube=load_cube)

        setattr(self, "nc_adcs_vars", nc_metadata_vars["adcs"])
        setattr(self, "nc_capture_config_vars", nc_metadata_vars["capture_config"])
        setattr(self, "nc_corrections_vars", nc_metadata_vars["corrections"])
        setattr(self, "nc_database_vars", nc_metadata_vars["database"])
        setattr(self, "nc_logfiles_vars", nc_metadata_vars["logfiles"])
        setattr(self, "nc_temperature_vars", nc_metadata_vars["temperature"])
        setattr(self, "nc_timing_vars", nc_metadata_vars["timing"])
        setattr(self, "nc_srf_vars", nc_metadata_vars["srf"])

        setattr(self, "nc_adcs_attrs", nc_metadata_attrs["adcs"])
        setattr(self, "nc_capture_config_attrs", nc_metadata_attrs["capture_config"])
        setattr(self, "nc_corrections_attrs", nc_metadata_attrs["corrections"])
        setattr(self, "nc_database_attrs", nc_metadata_attrs["database"])
        setattr(self, "nc_logfiles_attrs", nc_metadata_attrs["logfiles"])
        setattr(self, "nc_temperature_attrs", nc_metadata_attrs["temperature"])
        setattr(self, "nc_timing_attrs", nc_metadata_attrs["timing"])
        setattr(self, "nc_srf_attrs", nc_metadata_attrs["srf"])
 
        setattr(self, "nc_geometry_vars", nc_geometry_vars)
        setattr(self, "nc_geometry_attrs", nc_geometry_attrs)

        setattr(self, "nc_gcp_vars", nc_gcp_vars)
        setattr(self, "nc_gcp_attrs", nc_gcp_attrs)

        setattr(self, "nc_dimensions", nc_global_metadata["dimensions"])
        setattr(self, "nc_attrs", nc_global_metadata["ncattrs"])

        setattr(self, "nc_cube_attrs", nc_cube_attrs)

        # TODO: pass the dicts returned by load_func to _set_hypso_attributes()
        # Note: this MUST be run before writing datacubes in order to pass correct dimensions to DataArrayValidator
        self._set_hypso_attributes()
        self._check_capture_type()

        if load_cube:
            if self.product_level.lower() == "l2a":
                self.l2a_cubes[self.atmospheric_correction] = nc_cube
            else:
                setattr(self, cube_name, nc_cube)
        
        else:
            print("[WARNING] Datacube is not loaded!")


        self.ocsmart_l1d_input_nc_file = Path(path.parent, str(self.sensor).upper() + "_" + str(capture_name) + "-l1d.nc")
        self.ocsmart_l2a_output_h5_file = Path(path.parent, str(self.sensor).upper() + "_" + str(capture_name) + "-l1d_L2_OCSMART.h5") #HYPSO2_HSI_aeronetvenice_2025-06-22T10-46-15Z-l1d_L2_OCSMART


        dt = datetime.fromtimestamp(self.unixtime, tz=timezone.utc)
        # Moved into a capture_dir/acolite/ subfolder (2026-08-05, was
        # capture_dir directly) - matches ac_acolite_run_correction's own
        # settings['output'] below, and the PACE-side ACOLITE connector's
        # existing convention (ac_runners_pace.py). ACOLITE writes several
        # per-run log/settings .txt files alongside its L2R/L2W output
        # (delete_acolite_run_text_files defaults False), which had been
        # accumulating directly in the capture directory root with no
        # cleanup - one set per run, indefinitely.
        self.acolite_l2r_output_nc_file =  Path(self.capture_dir, "acolite", f"{self.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2R.nc")
        self.acolite_l2w_output_nc_file =  Path(self.capture_dir, "acolite", f"{self.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2W.nc")

        return None


    # TODO: Clean up this function. Use setattr, hasattr, getattr for setting class variables?
    def _set_hypso_attributes(self) -> None:

        # Capture config related attributes
        for attr in self.nc_capture_config_attrs.keys():
            setattr(self, attr, self.nc_capture_config_attrs[attr])
        # FPS has been renamed to framerate. Need to support both since old .nc files may still use FPS
        try:
            self.nc_capture_config_attrs['fps'] = self.nc_capture_config_attrs['framerate']
        except:
            self.nc_capture_config_attrs['framerate'] = self.nc_capture_config_attrs['fps']
            
        self.background_value = 8 * self.nc_capture_config_attrs["bin_factor"]
        self.exposure = self.nc_capture_config_attrs["exposure"] / 1000  # in seconds


        # Capture dimensions attributes
        self.x_start = self.nc_capture_config_attrs["aoi_x"]
        self.x_stop = self.nc_capture_config_attrs["aoi_x"] + self.nc_capture_config_attrs["column_count"]
        self.y_start = self.nc_capture_config_attrs["aoi_y"]
        self.y_stop = self.nc_capture_config_attrs["aoi_y"] + self.nc_capture_config_attrs["row_count"]
        self.bin_factor = self.nc_capture_config_attrs["bin_factor"]
        # Try/except here since not all captures have sample_div
        try:
            self.sample_div = self.nc_capture_config_attrs['sample_div']
        except:
            self.sample_div = 1
        self.row_count = self.nc_capture_config_attrs["row_count"]
        self.frame_count = self.nc_capture_config_attrs["frame_count"]
        self.column_count = self.nc_capture_config_attrs["column_count"]
        self.image_height = int(self.nc_capture_config_attrs["row_count"] / self.sample_div)
        self.image_width = int(self.nc_capture_config_attrs["column_count"] / self.nc_capture_config_attrs["bin_factor"])
        self.im_size = self.image_height * self.image_width
        self.bands = self.image_width
        self.lines = self.nc_capture_config_attrs["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols
        self.spatial_dimensions = (self.nc_capture_config_attrs["frame_count"], self.image_height)
        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")


        # Calibration related atrributes
        self.rad_coeffs = self.nc_corrections_vars['rad_matrix']

        try:
            self.spectral_coeffs = self.nc_corrections_vars['spec_coeffs']
        except KeyError:
            self.spectral_coeffs = self.nc_corrections_vars['wavelengths']

        if not hasattr(self, 'wavelengths'):
            if ('wavelengths' in self.nc_cube_attrs.keys()):
                self.wavelengths = self.nc_cube_attrs['wavelengths']
            else:
                self.wavelengths = np.array(range(0, self.image_width))

        if not hasattr(self, 'wavelengths_unbinned'):
            if ('wavelengths_unbinned' in self.nc_corrections_vars.keys()):
                self.wavelengths_unbinned = self.nc_corrections_vars['wavelengths_unbinned']
            else:
                self.wavelengths_unbinned = np.array(range(0, self.image_width))

        if not hasattr(self, 'fwhm'):
            if 'fwhm' in self.nc_cube_attrs.keys():
                self.fwhm = self.nc_cube_attrs['fwhm']
            else:
                #self.fwhm = [self.AVERAGE_FWHM] * self.bands
                self.fwhm = [self.AVERAGE_FWHM] * self.UNBINNED_BAND_COUNT


        if not hasattr(self, 'effective_fwhm'):
            if 'effective_fwhm' in self.nc_srf_vars.keys():
                self.effective_fwhm = self.nc_srf_vars['effective_fwhm']

        if not hasattr(self, 'esun'):
            if 'esun' in self.nc_srf_vars.keys():
                self.esun = self.nc_srf_vars['esun']

        if not hasattr(self, 'esun_wl'):
            if 'esun_wavelengths' in self.nc_srf_vars.keys():
                self.esun_wl = self.nc_srf_vars['esun_wavelengths']


        csiro_list =  ["csiro_ssi", "csiro_solar_wavelengths", "csiro_binned_srfs"
                       "csiro_effective_fwhm", "csiro_esun"]

        for csiro_key in csiro_list:
            if not hasattr(self, csiro_key):
                if csiro_key in self.nc_srf_vars.keys():
                    setattr(self, csiro_key, self.nc_srf_vars[csiro_key])


        # Geometry atrributes
        for key, value in self.nc_geometry_vars.items():
            if key == 'unixtime':
                continue
            elif key == 'latitude':
                setattr(self, 'latitudes', value)
            elif key == 'longitude':
                setattr(self, 'longitudes', value)

            elif key == 'latitude_direct':
                setattr(self, 'latitudes_direct', value)
            elif key == 'longitude_direct':
                setattr(self, 'longitudes_direct', value)


            elif key == 'sensor_zenith':
                setattr(self, 'sat_zenith_angles', value)
            elif key == 'sensor_azimuth':
                setattr(self, 'sat_azimuth_angles', value)

            elif key == 'sensor_zenith_direct':
                setattr(self, 'sat_zenith_angles_direct', value)
                if getattr(self, 'sat_zenith_angles', None) is None:
                    setattr(self, 'sat_zenith_angles', value)

            elif key == 'sensor_azimuth_direct':
                setattr(self, 'sat_azimuth_angles_direct', value)
                if getattr(self, 'sat_azimuth_angles', None) is None:
                    setattr(self, 'sat_azimuth_angles', value)

            elif key == 'solar_zenith':
                setattr(self, 'solar_zenith_angles', value)
            elif key == 'solar_azimuth':
                setattr(self, 'solar_azimuth_angles', value)

            elif key == 'solar_zenith_direct':
                setattr(self, 'solar_zenith_angles_direct', value)
                if getattr(self, 'solar_zenith_angles', None) is None:
                    setattr(self, 'solar_zenith_angles', value)

            elif key == 'solar_azimuth_direct':
                setattr(self, 'solar_azimuth_angles_direct', value)
                if getattr(self, 'solar_azimuth_angles', None) is None:
                    setattr(self, 'solar_azimuth_angles', value)

            elif key == 'relative_azimuth':
                setattr(self, 'relative_azimuth_angles', value)

            elif key == 'relative_azimuth_direct':
                setattr(self, 'relative_azimuth_angles_direct', value)
                if getattr(self, 'relative_azimuth_angles', None) is None:
                    setattr(self, 'relative_azimuth_angles', value)
                
            else:
                setattr(self, key, value)







        # Capture timing attributes
        try:
            self.start_timestamp_capture = int(self.timing['capture_start_unix']) + self.UNIX_TIME_OFFSET
        except:
            try:
                datestring = self.nc_attrs['date_aquired']
            except:
                datestring = self.nc_attrs['timestamp_acquired']


            try:
                dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            except ValueError:
                dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%f%zZ').replace(tzinfo=timezone.utc)

            self.start_timestamp_capture = dt.timestamp()

        #self.start_timestamp_capture = int(self.nc_timing_attrs['capture_start_unix']) + self.UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if framerate and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.nc_capture_config_attrs["frame_count"] / self.nc_capture_config_attrs["framerate"] + self.nc_capture_config_attrs["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] Framerate or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.nc_capture_config_attrs["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0
        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end
        self.unixtime = self.start_timestamp_capture

        #self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()
        self.iso_time = datetime.fromtimestamp(self.unixtime, tz=timezone.utc).isoformat()

        return None


    def _check_capture_type(self):

        #self.spatial_dimensions = (956, 684)  # 1092 x variable
        #self.standard_dimensions = {
        #    "nominal": 956,  # Along frame_count
        #    "wide": 1092  # Along image_height (row_count)
        #}

        if self.nc_capture_config_attrs["frame_count"] == 956:
        #if self.nc_capture_config_attrs["frame_count"] == self.standard_dimensions["nominal"]:
            self.capture_type = "nominal"
        elif self.nc_capture_config_attrs["frame_count"] == 106:
                    self.capture_type = "moon"
        elif self.image_height == 1092:
        #elif self.image_height == self.standard_dimensions["wide"]:
            self.capture_type = "wide"
        else:
            # EXPERIMENTAL_FEATURES
            if self.VERBOSE:
                print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
            self.capture_type = "custom"

        if self.VERBOSE:
            print(f"[INFO] Capture capture type: {self.capture_type}")


    def _set_calibration_coeff_files(self, coeff_type: str = 'moved', coeff_files: dict = None, **kwargs) -> None:
        """
        Set the absolute path for the calibration coefficients (radiometric, smile,
        destriping, spectral) for this capture's sensor. Three ways to supply them,
        checked in order:

        1. `coeff_files` - an explicit dict (radiometric/smile/destriping/spectral/
           spectral_full_frame -> path) for a true one-off set, no registration needed.
        2. `coeff_type` matching a name previously registered via
           hypso.calibration.register_calibration_coeffs(sat_id, name, files) - a
           custom, reusable, named set, plugged in without touching the bundled
           hypsoN_calibration packages.
        3. `coeff_type` falling through to self.sensor_profile's calibration_files
           resolver (the sensor's built-in presets, e.g. "moved"/"adjusted"/
           "original" for HYPSO-1/-2). Previously implemented separately per
           Hypso1/Hypso2 subclass, each hardcoding a call to
           get_hypsoN_calibration_files - now generic: the sensor-specific lookup
           lives in the SensorProfile instead, so this method works for any sensor.

        :return: None.
        """
        if self.sensor_profile is None:
            raise AttributeError(
                "_set_calibration_coeff_files requires self.sensor_profile to be set - "
                "either construct this capture with a SensorProfile (see hypso.sensors), "
                "or override this method in a subclass."
            )

        capture_type = self.capture_type

        if coeff_files is not None:
            logger.debug("Using explicitly-supplied calibration coefficient files (coeff_files=...)")
            calibration_files = {key: coeff_files.get(key) for key in
                                  ("radiometric", "smile", "destriping", "spectral", "spectral_full_frame")}
        else:
            custom = get_custom_calibration_coeffs(self.sat_id, coeff_type)
            if custom is not None:
                logger.debug("Using registered custom calibration coefficient set %r", coeff_type)
                calibration_files = custom
            else:
                logger.debug("Setting calibration coefficient files with coeff_type: %s", coeff_type)
                calibration_files = self.sensor_profile.calibration_files(capture_type, coeff_type=coeff_type)

        self.coeff_type = coeff_type if coeff_type is not None else "custom"
        self.rad_coeff_file = calibration_files['radiometric']
        self.smile_coeff_file = calibration_files['smile']
        self.destriping_coeff_file = calibration_files['destriping']
        self.spectral_coeff_file = calibration_files['spectral']

        return None

    def _run_calibration(self,
                         radiometric: bool = True,
                         smile: bool = True,
                         destripe: bool = True,
                         spectral: bool = True,
                         set_coeffs: bool = True,
                         coeff_type: str = None,
                         **kwargs) -> np.ndarray:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')


        if coeff_type is None:
            try:
                coeff_type = self.nc_corrections_attrs['radiometric_coefficients_version']
            except:
                pass
        else:
            self.nc_corrections_attrs['radiometric_coefficients_version'] = str(coeff_type).lower()
            

        # TODO: move this function call
        if set_coeffs:
            self._set_calibration_coeff_files(coeff_type=coeff_type, **kwargs)

        self._load_calibration_coeff_files()

        calibrated_cube = self.l1a_cube.to_numpy()

        if self.rad_coeffs is not None:
            if radiometric:

                if self.VERBOSE:
                    print("[INFO] Running radiometric calibration...")

                calibrated_cube = run_radiometric_calibration(cube=calibrated_cube, 
                                                background_value=self.background_value,
                                                exp=self.exposure,
                                                image_height=self.image_height,
                                                image_width=self.image_width,
                                                frame_count=self.frame_count,
                                                bin_factor=self.bin_factor,
                                                rad_coeffs=self.rad_coeffs
                                                )

        if self.smile_coeffs is not None:
            if smile:

                if self.VERBOSE:
                    print("[INFO] Running smile correction...")

                calibrated_cube = run_smile_correction(cube=calibrated_cube, 
                                                smile_coeffs=self.smile_coeffs)

        if self.destriping_coeffs is not None:
            if destripe:

                if self.VERBOSE:
                    print("[INFO] Running destriping correction...")

                calibrated_cube = run_destriping_correction(cube=calibrated_cube, 
                                                    destriping_coeffs=self.destriping_coeffs)

        if self.spectral_coeffs is not None:
            if spectral:
                if self.VERBOSE:
                    print("[INFO] Running spectral correction (binned)...")

                self.wavelengths = self.spectral_coeffs

        if self.spectral_coeffs_unbinned is not None:
            if spectral:
                if self.VERBOSE:
                    print("[INFO] Running spectral correction (unbinned)...")

                self.wavelengths_unbinned = self.spectral_coeffs_unbinned

        return calibrated_cube


    def _load_calibration_coeff_files(self) -> None:
        """
        Load the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        try:
            self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file, 'radiometric', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.rad_coeffs = None

        try:
            self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file, 'smile', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.smile_coeffs = None

        try:
            self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file, 'destriping', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.destriping_coeffs = None

        try:
            self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, self.bin_factor)
        except:
            self.spectral_coeffs = None

        try:
            self.spectral_coeffs_unbinned = read_coeffs_from_file(self.spectral_coeff_file, 'spectral', self.x_start, self.x_stop, self.y_start, self.y_stop, 1)
        except:
            self.spectral_coeffs_unbinned = None

        return None
    



    # Georeferencing orchestration (run_direct_georeferencing/run_georeferencing and
    # their private _run_* helpers) was extracted verbatim into hypso.geo - part of
    # the HypsoBase breakup called for in the approved refactor plan (self.geo
    # composition). These stay as thin delegating wrappers, not moved themselves,
    # because run_direct_georeferencing() is called externally
    # (hypso/ac/loading_acolite_output.py) and run_georeferencing() by
    # hypso-processing-pipeline - both names/signatures must keep working unchanged.
    # The private _run_frame_interpolation/_run_track_geometry/_run_angles_geometry
    # helpers had no external callers, so they moved to hypso.geo outright with no
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

        self.l1b_cube = self._run_calibration(coeff_type=coeff_type, **kwargs)

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

        if use_direct_georef and hasattr(self, 'solar_zenith_angles_direct'):

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


        toa_reflectance, \
        effective_fwhm, \
        srf, \
        srf_ssi, \
        srf_ssi_wl, \
        esun, \
        esun_wl = compute_toa_reflectance(sensor_wavelengths=sensor_wavelengths,
                                            sensor_fwhm=sensor_fwhm,
                                            bin_factor = sensor_bin_factor,
                                            toa_radiance=toa_radiance,
                                            iso_time=self.iso_time,
                                            solar_zenith_angles=solar_zenith_angles,
                                            use_thuillier = use_thuillier,
                                            generate_figures=generate_figures
                                            )

        self.l1d_cube = toa_reflectance
        
        self.srf = srf
        self.srf_ssi = srf_ssi
        self.srf_ssi_wl = srf_ssi_wl
        self.esun = esun
        self.esun_wl = esun_wl
        self.effective_fwhm = effective_fwhm

        return None


    def _spawn_next_level(self) -> "HypsoBase":
        """Shallow-copy this object into a new instance for the to_l1b/to_l1c/
        to_l1d family (see those methods) - self is left untouched, including
        its own cubes. A shallow copy.copy() is enough for the big-array
        attributes (cubes, latitudes/longitudes, calibration coefficient
        matrices): those are only ever read, never mutated in place, anywhere
        in this class, so aliasing the same array object between self and the
        new instance is safe and avoids duplicating potentially large data.

        Mutable *container* attributes are different - copy.copy() would alias
        the same dict/DataArrayDict object between self and the new instance,
        so a later mutation on one (e.g. new_obj.set_custom_mask(...)) would
        silently also change the other. Those are re-copied one level deep
        (the dict itself, not what's inside it) so self and the new object
        can diverge independently after this point.
        """
        new_obj = copy.copy(self)
        new_obj._custom_masks = dict(self._custom_masks)
        new_obj._l2a_cubes = copy.copy(self._l2a_cubes)
        return new_obj


    def to_l1b(self, coeff_type: str = None, **kwargs) -> "HypsoBase":
        """Like generate_l1b_cube(), but returns a NEW object instead of
        mutating self - self (including its own l1a_cube) is left completely
        untouched. The new object's l1a_cube is cleared once l1b_cube is
        generated, so it holds only the one cube its name promises - see
        docs/architecture.rst's "Producing a new object per level" section for
        why this exists alongside generate_l1b_cube() (kept unchanged - and
        this method does not replace it - for hypso-processing-pipeline's
        existing in-place, mutating usage).
        """
        new_obj = self._spawn_next_level()
        new_obj._generate_l1b_cube_impl(coeff_type=coeff_type, **kwargs)
        new_obj._l1a_cube = None
        return new_obj


    def to_l1c(self, coeff_type: str = None, **kwargs) -> "HypsoBase":
        """Like generate_l1c_cube(), but returns a NEW object instead of
        mutating self. See to_l1b()'s docstring for the general pattern. l1c
        has no independent cube storage (see l1c_cube's property getter) so
        there is nothing extra to clear beyond what generate_l1c_cube() itself
        populates.
        """
        new_obj = self._spawn_next_level()
        new_obj._generate_l1c_cube_impl(coeff_type=coeff_type, **kwargs)
        new_obj._l1a_cube = None
        return new_obj


    def to_l1d(self, use_direct_georef=False, use_thuillier=False, use_unbinned=True,
               generate_figures=False) -> "HypsoBase":
        """Like generate_l1d_cube(), but returns a NEW object instead of
        mutating self. See to_l1b()'s docstring for the general pattern. The
        new object's l1a_cube/l1b_cube are cleared once l1d_cube is generated,
        so it holds only l1d_cube.
        """
        new_obj = self._spawn_next_level()
        new_obj._generate_l1d_cube_impl(use_direct_georef=use_direct_georef, use_thuillier=use_thuillier,
                                        use_unbinned=use_unbinned, generate_figures=generate_figures)
        new_obj._l1a_cube = None
        new_obj._l1b_cube = None
        return new_obj




    def ac_ocsmart_stage_input(self):

        """
        Stages OC-SMART input file to the L1B directory located in the OC-SMART installation directory. The L1d file is copied and renamed to the L1B directory.

        :return: None
        """


        if self.ocsmart_dir is not None:
            try:
                
                dst_dir = Path(self.ocsmart_dir, "L1B/")
                dst_dir.mkdir(parents=True, exist_ok=True)

                src_file = self.l1d_nc_file
                dst_file = Path(dst_dir, self.ocsmart_l1d_input_nc_file.name)

                self.ocsmart_l1d_input_nc_file = dst_file

                import shutil
                shutil.copy2(src_file, dst_file)

                print("[INFO] Successfully staged OC-SMART input file to " + str(dst_file))

            except Exception as ex:
                print("[ERROR] Unable to stage OC-SMART input. An error occured.")
                print(ex)

        else:
            print("[ERROR] OC-SMART directory is not configured. The 'ocsmart_dir' attribute is empty.")

        return None


    def ac_ocsmart_run_correction(self):
        """
        Execute 'OCSMART.py' as a subprocess.

        :return: None
        """

        print("[INFO] Running OC-SMART atmospheric correction as a subprocess.")

        import subprocess
        ocsmart_run_script = Path(self.ocsmart_dir, "OCSMART.py")
        subprocess.run(["python3", ocsmart_run_script], cwd=self.ocsmart_dir, check=True)

        print("[INFO] Removing staged OC-SMART input file " + str(self.ocsmart_l1d_input_nc_file))
        self.ocsmart_l1d_input_nc_file.unlink(missing_ok=True)

        print("[INFO] OC-SMART atmospheric correction complete.")

        return None



    def ac_ocsmart_open_output(self, h5_file_path: Path = None):
        """
        Open and read OC-SMART atmospheric correction HDF5 output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

        :param h5_file_path: Path to the OC-SMART HDF5 file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the HDF5 and stored as xarray DataArrays.
        """


        if h5_file_path is not None:
            h5_file_path = Path(h5_file_path).absolute()
        else:
            ocsmart_output_dir = Path(self.ocsmart_dir, "L2/")
            h5_file_path = Path(ocsmart_output_dir, self.ocsmart_l2a_output_h5_file.name)


        if h5_file_path.is_file():
            print("[INFO] Opening OC-SMART output file " + str(h5_file_path))
            datasets = load_ocsmart_h5(h5_file_path = h5_file_path)

        else:
            print("[ERROR] OC-SMART output file " + str(h5_file_path) + " does not exist.")
            return None

        try:
            key = "Rrs"
            inferred_wavelengths = datasets[key].band.to_numpy()

            # Map inferred OC-SMART wavelengths to HYPSO wavelengths
            wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

            '''
            l2a_cube_wavelengths = inferred_wavelengths

            A = np.array(l2a_cube_wavelengths, dtype=float)
            B = np.array(self.wavelengths, dtype=float)

            index_map = {}
            indices_unique = []

            for a in A:
                ix = np.argmin(np.abs(B - a))
                if ix not in index_map: # ensure uniqueness
                    index_map[ix] = a
                    indices_unique.append(ix)
                else:
                    print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

            ocsmart_dataset_indices = np.array(indices_unique, dtype=int)

            wl_band_map = ocsmart_dataset_indices
            '''
            
            # Create empty cube with standard HYPSO cube dims
            shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
            cube = np.full(shape=shape, fill_value=np.nan)
            cube[:,:,wl_band_map] = datasets[key]

            self.l2a_cube["ocsmart"] = cube
            self.l2a_cube["ocsmart"].attrs['l2_variable_name'] = key

        except Exception as ex:
            print("[ERROR] Unable to load OC-SMART L2 Rrs dataset.")

        return datasets


    


    def ac_acolite_run_correction(self, settings_file: Path = None, 
                                  input_product_level: str = 'l1c',
                                  EARTHDATA_u: str = None,
                                  EARTHDATA_p: str = None
                                  ):
        

        

        acolite_path = Path(self.acolite_dir).absolute()
        
        print("[INFO] Running ACOLITE atmospheric correction installed in " + str(acolite_path))

        sys.path.append(str(acolite_path))
        #print(sys.path)

        import acolite as ac
        from acolite.acolite.settings import load
        from acolite.acolite import acolite_run
        
        # optional file with processing settings
        # if set to None defaults will be used

        # import settings
        #settings = ac.acolite.settings.load(settings_file)
        # load() only applies a sensor's own config/defaults/<name>.txt (e.g.
        # HYPSO2.txt's dsf_wave_range=450,750) when explicitly given that
        # name - it does not auto-detect sensor from the input file. Passing
        # None (as this always did before) silently fell back to ACOLITE's
        # fully generic defaults.txt (dsf_wave_range=400,2500) instead,
        # mirroring the fix ac_runners.py's PACE runner already has via its
        # explicit load("PACE_OCI") call.
        settings = load(settings_file if settings_file is not None else self.platform.upper())

        if EARTHDATA_u is not None and EARTHDATA_p is not None:
            settings['EARTHDATA_u'] = EARTHDATA_u
            settings['EARTHDATA_p'] = EARTHDATA_p
            settings['ancillary_data'] = True

        # set settings provided above

        if input_product_level.upper() == 'L1D':
            print("[INFO] Using L1d NetCDF as ACOLITE input.")
            settings['inputfile'] = str(self.l1d_nc_file) # L1d reflectance
        else:
            print("[INFO] Using L1c NetCDF as ACOLITE input.")
            settings['inputfile'] = str(self.l1c_nc_file) # default L1c (radiance)


        # capture_dir/acolite/, not capture_dir directly (2026-08-05) - see
        # the matching comment on self.acolite_l2r_output_nc_file above for
        # why (keeps ACOLITE's own per-run log/settings .txt files out of
        # the capture directory root).
        acolite_output_dir = Path(self.capture_dir, "acolite")
        acolite_output_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Writing ACOLITE output to " + str(acolite_output_dir))
        settings['output'] = str(acolite_output_dir)

        settings['polygon'] = None
        settings['rgb_rhot'] = True
        settings['rgb_rhos'] = True
        settings['map_l2w'] = False #produces blank .pngs
        settings['l2w_mask'] = False
        settings['l2w_mask_threshold'] = 0.2

        settings['l2w_parameters'] = ['Rrs_*', \
                                    'spm_nechad2010', \
                                    'spm_nechad2016', \
                                    'chl_re_mishra',\
                                    'chl_oc2', \
                                    'chl_oc3', \
                                    'chl_re_moses3b', \
                                    'chl_re_moses3b740', \
                                    'fai', \
                                    'fai_rhot', \
                                    'fait', \
                                    'ndci']


        processed = acolite_run(settings=settings)

        #acolite_l2_file = processed[0]['l2r'][0]

        print("[INFO] ACOLITE atmospheric correction complete.")

        return None
    



    def ac_acolite_open_output(self, acolite_l2r_output_nc_file: Path = None, acolite_l2w_output_nc_file: Path = None):
        
        """
        Open and read ACOLITE atmospheric correction L2R and L2W NetCDF output files. The remote sensing reflectance (Rrs) dataset is written to the satobj's 'l2a_cube' dictionary.

        :param h5_file_path: Path to the ACOLITE NetCDF file (optional)

        :return: "datasets" Dictionary containing 2D and 3D datasets read from the NetCDFs and stored as xarray DataArrays.
        """


        if acolite_l2r_output_nc_file is not None:
            acolite_l2r_output_nc_file = Path(acolite_l2r_output_nc_file).absolute()
        else:
            acolite_l2r_output_nc_file = Path(self.acolite_l2r_output_nc_file).absolute()

        if acolite_l2w_output_nc_file is not None:
            acolite_l2w_output_nc_file = Path(acolite_l2w_output_nc_file).absolute()
        else:
            acolite_l2w_output_nc_file = Path(self.acolite_l2w_output_nc_file).absolute()




        if acolite_l2r_output_nc_file.is_file():
            print("[INFO] Opening ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file))
            l2r_datasets = load_acolite_l2r_nc(acolite_l2r_output_nc_file)

            try:
                key = "rhos"
                inferred_wavelengths = l2r_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2r_datasets[key]

                self.l2a_cube["acolite_l2r"] = cube
                self.l2a_cube["acolite_l2r"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2R dataset.")
                l2r_datasets = None

        else:
            print("[ERROR] ACOLITE L2R NetCDF output file " + str(acolite_l2r_output_nc_file) + " does not exist.")
            l2r_datasets = None


        if acolite_l2w_output_nc_file.is_file():
            print("[INFO] Opening ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file))
            l2w_datasets = load_acolite_l2w_nc(acolite_l2w_output_nc_file)

            try:
                key = "Rrs"
                inferred_wavelengths = l2w_datasets[key].band.to_numpy()

                # Map inferred ACOLITE wavelengths to HYPSO wavelengths
                wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                # Create empty cube with standard HYPSO cube dims
                shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                cube = np.full(shape=shape, fill_value=np.nan)
                cube[:,:,wl_band_map] = l2w_datasets[key]

                self.l2a_cube["acolite_l2w"] = cube
                self.l2a_cube["acolite_l2w"].attrs['l2_variable_name'] = key

            except Exception as ex:
                print("[ERROR] Unable to load ACOLITE L2W dataset.")
                l2w_datasets = None

        else:
            print("[ERROR] ACOLITE L2W NetCDF output file " + str(acolite_l2w_output_nc_file) + " does not exist.")
            l2w_datasets = None


        return l2r_datasets, l2w_datasets


    def ac_polymer_get_id_sensor(self):

        sensor_version = "_" + str(self.coeff_type)

        # combine sensor name ("HYPSO-1" or "HYPSO-2") with coefficients version
        # Polymer expects format like "HYPSO-2_moved"
        id_sensor = str(self.sat_id) + sensor_version 

        return id_sensor


    def ac_polymer_get_srf_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        srf_nc_file = id_sensor + "_srf.nc"
        srf_nc_path = Path(self.parent_dir, srf_nc_file )

        self.srf_nc_file = srf_nc_file
        self.srf_nc_path = srf_nc_path

        return srf_nc_file, srf_nc_path
    

    def ac_polymer_get_ssi_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        ssi_nc_file = id_sensor + "_ssi.nc"
        ssi_nc_path = Path(self.parent_dir, ssi_nc_file )

        self.ssi_nc_file = ssi_nc_file
        self.ssi_nc_path = ssi_nc_path

        return ssi_nc_file, ssi_nc_path
    

    def ac_polymer_get_esun_nc_path(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        esun_nc_file = id_sensor + "_esun.nc"
        esun_nc_path = Path(self.parent_dir, esun_nc_file )

        self.ssi_nc_file = esun_nc_file
        self.ssi_nc_path = esun_nc_path

        return esun_nc_file, esun_nc_path










    def ac_polymer_generate_srf_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, srf_nc_path = self.ac_polymer_get_srf_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'Spectral response functions for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        for idx, wl in enumerate(self.wavelengths):
            
            # Construct band ID            
            bid = "Band_" + str(idx)

            # Read ith SRF and convert from CSR sparse array
            srf = self.srf[idx,:].toarray().flatten()
            srf_wavelengths = self.srf_ssi_wl

            # Find where SRF is non-zero
            nonzero_mask = srf > 0
            
            # Extract non-zero portion of SRF and SRF wavelength array
            if np.any(nonzero_mask):
                srf_nonzero = srf[nonzero_mask]
                srf_wavelengths_nonzero = srf_wavelengths[nonzero_mask]
            else:
                srf_nonzero = srf
                srf_wavelengths_nonzero = srf_wavelengths

            # Add band entry to dataset
            ds[bid] = xr.DataArray(
                srf_nonzero,
                coords={f"wav_{bid}": srf_wavelengths_nonzero},
                attrs={
                    "band_info": bid,
                    "band_wavelength": wl,
                    "index": idx,
                    "effective_fwhm": self.effective_fwhm[idx],
                    "center_fwhm": self.fwhm[idx]
                },
            )
            ds[f"wav_{bid}"].attrs["units"] = "nm"
            
        # Sort dataarrays within dataset based on index
        ds = ds[sorted(ds, key=lambda x: ds[x].attrs['index'])]



        ds.to_netcdf(srf_nc_path)

        return srf_nc_path        






    def ac_polymer_generate_ssi_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, ssi_nc_path = self.ac_polymer_get_ssi_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'TSIS-1 solar spectral irradiance for {id_sensor} (0.005 nm spectral resolution)'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        ds["ssi"] = xr.DataArray(
            self.srf_ssi,
            coords={f"wav_ssi": self.srf_ssi_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_ssi"].attrs["units"] = "nm"


        ds.to_netcdf(ssi_nc_path)

        return ssi_nc_path        





    def ac_polymer_generate_esun_nc(self):

        id_sensor = self.ac_polymer_get_id_sensor()

        _, esun_nc_path = self.ac_polymer_get_esun_nc_path()


        ds = xr.Dataset()
        ds.attrs["desc"] = f'ESUN for {id_sensor}'
        ds.attrs["sensor"] = id_sensor
        ds.attrs["platform"] = self.platform

        #ds.attrs["ssi"] = self.srf_ssi
        #ds.attrs["ssi_wavelengths"] = self.srf_ssi_wl
        #ds.attrs["ssi_units"] = "mW m-2 nm-1"

        #ds.attrs["esun"] = self.esun
        #ds.attrs["esun_wavlengths"] = self.esun_wl
        #ds.attrs["esun_units"] = "mW m-2 nm-1"


        ds["esun"] = xr.DataArray(
            self.esun,
            coords={f"wav_esun": self.esun_wl},
            attrs={
                "units": "mW m-2 nm-1",
            },
        )
        ds[f"wav_esun"].attrs["units"] = "nm"


        ds.to_netcdf(esun_nc_path)

        return esun_nc_path    








    def ac_polymer_run_correction(self,
                                  polymer_base_path: str,
                                  polymer_path: str = None,
                                  eoread_path: str = None,
                                  eotools_path: str = None,
                                  core_path: str = None,
                                  input_product_level: str = "l1c",
                                  #coeff_type: str = None,
                                  optional_output_datasets: list = ["SPM"],
                                  if_exists: str = "overwrite",
                                  polymer_version: str = "v1"):
        """
        polymer_version: which Polymer build polymer_path (etc.) point at -
            mirrors ac_polymer_open_output's version parameter.
            - "v1": Polymer_HYPSO_SRF_Oct_2025 - run_polymer's output
              selection is driven by output_datasets, and it writes a
              linear-scale "chla"/"fb" directly.
            - "v2": the newer stock Polymer build - run_polymer no longer
              has an output_datasets parameter (silently ignored if passed -
              it lands in **kwargs and is never used for selection), so
              output selection is driven by outputs_names instead; the
              solver only exposes log-scale "logchl"/"logfb", not "chla"/"fb".
        """

        #polymer_path = Path(self.polymer_dir).absolute()

        if polymer_path is not None:
            polymer_path = str(Path(polymer_path).absolute())
            sys.path.insert(0, polymer_path)

        if eotools_path is not None:
            eotools_path = str(Path(eotools_path).absolute())
            sys.path.insert(0, eotools_path)

        if eoread_path is not None:
            eoread_path = str(Path(eoread_path).absolute())
            sys.path.insert(0, eoread_path)

        if core_path is not None:
            core_path = str(Path(core_path).absolute())
            sys.path.insert(0, core_path)

        sys.path.insert(0, polymer_base_path)



        # TODO
        srf_nc_path, srf_nc_path = self.ac_polymer_get_srf_nc_path()

        run_polymer_kwargs = {"srf_getter": "hypso.ac.ac_polymer_srf_getter",
                                "srf_getter_arg": srf_nc_path}


        from eoread.hypso import Level1_HYPSO
        from polymer.main_v5 import run_polymer, run_polymer_dataset, default_output_datasets


        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        # Output (not input) moved into a parent_dir/polymer/ subfolder
        # (2026-08-05, was parent_dir directly) - same reasoning as
        # ac_acolite_run_correction's equivalent change above (keeps
        # per-run AC output out of the capture directory root; matches the
        # PACE-side Polymer connector's existing convention).
        polymer_output_dir = Path(self.parent_dir, "polymer")
        polymer_output_dir.mkdir(parents=True, exist_ok=True)

        match input_product_level.lower():

            case "l1c":
                polymer_l1_input_nc_file = Path(self.parent_dir, self.l1c_nc_file)
                polymer_l2_output_nc_file = Path(polymer_output_dir, str(self.l1c_name) + ".polymer.nc")
            case "l1d":
                polymer_l1_input_nc_file = Path(self.parent_dir, self.l1d_nc_file)
                polymer_l2_output_nc_file = Path(polymer_output_dir, str(self.l1d_name) + ".polymer.nc")
            case _:
                return None
            
        

        #import os
        #cwd = os.getcwd()
        #os.chdir(polymer_path)

        # This is from the Feb 2026 version of Polymer
        #from polymer.level1 import Level1
        #from polymer.level2 import Level2
        #from eoread.hypso import Level1_HYPSO
        #from polymer.main_v5 import run_polymer, run_polymer_dataset
        #from core.files.fileutils import mdir
        #polymer_output_file = run_polymer(Level1_HYPSO(polymer_input_file), dir_out=mdir(polymer_output_dir), split_bands=False)

        match polymer_version:
            case "v1":
                output_selection_kwargs = {
                    "output_datasets": default_output_datasets + optional_output_datasets,
                }
            case "v2":
                output_selection_kwargs = {
                    "outputs": "named",
                    "outputs_names": [
                        "latitude", "longitude", "rho_w", "logchl", "logfb",
                        "Rgli", "Rnir", "flags",
                    ] + optional_output_datasets,
                }
            case _:
                raise ValueError(f"Unknown polymer_version: {polymer_version!r}")

        # Run Polymer
        if True:
            output_file = run_polymer(
                Level1_HYPSO(polymer_l1_input_nc_file),
                dir_out=str(polymer_output_dir),
                if_exists = if_exists,
                srf_getter = "hypso.ac.ac_polymer_srf_getter",
                srf_getter_arg = srf_nc_path,
                **output_selection_kwargs,

            )

        try:
            polymer_l2_output_nc_file = Path(output_file).rename(polymer_l2_output_nc_file)
        except FileNotFoundError:
            print("[WARNING] Polymer L2 NetCDF output file has already been renamed.")
            pass

        print(output_file)
        print(polymer_l2_output_nc_file)

        return Path(polymer_l2_output_nc_file)


    





    def ac_polymer_open_output(self, 
                               polymer_l2_output_nc_file: Path = None, 
                               input_product_level="l1c",
                               version = "v1" 
                               #coeff_type: str = None
                               ):
        
        #if coeff_type is not None:
        #    coeff_type_str = "-" + str(coeff_type).lower()
        #else:
        #    coeff_type_str = ""

        if polymer_l2_output_nc_file is not None:
            polymer_l2_output_nc_file = Path(polymer_l2_output_nc_file)
        else:
            match input_product_level.lower():

                case "l1c":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1c product.")
                    # parent_dir/polymer/, not parent_dir directly - see
                    # ac_polymer_run_correction's matching change.
                    polymer_l2_output_nc_file = Path(self.parent_dir, "polymer", str(self.l1c_name)+ ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1c.nc.polymer.nc

                case "l1d":
                    print("[INFO] Reading Polymer L2 NetCDF output file generated using L1d product.")
                    polymer_l2_output_nc_file = Path(self.parent_dir, "polymer", str(self.l1d_name) + ".polymer.nc") #frohavet_2025-05-22T11-20-44Z-l1d.nc.polymer.nc
            

        polymer_l2_output_nc_file = polymer_l2_output_nc_file.absolute()
        

        if polymer_l2_output_nc_file.is_file():

            if version == "v1":
                polymer_datasets = load_polymer_l2_v1_nc(polymer_l2_output_nc_file)

                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    self.l2a_cube["polymer"] = cube
                    self.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.") 

            elif version == "v2":

                polymer_datasets = load_polymer_l2_v2_nc(polymer_l2_output_nc_file)
            
                try:
                    key = "rho_w"
                    inferred_wavelengths = polymer_datasets['bands'].data

                    # Map inferred Polymer wavelengths to HYPSO wavelengths
                    wl_band_map = self._get_inferred_wavelength_band_map(inferred_wavelengths=inferred_wavelengths)

                    # Create empty cube with standard HYPSO cube dims
                    shape = (self.spatial_dimensions[0], self.spatial_dimensions[1], self.bands)
                    cube = np.full(shape=shape, fill_value=np.nan)
                    cube[:,:,wl_band_map] = polymer_datasets[key]

                    self.l2a_cube["polymer"] = cube
                    self.l2a_cube["polymer"].attrs['l2_variable_name'] = key

                except Exception as ex:
                    print("[ERROR] Unable to load Polymer output dataset.")

        else:
            print("[ERROR] Polymer L2 NetCDF output file " + str(polymer_l2_output_nc_file) + " does not exist.")
            polymer_datasets = None

        
        return polymer_datasets










    def _get_inferred_wavelength_band_map(self, inferred_wavelengths):

        # Map inferred wavelengths to HYPSO wavelengths
        A = np.array(inferred_wavelengths, dtype=float)
        B = np.array(self.wavelengths, dtype=float)

        index_map = {}
        indices_unique = []

        for a in A:
            ix = np.argmin(np.abs(B - a))
            if ix not in index_map: # ensure uniqueness
                index_map[ix] = a
                indices_unique.append(ix)
            else:
                print("[WARNING] Duplicate prevented:", a, "mapped to", ix)

        wl_band_map = np.array(indices_unique, dtype=int)


        return wl_band_map


    def _get_fwhm(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths: 
            idx = np.argmin(np.abs(band - self.srf_wl))
            fwhm_per_band.append(self.srf_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm = fwhm
        
        return None


    def _get_fwhm_unbinned(self) -> None:
        
        fwhm_per_band = []
        for band in self.wavelengths_unbinned: 
            idx = np.argmin(np.abs(band - self.srf_wl))
            fwhm_per_band.append(self.srf_fwhm[idx])

        fwhm = fwhm_per_band

        self.fwhm_unbinned = fwhm
        
        return None


    from hypso.reflectance import compute_csiro_srfs

    compute_csiro_srfs = compute_csiro_srfs

    from hypso.ac import ac_dark_pixel_subtraction

    ac_dark_pixel_subtraction = ac_dark_pixel_subtraction
