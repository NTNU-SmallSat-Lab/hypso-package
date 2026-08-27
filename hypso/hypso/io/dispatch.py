"""Per-capture load dispatch, extracted from HypsoCapture (self.io composition - part
of the HypsoCapture breakup called for in the approved refactor plan, see
REFACTOR_PROGRESS.md). Bodies are moved verbatim from HypsoCapture.py, not rewritten -
same behavior, just relocated - each function takes `satobj` explicitly (matching
the pattern already used by hypso.georeferencing.geo and hypso.calibration.pipeline).

HypsoCapture's private _load_capture_file/_set_hypso_attributes/_check_capture_type/
_parse_filename/_compose_capture_name had no external callers (confirmed by grep
before moving), so they moved here outright with no wrapper kept on HypsoCapture.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from trollsift import Parser

from hypso.load import load_l1a_nc, \
                        load_l1b_nc, \
                        load_l1c_nc, \
                        load_l1d_nc, \
                        load_l2a_nc

from .metadata import CaptureMetadata

from hypso.capture_types import _get_fwhm, _get_fwhm_unbinned
from hypso.georeferencing import GeoAngles

logger = logging.getLogger(__name__)


def parse_filename(path) -> dict:
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


def compose_capture_name(satobj, fields: dict) -> str:
    if hasattr(satobj, '_use_old_filename_format'):
        p = Parser("{capture_target}_{capture_datetime:%Y-%m-%d_%H%MZ}")  # Old filename format
    else:
        p = Parser("{capture_target}_{capture_datetime:%Y-%m-%dT%H-%M-%SZ}")  # New filename format

    capture_name = p.compose(fields)

    return capture_name


def load_capture_file(satobj, path: Path, load_cube: bool = True) -> None:

    path = Path(path).absolute()

    fields = parse_filename(path)

    for key, value in fields.items():
        setattr(satobj, key, value)

    capture_name = compose_capture_name(satobj, fields=fields)

    satobj.capture_name = capture_name

    satobj.capture_dir = Path(path.parent.absolute())
    satobj.parent_dir = Path(path.parent.absolute())


    if satobj.label is not None:
        label = "-" + str(satobj.label)
    else:
        label = ""


    satobj.l1a_name = capture_name + label + "-l1a"
    satobj.l1b_name = capture_name + label + "-l1b"
    satobj.l1c_name = capture_name + label + "-l1c"
    satobj.l1d_name = capture_name + label + "-l1d"
    #satobj.l2a_name = capture_name + label + "-l2a"

    satobj.l1a_nc_file = Path(path.parent, satobj.l1a_name + ".nc")
    satobj.l1b_nc_file = Path(path.parent, satobj.l1b_name + ".nc")
    satobj.l1c_nc_file = Path(path.parent, satobj.l1c_name + ".nc")
    satobj.l1d_nc_file = Path(path.parent, satobj.l1d_name + ".nc")

    product_level = fields['product_level']

    match product_level:
        case "l1a":
            if satobj.VERBOSE: logger.info("Loading L1a capture %s", satobj.capture_name)

            load_func = load_l1a_nc
            cube_name = "l1a_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1a")
            setattr(satobj, "product_symbol", "DN")

        case "l1b":
            if satobj.VERBOSE: logger.info("Loading L1b capture %s", satobj.capture_name)

            load_func = load_l1b_nc
            cube_name = "l1b_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1b")
            setattr(satobj, "product_symbol", "Lt")

        case "l1c":
            if satobj.VERBOSE: logger.info("Loading L1c capture %s", satobj.capture_name)

            load_func = load_l1c_nc
            cube_name = "l1b_cube"  # L1c cube is the same as the L1b cube
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1c")
            setattr(satobj, "product_symbol", "lt")

        case "l1d":
            if satobj.VERBOSE: logger.info("Loading L1d capture %s", satobj.capture_name)

            load_func = load_l1d_nc
            cube_name = "l1d_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1d")
            setattr(satobj, "product_symbol", "rhot")

        case "l2a":
            if satobj.VERBOSE: logger.info("Loading L2a capture %s", satobj.capture_name)

            ac = getattr(satobj, 'atmospheric_correction', None)

            if ac is not None:
                logger.info("L2a Detected atmospheric correction: %s", ac)
            else:
                logger.warning("No L2a atmospheric correction detected.")
                setattr(satobj, "atmospheric_correction", "default")

            load_func = load_l2a_nc
            cube_name = "l2a_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l2a")
            setattr(satobj, "product_symbol", "Rrs")  # TODO: polymer and dps is rho_w

        case _:
            logger.error("Unsupported product level: %s", product_level)
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

    satobj.metadata = CaptureMetadata.from_load_result(
        nc_metadata_vars=nc_metadata_vars,
        nc_metadata_attrs=nc_metadata_attrs,
        nc_geometry_vars=nc_geometry_vars,
        nc_geometry_attrs=nc_geometry_attrs,
        nc_gcp_vars=nc_gcp_vars,
        nc_gcp_attrs=nc_gcp_attrs,
        nc_global_metadata=nc_global_metadata,
        nc_cube_attrs=nc_cube_attrs,
    )

    # TODO: pass the dicts returned by load_func to set_hypso_attributes()
    # Note: this MUST be run before writing datacubes in order to pass correct dimensions to as_dataarray()
    set_hypso_attributes(satobj)
    check_capture_type(satobj)

    if load_cube:
        if satobj.product_level.lower() == "l2a":
            satobj.l2a_cubes[satobj.atmospheric_correction] = nc_cube
        else:
            setattr(satobj, cube_name, nc_cube)

    else:
        logger.warning("Datacube is not loaded!")


    # OC-SMART's own staged-input/output naming (and its output directory)
    # is computed by hypso.ac.adapters.ocsmart.OCSMARTAdapter itself, not
    # here - it used to be set on satobj.ocsmart_l1d_input_nc_file/
    # ocsmart_l2a_output_h5_file at load time, using str(satobj.sensor).
    # upper() (e.g. "HYPSO2_HSI") as the filename prefix. That was a
    # confirmed bug: OC-SMART's own sensor autodetection only recognizes the
    # satellite-agnostic "HYPSO_HSI" prefix (no satellite digit) - staging
    # under the wrong prefix produced OC-SMART's "Unable to detect sensor"
    # warning with no output, silently (exit code 0, no exception).
    # Confirmed via hypso-processing-pipeline's own independent debugging of
    # the same issue (see its ac_runners_hypso.py). Fixed by moving this
    # naming into the adapter (which now uses the correct fixed prefix)
    # rather than patching the wrong value here - see OCSMARTAdapter.
    # HYPSO_PREFIX/.output_path(). Confirmed zero other readers of either
    # attribute anywhere in this repo, hypso-processing-pipeline, or the
    # original hypso-package before removing them from here.


    dt = datetime.fromtimestamp(satobj.unixtime, tz=timezone.utc)
    # Moved into a capture_dir/acolite/ subfolder (2026-08-05, was
    # capture_dir directly) - matches ac_acolite_run_correction's own
    # settings['output'] below, and the PACE-side ACOLITE connector's
    # existing convention (ac_runners_pace.py). ACOLITE writes several
    # per-run log/settings .txt files alongside its L2R/L2W output
    # (delete_acolite_run_text_files defaults False), which had been
    # accumulating directly in the capture directory root with no
    # cleanup - one set per run, indefinitely.
    satobj.acolite_l2r_output_nc_file = Path(satobj.capture_dir, "acolite", f"{satobj.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2R.nc")
    satobj.acolite_l2w_output_nc_file = Path(satobj.capture_dir, "acolite", f"{satobj.platform.upper()}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}_L2W.nc")

    return None


def set_hypso_attributes(satobj) -> None:

    # Capture config related attributes
    for attr in satobj.metadata.capture_config.attrs.keys():
        setattr(satobj, attr, satobj.metadata.capture_config.attrs[attr])
    # FPS has been renamed to framerate. Need to support both since old .nc files may still use FPS
    try:
        satobj.metadata.capture_config.attrs['fps'] = satobj.metadata.capture_config.attrs['framerate']
    except:
        satobj.metadata.capture_config.attrs['framerate'] = satobj.metadata.capture_config.attrs['fps']

    satobj.background_value = 8 * satobj.metadata.capture_config.attrs["bin_factor"]
    satobj.exposure = satobj.metadata.capture_config.attrs["exposure"] / 1000  # in seconds


    # Capture dimensions attributes
    satobj.x_start = satobj.metadata.capture_config.attrs["aoi_x"]
    satobj.x_stop = satobj.metadata.capture_config.attrs["aoi_x"] + satobj.metadata.capture_config.attrs["column_count"]
    satobj.y_start = satobj.metadata.capture_config.attrs["aoi_y"]
    satobj.y_stop = satobj.metadata.capture_config.attrs["aoi_y"] + satobj.metadata.capture_config.attrs["row_count"]
    satobj.bin_factor = satobj.metadata.capture_config.attrs["bin_factor"]
    # Try/except here since not all captures have sample_div
    try:
        satobj.sample_div = satobj.metadata.capture_config.attrs['sample_div']
    except:
        satobj.sample_div = 1
    satobj.row_count = satobj.metadata.capture_config.attrs["row_count"]
    satobj.frame_count = satobj.metadata.capture_config.attrs["frame_count"]
    satobj.column_count = satobj.metadata.capture_config.attrs["column_count"]
    satobj.image_height = int(satobj.metadata.capture_config.attrs["row_count"] / satobj.sample_div)
    satobj.image_width = int(satobj.metadata.capture_config.attrs["column_count"] / satobj.metadata.capture_config.attrs["bin_factor"])
    satobj.im_size = satobj.image_height * satobj.image_width
    satobj.bands = satobj.image_width
    satobj.lines = satobj.metadata.capture_config.attrs["frame_count"]  # AKA Frames AKA Rows
    satobj.samples = satobj.image_height  # AKA Cols
    satobj.spatial_dimensions = (satobj.metadata.capture_config.attrs["frame_count"], satobj.image_height)
    if satobj.VERBOSE:
        logger.info("Capture spatial dimensions: %s", satobj.spatial_dimensions)


    # Calibration related atrributes
    satobj.rad_coeffs = satobj.metadata.corrections.vars['rad_matrix']

    try:
        satobj.spectral_coeffs = satobj.metadata.corrections.vars['spec_coeffs']
    except KeyError:
        satobj.spectral_coeffs = satobj.metadata.corrections.vars['wavelengths']

    if not hasattr(satobj, 'wavelengths'):
        if ('wavelengths' in satobj.metadata.cube_attrs.keys()):
            satobj.wavelengths = satobj.metadata.cube_attrs['wavelengths']
        else:
            satobj.wavelengths = np.array(range(0, satobj.image_width))

    if not hasattr(satobj, 'wavelengths_unbinned'):
        if ('wavelengths_unbinned' in satobj.metadata.corrections.vars.keys()):
            satobj.wavelengths_unbinned = satobj.metadata.corrections.vars['wavelengths_unbinned']
        else:
            satobj.wavelengths_unbinned = np.array(range(0, satobj.image_width))

    # Always recompute fwhm/fwhm_unbinned from this capture's own wavelengths
    # via the sensor's nearest-wavelength lookup table, rather than trusting
    # a stored 'fwhm' cube attribute - every L1A/L1B/L1C file written before
    # this fix carries HypsoCapture.__init__'s old fixed-length sensor-
    # default fwhm array (wrong values, not just wrong length for a capture
    # with a non-standard band count - see REFACTOR_PROGRESS.md's capture-
    # dimensions plan, Bug A), and recomputing reproduces the identical
    # value for correctly L1D/L2A-generated files, so nothing is lost either
    # way. Same reasoning HypsoCapture.spectral_response's docstring already
    # gives for its own lazy rebuild - this makes it run unconditionally on
    # load instead of only lazily on the L1D path.
    if hasattr(satobj, 'fwhm_lookup_wl') and hasattr(satobj, 'fwhm_lookup_fwhm'):
        satobj.fwhm = _get_fwhm(satobj)
        satobj.fwhm_unbinned = _get_fwhm_unbinned(satobj)
    else:
        # No sensor lookup table (e.g. a subclass built without a
        # SensorProfile) - flat per-band average, sized to THIS capture's
        # actual wavelengths (established just above), not a hardcoded
        # constant.
        satobj.fwhm = [satobj.AVERAGE_FWHM] * len(satobj.wavelengths)
        satobj.fwhm_unbinned = [satobj.AVERAGE_FWHM] * len(satobj.wavelengths_unbinned)


    if not hasattr(satobj, 'effective_fwhm'):
        if 'effective_fwhm' in satobj.metadata.srf.vars.keys():
            satobj.effective_fwhm = satobj.metadata.srf.vars['effective_fwhm']

    if not hasattr(satobj, 'effective_fwhm_unbinned'):
        if 'effective_fwhm_unbinned' in satobj.metadata.srf.vars.keys():
            satobj.effective_fwhm_unbinned = satobj.metadata.srf.vars['effective_fwhm_unbinned']

    if not hasattr(satobj, 'esun'):
        if 'esun' in satobj.metadata.srf.vars.keys():
            satobj.esun = satobj.metadata.srf.vars['esun']

    if not hasattr(satobj, 'esun_wl'):
        if 'esun_wavelengths' in satobj.metadata.srf.vars.keys():
            satobj.esun_wl = satobj.metadata.srf.vars['esun_wavelengths']


    # Geometry atrributes. The 5 angle quantities (sensor/solar zenith/azimuth,
    # relative azimuth) are accumulated into plain dicts and built into
    # GeoAngles instances once after the loop, rather than setattr'd
    # incrementally - sat_zenith_angles/etc. are now read-only properties over
    # satobj.angles/angles_direct (see HypsoCapture.py), so the old
    # incremental setattr(satobj, 'sat_zenith_angles', value) calls would
    # raise. Order-independent: a later-processed non-direct key
    # unconditionally overwrites the dict entry (matching the original
    # setattr's unconditional overwrite), and a direct key only backfills the
    # non-direct dict entry if nothing's there yet (matching the original
    # getattr(satobj, name, None) is None check) - correct regardless of
    # which order the two keys appear in this dict.
    angle_kwargs = {}
    direct_angle_kwargs = {}

    for key, value in satobj.metadata.geometry.vars.items():
        if key == 'unixtime':
            continue
        elif key == 'latitude':
            setattr(satobj, 'latitudes', value)
        elif key == 'longitude':
            setattr(satobj, 'longitudes', value)

        elif key == 'latitude_direct':
            setattr(satobj, 'latitudes_direct', value)
        elif key == 'longitude_direct':
            setattr(satobj, 'longitudes_direct', value)


        elif key == 'sensor_zenith':
            angle_kwargs['sensor_zenith'] = value
        elif key == 'sensor_azimuth':
            angle_kwargs['sensor_azimuth'] = value

        elif key == 'sensor_zenith_direct':
            direct_angle_kwargs['sensor_zenith'] = value
            if angle_kwargs.get('sensor_zenith') is None:
                angle_kwargs['sensor_zenith'] = value

        elif key == 'sensor_azimuth_direct':
            direct_angle_kwargs['sensor_azimuth'] = value
            if angle_kwargs.get('sensor_azimuth') is None:
                angle_kwargs['sensor_azimuth'] = value

        elif key == 'solar_zenith':
            angle_kwargs['solar_zenith'] = value
        elif key == 'solar_azimuth':
            angle_kwargs['solar_azimuth'] = value

        elif key == 'solar_zenith_direct':
            direct_angle_kwargs['solar_zenith'] = value
            if angle_kwargs.get('solar_zenith') is None:
                angle_kwargs['solar_zenith'] = value

        elif key == 'solar_azimuth_direct':
            direct_angle_kwargs['solar_azimuth'] = value
            if angle_kwargs.get('solar_azimuth') is None:
                angle_kwargs['solar_azimuth'] = value

        elif key == 'relative_azimuth':
            angle_kwargs['relative_azimuth'] = value

        elif key == 'relative_azimuth_direct':
            direct_angle_kwargs['relative_azimuth'] = value
            if angle_kwargs.get('relative_azimuth') is None:
                angle_kwargs['relative_azimuth'] = value

        else:
            setattr(satobj, key, value)

    satobj.angles = GeoAngles(**angle_kwargs)
    satobj.angles_direct = GeoAngles(**direct_angle_kwargs)


    # Capture timing attributes
    try:
        satobj.start_timestamp_capture = int(satobj.timing['capture_start_unix']) + satobj.UNIX_TIME_OFFSET
    except:
        try:
            datestring = satobj.metadata.global_attrs['date_aquired']
        except:
            datestring = satobj.metadata.global_attrs['timestamp_acquired']


        try:
            dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
        except ValueError:
            dt = datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S.%f%zZ').replace(tzinfo=timezone.utc)

        satobj.start_timestamp_capture = dt.timestamp()

    #satobj.start_timestamp_capture = int(satobj.metadata.timing.attrs['capture_start_unix']) + satobj.UNIX_TIME_OFFSET

    # Get END_TIMESTAMP_CAPTURE
    # can't compute end timestamp using frame count and frame rate
    # assuming some default value if framerate and exposure not available
    try:
        satobj.end_timestamp_capture = satobj.start_timestamp_capture + satobj.metadata.capture_config.attrs["frame_count"] / satobj.metadata.capture_config.attrs["framerate"] + satobj.metadata.capture_config.attrs["exposure"] / 1000.0
    except Exception:
        if satobj.VERBOSE:
            logger.warning("Framerate or exposure values not found. Assuming 20.0 for each.")
        satobj.end_timestamp_capture = satobj.start_timestamp_capture + satobj.metadata.capture_config.attrs["frame_count"] / 20.0 + 20.0 / 1000.0

    # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
    time_margin_start = 641.0  # 70.0
    time_margin_end = 180.0  # 70.0
    satobj.start_timestamp_adcs = satobj.start_timestamp_capture - time_margin_start
    satobj.end_timestamp_adcs = satobj.end_timestamp_capture + time_margin_end
    satobj.unixtime = satobj.start_timestamp_capture

    #satobj.iso_time = datetime.utcfromtimestamp(satobj.unixtime).isoformat()
    satobj.iso_time = datetime.fromtimestamp(satobj.unixtime, tz=timezone.utc).isoformat()

    return None


def check_capture_type(satobj) -> None:
    """Classify satobj.capture_type using its sensor_profile's own
    capture_type_thresholds (see hypso.sensors.SensorProfile) instead of one
    hardcoded chain shared by every sensor - see REFACTOR_PROGRESS.md's
    capture-dimensions audit for why this changed and what it fixed."""

    for capture_type, attr, expected_value in satobj.sensor_profile.capture_type_thresholds:
        if getattr(satobj, attr) == expected_value:
            satobj.capture_type = capture_type
            break
    else:
        # EXPERIMENTAL_FEATURES
        if satobj.VERBOSE:
            logger.warning("Number of Rows (AKA frame_count) Is Not Standard.")
        satobj.capture_type = "custom"

    if satobj.VERBOSE:
        logger.info("Capture capture type: %s", satobj.capture_type)
