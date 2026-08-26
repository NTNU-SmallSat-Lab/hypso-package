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
            if satobj.VERBOSE: print('[INFO] Loading L1a capture ' + satobj.capture_name)

            load_func = load_l1a_nc
            cube_name = "l1a_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1a")
            setattr(satobj, "product_symbol", "DN")

        case "l1b":
            if satobj.VERBOSE: print('[INFO] Loading L1b capture ' + satobj.capture_name)

            load_func = load_l1b_nc
            cube_name = "l1b_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1b")
            setattr(satobj, "product_symbol", "Lt")

        case "l1c":
            if satobj.VERBOSE: print('[INFO] Loading L1c capture ' + satobj.capture_name)

            load_func = load_l1c_nc
            cube_name = "l1b_cube"  # L1c cube is the same as the L1b cube
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1c")
            setattr(satobj, "product_symbol", "lt")

        case "l1d":
            if satobj.VERBOSE: print('[INFO] Loading L1d capture ' + satobj.capture_name)

            load_func = load_l1d_nc
            cube_name = "l1d_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l1d")
            setattr(satobj, "product_symbol", "rhot")

        case "l2a":
            if satobj.VERBOSE: print('[INFO] Loading L2a capture ' + satobj.capture_name)

            ac = getattr(satobj, 'atmospheric_correction', None)

            if ac is not None:
                print("[INFO] L2a Detected atmospheric correction: " + str(ac))
            else:
                print("[WARNING] No L2a atmospheric correction detected.")
                setattr(satobj, "atmospheric_correction", "default")

            load_func = load_l2a_nc
            cube_name = "l2a_cube"
            setattr(satobj, "cube_name", cube_name)
            setattr(satobj, "product_level", "l2a")
            setattr(satobj, "product_symbol", "Rrs")  # TODO: polymer and dps is rho_w

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
        print("[WARNING] Datacube is not loaded!")


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
        print(f"[INFO] Capture spatial dimensions: {satobj.spatial_dimensions}")


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

    if not hasattr(satobj, 'fwhm'):
        if 'fwhm' in satobj.metadata.cube_attrs.keys():
            satobj.fwhm = satobj.metadata.cube_attrs['fwhm']
        else:
            #satobj.fwhm = [satobj.AVERAGE_FWHM] * satobj.bands
            satobj.fwhm = [satobj.AVERAGE_FWHM] * satobj.UNBINNED_BAND_COUNT


    if not hasattr(satobj, 'effective_fwhm'):
        if 'effective_fwhm' in satobj.metadata.srf.vars.keys():
            satobj.effective_fwhm = satobj.metadata.srf.vars['effective_fwhm']

    if not hasattr(satobj, 'esun'):
        if 'esun' in satobj.metadata.srf.vars.keys():
            satobj.esun = satobj.metadata.srf.vars['esun']

    if not hasattr(satobj, 'esun_wl'):
        if 'esun_wavelengths' in satobj.metadata.srf.vars.keys():
            satobj.esun_wl = satobj.metadata.srf.vars['esun_wavelengths']


    # Fixed a missing comma (inherited verbatim from the old HypsoCapture): the
    # implicit string concatenation "csiro_binned_srfs" "csiro_effective_fwhm"
    # produced one bogus name, so neither attribute was ever restored when
    # loading an L1D file.
    csiro_list = ["csiro_ssi", "csiro_solar_wavelengths", "csiro_binned_srfs",
                  "csiro_effective_fwhm", "csiro_esun"]

    for csiro_key in csiro_list:
        if not hasattr(satobj, csiro_key):
            if csiro_key in satobj.metadata.srf.vars.keys():
                setattr(satobj, csiro_key, satobj.metadata.srf.vars[csiro_key])


    # Geometry atrributes
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
            setattr(satobj, 'sat_zenith_angles', value)
        elif key == 'sensor_azimuth':
            setattr(satobj, 'sat_azimuth_angles', value)

        elif key == 'sensor_zenith_direct':
            setattr(satobj, 'sat_zenith_angles_direct', value)
            if getattr(satobj, 'sat_zenith_angles', None) is None:
                setattr(satobj, 'sat_zenith_angles', value)

        elif key == 'sensor_azimuth_direct':
            setattr(satobj, 'sat_azimuth_angles_direct', value)
            if getattr(satobj, 'sat_azimuth_angles', None) is None:
                setattr(satobj, 'sat_azimuth_angles', value)

        elif key == 'solar_zenith':
            setattr(satobj, 'solar_zenith_angles', value)
        elif key == 'solar_azimuth':
            setattr(satobj, 'solar_azimuth_angles', value)

        elif key == 'solar_zenith_direct':
            setattr(satobj, 'solar_zenith_angles_direct', value)
            if getattr(satobj, 'solar_zenith_angles', None) is None:
                setattr(satobj, 'solar_zenith_angles', value)

        elif key == 'solar_azimuth_direct':
            setattr(satobj, 'solar_azimuth_angles_direct', value)
            if getattr(satobj, 'solar_azimuth_angles', None) is None:
                setattr(satobj, 'solar_azimuth_angles', value)

        elif key == 'relative_azimuth':
            setattr(satobj, 'relative_azimuth_angles', value)

        elif key == 'relative_azimuth_direct':
            setattr(satobj, 'relative_azimuth_angles_direct', value)
            if getattr(satobj, 'relative_azimuth_angles', None) is None:
                setattr(satobj, 'relative_azimuth_angles', value)

        else:
            setattr(satobj, key, value)


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
    except:
        if satobj.VERBOSE:
            print("[WARNING] Framerate or exposure values not found. Assuming 20.0 for each.")
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

    #satobj.spatial_dimensions = (956, 684)  # 1092 x variable
    #satobj.standard_dimensions = {
    #    "nominal": 956,  # Along frame_count
    #    "wide": 1092  # Along image_height (row_count)
    #}

    if satobj.metadata.capture_config.attrs["frame_count"] == 956:
    #if satobj.metadata.capture_config.attrs["frame_count"] == satobj.standard_dimensions["nominal"]:
        satobj.capture_type = "nominal"
    elif satobj.metadata.capture_config.attrs["frame_count"] == 106:
                satobj.capture_type = "moon"
    elif satobj.image_height == 1092:
    #elif satobj.image_height == satobj.standard_dimensions["wide"]:
        satobj.capture_type = "wide"
    else:
        # EXPERIMENTAL_FEATURES
        if satobj.VERBOSE:
            print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
        satobj.capture_type = "custom"

    if satobj.VERBOSE:
        print(f"[INFO] Capture capture type: {satobj.capture_type}")
