import numpy as np

from datetime import datetime
import pandas as pd
import netCDF4 as nc
from hypso import georeference
from pathlib import Path
from hypso.utils import is_integer_num, find_file

EXPERIMENTAL_FEATURES = True


def load_nc(nc_file_path, standardDimensions):
    # Get metadata
    nc_info, spatialDim = get_metainfo_from_nc_file(nc_file_path, standardDimensions)

    # Create temporary position.csv and quaternion.csv
    nc_info = georeference.create_adcs_timestamps_files(nc_file_path, nc_info)

    FRAME_TIMESTAMP_TUNE_OFFSET = 0.0
    quaternion_csv_path = Path(nc_info["tmp_dir"], "quaternion.csv")
    position_csv_path = Path(nc_info["tmp_dir"], "position.csv")
    timestamps_srv_path = Path(nc_info["tmp_dir"], "timestamps_services.txt")
    frametime_pose_csv_path = Path(nc_info["tmp_dir"], "frametime-pose.csv")
    # TODO: Check if local-angles.csv is to be used or sun-azimuth.dat and etc
    local_angles_csv_path = Path(nc_info["tmp_dir"], "local-angles.csv")
    sat_azimuth_path = Path(nc_info["tmp_dir"], "sat-azimuth.dat")
    sat_zenith_path = Path(nc_info["tmp_dir"], "sat-zenith.dat")
    solar_azimuth_path = Path(nc_info["tmp_dir"], "sun-azimuth.dat")
    solar_zenith_path = Path(nc_info["tmp_dir"], "sun-zenith.dat")

    latitude_dataPath = Path(nc_info["tmp_dir"], "latitudes.dat")
    longitude_dataPath = Path(nc_info["tmp_dir"], "longitudes.dat")

    if not frametime_pose_csv_path.is_file():
        georeference.interpolate_at_frame(pos_csv_path=position_csv_path,
                                          quat_csv_path=quaternion_csv_path,
                                          flash_csv_path=timestamps_srv_path,
                                          additional_time_offset=FRAME_TIMESTAMP_TUNE_OFFSET,
                                          framerate=nc_info["fps"],
                                          exposure=nc_info["exposure"])

    if not local_angles_csv_path.is_file():
        georeference.geometry_computation(framepose_data_path=frametime_pose_csv_path,
                                          hypso_height=nc_info["row_count"])  # frame_height = row_count

    nc_info = get_local_angles(sat_azimuth_path, sat_zenith_path,
                               solar_azimuth_path, solar_zenith_path,
                               nc_info, spatialDim)

    nc_info = get_lat_lon_2d(latitude_dataPath, longitude_dataPath, nc_info, spatialDim)

    nc_rawcube = get_raw_cube_from_nc_file(nc_file_path)

    return nc_info, nc_rawcube, spatialDim


def get_lat_lon_2d(latitude_dataPath, longitude_dataPath, info, spatialDim):
    # Load Latitude
    info["lat"] = np.fromfile(latitude_dataPath, dtype="float32")
    info["lat"] = info["lat"].reshape(spatialDim)
    # Load Longitude
    info["lon"] = np.fromfile(longitude_dataPath, dtype="float32")
    info["lon"] = info["lon"].reshape(spatialDim)

    # Original Lat in case manual georeference is used
    info["lat_original"] = info["lat"]
    info["lon_original"] = info["lon"]

    return info


def get_local_angles(sat_azimuth_path, sat_zenith_path,
                     solar_azimuth_path, solar_zenith_path, info, spatialDim):
    info["solar_zenith_angle"] = np.fromfile(solar_zenith_path, dtype="float32")
    info["solar_zenith_angle"] = info["solar_zenith_angle"].reshape(spatialDim)

    info["solar_azimuth_angle"] = np.fromfile(solar_azimuth_path, dtype="float32")
    info["solar_azimuth_angle"] = info["solar_azimuth_angle"].reshape(spatialDim)

    info["sat_zenith_angle"] = np.fromfile(sat_zenith_path, dtype="float32")
    info["sat_zenith_angle"] = info["sat_zenith_angle"].reshape(spatialDim)

    info["sat_azimuth_angle"] = np.fromfile(sat_azimuth_path, dtype="float32")
    info["sat_azimuth_angle"] = info["sat_azimuth_angle"].reshape(spatialDim)

    return info


def get_metainfo_from_nc_file(nc_file_path: Path, standardDimensions) -> dict:
    """Get the metadata from the top folder of the data.

    Args:
        top_folder_name (str): The name of the top folder of the data.

    Returns:
        dict: The metadata.
    """

    info = {}
    nc_name = nc_file_path.stem
    temp_dir = Path(nc_file_path.parent.absolute(), nc_name.replace("-l1a", "") + "_tmp")
    info["tmp_dir"] = temp_dir
    # Add file name
    capture_name = nc_file_path.stem
    if "-l1a" in capture_name:
        capture_name = capture_name.replace("-l1a", "")
    info["capture_name"] = capture_name

    info["capture_region"] = capture_name.split('_')[0].strip('_')

    temp_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------------
    # Capture info -----------------------------------------------------------
    # ------------------------------------------------------------------------

    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["capture_config"]
        for attrname in group.ncattrs():
            value = getattr(group, attrname)
            try:
                if is_integer_num(float(value)):
                    info[attrname] = int(value)
                else:
                    info[attrname] = float(value)
            except BaseException:
                info[attrname] = value

    # ------------------------------------------------------------------------
    # Timestamps -------------------------------------------------------------
    # ------------------------------------------------------------------------
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["timing"]
        # timestamps_services = group.variables["timestamps_srv"][:]
        # timestamps = group.variables["timestamps"][:]
        # TODO: Cosider using attribute "capture_start_unix" instead of "capture_start_planned_unix"
        start_timestamp_capture = getattr(group, "capture_start_unix")  # unix time ms

        if start_timestamp_capture is None or start_timestamp_capture == 0:
            raise Exception("No Start Timestamp Capture Value available")

    # TODO: Verify offset validity. Sivert had 20 here
    UNIX_TIME_OFFSET = 20

    info["start_timestamp_capture"] = int(start_timestamp_capture) + UNIX_TIME_OFFSET

    # Get END_TIMESTAMP_CAPTURE
    #    cant compute end timestamp using frame count and frame rate
    #     assuming some default value if fps and exposure not available

    try:
        info["end_timestamp_capture"] = info["start_timestamp_capture"] + info["frame_count"] / info["fps"] + info[
            "exposure"] / 1000.0
    except:
        print("fps or exposure values not found assuming 20.0 for each")
        info["end_timestamp_capture"] = info["start_timestamp_capture"] + info["frame_count"] / 20.0 + 20.0 / 1000.0

    # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
    time_margin_start = 641.0  # 70.0
    time_margin_end = 180.0  # 70.0
    info["start_timestamp_adcs"] = info["start_timestamp_capture"] - time_margin_start
    info["end_timestamp_adcs"] = info["end_timestamp_capture"] + time_margin_end

    # data for geotiff processing -----------------------------------------------

    info["unixtime"] = info["start_timestamp_capture"]
    info["iso_time"] = datetime.utcfromtimestamp(
        info["unixtime"]
    ).isoformat()

    # ------------------------------------------------------------------------
    # Target Lat Lon ---------------------------------------------------------
    # ------------------------------------------------------------------------
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f
        try:
            # returns string 'False' if doesnt exist
            target_coords = getattr(group, "target_coords")
            if target_coords == 'False':
                target_lat = "-"
                target_lon = "-"
            else:
                target_coords = target_coords.split(" ")
                target_lat = target_coords[0]
                target_lon = target_coords[1]
        except AttributeError:
            target_lat = "-"
            target_lon = "-"

    info["latc"] = target_lat
    info["lonc"] = target_lon
    info['target_area'] = target_lat + ' ' + target_lon

    info["background_value"] = 8 * info["bin_factor"]

    info["x_start"] = info["aoi_x"]
    info["x_stop"] = info["aoi_x"] + info["column_count"]
    info["y_start"] = info["aoi_y"]
    info["y_stop"] = info["aoi_y"] + info["row_count"]
    info["exp"] = info["exposure"] / 1000  # in seconds

    info["image_height"] = info["row_count"]
    info["image_width"] = int(info["column_count"] / info["bin_factor"])
    info["im_size"] = info["image_height"] * info["image_width"]

    # Update Spatial Dim if not standard
    rows_img = info["frame_count"]  # Due to way image is captured
    cols_img = info["image_height"]

    if rows_img == standardDimensions["nominal"]:
        info["capture_type"] = "nominal"

    elif cols_img == standardDimensions["wide"]:
        info["capture_type"] = "wide"
    else:
        if EXPERIMENTAL_FEATURES:
            print("Number of Rows (AKA frame_count) Is Not Standard")
            info["capture_type"] = "custom"
        else:
            raise Exception("Number of Rows (AKA frame_count) Is Not Standard")

    spatialDim = (rows_img, cols_img)

    print(
        f"Processing *{info['capture_type']}* Image with Dimensions: {spatialDim}")

    return info, spatialDim


def get_raw_cube_from_nc_file(nc_file_path) -> np.ndarray:
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f.groups["products"]
        # 16-bit according to Original data Capture
        raw_cube = np.array(group.variables["Lt"][:], dtype='uint16')

        return raw_cube
