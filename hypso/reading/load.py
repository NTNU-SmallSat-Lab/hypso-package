import numpy as np
import os
from datetime import datetime
import pandas as pd
import glob
import netCDF4 as nc
from hypso import georeference
from pathlib import Path
from hypso.utils import is_integer_num, find_file
EXPERIMENTAL_FEATURES=True




def load_directory(top_folder_name, standardDimensions):
    dir_info,spatialDim=get_metainfo_from_directory(top_folder_name, standardDimensions)
    dir_rawcube=get_raw_cube_from_directory(top_folder_name,dir_info)

    return dir_info,dir_rawcube,spatialDim

def load_nc(nc_file_path, standardDimensions):
    # Get metadata
    nc_info,spatialDim = get_metainfo_from_nc_file(nc_file_path, standardDimensions)

    # Create temporary position.csv and quaternion.csv
    nc_info= georeference.create_adcs_timestamps_files(nc_file_path, nc_info)

    FRAME_TIMESTAMP_TUNE_OFFSET=0.0
    quaternion_csv_path=Path(nc_info["tmp_dir"], "quaternion.csv")
    position_csv_path=Path(nc_info["tmp_dir"], "position.csv")
    timestamps_srv_path=Path(nc_info["tmp_dir"], "timestamps_services.txt")
    frametime_pose_csv_path = Path(nc_info["tmp_dir"], "frametime-pose.csv")
    local_angles_csv_path = Path(nc_info["tmp_dir"], "local-angles.csv")
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
                                          hypso_height=nc_info["row_count"]) # frame_height = row_count


    nc_info=get_local_angles(local_angles_csv_path, nc_info)

    nc_info = get_lat_lon_2d(latitude_dataPath, longitude_dataPath, nc_info, spatialDim)

    nc_rawcube=get_raw_cube_from_nc_file(nc_file_path)

    return nc_info, nc_rawcube, spatialDim

def get_raw_cube_from_directory(top_folder_name,info) -> np.ndarray:
    # find file ending in .bip
    path_to_bip = None
    for file in os.listdir(top_folder_name):
        if file.endswith(".bip"):
            path_to_bip = os.path.join(top_folder_name, file)
            break

    cube = np.fromfile(path_to_bip, dtype="uint16")
    cube = cube.reshape(
        (-1, info["image_height"], info["image_width"]))

    # reverse the order of the third dimension
    cube = cube[:, :, ::-1]

    return cube

def get_lat_lon_2d(latitude_dataPath, longitude_dataPath, info, spatialDim):

    # Load Latitude
    info["lat"] = np.fromfile(latitude_dataPath, dtype="float32")
    info["lat"] = info["lat"].reshape(spatialDim)
    # Load Longitude
    info["lon"] = np.fromfile(longitude_dataPath, dtype="float32")
    info["lon"] = info["lon"].reshape(spatialDim)

    return info
def get_local_angles(local_angle_path, info):

    local_angle_df = pd.read_csv(local_angle_path)

    solar_za = local_angle_df["Solar Zenith Angle [degrees]"].tolist()
    solar_aa = local_angle_df["Solar Azimuth Angle [degrees]"].tolist()
    sat_za = local_angle_df["Satellite Zenith Angle [degrees]"].tolist()
    sat_aa = local_angle_df["Satellite Azimuth Angle [degrees]"].tolist()

    # Calculates the average solar/sat azimuth/zenith angle.
    average_solar_za = np.round(np.average(solar_za), 5)
    average_solar_aa = np.round(np.average(solar_aa), 5)
    average_sat_za = np.round((np.average(sat_za)), 5)
    average_sat_aa = np.round(np.average(sat_aa), 5)

    info["solar_zenith_angle"] = average_solar_za
    info["solar_azimuth_angle"] = average_solar_aa
    info["sat_zenith_angle"] = average_sat_za
    info["sat_azimuth_angle"] = average_sat_aa

    return info
def get_metainfo_from_nc_file(nc_file_path: Path, standardDimensions) -> dict:
    """Get the metadata from the top folder of the data.

    Args:
        top_folder_name (str): The name of the top folder of the data.

    Returns:
        dict: The metadata.
    """

    info = {}
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
        # Add file name
    info["capture_name"] = nc_file_path.stem

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

    # Data for geotiff processing -----------------------------------------------

    info["unixtime"] = info["start_timestamp_capture"]
    info["iso_time"] = datetime.utcfromtimestamp(
        info["unixtime"]
    ).isoformat()


    # ------------------------------------------------------------------------
    # Target Lat Lon ---------------------------------------------------------
    # ------------------------------------------------------------------------
    with nc.Dataset(nc_file_path, format="NETCDF4") as f:
        group = f
        target_coords = getattr(group, "target_coords")
        if target_coords is None:
            target_lat = "-"
            target_lon = "-"
        else:
            target_coords = target_coords.split(" ")
            target_lat=target_coords[0]
            target_lon=target_coords[1]

    info["latc"] =target_lat
    info["lonc"] = target_lon

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

    if (rows_img == standardDimensions["nominal"]):
        info["capture_type"] = "nominal"

    elif (cols_img == standardDimensions["wide"]):
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
        raw_cube = np.array(group.variables["Lt"][:])
        return raw_cube

def get_metainfo_from_directory(top_folder_name: Path, standardDimensions) -> dict:
    """Get the metadata from the top folder of the data.

    Args:
        top_folder_name (str): The name of the top folder of the data.

    Returns:
        dict: The metadata.
    """
    info = {}
    info["top_folder_name"] = top_folder_name
    info["folder_name"] = top_folder_name.name

    # find folder with substring "hsi0" or throw error
    for folder in top_folder_name.iterdir():
        if folder.is_dir():
            if "hsi0" in folder.name:
                raw_folder = folder
                break
    else:
        raise ValueError("No folder with metadata found.")

    # combine top_folder_name and raw_folder to get the path to the raw
    # data
    config_file_path = Path(
        top_folder_name, raw_folder, "capture_config.ini"
    )


    # read all lines in the config file
    with open(config_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # split the line at the equal sign
            line = line.split("=")
            # if the line has two elements, add the key and value to the
            # info dict
            if len(line) == 2:
                key = line[0].strip()
                value = line[1].strip()
                try:
                    if is_integer_num(float(value)):
                        info[key] = int(value)
                    else:
                        info[key] = float(value)
                except BaseException:
                    info[key] = value

    timetamp_file = Path(
        top_folder_name, raw_folder, "timestamps.txt")

    try:
        with open(timetamp_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "." not in line:
                    continue
                s_part = line.split(".")[0]
                if s_part.strip().isnumeric():
                    info["unixtime"] = int(s_part) + 20
                    info["iso_time"] = datetime.utcfromtimestamp(
                        info["unixtime"]
                    ).isoformat()
                    break
    except:
        print("No timestamps.txt file. Necessary for atmospheric correction.")
        info["unixtime"] = None
        info["iso_time"]=None

    # find local_angle_csv file with substring "local-angles.csv" or throw error
    for file in top_folder_name.iterdir():
        if file.is_file():
            if "local-angles.csv" in file.name:
                local_angle_csv = file
                break
    else:
        raise ValueError("No local-angles.csv file found.")


    info=get_local_angles(Path(top_folder_name, local_angle_csv), info)

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

    if (rows_img == standardDimensions["nominal"]):
        info["capture_type"] = "nominal"

    elif (cols_img == standardDimensions["wide"]):
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

    # Find Coordinates of the Center of the Image
    # TODO get center lat from .dat lat lon files , average it
    pos_file = ""
    foldername = info["top_folder_name"]
    for file in foldername.iterdir():
        if file.is_file():
            if file.name.endswith("geometric-meta-info.txt"):
                pos_file = os.path.join(foldername, file)
                break

    if pos_file == "":
        raise ValueError(f"Could not find position file in {foldername}")

    found_pos = False
    with open(pos_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "lat lon" in line:
                info["latc"] = float(line.split(
                    "lat lon")[1].split(" ")[1])
                info["lonc"] = float(line.split(
                    "lat lon")[1].split(" ")[2])
                found_pos = True
                break

    if not found_pos:
        raise ValueError(f"Could not find position in {pos_file}")

    # Find 2D Coordinate
    longitude_dataPath = find_file(top_folder_name, "longitudes",".dat")
    latitude_dataPath = find_file(top_folder_name, "latitudes",".dat")
    info=get_lat_lon_2d(latitude_dataPath, longitude_dataPath, info, spatialDim)

    return info, spatialDim