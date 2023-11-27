import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path

def create_adcs_timestamps_files(nc_path,nc_info, temp_dir="_tmp"):
    nc_name = nc_path.stem
    temp_dir=Path(nc_path.parent.absolute(),nc_name+temp_dir)
    nc_info["tmp_dir"]=temp_dir
    nc_info["tmp_dir"] = temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: ADCS Dat----------------------------------------------
    with nc.Dataset(nc_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["adcs"]

        # print(group.variables.keys())

        position_headers = ["timestamp", "eci x [m]", "eci y [m]", "eci z [m]"]
        quaternion_headers = ["timestamp", "quat_0", "quat_1", "quat_2", "quat_3", "Control error [deg]"]

        timestamps = np.array(group.variables["timestamps"][:])
        pos_x = np.array(group.variables["position_x"][:])
        pos_y = np.array(group.variables["position_y"][:])
        pos_z = np.array(group.variables["position_z"][:])

        position_array = np.column_stack((timestamps, pos_x, pos_y, pos_z))
        position_df = pd.DataFrame(position_array, columns=position_headers)

        quat_s = np.array(group.variables["quaternion_s"][:])
        quat_x = np.array(group.variables["quaternion_x"][:])
        quat_y = np.array(group.variables["quaternion_y"][:])
        quat_z = np.array(group.variables["quaternion_z"][:])
        control_error = np.array(group.variables["control_error"][:])

        quat_array = np.column_stack((timestamps, quat_s, quat_x, quat_y, quat_z, control_error))
        quat_df = pd.DataFrame(quat_array, columns=quaternion_headers)

        position_path=Path(temp_dir, "position.csv")
        position_df.to_csv(position_path, index=False)

        quaternion_path=Path(temp_dir, "quaternion.csv")
        quat_df.to_csv(quaternion_path, index=False)

    # Step 2: Timestamps dat----------------------------------------
    capture_start=nc_info["start_timestamp_capture"]
    capture_end=nc_info["end_timestamp_capture"]

    file_txt=['Services system time timestamps\n',
              f'Capture start: {capture_start}\n',
              f'Capture   end: {capture_end}\n',
              '\n',
              'Frames:\n']
    with nc.Dataset(nc_path, format="NETCDF4") as f:
        group = f.groups["metadata"]["timing"]
        timestamps_srv = np.array(group.variables["timestamps_srv"][:])
        for idx,stamp in enumerate(timestamps_srv):
            current_txt='{:4d}'.format(idx)+', '+str(stamp)+'\n'
            file_txt.append(current_txt)

    timestamps_srv_path=Path(temp_dir,"timestamps_services.txt")
    with open(timestamps_srv_path, "w") as f:# write mode
        for line in file_txt:
            f.write(line)

    return nc_info
