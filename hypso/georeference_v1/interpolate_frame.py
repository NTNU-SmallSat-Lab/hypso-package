import numpy as np
import pandas as pd
import scipy.interpolate as si
from pathlib import Path


#################################
############# SETUP #############
#################################

def interpolate_at_frame(pos_csv_path, quat_csv_path, flash_csv_path, additional_time_offset=0.0, framerate=-1.0,
                         exposure=-1.0) -> None:
    """
    Function to interpolate at the frame based on the quaternion, position and timestamps

    :param pos_csv_path:
    :param quat_csv_path:
    :param flash_csv_path:
    :param additional_time_offset:
    :param framerate:
    :param exposure:

    :return:
    """

    # print('This script requires paths to three files:', \
    #       '\n  1. a satellite position timeseries .csv file', \
    #       '\n  2. a satellite attitude quaternion timeseries .csv file', \
    #       '\n  3. a frame timestamps file (either timestamps.txt or timestamps_services.txt)', \
    #       '\n  as first, second and third arg in this order.\n')
    #
    # print('  Optionally provide a frame timestamp offset in seconds, the capture')
    # print('  framerate and exposure time in ms as fourth, fifth and sixth argument.')
    # print('  Providing framerate and exposure time extrapolates the timestamps, as')
    # print('  opposed to using the timestamps file.\n')
    #
    # print('  (framerate and exposure time must be specified BOTH, at the same')
    # print('  time, if only framerate is given, it is ignored)')
    #
    #
    # '''
    # The position timeseries .csv file shall look like
    # time, eci x [m], eci y [m], eci z [m]
    # 1648507140.0, 5050500, -1800000, 4545454
    #
    # The quaternion timeseries .csv file shall look like (q1 is scalar part)
    # time, q1, q2, q3, q4
    # 1648507140.0, 0.9, 0.03, 0.06, 0.3
    #
    # And the frame timestamps file shall be taken
    # from the metadata of an hsi capture, either 'timestamps.txt' or 'timestamps_services.txt'
    # '''

    print(f'  Position timeseries path:',
          f'\n    {pos_csv_path}',
          f'\n  Quaternion timeseries path:',
          f'\n    {quat_csv_path}',
          f'\n  Frame timestamps path:',
          f'\n    {flash_csv_path}\n')

    ######################################
    ############# SETUP DONE #############
    ######################################

    # 1. Reading .csv file with ECI position info
    posdata = pd.read_csv(pos_csv_path)
    pos_time_col_index = 0
    eci_x_col_index = 1
    eci_y_col_index = 2
    eci_z_col_index = 3

    print('  eci position samples', posdata.shape[0])

    # 2. Reading .csv file with GPS info
    quatdata = pd.read_csv(quat_csv_path)
    quat_time_col_index = 0
    q0_col_index = 1
    q1_col_index = 2
    q2_col_index = 3
    q3_col_index = 4

    print('  quaternion samples', quatdata.shape[0])

    # 3. Reading frame timestamps
    flashtimes = None
    frame_count = None
    if 'timestamps.txt' in flash_csv_path.name:
        capture_unix_ts = 100000000.0
        # Read starting unix timestamp from second line of file
        with open(flash_csv_path, 'r') as flash_ts_file:
            topline = flash_ts_file.readline()
            unix_ts_str = flash_ts_file.readline()
            capture_unix_ts = float(unix_ts_str)

        # reading .csv file with frame time info
        flashdata = pd.read_csv(flash_csv_path, header=None, skiprows=4)
        flash_time_col_index = 1

        flashdata_np = flashdata.values
        # stripping trailing zero values
        flashdata_np = flashdata_np[flashdata_np[:, flash_time_col_index] != 0, :]
        frame_count = flashdata_np.shape[0]

        flashtimes = flashdata_np[:, flash_time_col_index].astype('float64') / 1000000.0 + capture_unix_ts
    elif 'timestamps_services.txt' in flash_csv_path.name:

        # reading .csv file with frame time info
        flashdata = pd.read_csv(flash_csv_path, header=None, skiprows=5)
        flash_time_col_index = 1
        flashdata_np = flashdata.values
        frame_count = flashdata_np.shape[0]
        flashtimes = flashdata_np[:, flash_time_col_index]

    flashtimes = flashtimes + additional_time_offset
    ## workaround for making the timestamps smoother. Does not actually smooth out the data.
    if framerate > 0.0:
        starttime = flashtimes[0]
        for i in range(flashtimes.shape[0]):
            flashtimes[i] = starttime - exposure / (2 * 1000.0) + i / framerate

    # print('')
    # print('  Frame timestmps start', flashtimes[0], flashtimes[1], flashtimes[2])
    # print('  Frame timestmps end  ', flashtimes[-1], flashtimes[-2], flashtimes[-3])

    # Inclusion of frame times in ADCS telemetry time series check

    # ADCS data time boundary
    adcs_ts_start = posdata.iloc[0, 0]
    adcs_ts_end = posdata.iloc[-1, 0]

    frame_ts_start = flashtimes[0]
    frame_ts_end = flashtimes[-1]

    print(f'\n   ADCS time range: {adcs_ts_start:17.6f} to {adcs_ts_end:17.6f}')
    print(f'  Frame time range: {frame_ts_start:17.6f} to {frame_ts_end:17.6f}\n')

    if frame_ts_start < adcs_ts_start:
        print('ERROR: Frame timestamps begin earlier than ADCS data!')
        exit(-1)
    if frame_ts_end > adcs_ts_end:
        print('ERROR: Frame timestamps end later than ADCS data!')
        exit(-1)

    a = quatdata.values[:, 0] > flashtimes[0]
    b = quatdata.values[:, 0] < flashtimes[-1]
    # print(a)
    # print(b)
    # print(a & b)
    print(f'  {np.sum(a & b)} sample(s) inside frame time range')
    # print(np.diff(quatdata.values[:,0]))

    # print('Satellite positions:')
    # print(posdata)
    # print('Satellite orientations:')
    # print(quatdata)

    # print('Frame timestamps:')
    # print(flashtimes[0:9])
    # print('...')
    # print(flashtimes[-9:-1])
    # print('  Frames in log:', flashtimes.shape[0])
    print(f'  Interpolating {frame_count:d} frames')

    # USE THIS IF TIME IS GIVEN AS A DATETIME STRING
    # posdata_time_unix = np.zeros(posdata.shape[0]).astype('float64')
    # for i, dto in enumerate(posdata.values[:,pos_time_col_index]):
    #    dt = pd.to_datetime(dto)
    #    dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
    #    posdata_time_unix[i] = dt_utc.timestamp()

    # USE THIS IF TIME IS GIVEN AS A UNIX TIMESTAMP
    posdata_time_unix = posdata.values[:, pos_time_col_index].astype(np.float64)

    posdata_eci_x = posdata.values[:, eci_x_col_index].astype(np.float64)
    posdata_eci_y = posdata.values[:, eci_y_col_index].astype(np.float64)
    posdata_eci_z = posdata.values[:, eci_z_col_index].astype(np.float64)

    # USE THIS IF TIME IS GIVEN AS A DATETIME STRING
    # quatdata_time_unix = np.zeros(quatdata.shape[0]).astype('float64')
    # for i, dto in enumerate(posdata.values[:,quat_time_col_index]):
    #    dt = pd.to_datetime(dto)
    #    dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
    #    quatdata_time_unix[i] = dt_utc.timestamp()

    # USE THIS IF TIME IS GIVEN AS A UNIX TIMESTAMP
    quatdata_time_unix = quatdata.values[:, quat_time_col_index].astype(np.float64)

    quatdata_q0 = quatdata.values[:, q0_col_index].astype(np.float64)
    quatdata_q1 = quatdata.values[:, q1_col_index].astype(np.float64)
    quatdata_q2 = quatdata.values[:, q2_col_index].astype(np.float64)
    quatdata_q3 = quatdata.values[:, q3_col_index].astype(np.float64)

    ##############################################################################

    interpolation_method = 'linear'  # 'cubic' causes errors sometimes: "ValueError: Expect x to not have duplicates"

    posdata_eci_x_interp = si.griddata(posdata_time_unix, posdata_eci_x, flashtimes, method=interpolation_method)
    posdata_eci_y_interp = si.griddata(posdata_time_unix, posdata_eci_y, flashtimes, method=interpolation_method)
    posdata_eci_z_interp = si.griddata(posdata_time_unix, posdata_eci_z, flashtimes, method=interpolation_method)

    quatdata_q0_interp = si.griddata(quatdata_time_unix, quatdata_q0, flashtimes, method=interpolation_method)
    quatdata_q1_interp = si.griddata(quatdata_time_unix, quatdata_q1, flashtimes, method=interpolation_method)
    quatdata_q2_interp = si.griddata(quatdata_time_unix, quatdata_q2, flashtimes, method=interpolation_method)
    quatdata_q3_interp = si.griddata(quatdata_time_unix, quatdata_q3, flashtimes, method=interpolation_method)

    # print(quatdata_q0.shape, quatdata_q0_interp.shape)

    flashindices = np.linspace(0, frame_count - 1, frame_count).astype(np.int32)

    data = {
        'timestamp': flashtimes,
        'frame index': flashindices,
        'eci x [m]': posdata_eci_x_interp,
        'eci y [m]': posdata_eci_y_interp,
        'eci z [m]': posdata_eci_z_interp,
        'q_eta': quatdata_q0_interp,
        'q_e1': quatdata_q1_interp,
        'q_e2': quatdata_q2_interp,
        'q_e3': quatdata_q3_interp
    }
    datapandas = pd.DataFrame(data)
    # print('Interpolated data:')
    # print(datapandas)

    output_path = Path(pos_csv_path.parent.absolute(), 'frametime-pose.csv')
    datapandas.to_csv(output_path, index=False)

    print('Done')
