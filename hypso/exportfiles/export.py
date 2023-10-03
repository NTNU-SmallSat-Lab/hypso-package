import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone


def write_h1data_as_geojson(sat_object, path_to_save: str) -> None:
    # check if file ends with .geojson
    if not path_to_save.endswith('.geojson'):
        path_to_save = path_to_save + '.geojson'

    with open(path_to_save, 'w') as f:
        f.write(get_geojson_str(sat_object))


def write_h1data_as_NetCDF4(sat_object, path_to_save: str) -> None:
    DEBUG = True
    """
    Write the HYPSO-1 data as a NetCDF4 file.

    Args:
        path_to_save (str): The path to save the NetCDF4 file.
        path_to_h1data (str): The path to the HYPSO-1 data.

    Raises:
        ValueError: If the NetCDF4 file already exists.
    """

    # check if file ends with .nc
    if not path_to_save.endswith('.nc'):
        path_to_save = path_to_save + '.nc'

    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    h1 = sat_object
    path_to_h1data = sat_object.info["top_folder_name"]

    temp = h1.info["folder_name"].split("CaptureDL_")[-1]
    name = temp.split("_")[0]
    frames = h1.info["frame_count"]
    lines = h1.info["image_height"]
    bands = h1.info["image_width"]

    if DEBUG:
        print(h1.info)

    sata, satz, suna, sunz, lat, lon = get_nav_and_view(path_to_h1data)

    # get unixtimes
    posetime_file = None
    for file in os.listdir(path_to_h1data):
        if "frametime-pose" in file:
            posetime_file = os.path.join(path_to_h1data, file)
            break
    if posetime_file is None:
        raise ValueError("frametime-pose file is not available.")

    df = pd.read_csv(posetime_file)
    # df.columns = df.iloc[0]
    # df = df[1:]

    if DEBUG:
        print(df)

    # Create a new NetCDF file
    with nc.Dataset(path_to_save, 'w', format='NETCDF4') as f:

        f.instrument = "HYPSO-1 Hyperspectral Imager"
        f.institution = "Norwegian University of Science and Technology"
        f.resolution = "N/A"
        f.location_description = name
        f.license = "TBD"
        f.naming_authority = "NTNU SmallSat Lab"
        f.date_processed = datetime.utcnow().isoformat() + "Z"
        f.date_aquired = h1.info["iso_time"] + "Z"
        f.publisher_name = "NTNU SmallSat Lab"
        f.publisher_url = "https://hypso.space"
        # f.publisher_contact = "smallsat@ntnu.no"
        f.processing_level = "L1B"
        f.radiometric_file = str(h1.correction_coeffs_file_dict["radiometric"])
        f.smile_file = str(h1.correction_coeffs_file_dict["smile"])
        f.destriping_file = str(h1.correction_coeffs_file_dict["destriping"])
        f.spectral_file = str(h1.spectral_coeff_file)
        # Create dimensions
        f.createDimension('frames', frames)
        f.createDimension('lines', lines)
        f.createDimension('bands', bands)

        # create groups
        navigation_group = f.createGroup('navigation')

        navigation_group.iso8601time = h1.info["iso_time"] + "Z"

        # Create variables
        COMP_SCHEME = 'zlib'  # Default: zlib
        COMP_LEVEL = 4  # Default (when scheme != none): 4
        COMP_SHUFFLE = True  # Default (when scheme != none): True

        time = f.createVariable('navigation/unixtime', 'u8', ('frames',))
        time[:] = df["timestamp"].values

        sensor_z = f.createVariable(
            'navigation/sensor_zenith', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        sensor_z[:] = satz.reshape(frames, lines)
        sensor_z.long_name = "Sensor Zenith Angle"
        sensor_z.units = "degrees"
        # sensor_z.valid_range = [-180, 180]
        sensor_z.valid_min = -180
        sensor_z.valid_max = 180

        sensor_a = f.createVariable(
            'navigation/sensor_azimuth', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        sensor_a[:] = sata.reshape(frames, lines)
        sensor_a.long_name = "Sensor Azimuth Angle"
        sensor_a.units = "degrees"
        # sensor_a.valid_range = [-180, 180]
        sensor_a.valid_min = -180
        sensor_a.valid_max = 180

        solar_z = f.createVariable(
            'navigation/solar_zenith', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        solar_z[:] = sunz.reshape(frames, lines)
        solar_z.long_name = "Solar Zenith Angle"
        solar_z.units = "degrees"
        # solar_z.valid_range = [-180, 180]
        solar_z.valid_min = -180
        solar_z.valid_max = 180

        solar_a = f.createVariable(
            'navigation/solar_azimuth', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        solar_a[:] = suna.reshape(frames, lines)
        solar_a.long_name = "Solar Azimuth Angle"
        solar_a.units = "degrees"
        # solar_a.valid_range = [-180, 180]
        solar_a.valid_min = -180
        solar_a.valid_max = 180

        latitude = f.createVariable(
            'navigation/latitude', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        latitude[:] = lat.reshape(frames, lines)
        latitude.long_name = "Latitude"
        latitude.units = "degrees"
        # latitude.valid_range = [-180, 180]
        latitude.valid_min = -180
        latitude.valid_max = 180

        longitude = f.createVariable(
            'navigation/longitude', 'f4', ('frames', 'lines'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        longitude[:] = lon.reshape(frames, lines)
        longitude.long_name = "Longitude"
        longitude.units = "degrees"
        # longitude.valid_range = [-180, 180]
        longitude.valid_min = -180
        longitude.valid_max = 180

        f.createGroup('products')
        Lt = f.createVariable('products/Lt', 'f4',
                              ('frames', 'lines', 'bands'),
                              compression=COMP_SCHEME,
                              complevel=COMP_LEVEL,
                              shuffle=COMP_SHUFFLE,
                              # least_significant_digit=5, #Truncate data for extra compression. At 12 bits, 5 sigdigs should suffice?
                              )  # Default: lvl 4 w/ shuffling
        Lt.units = "W/m^2/micrometer/sr"
        Lt.long_name = "Top of Atmosphere Measured Radiance"
        Lt.wavelength_units = "nanometers"
        Lt.fwhm = [5.5] * bands
        Lt.wavelengths = np.around(h1.spectral_coefficients, 1)
        Lt[:] = h1.l1b_cube

        # NASA scan_lines_attributes:
        # Don't know if this is at all necessary
        # ...or correct, for that matter
        f.createGroup('scan_line_attributes')
        scan_quality_flags = f.createVariable('scan_line_attributes/scan_quality_flags',
                                              'uint8', ('frames', 'lines'))
        scan_quality_flags[:] = 255

        # NASA metadata
        # I'm assuming this is necessary for use with SNAP,
        # and part of the NASA CDF standard
        metadata_group = f.createGroup('metadata')

        meta_fgdc_group = f.createGroup('metadata/FGDC')
        meta_fgdc_ident_group = f.createGroup(
            'metadata/FGDC/Indentification_Information')
        meta_fgdc_ident_platform_group = f.createGroup(
            'metadata/FGDC/Identification_Information/Platform_and_Instrument_Identification')
        meta_fgdc_ident_platform_group.Instrument_Short_Name = 'hypso'

        meta_fgdc_ident_prclvl_group = f.createGroup(
            'metadata/FGDC/Identification_Information/Processing_Level')
        meta_fgdc_ident_prclvl_group.Processing_Level_Identifier = 'Level-1B'

        meta_fgdc_ident_time_group = f.createGroup(
            'metadata/FGDC/Identification_Information/Time_Period_of_Content')
        # will obvs need to be calculated from somewhere
        starttimestamp = datetime.fromtimestamp(time[0], timezone.utc)
        endtimestamp = datetime.fromtimestamp(time[-1], timezone.utc)
        meta_fgdc_ident_time_group.Beginning_Date = starttimestamp.date().isoformat()
        meta_fgdc_ident_time_group.Ending_Date = endtimestamp.date().isoformat()
        meta_fgdc_ident_time_group.Beginning_Time = starttimestamp.time().isoformat() + "Z"
        meta_fgdc_ident_time_group.Ending_Time = endtimestamp.time().isoformat() + "Z"

        # Hunch that we don't actually need this
        meta_hypso_calib_group = f.createGroup('metadata/HYPSO/Calibration')
        # Resolves to '+XVV' for HICO
        meta_hypso_calib_group.hypso_orientation_from_quaternion = '+XVV'


def print_nc(nc_file_path):
    recursive_print_nc(nc.Dataset(nc_file_path, format="NETCDF4"))


def recursive_print_nc(nc_file, path='', depth=0):

    indent = ''
    for i in range(depth):
        indent += '  '

    print(indent, '--- GROUP: "', path + nc_file.name, '" ---', sep='')

    print(indent, 'DIMENSIONS: ', sep='', end='')
    for d in nc_file.dimensions.keys():
        print(d, end=', ')
    print('')
    print(indent, 'VARIABLES: ', sep='', end='')
    for v in nc_file.variables.keys():
        print(v, end=', ')
    print('')

    print(indent, 'ATTRIBUTES: ', sep='', end='')
    for a in nc_file.ncattrs():
        print(a, end=', ')
    print('')

    print(indent, 'SUB-GROUPS: ', sep='', end='')
    for g in nc_file.groups.keys():
        print(g, end=', ')
    print('')
    print('')

    for g in nc_file.groups.keys():
        if nc_file.name == '/':
            newname = path + nc_file.name
        else:
            newname = path + nc_file.name + '/'
        recursive_print_nc(nc_file.groups[g], path=newname, depth=depth + 1)


# For Exporting on different Format

def get_geojson_str(sat_object) -> str:
    """Write the geojson metadata file.

    Args:
        writingmode (str, optional): The writing mode. Defaults to "w".

    Raises:
        ValueError: If the position file could not be found.
    """
    geojsondict = get_geojson_dict(sat_object)
    geojsonstr = json.dumps(geojsondict)

    return geojsonstr


def get_geojson_dict(sat_object) -> dict:
    """
    Get the data as a geojson dictionary.

    Returns:
        dict: The geojson dictionary.
    """

    # convert dictionary to json
    geojsondict = {}

    geojsondict["type"] = "Feature"

    geojsondict["geometry"] = {}
    geojsondict["geometry"]["type"] = "Point"
    geojsondict["geometry"]["coordinates"] = [
        sat_object.info["lon"], sat_object.info["lat"]]

    geojsondict["properties"] = {}
    name = sat_object.info["folder_name"].split("CaptureDL_")[-1].split("_")[0]
    geojsondict["properties"]["name"] = name
    geojsondict["properties"]["path"] = sat_object.info["top_folder_name"]

    geojsondict["metadata"] = {}
    date = sat_object.info["folder_name"].split(
        "CaptureDL_")[-1].split("20")[1].split("T")[0]
    date = f"20{date}"

    try:
        timestamp = datetime.strptime(date, "%Y-%m-%d_%H%MZ").isoformat()
    except BaseException:
        timestamp = datetime.strptime(date, "%Y-%m-%d").isoformat()

    geojsondict["metadata"]["timestamp"] = timestamp + "Z"
    geojsondict["metadata"]["frames"] = sat_object.info["frame_count"]
    geojsondict["metadata"]["bands"] = sat_object.info["image_width"]
    geojsondict["metadata"]["lines"] = sat_object.info["image_height"]
    geojsondict["metadata"]["satellite"] = "HYPSO-1"

    geojsondict["metadata"]["rad_coeff"] = os.path.basename(
        sat_object.radiometric_coeff_file)
    geojsondict["metadata"]["spec_coeff"] = os.path.basename(
        sat_object.spectral_coeff_file)

    geojsondict["metadata"]["solar_zenith_angle"] = sat_object.info["solar_zenith_angle"]
    geojsondict["metadata"]["solar_azimuth_angle"] = sat_object.info["solar_azimuth_angle"]
    geojsondict["metadata"]["sat_zenith_angle"] = sat_object.info["sat_zenith_angle"]
    geojsondict["metadata"]["sat_azimuth_angle"] = sat_object.info["sat_azimuth_angle"]

    return geojsondict


def get_nav_and_view(path_to_h1data):
    is_nav_data_available = False
    for file in os.listdir(path_to_h1data):
        if "sat-azimuth.dat" in file:
            sata = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
            is_nav_data_available = True
        elif "sat-zenith.dat" in file:
            satz = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "sun-azimuth.dat" in file:
            suna = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "sun-zenith.dat" in file:
            sunz = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "latitudes.dat" in file:
            lat = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)
        elif "longitudes.dat" in file:
            lon = np.fromfile(os.path.join(
                path_to_h1data, file), dtype=np.float32)

    if not is_nav_data_available:
        raise ValueError("Navigation data is not available.")
    return sata, satz, suna, sunz, lat, lon
