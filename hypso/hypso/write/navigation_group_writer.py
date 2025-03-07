from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np

def navigation_group_writer(satobj, netfile: nc.Dataset) -> None:
    """
    Write navigation group to NetCDF file. 

    :return: Nothing.
    """

    # Create Navigation Group --------------------------------------
    navigation_group = netfile.createGroup('navigation')

    # Unix time -----------------------
    #time = netfile.createVariable('navigation/unixtime', 'u8', ('lines',))
    #time[:] = np.array(satobj.nc_timing_vars['timestamps'])
    #time[:] = np.array(satobj.nc_timing_vars['timestamps_srv']) # Previous
    #df = satobj.framepose_df
    #time[:] = df["timestamp"].values


    # Direct Georeferencing Latitudes and Longitudes
    if (hasattr(satobj, 'latitudes') and satobj.latitudes is not None) and \
        (hasattr(satobj, 'longitudes') and satobj.longitudes is not None):
        try:

            # Latitude ---------------------------------
            latitude = netfile.createVariable(
                'navigation/latitude', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            latitude[:] = satobj.latitudes
            latitude.long_name = "Latitude"
            latitude.units = "degrees"
            # latitude.valid_range = [-180, 180]
            latitude.valid_min = -180
            latitude.valid_max = 180

            # Longitude ----------------------------------
            longitude = netfile.createVariable(
                'navigation/longitude', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            longitude[:] = satobj.longitudes
            longitude.long_name = "Longitude"
            longitude.units = "degrees"
            # longitude.valid_range = [-180, 180]
            longitude.valid_min = -180
            longitude.valid_max = 180

        except Exception as ex:
            print("[ERROR] Unable to write direct latitude and longitude information to NetCDF file. The file may be incomplete. Please run direct or indirect georeferencing.")
            print("[ERROR] Encountered exception: " + str(ex))


    # Indirect Georeferencing Latitudes and Longitudes
    if (hasattr(satobj, 'latitudes_indirect') and satobj.latitudes_indirect is not None) and \
        (hasattr(satobj, 'longitudes_indirect') and satobj.longitudes_indirect is not None):
        try:

            # Latitude (Indirect) ---------------------------------
            latitude_indirect = netfile.createVariable(
                'navigation/latitude_indirect', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            latitude_indirect[:] = satobj.latitudes_indirect
            latitude_indirect.long_name = "Latitude (Indirect)"
            latitude_indirect.units = "degrees"
            # latitude_indirect.valid_range = [-180, 180]
            latitude_indirect.valid_min = -180
            latitude_indirect.valid_max = 180

            # Longitude (Indirect) ----------------------------------
            longitude_indirect = netfile.createVariable(
                'navigation/longitude_indirect', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            longitude_indirect[:] = satobj.longitudes_indirect
            longitude_indirect.long_name = "Longitude (Indirect)"
            longitude_indirect.units = "degrees"
            # longitude_indirect.valid_range = [-180, 180]
            longitude_indirect.valid_min = -180
            longitude_indirect.valid_max = 180

        except Exception as ex:
            print("[ERROR] Unable to write indirect latitude and longitude information to NetCDF file. The file may be incomplete. Please run direct or indirect georeferencing.")
            print("[ERROR] Encountered exception: " + str(ex))


    # Direct Georeferenicng Solar and Satellite Angles
    if (hasattr(satobj, 'latitudes') and satobj.latitudes is not None) and \
        (hasattr(satobj, 'longitudes') and satobj.longitudes is not None):
        try:
            # Sensor Zenith --------------------------
            sensor_z = netfile.createVariable(
                'navigation/sensor_zenith', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_z[:] = satobj.sat_zenith_angles
            sensor_z.long_name = "Sensor Zenith Angle"
            sensor_z.units = "degrees"
            # sensor_z.valid_range = [-180, 180]
            sensor_z.valid_min = -180
            sensor_z.valid_max = 180

            # Sensor Azimuth ---------------------------
            sensor_a = netfile.createVariable(
                'navigation/sensor_azimuth', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_a[:] = satobj.sat_azimuth_angles
            sensor_a.long_name = "Sensor Azimuth Angle"
            sensor_a.units = "degrees"
            # sensor_a.valid_range = [-180, 180]
            sensor_a.valid_min = -180
            sensor_a.valid_max = 180

            # Solar Zenith ----------------------------------------
            solar_z = netfile.createVariable(
                'navigation/solar_zenith', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            solar_z[:] = satobj.solar_zenith_angles
            solar_z.long_name = "Solar Zenith Angle"
            solar_z.units = "degrees"
            # solar_z.valid_range = [-180, 180]
            solar_z.valid_min = -180
            solar_z.valid_max = 180

            # Solar Azimuth ---------------------------------------
            solar_a = netfile.createVariable(
            'navigation/solar_azimuth', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            solar_a[:] = satobj.solar_azimuth_angles
            solar_a.long_name = "Solar Azimuth Angle"
            solar_a.units = "degrees"
            # solar_a.valid_range = [-180, 180]
            solar_a.valid_min = -180
            solar_a.valid_max = 180

            # Relative Azimuth ---------------------------------------
            relative_a = netfile.createVariable(
            'navigation/relative_azimuth', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            relative_a[:] = satobj.relative_azimuth_angles
            relative_a.long_name = "Relative Azimuth Angle"
            relative_a.units = "degrees"
            # relative_a.valid_range = [-180, 180]
            relative_a.valid_min = -180
            relative_a.valid_max = 180

        except Exception as ex:
            print("[ERROR] Unable to write navigation angles to NetCDF file. The file may be incomplete. Please run geometry computations.")
            print("[ERROR] Encountered exception: " + str(ex))


    # Indirect Georeferenicng Solar and Satellite Angles
    if (hasattr(satobj, 'latitudes_indirect') and satobj.latitudes_indirect is not None) and \
        (hasattr(satobj, 'longitudes_indirect') and satobj.longitudes_indirect is not None):
        try:

            # Sensor Zenith (Indirect)--------------------------
            sensor_z_indirect = netfile.createVariable(
                'navigation/sensor_zenith_indirect', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_z_indirect[:] = satobj.sat_zenith_angles_indirect
            sensor_z_indirect.long_name = "Sensor Zenith Angle (Indirect)"
            sensor_z_indirect.units = "degrees"
            # sensor_z_indirect.valid_range = [-180, 180]
            sensor_z_indirect.valid_min = -180
            sensor_z_indirect.valid_max = 180

            # Sensor Azimuth (Indirect) ---------------------------
            sensor_a_indirect = netfile.createVariable(
                'navigation/sensor_azimuth_indirect', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_a_indirect[:] = satobj.sat_azimuth_angles_indirect
            sensor_a_indirect.long_name = "Sensor Azimuth Angle (Indirect)"
            sensor_a_indirect.units = "degrees"
            # sensor_a_indirect.valid_range = [-180, 180]
            sensor_a_indirect.valid_min = -180
            sensor_a_indirect.valid_max = 180

            # Solar Zenith (Indirect) ----------------------------------------
            solar_z_indirect = netfile.createVariable(
                'navigation/solar_zenith_indirect', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            solar_z_indirect[:] = satobj.solar_zenith_angles_indirect
            solar_z_indirect.long_name = "Solar Zenith Angle (Indirect)"
            solar_z_indirect.units = "degrees"
            # solar_z_indirect.valid_range = [-180, 180]
            solar_z_indirect.valid_min = -180
            solar_z_indirect.valid_max = 180

            # Solar Azimuth (Indirect) ---------------------------------------
            solar_a_indirect = netfile.createVariable(
            'navigation/solar_azimuth_indirect', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            solar_a_indirect[:] = satobj.solar_azimuth_angles_indirect
            solar_a_indirect.long_name = "Solar Azimuth Angle (Indirect)"
            solar_a_indirect.units = "degrees"
            # solar_a_indirect.valid_range = [-180, 180]
            solar_a_indirect.valid_min = -180
            solar_a_indirect.valid_max = 180

            # Relative Azimuth (Indirect) ---------------------------------------
            relative_a_indirect = netfile.createVariable(
            'navigation/relative_azimuth_indirect', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            relative_a_indirect[:] = satobj.relative_azimuth_angles_indirect
            relative_a_indirect.long_name = "Relative Azimuth Angle (Indirect)"
            relative_a_indirect.units = "degrees"
            # relative_a_indirect.valid_range = [-180, 180]
            relative_a_indirect.valid_min = -180
            relative_a_indirect.valid_max = 180


        except Exception as ex:
            pass


    return None