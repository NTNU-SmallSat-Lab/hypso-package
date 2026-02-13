from .utils import set_or_create_attr
from pathlib import Path
import netCDF4 as nc
import numpy as np

def geometry_group_writer(satobj, netfile: nc.Dataset, COMP_SCHEME = 'zlib', COMP_LEVEL = 4, COMP_SHUFFLE = True) -> None:
    """
    Write geometry group to NetCDF file. 

    :return: Nothing.
    """

    # Create geometry Group --------------------------------------
    geometry_group = netfile.createGroup('geometry')

    # Unix time -----------------------
    #time = netfile.createVariable('geometry/unixtime', 'u8', ('lines',))
    #time[:] = np.array(satobj.nc_timing_vars['timestamps'])
    #time[:] = np.array(satobj.nc_timing_vars['timestamps_srv']) # Previous
    #df = satobj.framepose_df
    #time[:] = df["timestamp"].values


    # Georeferencing Latitudes and Longitudes
    if (hasattr(satobj, 'latitudes') and satobj.latitudes is not None) and \
        (hasattr(satobj, 'longitudes') and satobj.longitudes is not None):
        try:

            # Latitude ---------------------------------
            latitude = netfile.createVariable(
                'geometry/latitude', 'f4', ('lines', 'samples'),
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
                'geometry/longitude', 'f4', ('lines', 'samples'),
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


    # Direct Georeferencing Latitudes and Longitudes
    if (hasattr(satobj, 'latitudes_direct') and satobj.latitudes_direct is not None) and \
        (hasattr(satobj, 'longitudes_direct') and satobj.longitudes_direct is not None):
        try:

            # Latitude (Indirect) ---------------------------------
            latitude_direct = netfile.createVariable(
                'geometry/latitude_direct', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            latitude_direct[:] = satobj.latitudes_direct
            latitude_direct.long_name = "Latitude (Indirect)"
            latitude_direct.units = "degrees"
            # latitude_direct.valid_range = [-180, 180]
            latitude_direct.valid_min = -180
            latitude_direct.valid_max = 180

            # Longitude (Indirect) ----------------------------------
            longitude_direct = netfile.createVariable(
                'geometry/longitude_direct', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            longitude_direct[:] = satobj.longitudes_direct
            longitude_direct.long_name = "Longitude (Indirect)"
            longitude_direct.units = "degrees"
            # longitude_direct.valid_range = [-180, 180]
            longitude_direct.valid_min = -180
            longitude_direct.valid_max = 180

        except Exception as ex:
            print("[ERROR] Unable to write indirect latitude and longitude information to NetCDF file. The file may be incomplete. Please run direct or indirect georeferencing.")
            print("[ERROR] Encountered exception: " + str(ex))


    # Direct Georeferenicng Solar and Satellite Angles
    if (hasattr(satobj, 'latitudes') and satobj.latitudes is not None) and \
        (hasattr(satobj, 'longitudes') and satobj.longitudes is not None):

        try:
            # Sensor Zenith --------------------------
            sensor_z = netfile.createVariable(
                'geometry/sensor_zenith', 'f4', ('lines', 'samples'),
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
        except Exception as ex:
            pass

        try:
            # Sensor Azimuth ---------------------------
            sensor_a = netfile.createVariable(
                'geometry/sensor_azimuth', 'f4', ('lines', 'samples'),
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
        except Exception as ex:
            pass

        try:
            # Solar Zenith ----------------------------------------
            solar_z = netfile.createVariable(
                'geometry/solar_zenith', 'f4', ('lines', 'samples'),
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
        except Exception as ex:
            pass

        try:
            # Solar Azimuth ---------------------------------------
            solar_a = netfile.createVariable(
            'geometry/solar_azimuth', 'f4', ('lines', 'samples'),
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
        except Exception as ex:
            pass

        try:
            # Relative Azimuth ---------------------------------------
            relative_a = netfile.createVariable(
            'geometry/relative_azimuth', 'f4', ('lines', 'samples'),
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
            pass



    # Indirect Georeferenicng Solar and Satellite Angles
    if (hasattr(satobj, 'latitudes_direct') and satobj.latitudes_direct is not None) and \
        (hasattr(satobj, 'longitudes_direct') and satobj.longitudes_direct is not None):
        
        try:
            # Sensor Zenith (Indirect)--------------------------
            sensor_z_direct = netfile.createVariable(
                'geometry/sensor_zenith_direct', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_z_direct[:] = satobj.sat_zenith_angles_direct
            sensor_z_direct.long_name = "Sensor Zenith Angle (Indirect)"
            sensor_z_direct.units = "degrees"
            # sensor_z_direct.valid_range = [-180, 180]
            sensor_z_direct.valid_min = -180
            sensor_z_direct.valid_max = 180

        except Exception as ex:
            pass

        try:
            # Sensor Azimuth (Indirect) ---------------------------
            sensor_a_direct = netfile.createVariable(
                'geometry/sensor_azimuth_direct', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            sensor_a_direct[:] = satobj.sat_azimuth_angles_direct
            sensor_a_direct.long_name = "Sensor Azimuth Angle (Indirect)"
            sensor_a_direct.units = "degrees"
            # sensor_a_direct.valid_range = [-180, 180]
            sensor_a_direct.valid_min = -180
            sensor_a_direct.valid_max = 180
        except Exception as ex:
            pass

        try:
            # Solar Zenith (Indirect) ----------------------------------------
            solar_z_direct = netfile.createVariable(
                'geometry/solar_zenith_direct', 'f4', ('lines', 'samples'),
                # compression=COMP_SCHEME,
                # complevel=COMP_LEVEL,
                # shuffle=COMP_SHUFFLE,
            )
            solar_z_direct[:] = satobj.solar_zenith_angles_direct
            solar_z_direct.long_name = "Solar Zenith Angle (Indirect)"
            solar_z_direct.units = "degrees"
            # solar_z_direct.valid_range = [-180, 180]
            solar_z_direct.valid_min = -180
            solar_z_direct.valid_max = 180
        except Exception as ex:
            pass

        try:
            # Solar Azimuth (Indirect) ---------------------------------------
            solar_a_direct = netfile.createVariable(
            'geometry/solar_azimuth_direct', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            solar_a_direct[:] = satobj.solar_azimuth_angles_direct
            solar_a_direct.long_name = "Solar Azimuth Angle (Indirect)"
            solar_a_direct.units = "degrees"
            # solar_a_direct.valid_range = [-180, 180]
            solar_a_direct.valid_min = -180
            solar_a_direct.valid_max = 180
        except Exception as ex:
            pass

        try:
            # Relative Azimuth (Indirect) ---------------------------------------
            relative_a_direct = netfile.createVariable(
            'geometry/relative_azimuth_direct', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
            )
            relative_a_direct[:] = satobj.relative_azimuth_angles_direct
            relative_a_direct.long_name = "Relative Azimuth Angle (Indirect)"
            relative_a_direct.units = "degrees"
            # relative_a_direct.valid_range = [-180, 180]
            relative_a_direct.valid_min = -180
            relative_a_direct.valid_max = 180
        except Exception as ex:
            pass

    return None