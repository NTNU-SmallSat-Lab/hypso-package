from .utils import set_or_create_attr
from hypso import Hypso
from pathlib import Path
import netCDF4 as nc
import numpy as np

def navigation_group_writer(satobj: Hypso, netfile: nc.Dataset, product_level: str) -> None:
    """
    Write navigation group to NetCDF file. 

    :return: Nothing.
    """

    # Create Navigation Group --------------------------------------
    navigation_group = netfile.createGroup('navigation')

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
        print("[ERROR] Unable to write latitude and longitude information to NetCDF file. The {0} file may be incomplete. Please run direct or indirect georeferencing.".format(product_level))
        print("[ERROR] Encountered exception: " + str(ex))


    try:
        sat_zenith_angle = satobj.sat_zenith_angles
        sat_azimuth_angle = satobj.sat_azimuth_angles

        solar_zenith_angle = satobj.solar_zenith_angles
        solar_azimuth_angle = satobj.solar_azimuth_angles

        # Unix time -----------------------
        time = netfile.createVariable('navigation/unixtime', 'u8', ('lines',))
        time[:] = np.array(satobj.timing_vars['timestamps_srv'])
        #df = satobj.framepose_df
        #time[:] = df["timestamp"].values

        
        # Sensor Zenith --------------------------
        sensor_z = netfile.createVariable(
            'navigation/sensor_zenith', 'f4', ('lines', 'samples'),
            # compression=COMP_SCHEME,
            # complevel=COMP_LEVEL,
            # shuffle=COMP_SHUFFLE,
        )
        sensor_z[:] = sat_zenith_angle.reshape(satobj.spatial_dimensions)
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
        sensor_a[:] = sat_azimuth_angle.reshape(satobj.spatial_dimensions)
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
        solar_z[:] = solar_zenith_angle.reshape(satobj.spatial_dimensions)
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
        solar_a[:] = solar_azimuth_angle.reshape(satobj.spatial_dimensions)
        solar_a.long_name = "Solar Azimuth Angle"
        solar_a.units = "degrees"
        # solar_a.valid_range = [-180, 180]
        solar_a.valid_min = -180
        solar_a.valid_max = 180

    except Exception as ex:
        print("[ERROR] Unable to write navigation angles to NetCDF file. The {0} file may be incomplete. Please run geometry computations.".format(product_level))
        print("[ERROR] Encountered exception: " + str(ex))

    return None