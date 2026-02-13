import os
from osgeo import gdal # install with `pip install gdal==3.8.4`
import numpy as np
from importlib.resources import files
from pathlib import Path
import Py6S
import dateutil
from .ac_6sv1_dem import MeanDEM
from .ac_6sv1_utils import get_lat_lon, get_image_extent_lat_lon, get_image_center_lat_lon



def get_6sv1_solar_angles_parameters(satobj, parameters):

    solar_azimuth_angles = satobj.solar_azimuth_angles
    solar_zenith_angles = satobj.solar_zenith_angles

    # Solar zenith angle, azimuth (average)
    parameters["SolarAzimuthAngle"] = np.mean(solar_azimuth_angles)
    parameters["SolarZenithAngle"] = np.mean(solar_zenith_angles)
    
    return parameters

def get_6sv1_datetime_parameters(satobj, parameters):

    iso_time = satobj.iso_time
    
    parameters['time'] = dateutil.parser.parse(iso_time)

    Date = dateutil.parser.isoparse(iso_time)
    parameters["ImgMonth"] = int(Date.month)
    parameters["ImgDay"] = int(Date.day)

    return parameters



def get_6sv1_wavelengths_parameters(satobj, parameters):

    wavelengths = satobj.wavelengths

    parameters['wavelengths'] = wavelengths

    return parameters


def get_6sv1_sensor_angles_parameters(satobj, parameters):


    sat_azimuth_angles = np.mean(satobj.sat_azimuth_angles)
    sat_zenith_angles = np.mean(satobj.sat_zenith_angles)

    parameters["SatAzimuthAngles"] = sat_azimuth_angles
    parameters["SatZenithAngles"] = sat_zenith_angles
    
    return parameters


def get_6sv1_relative_azimuth_angles_parameters(satobj, parameters):


    relative_azimuth_angles = np.mean(satobj.relative_azimuth_angles)

    parameters["RelativeAzimuthAngles"] = relative_azimuth_angles

    return parameters








def get_6sv1_atmospheric_profile_lut():

    # data from Table 2-2 in http://www.exelisvis.com/docs/FLAASH.html
    SAW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)
    SAS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
    MLS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
    MLW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)
    T = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    ap_JFMA = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: MLW,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_MJ = {
        80: SAW,
        70: MLW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_JA = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: SAW,
    }

    ap_SO = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_ND = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_dict = {
        1: ap_JFMA,
        2: ap_JFMA,
        3: ap_JFMA,
        4: ap_JFMA,
        5: ap_MJ,
        6: ap_MJ,
        7: ap_JA,
        8: ap_JA,
        9: ap_SO,
        10: ap_SO,
        11: ap_ND,
        12: ap_ND,
    }

    return ap_dict



def get_6sv1_atmospheric_profile_parameters(satobj, parameters):

    iso_time = satobj.iso_time
    Date = dateutil.parser.isoparse(iso_time)

    ImageCenterLat, ImageCenterLon = get_image_center_lat_lon(satobj)


    ap_dict = get_6sv1_atmospheric_profile_lut()

    # Atmospheric mode type
    if -15 < ImageCenterLat <= 15:
        parameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    elif (15 < ImageCenterLat <= 45) or (-45 <= ImageCenterLat < -15):
        if 4 < parameters["ImgMonth"] <= 9:
            parameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
        else:
            parameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)

    elif (45 < ImageCenterLat <= 60) or (-60 <= ImageCenterLat < -45):
        if 4 < parameters["ImgMonth"] <= 9:
            parameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
        else:
            parameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)

    rounded_lat = round(ImageCenterLat, -1)

    parameters["AtmosphericProfile"] = ap_dict[Date.month][rounded_lat]

    return parameters



def get_6sv1_mean_dem_parameters(satobj, parameters, dem_path = None):

    min_lat, max_lat, min_lon, max_lon = get_image_extent_lat_lon(satobj)

    if dem_path is not None:
        # Find the DEM height by studying the range of the area.
        pointUL = dict()
        pointDR = dict()

        # Modifications made due to HYPSO 2D Lat/Lon array not being squares, they may be skewed
        pointUL["lat"] = max_lat
        pointUL["lon"] = min_lon
        pointDR["lat"] = min_lat
        pointDR["lon"] = max_lon

        mean_elevation = (MeanDEM(pointUL, pointDR, dem_path)) * 0.001 # to kilometers
            
        print("meanDEM:")
        print(mean_elevation)
        parameters["meanDEM"] = mean_elevation
    
    else:
        parameters["meanDEM"] = 0

    return parameters



def get_6sv1_aerosol_profile_parameters(satobj, parameters):

    # aerosol type continent
    #parameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Maritime)
    parameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Continental)

    return parameters


def get_6sv1_ground_reflectance_parameters(satobj, parameters):

    #parameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0.26)
    #parameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0.05)
    parameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(Py6S.GroundReflectance.LakeWater)

    return parameters



def get_6sv1_parameters(satobj, dem_path: Path = None) -> dict:
    """
    Get the parameters you need for 6s atmospheric correction

    :param satobj: 
    :param dem_path:

    :return: Dictionary of the Basic Paramters for the PY6SV1 correction method
    """

    parameters = dict()

    parameters = get_6sv1_solar_angles_parameters(satobj, parameters)

    parameters = get_6sv1_datetime_parameters(satobj, parameters)

    parameters = get_6sv1_wavelengths_parameters(satobj, parameters)

    parameters = get_6sv1_sensor_angles_parameters(satobj, parameters)

    parameters = get_6sv1_relative_azimuth_angles_parameters(satobj, parameters)

    parameters = get_6sv1_atmospheric_profile_parameters(satobj, parameters)

    parameters = get_6sv1_mean_dem_parameters(satobj, parameters, dem_path = dem_path)

    parameters = get_6sv1_aerosol_profile_parameters(satobj, parameters)

    parameters = get_6sv1_ground_reflectance_parameters(satobj, parameters)

    parameters['aot550'] = 0.1 # 550nm aerosol optical thickness. Constant value changed later if supplied

    #parameters['radiance_cube'] = radiance_cube

    # TOA Approach without SRF *****************************************************************
    # Non-homogeneous lower bedding surface, Lambertian
    # This should be used when the SRF is not known and it can be obtained from the TOA Reflectance
    # This is non tested example --------------------------------
    # Running for each wavelength with its respective reflectance value
    # for (i, j) in zip(wave, toa_reflec):
    #     s.wavelength = Wavelength(i)
    #     s.atmos_corr = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(j)
    #     s.run()
    #     print "A18 wavelength: ", i, "Ref. : ", j, " ",
    #     boa_rec = s.outputs.atmos_corrected_reflectance_lambertian
    # s.atmos_corr = SixsInputParameter['AtmosCorrection']
    # Non-homogeneous lower bedding surface, Lambertian
    # parameters['AtmosCorrection'] = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(-0.1)
    # *****************************************************************

    return parameters

