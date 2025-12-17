import numpy as np
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
#from matplotlib.path import Path
import matplotlib

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import Polygon
from dateutil import parser


import Py6S
from tqdm import tqdm

from datetime import datetime, timedelta, timezone


from .ac_6sv1_aot550 import extract_footprint_and_date, download_viirs_aot, get_aot_in_swath
from .ac_6sv1_parameters import get_6sv1_parameters
from .ac_6sv1_lut import get_lut_filename
from .ac_6sv1_query_lut import LUTQuery, LUTQueryRegularGrid

from osgeo import gdal

from .ac_6sv1_utils import get_lat_lon


# TODO: Add relative azimuth angles (raa) to 6S simulations

class SixSParameters:

    def __init__(self):

        pass





class SixSResult(dict):
    KEYS = ["rho_R", "Tg_H20", "Tg_O3", "Tg_OG", "Ts_Tv", "S_atm",]

    def __init__(self, num_bands, wavelengths, fill_value=np.nan):
        super().__init__()
        self.num_bands = num_bands
        self.wavelengths = wavelengths
        for k in self.KEYS:
            self[k] = np.full((num_bands,), fill_value)

    def interpolate(self):

        for k in self.KEYS:

            # Linear 1D Interp to Fill Values skipped due to AOT Variances
            data = self[k][:]
            wl = self.wavelengths
            nans = np.isnan(data)
            data[nans] = np.interp(wl[nans], wl[~nans], data[~nans])
            self[k][:] = data




def run_6sv1_simulation_no_aerosol(parameters) -> SixSResult:

    wavelengths = parameters["wavelengths"]
    num_bands = len(parameters["wavelengths"])

    outputs = SixSResult(num_bands=num_bands, wavelengths=wavelengths)

    for BandId in tqdm(range(num_bands)):

        # Part I
        # Run 6S to calculate Rayleigh reflectance
        # https://blog.rtwilson.com/calculating-rayleigh-reflectance-using-py6s/

        # 6S Models
        s = Py6S.SixS()

        # Enable Sensor type customization
        s.geometry = Py6S.Geometry.User()

        # Add Geometry Parameters
        s.geometry.solar_z = parameters["SolarZenithAngle"]
        s.geometry.solar_a = parameters["SolarAzimuthAngle"]
        s.geometry.view_z = parameters["SatZenithAngles"]
        s.geometry.view_a = parameters["SatAzimuthAngles"]

        # Date: Month, Day
        s.geometry.month = parameters["ImgMonth"]
        s.geometry.day = parameters["ImgDay"]

        # Type of atmospheric pattern
        s.atmos_profile = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.NoGaseousAbsorption)

        # Target Features
        s.ground_reflectance = parameters["GroundReflectance"]

        # Aerosol Profile
        s.aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.NoAerosols)
        
        # Study area altitude, satellite sensor orbit altitude
        s.altitudes = Py6S.Altitudes()
        s.altitudes.set_target_custom_altitude(parameters["meanDEM"])
        s.altitudes.set_sensor_satellite_level()

        # Wavelength
        current_band_wl = parameters["wavelengths"][BandId] / 1000 # convert to micrometers
        s.wavelength = Py6S.Wavelength(current_band_wl)

        s.run()

        outputs['rho_R'][BandId] = s.outputs.atmospheric_intrinsic_reflectance # Intrinsic reflectance without aerosol
        outputs['Tg_H20'][BandId] = s.outputs.trans['water'].total
        outputs['Tg_O3'][BandId] = s.outputs.trans['ozone'].total
        outputs['Tg_OG'][BandId] = 1.0
        outputs['Ts_Tv'][BandId] = s.outputs.trans['total_scattering'].total
        outputs['S_atm'][BandId] = s.outputs.spherical_albedo.total

    outputs.interpolate()

    return outputs


def run_6sv1_simulation(parameters) -> SixSResult:

    wavelengths = parameters["wavelengths"]
    num_bands = len(parameters["wavelengths"])

    outputs = SixSResult(num_bands=num_bands, wavelengths=wavelengths)

    for BandId in tqdm(range(num_bands)):

        # Part II
        # Run 6S with AOD

        s = Py6S.SixS()

        # Enable Sensor type customization
        s.geometry = Py6S.Geometry.User()

        # Add Geometry Parameters
        s.geometry.solar_z = parameters["SolarZenithAngle"]
        s.geometry.solar_a = parameters["SolarAzimuthAngle"]
        s.geometry.view_z = parameters["SatZenithAngles"]
        s.geometry.view_a = parameters["SatAzimuthAngles"]

        # Date: Month, Day
        s.geometry.month = parameters["ImgMonth"]
        s.geometry.day = parameters["ImgDay"]

        # Type of atmospheric pattern
        s.atmos_profile = parameters["AtmosphericProfile"]

        # Target Features
        s.ground_reflectance = parameters["GroundReflectance"]

        # Aerosol Profile
        s.aero_profile = parameters["AeroProfile"] 

        if 'aot550' in parameters.keys():
            s.aot550 = parameters['aot550']
        elif 'aeronet' in parameters.keys():
            s = Py6S.SixSHelpers.Aeronet.import_aeronet_data(s, parameters['aeronet'], parameters['time'])
        else:
            s.aot550 = 0.1  # Use Default Values

        # Study area altitude, satellite sensor orbit altitude
        s.altitudes = Py6S.Altitudes()
        s.altitudes.set_target_custom_altitude(parameters["meanDEM"])
        s.altitudes.set_sensor_satellite_level()

        # Wavelength
        current_band_wl = parameters["wavelengths"][BandId] / 1000 # convert to micrometers
        s.wavelength = Py6S.Wavelength(current_band_wl)

        s.run()

        rho_A_R = s.outputs.atmospheric_intrinsic_reflectance
        Tg_H20 = s.outputs.trans['water'].total
        Tg_O3 = s.outputs.trans['ozone'].total
        Tg_OG = 1.0
        Ts_Tv = s.outputs.trans['total_scattering'].total
        S_atm = s.outputs.spherical_albedo.total

        outputs['rho_R'][BandId] = s.outputs.atmospheric_intrinsic_reflectance # Intrinsic reflectance with aerosol
        outputs['Tg_H20'][BandId] = s.outputs.trans['water'].total
        outputs['Tg_O3'][BandId] = s.outputs.trans['ozone'].total
        outputs['Tg_OG'][BandId] = 1.0
        outputs['Ts_Tv'][BandId] = s.outputs.trans['total_scattering'].total
        outputs['S_atm'][BandId] = s.outputs.spherical_albedo.total


        #with open(str(BandId)+'_' + str(current_band_wl) + 'nm_output.txt', 'w') as file:
        #    file.write(s.outputs.fulltext)

    outputs.interpolate()

    return outputs








def get_mean_aot550(satobj, VERBOSE: bool = True):

    latitudes, longitudes = get_lat_lon(satobj)

    aot550_path = Path(satobj.capture_dir)
    aot550_path = aot550_path.joinpath("data_aerosol")
    aot550_path.mkdir(parents=True, exist_ok=True)

    footprint, bbox, temporal = extract_footprint_and_date(satobj=satobj)

    files = download_viirs_aot(footprint_polygon=footprint, temporal_range=temporal, local_path=aot550_path)

    aot_inside_NOAA = None
    aot_inside_SNPP = None

    try:
        files_f = [f for f in files if 'NOAA' in str(f)]
        all_aot_NOAA, all_lat_NOAA, all_lon_NOAA, aot_inside_NOAA = get_aot_in_swath(files_f, footprint, latitudes, longitudes, name='NOAA', local_path=aot550_path)
        if VERBOSE:
            print(np.mean(aot_inside_NOAA))
    except Exception as ex:
        print(ex)

    try:
        files_f = [f for f in files if 'SNPP' in str(f)]
        all_aot_SNPP, all_lat_SNPP, all_lon_SNPP, aot_inside_SNPP = get_aot_in_swath(files_f, footprint, latitudes, longitudes, name='SNPP', local_path=aot550_path)
        if VERBOSE:
            print(np.mean(aot_inside_SNPP))
    except Exception as ex:
        print(ex)

    if aot_inside_NOAA is None and aot_inside_SNPP is not None:
        aot550 = np.mean(aot_inside_SNPP)

    elif aot_inside_SNPP is None and aot_inside_NOAA is not None:
        aot550 = np.mean(aot_inside_NOAA)
    
    elif aot_inside_SNPP is not None and aot_inside_NOAA is not None:
        aot550 = np.mean(np.concatenate([aot_inside_NOAA, aot_inside_SNPP]))
    else:

        print("[WARNING] No AOT at 550nm value found. Defaulting to AOT550 of 0.1.")
        aot550 = 0.1

    if VERBOSE:
        print("Mean AOT at 550nm:")
        print(aot550)


    return aot550








def run_6sv1_atmospheric_correction(satobj, dem_path: Path = None, VERBOSE: bool = True, use_luts=False, luts_dir=None):

    if VERBOSE: 
        print("[INFO] Running 6SV1 atmospheric correction")
        print("\n-------  Py6S Atmospheric Correction  ----------")

    # Original units mW  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 1000 # mW to W -> W  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 0.001


    aot550 = get_mean_aot550(satobj=satobj, VERBOSE=VERBOSE)

    parameters = get_6sv1_parameters(satobj=satobj, dem_path=dem_path)

    parameters['aot550'] = aot550 # Update AOT at 550nm if value is provided


    print(parameters)

    if use_luts:

        # No Aerosols
        aero_profile = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.NoAerosols)
        atmos_profile = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.NoGaseousAbsorption)
        lut_file_no_aerosol, _, lut_base_filename = get_lut_filename(luts_dir=luts_dir, aero_profile=aero_profile, atmos_profile=atmos_profile)
        lut_query_sys_no_aerosol = LUTQueryRegularGrid(lut_file=lut_file_no_aerosol)


        # With Aerosols
        aero_profile = parameters["AeroProfile"]
        atmos_profile = parameters["AtmosphericProfile"]
        lut_file, _, lut_base_filename = get_lut_filename(luts_dir=luts_dir, aero_profile=aero_profile, atmos_profile=atmos_profile)
        lut_query_sys = LUTQueryRegularGrid(lut_file=lut_file)


        cube = np.full((satobj.l1d_cube.shape), fill_value=np.nan)

        height, width, bands = cube.shape


        if 'aot550' in parameters.keys():
            aot550 = parameters['aot550']
        else:
            aot550 = 0.1  # Use Default Values

        sza_array = satobj.solar_zenith_angles
        vza_array = satobj.sat_zenith_angles
        raa_array = satobj.relative_azimuth_angles

        aot550_array = np.full((height, width), fill_value=aot550)

        
        for band in tqdm(range(0,bands)):

            wavelength = satobj.wavelengths[band]

            wavelength_array = np.full((height, width), fill_value=wavelength)

            #query_point = np.array([sza, vza, raa, aot550, wavelength])
            #query_point = query_point.reshape(1, -1)
            #response_point = interp_func(query_point)

            query_array = np.stack((sza_array, vza_array, raa_array, aot550_array, wavelength_array), axis=-1)
            query_array = query_array.reshape((height*width, -1))

            interp_func = lut_query_sys_no_aerosol.interpolators['rho_R']
            response_array = interp_func(query_array)
            rho_A_R = response_array.reshape(height, width)

            interp_func = lut_query_sys.interpolators['rho_R']
            response_array = interp_func(query_array)
            rho_R = response_array.reshape(height, width)

            interp_func = lut_query_sys.interpolators['Tg_H20']
            response_array = interp_func(query_array)
            Tg_H20 = response_array.reshape(height, width)

            interp_func = lut_query_sys.interpolators['Tg_O3']
            response_array = interp_func(query_array)
            Tg_O3 = response_array.reshape(height, width)

            interp_func = lut_query_sys.interpolators['Ts_Tv']
            response_array = interp_func(query_array)
            Ts_Tv = response_array.reshape(height, width)

            interp_func = lut_query_sys.interpolators['S_atm']
            response_array = interp_func(query_array)
            S_atm = response_array.reshape(height, width)


            rho_toa = satobj.l1d_cube[:,:,band].to_numpy()

            rho_atm = rho_R + (rho_A_R - rho_R)*Tg_H20

            Y = rho_toa - rho_atm * Tg_O3

            numerator = Y
            denominator = (S_atm * Y) + (Ts_Tv * Tg_O3 * Tg_H20)

            rho_s = numerator / denominator

            cube[:,:,band] = rho_s   


    else:
        outputs_no_aerosol = run_6sv1_simulation_no_aerosol(parameters=parameters)
        outputs = run_6sv1_simulation(parameters=parameters)

        cube = np.full((satobj.l1d_cube.shape), fill_value=np.nan)

        height, width, bands = cube.shape

        for band in tqdm(range(0,bands)):

            rho_toa = satobj.l1d_cube[:,:,band]

            rho_R = outputs['rho_R'][band]
            rho_A_R = outputs_no_aerosol['rho_R'][band]
            Tg_H20 = outputs['Tg_H20'][band]
            Tg_O3 = outputs['Tg_O3'][band]
            Ts_Tv = outputs['Ts_Tv'][band]
            S_atm = outputs['S_atm'][band]

            rho_atm = rho_R + (rho_A_R - rho_R)*Tg_H20

            Y = rho_toa - rho_atm * Tg_O3

            numerator = Y
            denominator = (S_atm * Y) + (Ts_Tv * Tg_O3 * Tg_H20)

            rho_s = numerator / denominator

            cube[:,:,band] = rho_s

    return cube






def run_6sv1_atmospheric_correction_luts(satobj, dem_path: Path = None, VERBOSE: bool = True):

    if VERBOSE: 
        print("[INFO] Running 6SV1 atmospheric correction")
        print("\n-------  Py6S Atmospheric Correction  ----------")

    # Original units mW  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 1000 # mW to W -> W  (m^{-2} sr^{-1} nm^{-1})
    # radiance_cube = radiance_cube / 0.001


    aot550 = get_mean_aot550(satobj=satobj, VERBOSE=VERBOSE)

    parameters = get_6sv1_parameters(satobj=satobj, dem_path=dem_path)

    parameters['aot550'] = aot550

    wavelengths = parameters["wavelengths"]
    num_bands = len(parameters["wavelengths"])

    outputs = SixSResult(num_bands=num_bands, wavelengths=wavelengths)

    aero_profile = parameters["AeroProfile"]
    atmos_profile = parameters["AtmosphericProfile"]

    luts_dir = "/home/cameron/Nedlastinger/6S_HYPSO_LUTS"

    lut, _, _ = get_lut_filename(luts_dir=luts_dir, aero_profile=aero_profile, atmos_profile=atmos_profile)


    # write fast function for query























    outputs_no_aerosol = run_6sv1_simulation_no_aerosol(parameters=parameters)
    outputs = run_6sv1_simulation(parameters=parameters)


    cube = np.full((satobj.l1d_cube.shape), fill_value=np.nan)

    height, width, bands = cube.shape

    for band in tqdm(range(0,bands)):

        rho_toa = satobj.l1d_cube[:,:,band]

        rho_R = outputs['rho_R'][band]
        rho_A_R = outputs_no_aerosol['rho_R'][band]
        Tg_H20 = outputs['Tg_H20'][band]
        Tg_O3 = outputs['Tg_O3'][band]
        Ts_Tv = outputs['Ts_Tv'][band]
        S_atm = outputs['S_atm'][band]

        rho_atm = rho_R + (rho_A_R - rho_R)*Tg_H20

        Y = rho_toa - rho_atm * Tg_O3

        numerator = Y
        denominator = (S_atm * Y) + (Ts_Tv * Tg_O3 * Tg_H20)

        rho_s = numerator / denominator

        cube[:,:,band] = rho_s

    return cube








def toa_radiance_to_surface_reflectance(rho_toa, rho_ra, Tg, Ts, S):
    
    # Vermote (1997), Equation 1) inverted

    A = 1 / (Tg * Ts)
    B = -rho_ra / Ts
    Y = A*rho_toa + B 
    rho_s = Y / (1 + S*Y) 

    return rho_s