import os
from osgeo import gdal # install with `pip install gdal==3.8.4`
import numpy as np
from importlib.resources import files
from pathlib import Path
import Py6S
import dateutil

import itertools
from Py6S import SixS, Geometry, AtmosProfile, AeroProfile, GroundReflectance, Wavelength

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import tqdm
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import warnings


SZA_VALUES = np.linspace(0,80,5)
VZA_VALUES = np.linspace(0,80,5)
RAA_VALUES = np.linspace(0,180,10)

AOT550_VALUES = np.linspace(0,0.5,11)

#WAVELENGTH_VALUES = np.linspace(380,800,120) 
WAVELENGTH_VALUES = np.arange(380, 800 ,3.33)

MEAN_DEM_VALUES = [0, 1, 2]




NA = Py6S.AeroProfile.PredefinedType(Py6S.AeroProfile.NoAerosols)
M = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Maritime)
C = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Continental)

#AERO_PROFILE_VALUES = [NA, C]

AERO_PROFILE_VALUES = {"NA": NA,
                       "C": C
                       }

NGA = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.NoGaseousAbsorption)
SAW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)
SAS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
MLS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
MLW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)
T = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

ATMOS_PROFILE_VALUES = {"NGA": NGA, 
                        "SAW": SAW, 
                        "SAS": SAS, 
                        "MLS": MLS, 
                        "MLW": MLW, 
                        "T": T
                        }




def run_single_simulation(params):
    """Run a single 6S simulation with given parameters"""

    atmos_profile, aero_profile, sza, vza, raa, aot550, wavelength  = params
    
    try:


        s = Py6S.SixS()

        # Enable Sensor type customization
        s.geometry = Py6S.Geometry.User()

        s.geometry.solar_z = sza
        s.geometry.solar_a = 0
        s.geometry.view_z = vza
        s.geometry.view_a = raa


        s.aot550 = aot550
        s.atmos_profile = atmos_profile
        s.aero_profile = aero_profile
        s.altitudes.set_target_custom_altitude(0)
        s.altitudes.set_sensor_satellite_level()
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)

        s.wavelength = Py6S.Wavelength(wavelength / 1000) # convert to micrometers

        s.run()

        return {
            'solar_zenith': sza,
            'view_zenith': vza,
            'relative_azimuth': raa,
            'aot550': aot550,
            'wavelength': wavelength,
            'atmos_profile': atmos_profile,
            'aero_profile': aero_profile,
            'success': True,
            'rho_R': s.outputs.atmospheric_intrinsic_reflectance,
            'Tg_H20': s.outputs.trans['water'].total,
            'Tg_O3': s.outputs.trans['ozone'].total,
            'Ts_Tv': s.outputs.trans['total_scattering'].total,
            'S_atm': s.outputs.spherical_albedo.total
        }
    
    except Exception as e:
        print(f"Failed: {params}, Error: {e}")
        return {'success': False}





def create_parallel_lut():
    """Create LUT using parallel processing"""
    
    # Define all parameter combinations
    
    

    for ATMOS_PROFILE_KEY in ATMOS_PROFILE_VALUES.keys():

        for AERO_PROFILE_KEY in AERO_PROFILE_VALUES.keys():

            ATMOS_PROFILE_VALUE = ATMOS_PROFILE_VALUES[ATMOS_PROFILE_KEY]
            AERO_PROFILE_VALUE = AERO_PROFILE_VALUES[AERO_PROFILE_KEY]

            print('[INFO] Building Look-Up Table for: ')
            print('\t'*7 + 'Aerosol Profile: ' + str(AERO_PROFILE_KEY))
            print('\t'*7 + 'Atmospheric Profile: ' + str(ATMOS_PROFILE_KEY))
            
            all_params = []

            param_grid = list(itertools.product(ATMOS_PROFILE_VALUE, AERO_PROFILE_VALUE, SZA_VALUES, VZA_VALUES, RAA_VALUES, AOT550_VALUES, WAVELENGTH_VALUES))

            for sza, vza, raa, aot550, wavelength, atmos_profile, aero_profile in param_grid:
                all_params.append((sza, vza, raa, aot550, wavelength, atmos_profile, aero_profile))
            
            print('\t'*7 + f"Total simulations: {len(all_params)}")
    

            #result = run_single_simulation(all_params[2])

            # Use multiprocessing
            #with mp.Pool(processes=mp.cpu_count()-1) as pool:
            #with mp.Pool(processes=2) as pool:
            
            pool = Pool(processes=32)

            results = []
            for result in tqdm.tqdm(pool.imap_unordered(run_single_simulation, all_params), total=len(all_params)): 
                results.append(result)


            # Filter successful results
            successful_results = [r for r in results if r['success']]
            
            df = pd.DataFrame(successful_results)

            BASE_FILENAME = "py6s_lut_aero" + str(AERO_PROFILE_KEY) + "_atmos" + str(ATMOS_PROFILE_KEY)

            CSV_FILENAME = BASE_FILENAME  + ".csv"
            PKL_FILENAME = BASE_FILENAME  + ".pkl"

            df.to_csv(CSV_FILENAME, index=False)
            print(f"[INFO] Parallel LUT created with {len(df)} successful entries")

            df.to_pickle(PKL_FILENAME)

# Uncomment to run parallel version (be careful with system resources)
# parallel_lut = create_parallel_lut()


def query_lut():
    
    #val = LUT_xr.R_toa.interp(
    #AOT=0.23,
    #SZA=22.0,
    #WL=0.54
    #).item()
    
    return











class LUTQuery:
    def __init__(self, lut_file='py6s_parallel_lut.csv', method='linear'):
        """
        Initialize LUT query system
        
        Parameters:
        - lut_file: path to LUT CSV file
        - method: 'linear' for interpolation, 'nearest' for nearest neighbor
        """
        self.df = pd.read_csv(lut_file)
        self.method = method
        self.interpolators = {}
        self._build_interpolators()
        
    def _build_interpolators(self):
        """Build interpolators for each output parameter"""
        # Convert string profiles back to numeric codes for interpolation
        self.df['atmos_profile_code'] = self.df['atmos_profile'].astype('category').cat.codes
        self.df['aero_profile_code'] = self.df['aero_profile'].astype('category').cat.codes
        
        # Create mapping for profile lookups
        self.atmos_profile_map = dict(enumerate(self.df['atmos_profile'].astype('category').cat.categories))
        self.aero_profile_map = dict(enumerate(self.df['aero_profile'].astype('category').cat.categories))
        
        # Parameters for interpolation
        interpolation_params = ['solar_zenith', 'view_zenith', 'relative_azimuth', 
                               'aot550', 'wavelength', 'atmos_profile_code', 'aero_profile_code']
        
        points = self.df[interpolation_params].values
        
        # Build interpolators for each output variable
        output_vars = ['rho_R', 'Tg_H20', 'Tg_O3', 'Ts_Tv', 'S_atm']
        
        for var in output_vars:
            if self.method == 'linear':
                self.interpolators[var] = LinearNDInterpolator(points, self.df[var].values)
            else:  # nearest
                self.interpolators[var] = NearestNDInterpolator(points, self.df[var].values)
    
    def _convert_profiles_to_codes(self, atmos_profile, aero_profile):
        """Convert profile objects to numeric codes"""
        atmos_str = str(atmos_profile)
        aero_str = str(aero_profile)
        
        # Find closest matching profile codes
        atmos_codes = {v: k for k, v in self.atmos_profile_map.items()}
        aero_codes = {v: k for k, v in self.aero_profile_map.items()}
        
        atmos_code = atmos_codes.get(atmos_str, 0)  # default to first profile
        aero_code = aero_codes.get(aero_str, 0)     # default to first profile
        
        return atmos_code, aero_code
    
    def query(self, sza, vza, raa, aot550, wavelength, atmos_profile, aero_profile):
        """
        Query the LUT for specific parameters
        
        Parameters:
        - sza: solar zenith angle (degrees)
        - vza: view zenith angle (degrees) 
        - raa: relative azimuth angle (degrees)
        - aot550: aerosol optical thickness at 550nm
        - wavelength: wavelength (nm)
        - atmos_profile: Py6S atmospheric profile object
        - aero_profile: Py6S aerosol profile object
        
        Returns:
        - Dictionary with all output parameters
        """
        # Convert profiles to codes
        atmos_code, aero_code = self._convert_profiles_to_codes(atmos_profile, aero_profile)
        
        # Create query point
        query_point = np.array([[sza, vza, raa, aot550, wavelength, atmos_code, aero_code]])
        
        results = {}
        for var, interp in self.interpolators.items():
            try:
                results[var] = float(interp(query_point)[0])
            except ValueError as e:
                warnings.warn(f"Could not interpolate {var}: {e}")
                results[var] = np.nan
        
        return results




















if __name__ == "__main__":

    create_parallel_lut()


