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

from osgeo import gdal

import pandas as pd

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import warnings



class LUTQuery:
    def __init__(self, lut_file: str = None, method='linear'):
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

        
        # Parameters for interpolation
        interpolation_params = ['solar_zenith', 'view_zenith', 'relative_azimuth', 
                               'aot550', 'wavelength']
        
        points = self.df[interpolation_params].values
        
        # Build interpolators for each output variable
        output_vars = ['rho_R', 'Tg_H20', 'Tg_O3', 'Ts_Tv', 'S_atm']
        
        for var in output_vars:
            if self.method == 'linear':
                self.interpolators[var] = LinearNDInterpolator(points, self.df[var].values)
            else:  # nearest
                self.interpolators[var] = NearestNDInterpolator(points, self.df[var].values)
    

    
    def query(self, sza, vza, raa, aot550, wavelength):
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

        
        # Create query point
        query_point = np.array([[sza, vza, raa, aot550, wavelength]])
        
        results = {}
        for var, interp in self.interpolators.items():
            try:
                results[var] = float(interp(query_point)[0])
            except ValueError as e:
                warnings.warn(f"Could not interpolate {var}: {e}")
                results[var] = np.nan
        
        return results
