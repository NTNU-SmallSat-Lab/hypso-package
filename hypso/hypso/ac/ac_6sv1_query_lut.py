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

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

import warnings



class LUTQuery:
    def __init__(self, lut_file: str = None, method='regular'):
        """

        Initialize LUT query system
        
        Parameters:
        - lut_file: path to LUT CSV file
        - method: 'regular' for interpolation, 'nearest' for nearest neighbor
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
        
        # Interpolate missing values in dataset
        self.df = fill_missing_in_df(self.df, output_vars)
        
        #self.df.to_csv("test.csv", index=False)

        for var in output_vars:
            if self.method == 'regular':
                self.interpolators[var] = RegularGridInterpolator(points, self.df[var].values)
            elif self.method == 'linear':
                self.interpolators[var] = LinearNDInterpolator(points, self.df[var].values)
            else:
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














class LUTQueryRegularGrid:
    def __init__(self, lut_file: str = None, method='linear'):
        """
        Initialize LUT query using RegularGridInterpolator
        
        Assumes LUT is on a regular grid in parameter space
        """
        self.df = pd.read_csv(lut_file)
        self.method = method
        self._extract_grid_structure()
        self._build_interpolators()
    
    def _extract_grid_structure(self):
        """Extract regular grid structure from the LUT DataFrame"""
        
        # Parameter dimensions (in order)
        self.param_names = ['solar_zenith', 'view_zenith', 'relative_azimuth', 
                           'aot550', 'wavelength']
        
        # Get unique sorted values for each parameter
        self.grid_points = []
        self.grid_arrays = {}  # Store grid arrays for each parameter
        
        for param in self.param_names:
            unique_vals = np.sort(self.df[param].dropna().unique())
            self.grid_points.append(unique_vals)
            self.grid_arrays[param] = unique_vals
            
            print(f"{param}: {len(unique_vals)} values from {unique_vals[0]:.2f} to {unique_vals[-1]:.2f}")
        
        # Verify the grid is regular (all combinations should exist)
        expected_size = np.prod([len(points) for points in self.grid_points])
        actual_size = len(self.df)
        
        print(f"\nExpected grid size: {expected_size}")
        print(f"Actual data size: {actual_size}")
        
        if expected_size != actual_size:
            print("Warning: Data may not be on a complete regular grid!")
            print("Using RegularGridInterpolator may give unexpected results.")
    
    def _build_interpolators(self):
        """Build RegularGridInterpolator for each output variable"""
        
        self.interpolators = {}
        output_vars = ['rho_R', 'Tg_H20', 'Tg_O3', 'Ts_Tv', 'S_atm']
        
        for var in output_vars:
            # Reshape data into a 5D grid
            grid_data = self._reshape_to_grid(var)
            
            # Create interpolator
            self.interpolators[var] = RegularGridInterpolator(
                self.grid_points,
                grid_data,
                method=self.method,
                bounds_error=False,
                fill_value=None  # We'll handle extrapolation
            )
            
            print(f"Built {self.method} interpolator for {var}")
    
    def _reshape_to_grid(self, variable_name):
        """
        Reshape 1D column data into a 5D grid array
        """
        # Get the variable data
        var_data = self.df[variable_name].values
        
        # Create an empty array with correct shape
        grid_shape = tuple(len(points) for points in self.grid_points)
        grid_array = np.empty(grid_shape)
        grid_array[:] = np.nan
        
        # Create a multi-index to map grid positions
        # We need to find which row in df corresponds to which grid position
        grid_indices = {}
        
        # For each parameter, create mapping from value to index
        for i, param in enumerate(self.param_names):
            grid_indices[param] = {val: idx for idx, val in enumerate(self.grid_points[i])}
        
        # Fill the grid array
        for idx, row in self.df.iterrows():
            # Get grid indices for this row
            grid_pos = []
            for param in self.param_names:
                grid_pos.append(grid_indices[param][row[param]])
            
            # Assign value to grid position
            grid_array[tuple(grid_pos)] = row[variable_name]
        
        # Check for NaN values in the grid
        nan_count = np.isnan(grid_array).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in {variable_name} grid")
            
            # Option 1: Fill NaNs with nearest value
            if nan_count < grid_array.size * 0.1:  # Less than 10% missing
                from scipy.ndimage import distance_transform_edt
                mask = np.isnan(grid_array)
                if mask.any():
                    # Find indices of known values
                    known_idx = np.where(~mask)
                    known_values = grid_array[~mask]
                    
                    # Create interpolator for known values
                    from scipy.interpolate import NearestNDInterpolator
                    interp = NearestNDInterpolator(np.array(known_idx).T, known_values)
                    
                    # Fill missing values
                    missing_idx = np.where(mask)
                    if len(missing_idx[0]) > 0:
                        grid_array[missing_idx] = interp(np.array(missing_idx).T)
        
        return grid_array
    
    def query(self, sza, vza, raa, aot550, wavelength):
        """
        Query the LUT for specific parameters
        
        Parameters:
        - sza: solar zenith angle (degrees)
        - vza: view zenith angle (degrees) 
        - raa: relative azimuth angle (degrees)
        - aot550: aerosol optical thickness at 550nm
        - wavelength: wavelength (nm)
        
        Returns:
        - Dictionary with all output parameters
        """
        
        # Prepare query point (in correct order)
        query_point = np.array([sza, vza, raa, aot550, wavelength])
        
        results = {}
        for var, interp in self.interpolators.items():
            try:
                # Reshape to 2D array with shape (1, 5) for single query
                val = interp(query_point.reshape(1, -1))[0]
                results[var] = float(val)
            except Exception as e:
                print(f"Error interpolating {var}: {e}")
                results[var] = np.nan
        
        return results
    
    def query_batch(self, query_points):
        """
        Query multiple points at once for efficiency
        
        Parameters:
        - query_points: numpy array of shape (n_points, 5)
                      Each row: [sza, vza, raa, aot550, wavelength]
        
        Returns:
        - Dictionary with arrays of results
        """
        results = {}
        for var, interp in self.interpolators.items():
            try:
                results[var] = interp(query_points).astype(float)
            except Exception as e:
                print(f"Error interpolating {var}: {e}")
                results[var] = np.full(len(query_points), np.nan)
        
        return results














def fill_missing_in_df(df, target_cols):
    """
    Fill missing values in the DataFrame using interpolation
    """
    print("Filling missing values in DataFrame...")
    
    # Check for missing values
    for col in target_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values")

    # Linear interpolation along the index
    df[target_cols] = df[target_cols].interpolate(method='linear', axis=0)
        

    # Check if any missing values remain
    remaining_missing = df[target_cols].isnull().sum().sum()
    if remaining_missing > 0:
        print(f"  Warning: {remaining_missing} missing values remain")
        # Try one more pass with forward/backward fill
        df[target_cols] = df[target_cols].fillna(method='ffill').fillna(method='bfill')
    
    print("Missing values filled.")

    return df