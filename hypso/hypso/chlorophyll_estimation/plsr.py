import os
import numpy as np
import xarray as xr
from pathlib import Path

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import pickle

def generate_plsr_chl_estimates(satobj, model_path: Path = None):

    if satobj.l1d_cube is not None:
        print('Run PLSR')


        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        if not isinstance(model, PLSRegression):
            print('[ERROR] Invalid model or wrong format.')
            return None

        X = satobj.masked_l1d_cube

    else:
        print('[ERROR] L1d top-of-atmosphere reflectance has not been generated!')


    return None