import pathlib

import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from importlib.resources import files
from joblib import dump, load
from typing import Tuple

#from hypso.chlorophyll.utilities_chl import convolve2d
#from hypso.chlorophyll.indices import *
from .indices import TBVI, TBM, BR, BR_LOG, BDIFF
from .utils import convolve2d

simplefilter("ignore", category=ConvergenceWarning)

def run_tuned_chlorophyll_estimation(l2a_cube: np.ndarray, 
                         model: str,
                         spatial_dimensions: tuple[int, int],
                         mask: np.ndarray = None
                         ) -> np.ndarray:
    """
    Estimates the chlorophyll using a trained ML Scikit-Learn Model

    :param model: Absolute path for the pre-trained Scikit-Learn model

    :return: np.ndarray
    """
    
    if mask is None:
        mask = np.zeros(spatial_dimensions, dtype=bool)

    # Load Model -----------------------------------------------------
    model = load(model)

    # Get Cube with proper data -------------------------------------
    rrs_cube = l2a_cube

    # Processing mask only for water pixels
    mask = ~mask

    # Reshape 3D to 2D array --------------------------------------------------
    channels = rrs_cube.shape[2]
    rrs_array = rrs_cube.reshape((-1, channels))

    # Get DF with proper headers ---------------------------------------------
    hypso_headers = files('hypso.chlorophyll').joinpath('data/hypso_wl_headers.csv')
    
    # TODO: use wavelengths?
    hypso_string_wl = list(np.loadtxt(hypso_headers, delimiter=",", dtype=str))

    rrs_df = pd.DataFrame(rrs_array, columns=hypso_string_wl)

    chl = None


    # Features -----------------------------------------------------------------
    hypso_optimal_features_df, _ = get_best_features(rrs_df, None, rrs_df, None,
                                                        dataset_name="hypso")
    hypso_optimal_features_df = hypso_optimal_features_df.loc[:,
                                hypso_optimal_features_df.columns != 'Chla']

    # Filter -----------------------------------------------------------------
    sub_params = ["bratio"]
    sub_params = ["bratio", "bratio_log", "tbvi", "diff"]
    pre_filter = [k for k in hypso_optimal_features_df.columns for w in k.split("-") if w in sub_params]

    hypso_optimal_features_df = hypso_optimal_features_df[
        hypso_optimal_features_df.columns.intersection(pre_filter)]

    chl = model.predict(hypso_optimal_features_df)
    chl = np.reshape(chl, spatial_dimensions)

    # Smoothing ------------------------------------------------------------
    kernel = np.array([[1 / 16, 1 / 8, 1 / 16],  # 3x3 kernel
                        [1 / 8, 1 / 4, 1 / 8],
                        [1 / 16, 1 / 8, 1 / 16]], dtype=float)
    chl = convolve2d(chl, kernel, max_missing=0.5, verbose=True)


    # Masking -----------------------------------------------------------------
    chl = np.multiply(chl, mask.astype(int))

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                chl[r, c] = np.nan

    return chl



def get_best_features(X_train_rrs: np.ndarray, 
                      y_train_rrs: np.ndarray, 
                      X_test_rrs: np.ndarray, 
                      y_test_rrs: np.ndarray,
                      dataset_name: str = "hypso"
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the optimal features based on the findings of the Master thesis "Improving Ocean Chlorophyll Estimation in
    Satellite Hyperspectral Images Using Ensemble Machine Learning" by Flores-Romero Alvaro

    :param X_train_rrs: Surface reflectance values for training
    :param y_train_rrs: Surface chlorophyll values for training
    :param X_test_rrs: Surface reflectance values for testing
    :param y_test_rrs: Surface chlorophyll values for testing
    :param dataset_name: Dataset name to estimate values. Defaults to "hypso"

    :return: Returns the features for training and testing
    """
    hypso_optimal_dict = {
        "tbvi": [["Rrs_701", "Rrs_460"], ["Rrs_569", "Rrs_489"]],
        "tbm": [["Rrs_503", "Rrs_601", "Rrs_801"]],
        "ocvi": [["Rrs_499", "Rrs_503", "Rrs_701"],
                 ["Rrs_499", "Rrs_503", "Rrs_615"]],
        "bratio": [["Rrs_701", "Rrs_503"],
                   ["Rrs_555", "Rrs_489"]],
        "bratio_log": [["Rrs_701", "Rrs_478"],
                       ["Rrs_569", "Rrs_489"]],
        "bdiff": [["Rrs_615", "Rrs_766"],
                  ["Rrs_517", "Rrs_489"]],
        "custom": [["Rrs_443", "Rrs_562"],
                   ["Rrs_450", "Rrs_670"],
                   ["Rrs_531", "Rrs_670"],
                   ["Rrs_670", "Rrs_450"],
                   ["Rrs_656", "Rrs_562"]]
    }

    hypso_dxdx_optimal_dict = {
        "tbvi": [["Rrs_715", "Rrs_478"], ["Rrs_687", "Rrs_481"]],
        "tbm": [["Rrs_481", "Rrs_687", "Rrs_804"]],
        "bratio": [["Rrs_517", "Rrs_684"]],
        "bdiff": [["Rrs_506", "Rrs_687"],
                  ["Rrs_506", "Rrs_715"],
                  ["Rrs_722", "Rrs_517"]],
        "custom": [["Rrs_443", "Rrs_562"],
                   ["Rrs_450", "Rrs_670"],
                   ["Rrs_531", "Rrs_670"],
                   ["Rrs_670", "Rrs_450"],
                   ["Rrs_656", "Rrs_562"]]
    }

    gloria_optimal_dict = {
        "tbvi": [["Rrs_699", "Rrs_496"], ["Rrs_699", "Rrs_599"]],
        "tbm": [["Rrs_599", "Rrs_698", "Rrs_701"],
                ["Rrs_497", "Rrs_533", "Rrs_602"],
                ["Rrs_497", "Rrs_698", "Rrs_704"]],
        "ocvi": [["Rrs_482", "Rrs_503", "Rrs_761"],
                 ["Rrs_464", "Rrs_500", "Rrs_698"],
                 ["Rrs_461", "Rrs_500", "Rrs_701"],
                 ["Rrs_599", "Rrs_605", "Rrs_701"]],
        "bratio": [["Rrs_516", "Rrs_499"],
                   ["Rrs_700", "Rrs_591"]],
        "bratio_log": [["Rrs_699", "Rrs_500"],
                       ["Rrs_709", "Rrs_497"],
                       ["Rrs_700", "Rrs_670"]],
        "bdiff": [["Rrs_705", "Rrs_489"]],
        "custom": [["Rrs_443", "Rrs_562"],
                   ["Rrs_450", "Rrs_670"],
                   ["Rrs_531", "Rrs_670"],
                   ["Rrs_670", "Rrs_450"],
                   ["Rrs_656", "Rrs_562"]]
    }

    gloria_dxdx_optimal_dict = {
        "tbvi": [["Rrs_798", "Rrs_575"]],
        "tbm": [["Rrs_599", "Rrs_698", "Rrs_701"],
                ["Rrs_497", "Rrs_698", "Rrs_704"],
                ["Rrs_497", "Rrs_542", "Rrs_698"]],
        "bratio_log": [["Rrs_698", "Rrs_750"],
                       ["Rrs_500", "Rrs_600"]],
        "bdiff": [["Rrs_669", "Rrs_705"],
                  ["Rrs_504", "Rrs_701"]],
        "custom": [["Rrs_443", "Rrs_562"],
                   ["Rrs_450", "Rrs_670"],
                   ["Rrs_531", "Rrs_670"],
                   ["Rrs_670", "Rrs_450"],
                   ["Rrs_656", "Rrs_562"]]
    }

    optimal_dict = None
    if dataset_name == "hypso":
        optimal_dict = hypso_optimal_dict
    elif dataset_name == "hypsodxdx":
        optimal_dict = hypso_dxdx_optimal_dict
    elif dataset_name == "gloria":
        optimal_dict = gloria_optimal_dict
    elif dataset_name == "gloriadxdx":
        optimal_dict = gloria_dxdx_optimal_dict

    global_fx = None
    feature_names = []
    for chl_idx_key in optimal_dict.keys():
        for model_rrs in optimal_dict[chl_idx_key]:
            feature_names.append(chl_idx_key + "-" + dataset_name + ("-".join(model_rrs)))

    tmp_train_features = pd.DataFrame(np.ones((X_train_rrs.shape[0], len(feature_names) + 1),
                                              dtype=float), columns=["Chla"] + feature_names)
    tmp_test_features = pd.DataFrame(np.ones((X_test_rrs.shape[0], len(feature_names) + 1),
                                             dtype=float), columns=["Chla"] + feature_names)

    yttrain = 1
    yttest = 1

    if y_train_rrs is None:
        y_train_rrs = [1 for k in range(X_train_rrs.shape[0])]
    if y_test_rrs is None:
        y_test_rrs = [1 for k in range(X_train_rrs.shape[0])]

    for chl_idx_key in optimal_dict.keys():
        if chl_idx_key == "tbvi":
            global_fx = TBVI
        elif chl_idx_key == "tbm":
            global_fx = TBM
        elif chl_idx_key == "bratio" or chl_idx_key == "custom":
            global_fx = BR
        elif chl_idx_key == "bratio_log":
            global_fx = BR_LOG
        elif chl_idx_key == "bdiff":
            global_fx = BDIFF

        for model_rrs in optimal_dict[chl_idx_key]:
            # Get Reflectance
            wl_train = [X_train_rrs[wx] for wx in model_rrs]
            wl_test = [X_test_rrs[wx] for wx in model_rrs]

            chl_train_descriptor = global_fx(wl_train)
            chl_test_descriptor = global_fx(wl_test)

            # yttrain = np.log(y_train_rrs)
            # yttest = np.log(y_test_rrs)

            yttrain = y_train_rrs
            yttest = y_test_rrs

            name_comb = chl_idx_key + "-" + dataset_name + ("-".join(model_rrs))

            tmp_train_features[name_comb] = chl_train_descriptor
            tmp_test_features[name_comb] = chl_test_descriptor

    tmp_train_features["Chla"] = yttrain
    tmp_test_features["Chla"] = yttest

    # tmp_train_features.dropna(inplace=True)
    # tmp_test_features.dropna(inplace=True)

    tmp_train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp_test_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    tmp_train_features.fillna(-666, inplace=True)
    tmp_test_features.fillna(-666, inplace=True)

    tmp_test_features_array = tmp_test_features.to_numpy()

    sub_params = ["custom", "BR"]
    for idx, k in enumerate(tmp_test_features.columns):
        for w in k.split("-"):
            if w in sub_params:
                tmp_test_features_array = tmp_test_features_array[tmp_test_features_array[:, idx] > 0]
                break

    tmp_test_features = pd.DataFrame(tmp_test_features_array, columns=tmp_test_features.columns)

    tmp_train_features = tmp_train_features.reset_index(drop=True)
    tmp_test_features = tmp_test_features.reset_index(drop=True)

    return tmp_train_features, tmp_test_features

def validate_tuned_model(model: str) -> bool:

    if model is None:
        return False

    if "tuned" not in model.stem:
        return False

    if "acolite" in model.stem or "6sv1" in model.stem:

        return True
    
    return False



def start_chl_estimation(sat_obj, model=None) -> None:
    """
    Estimates the chlorophyll using a trained ML Scikit-Learn Model

    :param sat_obj: Hypso satellite object
    :param model: Absolute path for the pre-trained Scikit-Learn model

    :return: No return.
    """
    # Load Mode -----------------------------------------------------
    if model is None:
        raise Exception("Please specify the model path")
    else:
        model = pathlib.Path(model)

    model = load(model)

    # Processing mask only for water pixels
    waterMask = sat_obj.waterMask

    # Get Cube with proper data -------------------------------------
    try:
        if "acolite" in model.stem:
            rrs_cube = sat_obj.l2a_cube["ACOLITE"]
        elif "6sv1" in model.stem:
            rrs_cube = sat_obj.l2a_cube["6SV1"]
        else:
            raise Exception("Only ACOLITE and 6SV1 correction are supported. Check model supplied")
    except TypeError as e:
        raise Exception("Generate ACOLITE and 6SV1 L2 Cube First. ", e)

    # Reshape 3D to 2D array --------------------------------------------------
    channels = rrs_cube.shape[2]
    rrs_array = rrs_cube.reshape((-1, channels))

    # Get DF with proper headers ---------------------------------------------
    hypso_headers = files(
        'hypso.experimental').joinpath('chlorophyll/data/hypso_wl_headers.csv')
    hypso_string_wl = list(np.loadtxt(hypso_headers, delimiter=",", dtype=str))

    rrs_df = pd.DataFrame(rrs_array, columns=hypso_string_wl)

    estimation = None
    if "tuned" in model.stem:
        hypso_optimal_features_df, _ = get_best_features(rrs_df, None, rrs_df, None,
                                                         dataset_name="hypso")
        hypso_optimal_features_df = hypso_optimal_features_df.loc[:,
                                    hypso_optimal_features_df.columns != 'Chla']

        # Filter -----------------------------------------------------------------
        sub_params = ["bratio"]
        sub_params = ["bratio", "bratio_log", "tbvi", "diff"]
        pre_filter = [k for k in hypso_optimal_features_df.columns for w in k.split("-") if w in sub_params]

        hypso_optimal_features_df = hypso_optimal_features_df[
            hypso_optimal_features_df.columns.intersection(pre_filter)]

        estimation = model.predict(hypso_optimal_features_df)
        estimation = np.reshape(estimation, sat_obj.spatialDim)

        # smooth ------------------------------------------------------------
        kernel = np.array([[1 / 16, 1 / 8, 1 / 16],  # 3x3 kernel
                           [1 / 8, 1 / 4, 1 / 8],
                           [1 / 16, 1 / 8, 1 / 16]], dtype=float)
        estimation = convolve2d(estimation, kernel, max_missing=0.5, verbose=True)

    else:
        raise Exception("No valid method found from model name.")

    estimation = np.multiply(estimation, waterMask.astype(int))

    for r in range(waterMask.shape[0]):
        for c in range(waterMask.shape[1]):
            if not waterMask[r, c]:
                estimation[r, c] = np.nan

    sat_obj.chl = estimation
    