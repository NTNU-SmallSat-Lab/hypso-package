import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)




def get_best_features(rrs_df):
    y_rrs = rrs_df["Chla"]
    y_rrs = transformation(y_rrs)
    y_rrs = np.squeeze(y_rrs)

    rrs_df.drop("Chla", axis=1, inplace=True)
    X_rrs = rrs_df

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

    # hypso_optimal_dict = {
    #     "tbvi": [["Rrs_701", "Rrs_460"], ["Rrs_569", "Rrs_489"]],
    #     "bratio": [["Rrs_701", "Rrs_503"],
    #                ["Rrs_555", "Rrs_489"]],
    #     "bratio_log": [["Rrs_701", "Rrs_478"],
    #                    ["Rrs_569", "Rrs_489"]],
    #     "bdiff": [["Rrs_615", "Rrs_766"],
    #               ["Rrs_517", "Rrs_489"]]
    # }

    optimal_dict = hypso_optimal_dict

    global_fx = None
    feature_names = []
    for chl_idx_key in optimal_dict.keys():
        for model_rrs in optimal_dict[chl_idx_key]:
            feature_names.append(chl_idx_key + "-hypso" + ("-".join(model_rrs)))

    tmp_features = pd.DataFrame(np.ones((X_rrs.shape[0], len(feature_names) + 1),
                                        dtype=float), columns=["Chla"] + feature_names)

    yttrain = 1

    if y_rrs is None:
        y_rrs = [1 for k in range(X_rrs.shape[0])]

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
            wl = [X_rrs[wx] for wx in model_rrs]

            chl_train_descriptor = global_fx(wl)

            name_comb = chl_idx_key + "-hypso" + ("-".join(model_rrs))

            tmp_features[name_comb] = chl_train_descriptor

    tmp_features["Chla"] = y_rrs

    tmp_features = tmp_features.dropna()

    tmp_features = tmp_features.reset_index(drop=True)

    return tmp_features

def create_features(sat_obj):
