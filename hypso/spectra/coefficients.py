from importlib.resources import files
import numpy as np


def print_this():
    spectral_coeff_file = files(
        'hypso.spectra').joinpath('data/hypso_wl_headers.csv')


    hypso_string_wl = list(np.loadtxt(spectral_coeff_file,
                                      delimiter=",", dtype=str))

    print(hypso_string_wl)
