import numpy as np
def TBVI(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    return (wl_1 - wl_2) /(wl_1 + wl_2)


def BR(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    return wl_1/ wl_2


def FBM(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    return (wl_1 - wl_2)/ (wl_3 + wl_4)


def SINGLE(refl):
    wl_1 = refl[0]
    return wl_1


def BDRATIO(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]
    return (wl_1 - wl_2)/ (wl_3 - wl_4)


def BR_LOG(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    return np.log(wl_1/ wl_2)


def BDIFF(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    return wl_1 - wl_2


def BDIFF_LOG(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    return np.log(wl_1 - wl_2)


def BR_DIFF(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]
    return (wl_1/ wl_2) - (wl_3/ wl_4)


def BR_DIFF_LOG(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    return np.log((wl_1/ wl_2) - (wl_3/ wl_4))


def TBM(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return wl_3 * ((1/ wl_1) - (1/ wl_2))


def EXG(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return (2 * wl_1) - wl_2 - wl_3


def GLI(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return ((2 * wl_1) - wl_2 - wl_3)/ ((2 * wl_1) + wl_2 + wl_3)


def OCVI(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    return (wl_1/ wl_2) * ((wl_3/ wl_2) ** 0.64)


def EVI(refl):
    wl_1 = refl[0]
    wl_2 = refl[1]
    wl_3 = refl[2]
    wl_4 = refl[3]

    L = 1
    c1 = 6
    c2 = 7.5
    return wl_1 * (wl_2 - wl_3)/ (wl_2 + (c1 * wl_3) - (c2 * wl_4) + L)