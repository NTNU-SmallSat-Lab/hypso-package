"""
Created on Wed June 12, 2019

This module contains a functions to calculate the tropical Pacific chlorophyll algorithm (TPCA) concentrations for the satellite sensors; SeaWiFS, MODIS-Aqua and MERIS
Developed for Pittman et al., 2019. JGR: Oceans (2019JC015498)

Notes:
    Numpy has been used over pure python or the math library for more efficient implementations. For increased efficiency, dask (dask.array) can be used instead of numpy.
    These functions are designed for Level 3 Mapped products, which may not precisely reproduce the implementations available on the NASA website (https://oceandata.sci.gsfc.nasa.gov/). 

General functions include: 
    blended_chl
    calculate_chl_ocx
    calculate_chl_ci
    
Sensor specific functions include:
    calculate_seawifs_chl
    calcuate_modis_chl
    calculate_meris_chl

@author: Nicholas.Pittman
@email: Nic.Pittman@utas.edu.au
@position: PhD Candidate; Biogeochemistry, remote sensing, oceanography, Tropical Pacific
@affiliation1: Institute of Marine and Antarctic Studies, University of Tasmania
@affiliation2: Australian Research Council Centre of Excellence for Climate Extremes.

References:
    Tropical Pacific Chlorophyll Algorithm (2019JC015498)
    'An assessment and improvement of satellite ocean color algorithms for the tropical Pacific Ocean'
    Pittman, N., Strutton, P., Matear, R., Johnson, R., (2019)
    Submitted: Journal of Geophysical Research: Oceans
    
    CI and blending window
    Hu, C., Lee, Z., and Franz, B. (2012). 
    Chlorophyll a algorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference.
    Journal of Geophysical Research: Oceans 117.
    
    Traditional OCx Algorithm
    O’Reilly, J.E., Maritorena, S., Mitchell, B.G., Siegel, D.A., 
    Carder, K.L., Garver, S.A., Kahru, M., and McClain, C. (1998).
    Ocean color chlorophyll algorithms for SeaWiFS. 
    Journal of Geophysical Research: Oceans 103, 24937–24953.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cf
from typing import Union, Tuple


### The band index that is closes to the required value
def closest_index(input_list, lookup_value) -> np.ndarray:
    """
    Find the closest index to lookup_value

    :param input_list: List up to look on the value
    :param lookup_value: Value to lookup

    :return: Closest indext to the loookup_value
    """
    difference = lambda input_list: abs(input_list - lookup_value)
    res = min(input_list, key=difference)
    res_idx = np.where(input_list == res)[0][0]
    return res_idx


def blended_chl(chl_ci, chl_ocx, t1, t2) -> np.ndarray:  # Default blending window of 0.15 to 0.2
    """
    A general Chl algorithm blending function between Chl_CI to Chl_OCx

    :param chl_ci:
    :param chl_ocx:
    :param t1:
    :param t2:
    :return:
    """

    # TODO: Check if blending with mean is better
    # Blended with Mean
    # https://www-cdn.eumetsat.int/files/2021-02/Sentinel-3%20OLCI%20Chlorophyll%20Index%20switch%20for%20low-chlorophyll%20waters_ATBD.pdf
    # https://earth.esa.int/eogateway/documents/20142/37627/MERIS_ATBD_2.9_v4.3+-+2011.pdf/e03662c8-7b57-fb00-f6c7-46f5a950c4be
    # chl_mean = (chl_ocx + chl_ci) / 2.0
    #
    # alpha = (chl_mean - t1) / (t2 - t1)
    #
    # beta = (t2 - chl_mean) / (t2 - t1)

    alpha = (chl_ci - t1) / (t2 - t1)

    beta = (t2 - chl_ci) / (t2 - t1)

    chl = (alpha * chl_ocx) + beta * chl_ci

    chl = np.where(chl_ci < t1, chl_ci, chl)
    chl = np.where(chl_ci > t2, chl_ocx, chl)
    return chl


def calculate_chl_ocx(ocx_poly: list, lmbr: np.ndarray) -> Union[int, float]:
    """
    A general Chl_OCx algorithm for fourth order polynomials (O'Reilly et al., 1998)

    :param ocx_poly:
    :param lmbr:
    :return:
    """

    chl_ocx = 10 ** (ocx_poly[0] + (ocx_poly[1] * lmbr ** 1) + (ocx_poly[2] * lmbr ** 2) + (ocx_poly[3] * lmbr ** 3) + (
            ocx_poly[4] * lmbr ** 4))
    return chl_ocx


def calculate_chl_ci(ci_poly: list, CI: np.ndarray) -> list:
    """
    A general Chl_CI algorithm for a linear polynomial (Hu et al., 2012)

    :param ci_poly:
    :param CI:
    :return:
    """

    chl_ci = 10 ** (ci_poly[0] + ci_poly[1] * CI)
    return chl_ci


def modis_aqua_ocx(wl, hypercube, ocx_version):
    """
    Calculate the MODIS AQUA OCX for Chlorophyll estimation

    :param wl:
    :param hypercube:
    :param ocx_version:
    :return:
    """
    ci_poly = [-0.4287, 230.47]
    t1 = 0.25
    t2 = 0.35

    # -------------------------------------
    # Blue 1
    index_r412 = closest_index(wl, 412)
    r412 = hypercube[:, :, index_r412]

    # Blue 1
    index_r443 = closest_index(wl, 443)
    r443 = hypercube[:, :, index_r443]

    # Blue 2
    index_r488 = closest_index(wl, 488)
    r488 = hypercube[:, :, index_r488]

    # Green
    index_r531 = closest_index(wl, 531)
    r531 = hypercube[:, :, index_r531]

    # Green
    index_r547 = closest_index(wl, 547)
    r547 = hypercube[:, :, index_r547]

    # Green
    index_r555 = closest_index(wl, 555)
    r555 = hypercube[:, :, index_r555]

    # Red
    index_r667 = closest_index(wl, 667)
    r667 = hypercube[:, :, index_r667]

    # Note: M(555&670), for example, represents the mean of Rrs555 and Rrs670
    ocx_poly = None
    lambda_blue = 412
    lambda_green = 555
    lambda_red = 667

    rrs_green = r555
    rrs_blue = r412
    rrs_red = r667
    if ocx_version == 4:  # MBR = Rrs(412 > 443 > 488)/Rrs555
        ocx_poly = [0.27015, -2.47936, 1.53752, -0.13967, -0.66166]

        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

            div = np.where(divide_condition, np.nan, r412 / r555)
            div2 = np.where(divide_condition, np.nan, r443 / r555)
            div3 = np.where(divide_condition, np.nan, r488 / r555)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)

    elif ocx_version == 5:  # MBR = Rrs(412 > 443 > 488 > 531)/ Rrs 555
        ocx_poly = [0.42919, -4.88411, 9.57678, -9.24289, 2.51916]

        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

            div = np.where(divide_condition, np.nan, r412 / r555)
            div2 = np.where(divide_condition, np.nan, r443 / r555)
            div3 = np.where(divide_condition, np.nan, r488 / r555)
            div4 = np.where(divide_condition, np.nan, r531 / r555)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)

    elif ocx_version == 6:  # MBR = Rrs(412 > 443 > 488 > 531)/M(555&667)
        ocx_poly = [1.22914, -4.99423, 5.64706, -3.53426, 0.69266]

        with np.errstate(divide='ignore'):
            denominator_rrs = (r555 + r667) / 2.0
            divide_condition = np.logical_or(denominator_rrs == 0, denominator_rrs == np.nan)

            div = np.where(divide_condition, np.nan, r412 / denominator_rrs)
            div2 = np.where(divide_condition, np.nan, r443 / denominator_rrs)
            div3 = np.where(divide_condition, np.nan, r488 / denominator_rrs)
            div4 = np.where(divide_condition, np.nan, r531 / denominator_rrs)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)

            # Calculate Chl OCX (O'Reilly et al., 1998)
    with np.errstate(divide='ignore'):
        lmbr = np.where(np.logical_or(mbr == 0, mbr == np.nan), np.nan, np.log10(mbr))

        chl_ocx = calculate_chl_ocx(ocx_poly, lmbr)
        # Calculate Chl CI (Hu et al., 2012)
        CI = rrs_green - (rrs_blue + (lambda_green - lambda_blue) / (lambda_red - lambda_blue) * (rrs_red - rrs_blue))
        chl_ci = calculate_chl_ci(ci_poly, CI)

    blended = blended_chl(chl_ci, chl_ocx, t1, t2)  # Blending cutoff
    return blended, mbr, chl_ocx, chl_ci, ocx_poly


def sentinel_ocx(wl, hypercube, ocx_version):
    """
    Calculate chlorophyll using the Sentinel OCX Polynomial.\n
    **Given:**\n
    ------ MODIS-Aqua RRS values for 443,488,547,667\n
    ------ OCx: Polynomial\n
    ------ CI: Polynomial\n
    ------ l: Low blending cutoff\n
    ------ h: High blending cutoff\n
    **Calculate:**\n
    ------ Calculate Chl OCx\n
    ------ Calculate Chl CI\n
    ------ Return blended chlorophyll esimate (Default product is the Pittman et al., 2019 TPCA)\n

    :param wl:
    :param hypercube:
    :param ocx_version:
    :return:
    """

    # Polynomials: Chlorophyll algorithms for ocean color sensors - OC4, OC5 & OC6
    # OCI Parameters -> Hu2012: [-0.4287, 230.47]  # Same for both algorithms (CI)
    # Chl breakup ranges: NASA https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/

    # Calculate Chl OCX (O'Reilly et al., 1998)

    ci_poly = [-0.4287, 230.47]
    t1 = 0.25
    t2 = 0.35

    # Blue 1
    index_r413 = closest_index(wl, 413)
    r413 = hypercube[:, :, index_r413]

    # Blue 1
    index_r443 = closest_index(wl, 443)
    r443 = hypercube[:, :, index_r443]

    # Blue 2
    index_r490 = closest_index(wl, 490)
    r490 = hypercube[:, :, index_r490]

    # Blue 3
    index_r510 = closest_index(wl, 510)
    r510 = hypercube[:, :, index_r510]

    # Green
    index_r560 = closest_index(wl, 560)
    r560 = hypercube[:, :, index_r560]

    # Red
    index_r665 = closest_index(wl, 665)
    r665 = hypercube[:, :, index_r665]

    # Note: M(555&670), for example, represents the mean of Rrs555 and Rrs670
    ocx_poly = None
    lambda_blue = 442.5
    lambda_green = 560
    lambda_red = 665

    rrs_green = r560
    rrs_blue = r443
    rrs_red = r665
    if ocx_version == 4:  # MBR = Rrs(443 > 490 > 510)/Rrs560
        ocx_poly = [0.42540, -3.21679, 2.86907, -0.62628, -1.09333]
        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r560 == 0, r560 == np.nan)
            div = np.where(divide_condition, np.nan, r443 / r560)
            div2 = np.where(divide_condition, np.nan, r490 / r560)
            div3 = np.where(divide_condition, np.nan, r510 / r560)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)

    elif ocx_version == 5:  # MBR = Rrs(413 > 443 > 490 > 510)/ Rrs 560
        ocx_poly = [0.43213, -3.13001, 3.05479, -1.45176, -0.24947]
        lambda_blue = 413.0
        rrs_blue = r413
        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r560 == 0, r560 == np.nan)

            div = np.where(divide_condition, np.nan, r413 / r560)
            div2 = np.where(divide_condition, np.nan, r443 / r560)
            div3 = np.where(divide_condition, np.nan, r490 / r560)
            div4 = np.where(divide_condition, np.nan, r510 / r560)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)

    elif ocx_version == 6:  # MBR = Rrs(413 > 443 > 490 > 510)/M(560&665)
        ocx_poly = [0.95039, -3.05404, 2.17992, -1.12097, 0.15262]
        lambda_blue = 413.0
        rrs_blue = r413
        with np.errstate(divide='ignore'):
            denominator_rrs = (r560 + r665) / 2.0
            divide_condition = np.logical_or(denominator_rrs == 0, denominator_rrs == np.nan)

            div = np.where(divide_condition, np.nan, r413 / denominator_rrs)
            div2 = np.where(divide_condition, np.nan, r443 / denominator_rrs)
            div3 = np.where(divide_condition, np.nan, r490 / denominator_rrs)
            div4 = np.where(divide_condition, np.nan, r510 / denominator_rrs)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)

    with np.errstate(divide='ignore'):

        lmbr = np.where(np.logical_or(mbr == 0, mbr == np.nan), np.nan, np.log10(mbr))

        chl_ocx = calculate_chl_ocx(ocx_poly, lmbr)
        # Calculate Chl CI (Hu et al., 2012)
        CI = rrs_green - (rrs_blue + (lambda_green - lambda_blue) / (lambda_red - lambda_blue) * (rrs_red - rrs_blue))
        chl_ci = calculate_chl_ci(ci_poly, CI)

    blended = blended_chl(chl_ci, chl_ocx, t1, t2)  # Blending cutoff
    return blended, mbr, chl_ocx, chl_ci, ocx_poly


