#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



### The band index that is closes to the required value
def closest_index(input_list, lookup_value):
    difference = lambda input_list: abs(input_list - lookup_value)
    res = min(input_list, key=difference)
    res_idx = np.where(input_list == res)[0][0]
    return res_idx


def blended_chl(chl_ci, chl_ocx, t1, t2):  # Default blending window of 0.15 to 0.2
    """A general Chl algorithm blending function between Chl_CI to Chl_OCx"""

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


def calculate_chl_ocx(ocx_poly, lmbr):
    """A general Chl_OCx algorithm for fourth order polynomials (O'Reilly et al., 1998)"""
    chl_ocx = 10 ** (ocx_poly[0] + (ocx_poly[1] * lmbr ** 1) + (ocx_poly[2] * lmbr ** 2) + (ocx_poly[3] * lmbr ** 3) + (
            ocx_poly[4] * lmbr ** 4))
    return chl_ocx


def calculate_chl_ci(ci_poly, CI):
    """A general Chl_CI algorithm for a linear polynomial (Hu et al., 2012)"""
    chl_ci = 10 ** (ci_poly[0] + ci_poly[1] * CI)
    return chl_ci


def modis_aqua_ocx(wl, hypercube, ocx_version):
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


def hico_ocx_noOCI(wl, hypercube):
    """
        Given:
            MODIS-Aqua RRS values for 443,488,547,667
            OCx Polynomial
            CI Polynomial
            l - Low blending cutoff
            h - High blending cutoff

        Calculate:
            Calculate Chl OCx
            Calculate Chl CI
            Return blended chlorophyll esimate (Default product is the Pittman et al., 2019 TPCA)
    """
    # Polynomials: Chlorophyll algorithms for ocean color sensors - OC4, OC5 & OC6
    # OCI Parameters -> Hu2012: [-0.4287, 230.47]  # Same for both algorithms (CI)
    # Chl breakup ranges: NASA https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/

    # Calculate Chl OCX (O'Reilly et al., 1998)
    # Blue 1
    index_r416 = closest_index(wl, 416)
    r416 = hypercube[:, :, index_r416]

    # Blue 2
    index_r444 = closest_index(wl, 444)
    r444 = hypercube[:, :, index_r444]

    # Blue 3
    index_r490 = closest_index(wl, 490)
    r490 = hypercube[:, :, index_r490]

    # Blue 3
    index_r513 = closest_index(wl, 513)
    r513 = hypercube[:, :, index_r513]

    # Green
    index_r553 = closest_index(wl, 553)
    r553 = hypercube[:, :, index_r553]

    # Red
    index_r668 = closest_index(wl, 668)
    r668 = hypercube[:, :, index_r668]

    # Note: M(555&670), for example, represents the mean of Rrs555 and Rrs670
    ocx_poly = [0.26869, 0.96178, -3.43787, 2.80047, -1.59267]

    with np.errstate(divide='ignore'):  # 416, 444, 490, 513)}{mean(553,668)
        denominator_rrs = (r553 + r668) / 2.0
        divide_condition = np.logical_or(denominator_rrs == 0, denominator_rrs == np.nan)

        div = np.where(divide_condition, np.nan, r416 / denominator_rrs)
        div2 = np.where(divide_condition, np.nan, r444 / denominator_rrs)
        div3 = np.where(divide_condition, np.nan, r490 / denominator_rrs)
        div4 = np.where(divide_condition, np.nan, r513 / denominator_rrs)

        mbr = np.maximum(div, div2)  # Calculate max band ratio
        mbr = np.maximum(mbr, div3)
        mbr = np.maximum(mbr, div4)

    with np.errstate(divide='ignore'):
        lmbr = np.where(np.logical_or(mbr == 0, mbr == np.nan), np.nan, np.log10(mbr))

        chl_ocx = calculate_chl_ocx(ocx_poly, lmbr)

    return chl_ocx, mbr


def sentinel_ocx(wl, hypercube, ocx_version):
    """
        Given:
            MODIS-Aqua RRS values for 443,488,547,667
            OCx Polynomial
            CI Polynomial
            l - Low blending cutoff
            h - High blending cutoff

        Calculate:
            Calculate Chl OCx
            Calculate Chl CI
            Return blended chlorophyll esimate (Default product is the Pittman et al., 2019 TPCA)
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


def plot_chl_values(chl_array, lat, lon, plotTitle='Cool Plot', customDPI=450):
    extent_lon_min = lon.min()
    extent_lon_max = lon.max()

    extent_lat_min = lat.min()
    extent_lat_max = lat.max()
    plotZoomFactor = 1.0  # 1.05 for normal and 1.0 for debug
    projection = ccrs.Mercator()
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    crs = ccrs.PlateCarree()
    # Now we will create axes object having specific projection

    fig = plt.figure(figsize=(5, 10), dpi=customDPI)
    fig.patch.set_alpha(1)
    ax = fig.add_subplot(projection=projection, frameon=True)

    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(draw_labels=True,
                      linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land

    ax.set_extent([extent_lon_min / plotZoomFactor, extent_lon_max * plotZoomFactor,
                   extent_lat_min / plotZoomFactor, extent_lat_max * plotZoomFactor], crs=crs)

    # fig, ax = plt.subplots(figsize=(10, 10), dpi=450)
    # fig.patch.set_alpha(1)

    min_chlr_val = np.nanmin(chl_array)
    lower_limit_chl = 0.01 if min_chlr_val < 0.01 else min_chlr_val

    max_chlr_val = np.nanmax(chl_array)
    upper_limit_chl = 100 if max_chlr_val > 100 else max_chlr_val

    # Set Range to next full log
    for i in range(-2, 3):
        full_log = 10 ** i
        if full_log < lower_limit_chl:
            lower_limit_chl = full_log
    for i in range(2, -3, -1):
        full_log = 10 ** i
        if full_log > upper_limit_chl:
            upper_limit_chl = full_log

    chl_range = [lower_limit_chl, upper_limit_chl]  # old: [0.01, 100] [0.3, 1]
    # chl_range = [0.01, 100]  # OLD
    # chl_range = [0.3, 1] # OLD

    print('Chl Range: ', chl_range)

    # Log Normalize Color Bar colors
    norm = colors.LogNorm(chl_range[0], chl_range[1])
    # im = plt.pcolormesh(chl_array, cmap=plt.cm.jet, norm=norm)
    im = ax.pcolormesh(lon, lat, chl_array,
                       cmap=plt.cm.jet, transform=ccrs.PlateCarree(), norm=norm, zorder=0)

    # Colourmap with axes to match figure size
    cbar = plt.colorbar(im, location='right', shrink=1, ax=ax, pad=0.05)

    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

    # cbar.ax.yaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))
    # cbar.ax.xaxis.set_minor_formatter(mticker.ScalarFormatter(useMathText=False, useOffset=False))

    cbar.set_label(f" Chlorophyll Concentration [mg m^-3]")

    plt.title(plotTitle)
    plt.show()


def myLogFormat(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    if decimalplaces == 0:
        # Insert that number into a format string
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
        # Return the formatted tick label
        return formatstring.format(y)
    else:
        formatstring = '{:2.1e}'.format(y)
        return formatstring


def custom_MCI(wl, hypercube, lambda1, lambda2, lambda3):
    index_lambda1 = closest_index(wl, lambda1)
    rrs_lambda1 = hypercube[:, :, index_lambda1]

    index_lambda2 = closest_index(wl, lambda2)
    rrs_lambda2 = hypercube[:, :, index_lambda2]

    index_lambda3 = closest_index(wl, lambda3)
    rrs_lambda3 = hypercube[:, :, index_lambda3]

    MCI = rrs_lambda2 - rrs_lambda1 + (
            rrs_lambda1 -
            (rrs_lambda3 *
             (lambda2 - lambda1) /
             (lambda3 - lambda1)
             )
    )

    # MCI = rrs_lambda2 - rrs_lambda1 - ((rrs_lambda3 - rrs_lambda1) * (lambda2 - lambda1) / (lambda3 - lambda1))

    return MCI


def hypso_ocx(wl, hypercube, ocx_version, ocx_poly=[-0.4287, 230.47], numerator_bands=[443, 490, 510],
              denominator_bands=[560], t=[0.25, 0.35]):
    ci_poly = [-0.4287, 230.47]
    t1 = 0.25
    t2 = 0.35

    # -------------------------------------
    # Blue 1
    index_r412 = closest_index(wl, 412)
    r412 = hypercube[:, :, index_r412]

    index_r432 = closest_index(wl, 432)
    r432 = hypercube[:, :, index_r432]

    # Blue 1
    index_r443 = closest_index(wl, 443)
    r443 = hypercube[:, :, index_r443]

    # Blue 2
    index_r488 = closest_index(wl, 488)
    r488 = hypercube[:, :, index_r488]

    # Green
    index_r500 = closest_index(wl, 500)
    r500 = hypercube[:, :, index_r500]

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
    lambda_blue = 432
    lambda_green = 555
    lambda_red = 667

    rrs_green = r555
    rrs_blue = r432
    rrs_red = r667
    if ocx_version == 4:  # MBR = Rrs(412 > 443 > 488)/Rrs555
        ocx_poly = [0.27015, -2.47936, 1.53752, -0.13967, -0.66166]

        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

            div = np.where(divide_condition, np.nan, r432 / r555)
            div2 = np.where(divide_condition, np.nan, r443 / r555)
            div3 = np.where(divide_condition, np.nan, r488 / r555)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)

            max_mbr = -np.inf
            for i in range(390, 550):
                index_rexp = closest_index(wl, i)
                rexp = hypercube[:, :, index_rexp]
                divexp = np.where(divide_condition, np.nan, rexp / r555)

                mbrxx = np.maximum(div, div2)  # Calculate max band ratio
                mbrxx = np.maximum(mbrxx, div3)
                mbrxx = np.maximum(mbrxx, divexp)

                if np.nanmax(mbrxx) > max_mbr:
                    max_mbr = np.nanmax(mbrxx)
            print("Max MBR Experimental OC4: ", np.nanmax(mbrxx))

    elif ocx_version == 5:  # MBR = Rrs(412 > 443 > 488 > 531)/ Rrs 555
        ocx_poly = [0.42919, -4.88411, 9.57678, -9.24289, 2.51916]

        with np.errstate(divide='ignore'):
            divide_condition = np.logical_or(r555 == 0, r555 == np.nan)

            div = np.where(divide_condition, np.nan, r432 / r555)
            div2 = np.where(divide_condition, np.nan, r443 / r555)
            div3 = np.where(divide_condition, np.nan, r488 / r555)
            div4 = np.where(divide_condition, np.nan, r531 / r555)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)

    elif ocx_version == 6:  # MBR = Rrs(412 > 443 > 488 > 531)/M(555&667)
        ocx_poly = [1.22914, -4.99423, 5.64706, -3.53426, 0.69266]

        with np.errstate(divide='ignore'):
            denominator_rrs = (r555 + r667) / 5.0
            divide_condition = np.logical_or(denominator_rrs == 0, denominator_rrs == np.nan)

            div = np.where(divide_condition, np.nan, r432 / denominator_rrs)
            div2 = np.where(divide_condition, np.nan, r443 / denominator_rrs)
            div3 = np.where(divide_condition, np.nan, r488 / denominator_rrs)
            div4 = np.where(divide_condition, np.nan, r500 / denominator_rrs)
            div5 = np.where(divide_condition, np.nan, r531 / denominator_rrs)

            mbr = np.maximum(div, div2)  # Calculate max band ratio
            mbr = np.maximum(mbr, div3)
            mbr = np.maximum(mbr, div4)
            mbr = np.maximum(mbr, div5)

            max_mbr = -np.inf
            for i in range(390, 550):
                index_rexp = closest_index(wl, i)
                rexp = hypercube[:, :, index_rexp]
                divexp = np.where(divide_condition, np.nan, rexp / denominator_rrs)

                mbrxx = np.maximum(div, div2)  # Calculate max band ratio
                mbrxx = np.maximum(mbrxx, div3)
                mbrxx = np.maximum(mbrxx, div4)
                mbrxx = np.maximum(mbrxx, divexp)

                if np.nanmax(mbrxx) > max_mbr:
                    max_mbr = np.nanmax(mbrxx)
            print("Max MBR Experimental OC6: ", np.nanmax(mbrxx))

    # Calculate Chl OCX (O'Reilly et al., 1998)
    with np.errstate(divide='ignore'):
        lmbr = np.where(np.logical_or(mbr == 0, mbr == np.nan), np.nan, np.log10(mbr))

        chl_ocx = calculate_chl_ocx(ocx_poly, lmbr)
        # Calculate Chl CI (Hu et al., 2012)
        CI = rrs_green - (rrs_blue + (lambda_green - lambda_blue) / (lambda_red - lambda_blue) * (rrs_red - rrs_blue))
        chl_ci = calculate_chl_ci(ci_poly, CI)

    blended = blended_chl(chl_ci, chl_ocx, t1, t2)  # Blending cutoff
    return blended, mbr, chl_ocx, chl_ci, ocx_poly
