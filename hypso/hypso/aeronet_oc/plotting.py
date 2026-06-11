import os
import sys
import numpy as np
from pathlib import Path

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import re



# --------------------------------------------------------------------------- #
#                              Plotting Utilities                             #
# --------------------------------------------------------------------------- #


def compute_bland_altman_metrics(xx, yy, xx_unc_modl, yy_unc_modl):
    """
    Compute metrics for Bland-Altman plot.

    Parameters
    ----------
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    xx_unc_modl : float
        Uncertainty in X.
    yy_unc_modl : float
        Uncertainty in Y.

    Returns
    -------
    dict
        Dictionary of Bland-Altman metrics.
    """
    jj = (xx + yy) / 2
    kk = (yy - xx) / np.sqrt((xx_unc_modl**2) + (yy_unc_modl**2))

    meanbias = np.mean(kk)
    stdbias = np.std(kk)
    LOAlow = meanbias - stdbias
    LOAhgh = meanbias + stdbias

    ba_stat, ba_p = stats.spearmanr(jj, kk)
    ba_independ = ba_p > 0.05

    return {
        "count": kk.shape[0],
        "jj": jj,
        "kk": kk,
        "meanbias": meanbias,
        "LOAlow": LOAlow,
        "LOAhgh": LOAhgh,
        "ba_stat": ba_stat,
        "ba_p": ba_p,
        "ba_independ": ba_independ
    }


def compute_regression_metrics(xx, yy, is_type2=False):
    """
    Compute regression metrics using specified type.

    Parameters
    ----------
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    is_type2 : bool, optional
        Whether to use Type 2 regression (orthogonal distance regression).
        Default is False, for Type 1 regression (ordinary least squares).

    Returns
    -------
    dict
        Dictionary of regression metrics.
    """
    if is_type2:
        # Perform Type 2 regression (orthogonal distance regression)
        def linear_model(B, x):
            """
            Linear function y = m*x + b.

            B is a vector of the parameters.
            x is an array of the current x values.
            x is in the same format as the x passed to Data or RealData.
            Return an array in the same format as y passed to Data or RealData.
            """
            return B[0] * x + B[1]

        # Create a model instance
        linear = odr.Model(linear_model)

        # Create a RealData object using the data
        data = odr.RealData(xx, yy)

        # Set up ODR with the model and data
        odr_instance = odr.ODR(data, linear, beta0=[1., 0.])

        # Run the regression
        odr_result = odr_instance.run()
        slope = odr_result.beta[0]
        intercept = odr_result.beta[1]
    else:
        # Perform Type 1 regression (ordinary least squares)
        regress_result = stats.linregress(xx, yy)
        slope = regress_result.slope
        intercept = regress_result.intercept

    spearman_r = stats.spearmanr(xx, yy)
    pearson_r = stats.pearsonr(xx, yy)
    rmse_all = np.sqrt(np.mean((yy - xx) ** 2))
    mae_all = np.mean(np.abs(yy - xx))

    return {
        "count": len(xx),
        "slope": slope,
        "intercept": intercept,
        "r_spear": spearman_r.correlation,
        "r_pear": pearson_r[0],
        "rmse": rmse_all,
        "mae": mae_all
    }


def add_text_annotations(ax, text_lines, position='top right',
                         fontsize=SIZE_TEXTLABEL):
    """
    Add text annotations to the plot.

    Parameters
    ----------
    ax : Axes
        The axis to add text to.
    text_lines : list of str
        List of strings to be displayed as text.
    position : str, default 'top right'
        Position of the text on the plot.
    fontsize : int, default 12
        Font size of the text.
    """
    if position == 'top right':
        x = 0.95
        y = 0.95
        ha = 'right'
        va = 'top'
    elif position == 'top left':
        x = 0.05
        y = 0.95
        ha = 'left'
        va = 'top'
    elif position == 'bottom left':
        x = 0.05
        y = 0.05
        ha = 'left'
        va = 'bottom'
    elif position == 'bottom right':
        x = 0.95
        y = 0.05
        ha = 'right'
        va = 'bottom'

    text = '\n'.join(text_lines)
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=fontsize,
        verticalalignment=va, horizontalalignment=ha,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )


def setup_plot(label):
    """
    Set up the plot with titles and labels.

    Parameters
    ----------
    label : str
        Title of the plot.

    Returns
    -------
    tuple
        Figure and axes of the plot.
    """
    style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    fig.suptitle(label, fontsize=22)
    return fig, ax1, ax2


def format_ticks(ax):
    """Format the tick labels on the axes to be more readable."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3g}'))
    ax.tick_params(axis='both', which='major', width=2, length=6)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


def plot_bland_altman(ax1, metrics, binscale, scat, xx_unc_modl,
                      x_label="x", y_label="y"):
    """
    Plot Bland-Altman plot.

    Parameters
    ----------
    ax1 : Axes
        Axis for the Bland-Altman plot.
    metrics : dict
        Bland-Altman metrics.
    binscale : float
        Scaling factor for bin size.
    scat : bool
        If False, plot as 2D histogram.
    xx_unc_modl : float
        Uncertainty in X.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data
    """
    jj = metrics["jj"]
    kk = metrics["kk"]
    npoints = metrics["count"]
    meanbias = metrics["meanbias"]
    LOAlow = metrics["LOAlow"]
    LOAhgh = metrics["LOAhgh"]
    ba_independ = metrics["ba_independ"]
    ba_stat = metrics["ba_stat"]

    nbin = int(0.5 * binscale * np.sqrt(len(jj)))
    min_kk = meanbias - 5 * np.std(kk)
    max_kk = meanbias + 5 * np.std(kk)

    gamma = 0.5
    if scat:
        min_jj = np.min(jj)
        max_jj = np.max(jj)
        lineclr, loaclr, fitclr = (COLOR_LINE, COLOR_LOA, COLOR_FITLINE)
        ax1.scatter(jj, kk, color=COLOR_SCATTER)
        ax1.set_xlim([min_jj, max_jj])
        ax1.set_ylim([min_kk, max_kk])
    else:
        jj_sorted = np.sort(jj)
        min_jj = jj_sorted[int(0.01 * len(jj))]
        max_jj = jj_sorted[int(0.99 * len(jj))]
        lineclr, loaclr, fitclr = ('white', 'yellow', 'cyan')
        h = ax1.hist2d(jj, kk, bins=(nbin, nbin),
                       norm=mcolors.PowerNorm(gamma), cmap=plt.cm.inferno,
                       range=[[min_jj, max_jj], [min_kk, max_kk]])
        plt.colorbar(h[3], ax=ax1)

    ax1.set_title('Bland-Altman plot', fontsize=SIZE_TITLE)
    ylabel = ('Uncertainty normalized bias' if xx_unc_modl != np.sqrt(0.5)
              else f'Bias, ${y_label}-{x_label}$')
    ax1.set_ylabel(ylabel, fontsize=SIZE_AXLABEL)
    ax1.set_xlabel(f'Paired mean, $({x_label}+{y_label})/2$',
                   fontsize=SIZE_AXLABEL)
    ax1.plot([min_jj, max_jj], [0, 0],
             color=lineclr, linestyle='solid', linewidth=4.0)

    if ba_independ:
        ax1.plot([min_jj, max_jj], [meanbias, meanbias],
                 color=fitclr, linestyle='dashed', linewidth=3.0,
                 label='Mean Bias')
        ax1.plot([min_jj, max_jj], [LOAlow, LOAlow],
                 color=loaclr, linestyle='dashed', linewidth=2.0,
                 label='Lower LOA')
        ax1.plot([min_jj, max_jj], [LOAhgh, LOAhgh],
                 color=loaclr, linestyle='dashed', linewidth=2.0,
                 label='Upper LOA')
        ax1.fill_between([min_jj, max_jj], LOAlow, LOAhgh,
                         color=loaclr, alpha=0.1)
    else:
        ba_regress_result = stats.linregress(jj, kk)
        ba_min_fit_yy = (ba_regress_result.slope * min_jj
                         + ba_regress_result.intercept)
        ba_max_fit_yy = (ba_regress_result.slope * max_jj
                         + ba_regress_result.intercept)
        ax1.plot([min_jj, max_jj], [ba_min_fit_yy, ba_max_fit_yy],
                 color=fitclr, linestyle='dashed', linewidth=3.0,
                 label='Linear Fit')
    if SHOW_LEGEND:
        ax1.legend()
    ax1.grid(True)
    format_ticks(ax1)

    text_lines = [
        f"Number of Points: {npoints}",
        f"Mean Bias: {meanbias:.2e}",
        f"Limits of Agreement: [{LOAlow:.2e}, {LOAhgh:.2e}]",
        f"Rank Correlation: {ba_stat:.3f}",
        "Bias Independent" if ba_independ else "Bias Dependent"
    ]
    add_text_annotations(ax1, text_lines, position='bottom right')


def plot_scatter(ax2, xx, yy, regress_metrics, binscale, scat,
                 x_label="x", y_label="y"):
    """
    Plot scatter plot with regression line.

    Parameters
    ----------
    ax2 : Axes
        Axis for the scatter plot.
    xx : array
        Array of X data values.
    yy : array
        Array of Y data values.
    regress_metrics : dict
        Regression metrics.
    binscale : float
        Scaling factor for bin size.
    scat : bool
        If False, plot as 2D histogram.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data
    """
    nbin = int(0.5 * binscale * np.sqrt(len(xx)))
    min_val = min(np.min(xx), np.min(yy))
    max_val = max(np.max(xx), np.max(yy))
    gamma = 0.5

    if scat:
        ax2.scatter(xx, yy, color=COLOR_SCATTER)
        ax2.set_xlim([min_val, max_val])
        ax2.set_ylim([min_val, max_val])
    else:
        g = ax2.hist2d(xx, yy, bins=(nbin, nbin),
                       norm=mcolors.PowerNorm(gamma), cmap=plt.cm.inferno,
                       range=[[min_val, max_val], [min_val, max_val]])
        plt.colorbar(g[3], ax=ax2)

    ax2.set_title('Scatterplot', fontsize=SIZE_TITLE)
    ax2.set_xlabel(f'${x_label}$', fontsize=SIZE_AXLABEL)
    ax2.set_ylabel(f'${y_label}$', fontsize=SIZE_AXLABEL)
    ax2.plot([min_val, max_val], [min_val, max_val],
             color=COLOR_LINE, linestyle='solid', linewidth=4.0)

    slope = regress_metrics["slope"]
    intercept = regress_metrics["intercept"]
    min_fit_yy = slope * min_val + intercept
    max_fit_yy = slope * max_val + intercept
    ax2.plot([min_val, max_val], [min_fit_yy, max_fit_yy],
             color=COLOR_FITLINE, linestyle='dashed', linewidth=3.0,
             label='Regression Line')
    if SHOW_LEGEND:
        ax2.legend()
    ax2.grid(True)
    format_ticks(ax2)

    text_lines = [
        f"Slope: {slope:.3f}",
        f"Intercept: {intercept:.2e}",
        f"Linear Correlation: {regress_metrics['r_pear']:.3f}",
        f"Rank Correlation: {regress_metrics['r_spear']:.3f}",
        f"RMSE: {regress_metrics['rmse']:.2e}",
        f"MAE: {regress_metrics['mae']:.2e}"
    ]
    add_text_annotations(ax2, text_lines, position='bottom right')


def plot_BAvsScat(x_input, y_input, label='',
                  saveplot=None, scat=True, binscale=1.0,
                  xx_unc_modl=np.sqrt(0.5), yy_unc_modl=np.sqrt(0.5),
                  x_label="x", y_label="y", is_type2=True):
    """
    Routine to plot paired data as Bland-Altman and scatter plot.

    Parameters
    ----------
    x_input : array-like
        Array of X data values.
    y_input : array-like
        Corresponding array of Y data values.
    label : string, default ''
        Text label for plotting.
    saveplot : string, default None
        Set to save plot in ../output/ with the string as the filename.
    scat : boolean, default True
        Make a 2D histogram if False, regular scatter plot if True.
    binscale : float, default 1.0
        Scaling factor for how many bins to include in a 2D histogram.
    xx_unc_modl : float, default np.sqrt(0.5)
        Uncertainty in X.
    yy_unc_modl : float, default np.sqrt(0.5)
        Uncertainty in Y.
    x_label : string, default "x"
        String for labels for x data
    y_label : string, default "y"
        String for labels for y data

    Returns
    -------
    dict
        Dictionary of computed statistics.
    """
    xx = np.asarray(x_input)
    yy = np.asarray(y_input)
    valid_indices = (np.isfinite(x_input) & np.isfinite(y_input)
                     & (x_input != -999) & (y_input != -999))
    xx = x_input[valid_indices]
    yy = y_input[valid_indices]

    ba_metrics = compute_bland_altman_metrics(xx, yy, xx_unc_modl, yy_unc_modl)
    regress_metrics = compute_regression_metrics(xx, yy, is_type2=is_type2)

    fig, ax1, ax2 = setup_plot(label)
    plot_bland_altman(ax1, ba_metrics, binscale, scat, xx_unc_modl,
                      x_label, y_label)
    plot_scatter(ax2, xx, yy, regress_metrics, binscale, scat,
                 x_label, y_label)

    if saveplot is not None:
        figpath = Path("../output") / saveplot
        fig.savefig(figpath)
        print('Saved figure to:', figpath)

    plt.show()

    return {
        "Number_of_Points": ba_metrics["count"],
        "Scale_Independence": ba_metrics["ba_independ"],
        "Mean_Bias": ba_metrics["meanbias"],
        "Limits_of_Agreement_low": (ba_metrics["LOAlow"]
                                    if ba_metrics["ba_independ"]
                                    else float("nan")),
        "Limits_of_Agreement_high": (ba_metrics["LOAhgh"]
                                     if ba_metrics["ba_independ"]
                                     else float("nan")),
        "Linear_Slope": regress_metrics["slope"],
        "Linear_Intercept": regress_metrics["intercept"],
        "Linear_Correlation": regress_metrics["r_pear"],
        "Rank_Correlation": regress_metrics["r_spear"],
        "RMSE": regress_metrics["rmse"],
        "MAE": regress_metrics["mae"]
    }