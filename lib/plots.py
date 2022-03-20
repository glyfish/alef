import numpy
from matplotlib import pyplot
from lib import config
from enum import Enum
from matplotlib.ticker import (MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogFormatterMathtext)

class RegressionPlotType(Enum):
    LINEAR = 1
    FBM_AGG_VAR = 2
    FBM_PSPEC = 3

class PlotType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

def time_series(samples, time, title):
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$X_t$")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)

def pspec(ps, time, title):
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$\omega$")
    axis.set_ylabel(r"$\rho_\omega$")
    axis.set_title(title)
    logStyle(axis, ps)
    axis.loglog(time, ps, lw=1)

def time_series_comparison(samples, time, labels, lengend_location, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$X_t$")
    axis.set_title(title)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1, label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)

def time_series_stack(series, ylim, labels, title, time=None):
    nplot = len(series)
    if time is None:
        nsample = len(series[0])
        time = numpy.tile(numpy.linspace(0, nsample-1, nsample), (nplot,1))
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(15, 12))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    for i in range(nplot):
        nsample = len(series[i])
        axis[i].set_ylabel(r"$X_t$")
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, time[i][-1]])
        text = axis[i].text(time[i][int(0.9*nsample)], 0.65*ylim[-1], labels[i], fontsize=18)
        text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
        axis[i].plot(time[i], series[i], lw=1.0)

def cumulative(accum, target, title, label):
    range = max(accum) - min(accum)
    legend_pos=[0.85, 0.95]
    nsample = len(accum)
    time = numpy.linspace(1.0, nsample, nsample)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_ylim([min(accum)-0.25*range, max(accum)+0.25*range])
    axis.set_xlabel("Time")
    axis.set_ylabel(label)
    axis.set_title(title)
    axis.set_xlim([1.0, nsample])
    axis.semilogx(time, accum, label=f"Cumulative "+label)
    axis.semilogx(time, numpy.full((len(time)), target), label="Target "+label)
    axis.legend(bbox_to_anchor=legend_pos)

def acf_pacf(title, acf, pacf, max_lag):
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(max_lag+1), acf, label="ACF")
    axis.plot(range(1, max_lag+1), pacf, label="PACF")
    axis.legend(fontsize=16)

def regression(y, x, results, title, type=RegressionPlotType.LINEAR):
    β = results.params
    σ = results.bse[1] / 2
    r2 = results.rsquared

    if β[1] < 0:
        x_text = 0.8
        y_text = 0.825
        legend_loc = [0.4, 0.3]
    else:
        x_text = 0.2
        y_text = 0.8
        legend_loc = [0.8, 0.3]

    if type == RegressionPlotType.FBM_AGG_VAR:
        plot_type = PlotType.LOG
        y_fit = 10**β[0]*x**(β[1])
        h = float(1.0 + β[1]/2.0)
        ylabel = r"$Var(X^{m})$"
        xlabel = r"$m$"
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{\sigma}^2=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        label = r"$Var(X^{m})=\sigma^2 m^{2H-2}$"
    elif type == RegressionPlotType.FBM_PSPEC:
        plot_type = PlotType.LOG
        y_fit = 10**β[0]*x**(β[1])
        h = float(1.0 - β[1])/2.0
        ylabel = r"$\hat{\rho}^H_\omega$"
        xlabel = r"$\omega$"
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{C}=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        label = r"$\hat{\rho}^H_\omega = C | \omega |^{1 - 2H}$"
    else:
        y_fit = β[0] + x*β[1]
        ylabel = "y"
        xlabel = "x"
        results_text = r"$\alpha=$" + f"{format(β[1], '2.2f')}\n" + \
                       r"$\beta=$" + f"{format(β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        label = r"$y=\beta + \alpha x$"

    figure, axis = pyplot.subplots(figsize=(15, 12))

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_type == PlotType.LOG:
        logStyle(axis, x)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Data")
        axis.loglog(x, y_fit, zorder=10, label=label)
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Data")
        axis.plot(x, y_fit, zorder=10, label=label)

    axis.legend(bbox_to_anchor=legend_loc)

def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)

def logStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 3:
        axis.tick_params(axis='both', which='minor', length=8, color="#c0c0c0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#c0c0c0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#c0c0c0")
        axis.spines['left'].set_color("#c0c0c0")
        axis.set_xlim([min(x), max(x)])
