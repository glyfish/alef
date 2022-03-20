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

class PlotDataType(Enum):
    TIME_SERIES = 1
    PSPEC = 2

def curve(x, y, title, data_type=PlotDataType.TIME_SERIES):
    if data_type == PlotDataType.TIME_SERIES:
        plot_type = PlotType.LINEAR
        xlabel = r"$t$"
        ylabel = r"$X_t$"
    elif data_type == PlotDataType.PSPEC:
        plot_type = PlotType.LOG
        xlabel = r"$\omega$"
        ylabel = r"$\rho_\omega$"
    else:
        plot_type = PlotType.LINEAR
        xlabel = "y"
        ylabel = "x"

    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if plot_type == PlotType.LOG:
        _logStyle(axis, y)
        axis.loglog(x, y, lw=1)
    elif plot_type == PlotType.XLOG:
        _logXStyle(axis, y)
        axis.semilogx(x, y, lw=1)
    elif plot_type == PlotType.YLOG:
        _logYStyle(axis, y)
        axis.semilog(x, y, lw=1)
    else:
        axis.plot(x, y, lw=1)

def comparison(x, y, labels, lengend_location, title, data_type=PlotDataType.TIME_SERIES):
    if data_type == PlotDataType.TIME_SERIES:
        plot_type= PlotType.LINEAR
        xlabel = r"$t$"
        ylabel = r"$X_t$"
    elif data_type == PlotDataType.PSPEC:
        plot_type = PlotType.LOG
        xlabel = r"$\omega$"
        ylabel = r"$\rho_\omega$"
    else:
        plot_type = PlotType.LINEAR
        xlabel = "y"
        ylabel = "x"

    nplot = len(y)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    for i in range(nplot):
        if plot_type == PlotType.LOG:
            axis.loglog(x, y[i], lw=1, label=labels[i])
        elif plot_type == PlotType.XLOG:
            _logXStyle(axis, ps)
            axis.semilogx(x, y[i], lw=1, label=labels[i])
        elif plot_type == PlotType.YLOG:
            _logYStyle(axis, ps)
            axis.semilogy(x, y[i], lw=1, label=labels[i])
        else:
            axis.plot(x, y[i], lw=1, label=labels[i])

    axis.legend(ncol=2, bbox_to_anchor=lengend_location)

def stack(y, ylim, labels, title, x=None, data_type=PlotDataType.TIME_SERIES):
    if data_type == PlotDataType.TIME_SERIES:
        plot_type = PlotType.LINEAR
        xlabel = r"$t$"
        ylabel = r"$X_t$"
    else:
        plot_type = PlotType.LINEAR
        xlabel = "y"
        ylabel = "x"

    nplot = len(y)
    if x is None:
        nx = len(y[0])
        x = numpy.tile(numpy.linspace(0, nx-1, nx), (nplot,1))

    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(15, 12))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(xlabel)

    for i in range(nplot):
        nsample = len(y[i])
        axis[i].set_ylabel(ylabel)
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, x[i][-1]])
        text = axis[i].text(x[i][int(0.9*nsample)], 0.65*ylim[-1], labels[i], fontsize=18)
        text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
        if plot_type == PlotType.LOG:
            axis[i].loglog(x[i], y[i], lw=1.0)
        elif plot_type == PlotType.XLOG:
            axis[i].semilogx(x[i], y[i], lw=1.0)
        elif plot_type == PlotType.YLOG:
            axis[i].semilogy(x[i], y[i], lw=1.0)
        else:
            axis[i].plot(x[i], y[i], lw=1.0)

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
        _logStyle(axis, x)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Data")
        axis.loglog(x, y_fit, zorder=10, label=label)
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Data")
        axis.plot(x, y_fit, zorder=10, label=label)

    axis.legend(bbox_to_anchor=legend_loc)

def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)

### -- Private ---
def _logStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 3:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def _logXStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 3:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def _logYStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 3:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['left'].set_color("#b0b0b0")
