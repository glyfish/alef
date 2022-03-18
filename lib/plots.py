import numpy
from matplotlib import pyplot
from lib import config
from enum import Enum

class RegressionPlotType(Enum):
    LINEAR = 1
    AGG_VAR = 2

def time_series(samples, time, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$X_t$")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)

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

def regression(y, x, result, title, type=RegressionPlotType.LINEAR):
    β = results.params
    σ = results.bse
    r2 = results.rsquared

    x_text = int(0.8*len(x))
    y_text = int(0.4*len(y))
    σ = σ[1] / 2.0

    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_title(title)
    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')

    if type == RegressionPlotType.AGG_VAR:
        y_fit = 10**β[0]*x**(β[1])
        h = float(1.0 + β[1]/2.0)
        ylabel = r"$Var(X^{m})$"
        xlabel = r"$m$"
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.3f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.3f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.3f')}"
        label = r"$Var(X^{m})=/beta *m^{2H-2}$"
    else:
        ylabel = "y"
        xlabel = "x"
        axis.set_ylabel("y")
        axis.set_xlabel("x")
        y_fit = β[0] + x*β[1]
        results_text = r"$\alpha=$" + f"{format(β[1], '2.3f')}\n" + \
                       r"$\beta=$" + f"{format(β[0], '2.3f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.3f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.3f')}"
        label = r"$y=\beta +\alpha x$"

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.text(x[x_text], y[y_text], results_text, bbox=bbox, fontsize=14.0, zorder=7)

    if type == RegressionPlotType.AGG_VAR:
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=10, label="Simulation")
        axis.loglog(x, y_fit, zorder=5, label=label)
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=10, label="Simulation")
        axis.plot(x, y_fit, zorder=5, label=label)

    axis.legend(bbox_to_anchor=[0.4, 0.4])

def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)
