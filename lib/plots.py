import numpy
from matplotlib import pyplot
from lib import config

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

def time_series_stack(series, labels, ylim, title):
    nplot = len(series)
    nsample = len(series[0])
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(15, 12))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        axis[i].set_ylabel(r"$X_t$")
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, nsample])
        axis[i].text(time[int(0.9*nsample)], 0.65*ylim[-1], labels[i], fontsize=18)
        axis[i].plot(time, series[i], lw=1.0)

def cumulative_mean(acum, μ, title, ylim=None, legend_pos = None):
    nsample = len(acum)
    time = numpy.linspace(1.0, nsample, nsample)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    if ylim is not None:
        axis.set_ylim(ylim)
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$μ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
    axis.semilogx(time, acum, label=f"Cumulative μ")
    if legend_pos is None:
        axis.legend()
    else:
        axis.legend(bbox_to_anchor=legend_pos)

def cumulative_std(acum, σ, title, ylim=None, legend_pos = None):
    nsample = len(acum)
    time = numpy.linspace(1.0, nsample, nsample)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    if ylim is not None:
        axis.set_ylim(ylim)
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$σ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), σ), label="Target σ", color="#000000")
    axis.semilogx(time, acum, label=f"Cumulative σ")
    if legend_pos is None:
        axis.legend()
    else:
        axis.legend(bbox_to_anchor=legend_pos)
