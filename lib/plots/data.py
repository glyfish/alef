import numpy
from enum import Enum
from matplotlib import pyplot

from lib.plots.config import (PlotType, logStyle, logXStyle, logYStyle)

# Specify PlotConfig for curve, comparison and stack plots
class DataPlotType(Enum):
    GENERIC = 1         # Unknown data type
    TIME_SERIES = 2     # Time Series
    PSPEC = 3           # Power Spectrum
    ACF = 4             # Autocorrelation function
    VR_STAT = 5         # FBM variance ratio test statistic
    DIFF_1 = 6          # First time series difference
    DIFF_2 = 7          # Second time series difference

# Configurations used in plots
class DataPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels

## plot data type
def create_data_plot_type(plot_type):
    if plot_type.value == DataPlotType.TIME_SERIES.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$X_t$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.PSPEC.value:
        return DataPlotConfig(xlabel=r"$\omega$", ylabel=r"$\rho_\omega$", plot_type=PlotType.LOG)
    elif plot_type.value == DataPlotType.ACF.value:
        return DataPlotConfig(xlabel=r"$\tau$", ylabel=r"$\rho_\tau$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VR_STAT.value:
        return DataPlotConfig(xlabel=r"$s$", ylabel=r"$Z(s)$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_1.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta X_t$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_2.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta^2 X_t$", plot_type=PlotType.LINEAR)
    else:
        return DataPlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(y, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2

    plot_config = create_data_plot_type(plot_type)

    if x is None:
        npts = len(y)
        if plot_config.plot_type.value == PlotType.XLOG.value or plot_config.plot_type.value == PlotType.LOG.value:
            x = numpy.linspace(1.0, float(npts), npts)
        else:
            x = numpy.linspace(0.0, float(npts), npts)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, lw=lw)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, lw=lw)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, lw=lw)
    else:
        axis.plot(x, y, lw=lw)

###############################################################################################
# Plot multiple curves using the same axes  (Uses DataPlotType config)
def comparison(y, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2
    labels    = kwargs["labels"]    if "labels"    in kwargs else None

    plot_config = create_data_plot_type(plot_type)
    nplot = len(y)
    ncol = int(nplot/6) + 1

    if x is None:
        nx = len(y[0])
        if plot_config.plot_type.value == PlotType.XLOG.value or plot_config.plot_type.value == PlotType.LOG.value:
            x = numpy.tile(numpy.linspace(1, nx-1, nx), (nplot, 1))
        else:
            x = numpy.tile(numpy.linspace(0, nx-1, nx), (nplot, 1))

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    for i in range(nplot):
        if plot_config.plot_type.value == PlotType.LOG.value:
            logStyle(axis, x[i], y[i])
            if labels is None:
                axis.loglog(x[i], y[i], lw=lw)
            else:
                axis.loglog(x[i], y[i], label=labels[i], lw=lw)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            logXStyle(axis, x[i], y[i])
            if labels is None:
                axis.semilogx(x[i], y[i], lw=lw)
            else:
                axis.semilogx(x[i], y[i], label=labels[i], lw=lw)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            logYStyle(axis, x[i], y[i])
            if labels is None:
                axis.semilogy(x[i], y[i], lw=lw)
            else:
                axis.semilogy(x[i], y[i], label=labels[i], lw=lw)
        else:
            if labels is None:
                axis.plot(x[i], y[i], lw=lw)
            else:
                axis.plot(x[i], y[i], label=labels[i], lw=lw)

    if nplot <= 12 and labels is not None:
        axis.legend(ncol=ncol, loc='best', bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))

###############################################################################################
# Plot a single curve in a stack of plots that use the same x-axis (Uses PlotDataType config)
def stack(y, **kwargs):
    ylim      = kwargs["ylim"]      if "ylim"      in kwargs else None
    x         = kwargs["x"]         if "x"         in kwargs else None
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    labels    = kwargs["labels"]    if "labels"    in kwargs else None

    nplot = len(y)
    if x is None:
        nx = len(y[0])
        x = numpy.tile(numpy.linspace(0, nx-1, nx), (nplot, 1))

    if isinstance(plot_type, list):
        m = len(plot_type)
        if m < nplot:
            plot_type.append([plot_type[m-1] for i in range(m, nplot)])
    else:
        plot_type = [plot_type for i in range(nplot)]

    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(13, 10))

    if title is not None:
        axis[0].set_title(title)


    for i in range(nplot):
        plot_config = create_data_plot_type(plot_type[i])

        if i == nplot-1:
            axis[nplot-1].set_xlabel(plot_config.xlabel)

        nsample = len(y[i])
        axis[i].set_ylabel(plot_config.ylabel)

        if ylim is None:
            ylim_plot = [1.1*numpy.amin(y[i]), 1.1*numpy.amax(y[i])]
        else:
            ylim_plot = ylim

        axis[i].set_ylim(ylim_plot)
        axis[i].set_xlim([x[i][0], x[i][-1]])

        if labels is not None:
            ypos = 0.8*(ylim_plot[1]-ylim_plot[0])+ylim_plot[0]
            xpos = 0.8*(x[i][-1]-x[i][0])+x[i][0]
            text = axis[i].text(xpos, ypos, labels[i], fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

        if plot_config.plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x[i], y[i], lw=1)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x[i], y[i], lw=1)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x[i], y[i], lw=1)
        else:
            axis[i].plot(x[i], y[i], lw=1)
