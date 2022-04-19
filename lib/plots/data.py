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
    VAR = 8             # Variance
    COV = 9             # Covariance
    MEAN = 10           # Mean
    STD = 11            # Standard deviation

# Configurations used in plots
class DataPlotConfig:
    def __init__(self, xlabel, ylabel, xcol, ycol, plot_type=PlotType.LINEAR):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.xcol = xcol
        self.ycol = ycol

## plot data type
def create_data_plot_type(plot_type):
    if plot_type.value == DataPlotType.TIME_SERIES.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$S_t$", xcol="Time", ycol="Xt", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.PSPEC.value:
        return DataPlotConfig(xlabel=r"$\omega$", ylabel=r"$\rho_\omega$", xcol="Frequency", ycol="Power Spectrum", plot_type=PlotType.LOG)
    elif plot_type.value == DataPlotType.ACF.value:
        return DataPlotConfig(xlabel=r"$\tau$", ylabel=r"$\rho_\tau$", xcol="Lag", ycol="Autocorrelation", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VR_STAT.value:
        return DataPlotConfig(xlabel=r"$s$", ylabel=r"$Z(s)$", xcol="Lag", ycol="Variance Ratio", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_1.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta S_t$", xcol="Time", ycol="Difference 1", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_2.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta^2 S_t$", xcol="Time", ycol="Difference 2", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VAR.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\sigma_t^2$", xcol="Time", ycol="Variance", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.COV.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"Cov($S_t S_s$)", xcol="Time", ycol="Covariance", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.MEAN.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\mu_t$", xcol="Time", ycol="Mean", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.STD.value:
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\sigma_t$", xcol="Time", ycol="STD", plot_type=PlotType.LINEAR)
    else:
        return DataPlotConfig(xlabel="x", ylabel="y", xcol="x", ycol="y", plot_type=PlotType.LINEAR)

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(df, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2

    plot_config = create_data_plot_type(plot_type)

    if plot_config.ycol in meta_data.keys():
        npts = meta_data[plot_config.ycol]["npts"]
    else:
        npts = len(df[plot_config.ycol])

    x = df[plot_config.xcol][:npts]
    y = df[plot_config.ycol][:npts]

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
def comparison(dfs, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2
    labels    = kwargs["labels"]    if "labels"    in kwargs else None

    plot_config = create_data_plot_type(plot_type)
    nplot = len(dfs)
    ncol = int(nplot/6) + 1

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    for i in range(nplot):
        df = dfs[i]
        meta_data = df.attrs

        if plot_config.ycol in meta_data.keys():
            npts = meta_data[plot_config.ycol]["npts"]
        else:
            npts = len(df[plot_config.ycol])

        x = df[plot_config.xcol][:npts]
        y = df[plot_config.ycol][:npts]

        if plot_config.plot_type.value == PlotType.LOG.value:
            logStyle(axis, x, y)
            if labels is None:
                axis.loglog(x, y, lw=lw)
            else:
                axis.loglog(x, y, label=labels[i], lw=lw)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            logXStyle(axis, x, y)
            if labels is None:
                axis.semilogx(x, y, lw=lw)
            else:
                axis.semilogx(x, y, label=labels[i], lw=lw)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            logYStyle(axis, x, y)
            if labels is None:
                axis.semilogy(x, y, lw=lw)
            else:
                axis.semilogy(x, y, label=labels[i], lw=lw)
        else:
            if labels is None:
                axis.plot(x, y, lw=lw)
            else:
                axis.plot(x, y, label=labels[i], lw=lw)

    if nplot <= 12 and labels is not None:
        axis.legend(ncol=ncol, loc='best', bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))

###############################################################################################
# Plot a single curve in a stack of plots that use the same x-axis (Uses PlotDataType config)
def stack(dfs, **kwargs):
    ylim      = kwargs["ylim"]      if "ylim"      in kwargs else None
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else DataPlotType.GENERIC
    labels    = kwargs["labels"]    if "labels"    in kwargs else None

    nplot = len(dfs)

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
        df = dfs[i]
        meta_data = df.attrs

        if plot_config.ycol in meta_data.keys():
            npts = meta_data[plot_config.ycol]["npts"]
        else:
            npts = len(df[plot_config.ycol])

        x = df[plot_config.xcol][:npts]
        y = df[plot_config.ycol][:npts]

        if i == nplot-1:
            axis[nplot-1].set_xlabel(plot_config.xlabel)

        axis[i].set_ylabel(plot_config.ylabel)

        if ylim is None:
            ylim_plot = [1.1*numpy.amin(y), 1.1*numpy.amax(y)]
        else:
            ylim_plot = ylim

        axis[i].set_ylim(ylim_plot)
        axis[i].set_xlim([x[0], x[npts-1]])

        if labels is not None:
            ypos = 0.8*(ylim_plot[1]-ylim_plot[0])+ylim_plot[0]
            xpos = 0.8*(x[npts-1]-x[0])+x[0]
            text = axis[i].text(xpos, ypos, labels[i], fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

        if plot_config.plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x, y, lw=1)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x, y, lw=1)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x, y, lw=1)
        else:
            axis[i].plot(x, y, lw=1)
