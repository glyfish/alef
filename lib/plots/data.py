import numpy
from enum import Enum
from matplotlib import pyplot

from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)

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
    def __init__(self, xlabel, ylabel, data_type, plot_type=PlotType.LINEAR):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.data_type = data_type

## plot data type
def create_data_plot_type(plot_type):
    if plot_type.value == DataPlotType.TIME_SERIES.value:
        data_type=create_schema(DataType.TIME_SERIES)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$S_t$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.PSPEC.value:
        data_type = create_schema(DataType.PSPEC)
        return DataPlotConfig(xlabel=r"$\omega$", ylabel=r"$\rho_\omega$", data_type=data_type, plot_type=PlotType.LOG)
    elif plot_type.value == DataPlotType.ACF.value:
        data_type = create_schema(DataType.ACF)
        return DataPlotConfig(xlabel=r"$\tau$", ylabel=r"$\rho_\tau$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VR_STAT.value:
        data_type = create_schema(DataType.VR_STAT)
        return DataPlotConfig(xlabel=r"$s$", ylabel=r"$Z(s)$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_1.value:
        data_type = create_schema(DataType.DIFF_1)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta S_t$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.DIFF_2.value:
        data_type = create_schema(DataType.DIFF_2)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\Delta^2 S_t$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VAR.value:
        data_type = create_schema(DataType.VAR)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\sigma_t^2$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.COV.value:
        data_type = create_schema(DataType.COV)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"Cov($S_t S_s$)", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.MEAN.value:
        data_type = create_schema(DataType.MEAN)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\mu_t$", data_type=data_type, plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.STD.value:
        data_type = create_schema(DataType.STD)
        return DataPlotConfig(xlabel=r"$t$", ylabel=r"$\sigma_t$", data_type=data_type, plot_type=PlotType.LINEAR)
    else:
        raise Exception(f"Data plot type is invalid: {plot_type}")

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(df, **kwargs):
    title        = kwargs["title"]        if "title"        in kwargs else None
    plot_type    = kwargs["plot_type"]    if "plot_type"    in kwargs else DataPlotType.GENERIC
    lw           = kwargs["lw"]           if "lw"           in kwargs else 2
    npts         = kwargs["npts"]         if "npts"         in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    plot_config = create_data_plot_type(plot_type)
    x, y = plot_config.data_type.get_data(df)

    if npts is None or npts > len(y):
        npts = len(y)

    x = x[:npts]
    y = y[:npts]

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

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
    title        = kwargs["title"]        if "title"        in kwargs else None
    plot_type    = kwargs["plot_type"]    if "plot_type"    in kwargs else DataPlotType.GENERIC
    lw           = kwargs["lw"]           if "lw"           in kwargs else 2
    labels       = kwargs["labels"]       if "labels"       in kwargs else None
    npts         = kwargs["npts"]         if "npts"         in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    plot_config = create_data_plot_type(plot_type)
    nplot = len(dfs)
    ncol = int(nplot/6) + 1

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    for i in range(nplot):
        x, y = plot_config.data_type.get_data(dfs[i])

        if npts is None or npts > len(y):
            npts = len(y)

        x = x[:npts]
        y = y[:npts]

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
    ylim         = kwargs["ylim"]         if "ylim"         in kwargs else None
    title        = kwargs["title"]        if "title"        in kwargs else None
    plot_type    = kwargs["plot_type"]    if "plot_type"    in kwargs else DataPlotType.GENERIC
    labels       = kwargs["labels"]       if "labels"       in kwargs else None
    npts         = kwargs["npts"]         if "npts"         in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    nplot = len(dfs)

    if isinstance(plot_type, list):
        m = len(plot_type)
        if m < nplot:
            plot_type.append([plot_type[m-1] for i in range(m, nplot)])
    else:
        plot_type = [plot_type for i in range(nplot)]

    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(13, 10))

    if title is not None:
        axis[0].set_title(title, y=title_offset)

    for i in range(nplot):
        plot_config = create_data_plot_type(plot_type[i])
        x, y = plot_config.data_type.get_data(dfs[i])

        if npts is None or npts > len(y):
            npts = len(y)

        x = x[:npts]
        y = y[:npts]

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
