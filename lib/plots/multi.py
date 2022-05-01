import numpy
from enum import Enum
from matplotlib import pyplot

from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing, calculate_ticks)

# Specify CompPlotType
class CompPlotType(Enum):
    GENERIC = 1             # Generic comparison plot
    ACF_PACF = 2            # ACF-PACF comparison plot

# Plot Configurations
class CompPlotConfig:
    def __init__(self, xlabel, ylabels, schemas, plot_type=PlotType.LINEAR):
        self.xlabel=xlabel
        self.ylabels=ylabels
        self.plot_type=plot_type
        self.schemas=schemas

## plot data type takes multiple data types
def create_comp_plot_type(plot_type):
    if plot_type.value == CompPlotType.GENERIC.value:
        schemas = [create_schema(DataType.TIME_SERIES), create_schema(DataType.GENERIC)]
        return CompPlotConfig(xlabel=r"$t$",
                              ylabels=[r"$S_t$", r"$y$"],
                              schemas=schemas,
                              plot_type=PlotType.LINEAR)
    if plot_type.value == CompPlotType.ACF_PACF.value:
        schemas = [create_schema(DataType.ACF), create_schema(DataType.PACF)]
        return CompPlotConfig(xlabel=r"$\tau$",
                              ylabels=[r"$\rho_\tau$", r"$\varphi_\tau$"],
                              schemas=schemas,
                              plot_type=PlotType.LINEAR)
    else:
        raise Exception(f"Data plot type is invalid: {plot_type}")

###############################################################################################
# Plot two curves with different data_types using different y axis scales, same xaxis
# and data in the same DataFrame
def twinx(df, **kwargs):
    title        = get_param_default_if_missing("title", None, **kwargs)
    title_offset = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    plot_type    = get_param_default_if_missing("plot_type", CompPlotType.GENERIC, **kwargs)
    labels       = get_param_default_if_missing("labels", None, **kwargs)
    nticks       = get_param_default_if_missing("nticks", 5, **kwargs)
    legend_loc   = get_param_default_if_missing("legend_loc", "upper right", **kwargs)

    plot_config = create_comp_plot_type(plot_type)

    if len(plot_config.schemas) < 2:
        raise Exception(f"Must have at least two schemas: {plot_type}")

    figure, axis1 = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis1.set_title(title, y=title_offset)

    axis1.set_xlabel(plot_config.xlabel)

    # first plot left axis1
    schema = plot_config.schemas[0]
    axis1.set_ylabel(plot_config.ylabels[0])
    _plot_curve(axis1, df, schema, plot_config, labels[0], **kwargs)

    # second plot right axis2
    schema = plot_config.schemas[1]
    axis2 = axis1.twinx()
    axis2._get_lines.prop_cycler = axis1._get_lines.prop_cycler
    axis2.set_ylabel(plot_config.ylabels[1])
    _plot_curve(axis2, df, schema, plot_config, labels[1], **kwargs)

    axis1.set_yticks(calculate_ticks(axis1, nticks))
    axis2.set_yticks(calculate_ticks(axis2, nticks))
    axis2.grid(False)

    figure.legend(loc=legend_loc, bbox_to_anchor=(0.2, 0.2, 0.6, 0.6))

###############################################################################################
# plot curve on specified axis
def _plot_curve(axis, df, schema, plot_config, label, **kwargs):
    lw   = get_param_default_if_missing("lw", 2, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)

    x, y = schema.get_data(df)

    if npts is None or npts > len(y):
        npts = len(y)

    x = x[:npts]
    y = y[:npts]

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        if label is None:
            axis.loglog(x, y, lw=lw)
        else:
            axis.loglog(x, y, label=label, lw=lw)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        if label is None:
            axis.semilogx(x, y, lw=lw)
        else:
            axis.semilogx(x, y, label=label, lw=lw)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        if label is None:
            axis.semilogy(x, y, lw=lw)
        else:
            axis.semilogy(x, y, label=label, lw=lw)
    else:
        if label is None:
            axis.plot(x, y, lw=lw)
        else:
            axis.plot(x, y, label=label, lw=lw)

###############################################################################################
# Plot multiple curves of the same DataType using the same axes  (Uses DataPlotType config)
def comparison(df, **kwargs):
    title        = get_param_default_if_missing("title", None, **kwargs)
    title_offset = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    plot_type    = get_param_default_if_missing("plot_type", CompPlotType.GENERIC, **kwargs)
    labels       = get_param_default_if_missing("labels", None, **kwargs)
    lw           = get_param_default_if_missing("lw", 2, **kwargs)
    npts         = get_param_default_if_missing("npts", None, **kwargs)

    plot_config = create_data_plot_type(plot_type)
    nplot = len(plot_config.schemas)

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
