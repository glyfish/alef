import numpy
from enum import Enum
from matplotlib import pyplot
import matplotlib.ticker

from lib.data.schema import (DataType, DataSchema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

###############################################################################################
# Configurations used in plots
class DataPlotConfig:
    def __init__(self, df):
        self.meta_data = MetaData.get(df)
        self.source_meta_data = MetaData.get_source_meta_data(df)

    def __repr__(self):
        return f"DataPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"meta_data=({self.meta_data})"

    def xlabel(self):
        return self.meta_data.xlabel

    def ylabel(self):
        return self.meta_data.ylabel

    def title(self):
        title = f"{self.meta_data.desc}"
        params = self.meta_data.params
        formula = self.meta_data.formula

        if self.source_meta_data is not None:
            params = params | self.source_meta_data.params
        elif formula is not None:
            title = f"{title} {self.meta_data.formula()}"

        if not params:
            return title
        else:
            return f"{title}: {MetaData.params_to_str(params)}"

###############################################################################################
# Configurations used in plots of data lists
class DataListPlotConfig:
    def __init__(self, dfs):
        self.nplot = len(dfs)
        self.meta_datas = [MetaData.get(df) for df in dfs]

    def __repr__(self):
        return f"DataListPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"meta_data=({self.meta_data}), nplot=({self.nplot})"

    def xlabel(self):
        return self.meta_datas[0].xlabel

    def ylabel(self):
        return self.meta_datas[0].ylabel

    def ylabels(self):
        return [self.meta_datas[i].ylabel for i in range(self.nplot)]

###############################################################################################
# twin plot configurations
class TwinPlotConfig:
    def __init__(self, left, right):
        self.left_meta_data = MetaData.get(left)
        self.right_meta_data = MetaData.get(right)
        self.source_meta_data = MetaData.get_source_meta_data(left)

    def __repr__(self):
        return f"TwinPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"left_meta_datas=({self.left_meta_datas}), " \
               f"right_meta_data=({self.right_meta_datas}), " \
               f"source_meta_data=({self.source_meta_data})"

    def xlabel(self):
        return self.left_meta_data.xlabel

    def left_ylabel(self):
        return self.left_meta_data.ylabel

    def right_ylabel(self):
        return self.right_meta_data.ylabel

    def labels(self):
        return [self.left_meta_data.desc + " (" + self.left_meta_data.ylabel + ")",
                self.right_meta_data.desc + " (" + self.right_meta_data.ylabel + ")"]

    def title(self):
        params = self.source_meta_data.params | self.left_meta_data.params | self.right_meta_data.params
        var_desc  = self.left_meta_data.desc + "-" + self.right_meta_data.desc
        return f"{self.source_meta_data.desc} {var_desc}: {MetaData.params_to_str(params)}"

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(df, **kwargs):
    plot_config    = DataPlotConfig(df)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    x, y = plot_config.meta_data.get_data(df)

    if npts is None or npts > len(y):
        npts = len(y)

    x = x[:npts]
    y = y[:npts]

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, lw=lw)
    elif plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, lw=lw)
    elif plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, lw=lw)
    else:
        axis.plot(x, y, lw=lw)

###############################################################################################
# Plot multiple curves of the same DataType using the same axes  (Uses DataPlotType config)
def comparison(dfs, **kwargs):
    plot_config    = DataListPlotConfig(dfs)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabels", plot_config.ylabel(), **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    ncol = int(plot_config.nplot/6) + 1

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    for i in range(plot_config.nplot):
        x, y = plot_config.meta_datas[i].get_data(dfs[i])

        if npts is None or npts > len(y):
            npts = len(y)

        x = x[:npts]
        y = y[:npts]

        if plot_type.value == PlotType.LOG.value:
            logStyle(axis, x, y)
            if labels is None:
                axis.loglog(x, y, lw=lw)
            else:
                axis.loglog(x, y, label=labels[i], lw=lw)
        elif plot_type.value == PlotType.XLOG.value:
            logXStyle(axis, x, y)
            if labels is None:
                axis.semilogx(x, y, lw=lw)
            else:
                axis.semilogx(x, y, label=labels[i], lw=lw)
        elif plot_type.value == PlotType.YLOG.value:
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

    if plot_config.nplot <= 12 and labels is not None:
        axis.legend(ncol=ncol, loc='best', bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))

###############################################################################################
# Plot a single curve in a stack of plots that use the same x-axis (Uses PlotDataType config)
def stack(dfs, **kwargs):
    plot_config    = DataListPlotConfig(dfs)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabels        = get_param_default_if_missing("ylabel", plot_config.ylabels(), **kwargs)
    ylim           = get_param_default_if_missing("ylim", None, **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    figure, axis = pyplot.subplots(plot_config.nplot, sharex=True, figsize=(13, 10))

    axis[plot_config.nplot-1].set_xlabel(xlabel)

    if title is not None:
        axis[0].set_title(title, y=title_offset)

    for i in range(plot_config.nplot):
        x, y = plot_config.meta_datas[i].get_data(dfs[i])

        if npts is None or npts > len(y):
            npts = len(y)

        x = x[:npts]
        y = y[:npts]

        axis[i].set_ylabel(ylabels[i])

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

        if plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x, y, lw=1)
        elif plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x, y, lw=1)
        elif plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x, y, lw=1)
        else:
            axis[i].plot(x, y, lw=1)

###############################################################################################
# Plot two curves with different data_types using different y axis scales, same xaxis
# with data in the same DataFrame
def twinx(**kwargs):
    left            = get_param_throw_if_missing("left", **kwargs)
    right           = get_param_throw_if_missing("right", **kwargs)
    plot_config     = TwinPlotConfig(left, right)

    plot_type       = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title           = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    left_ylabel     = get_param_default_if_missing("left_ylabel", plot_config.left_ylabel(), **kwargs)
    right_ylabel    = get_param_default_if_missing("right_ylabel", plot_config.right_ylabel(), **kwargs)
    labels          = get_param_default_if_missing("labels", plot_config.labels(), **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "upper right", **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)

    figure, axis1 = pyplot.subplots(figsize=(13, 10))

    axis1.set_title(title, y=title_offset)

    # first plot left axis1
    axis1.set_ylabel(left_ylabel)
    axis1.set_xlabel(xlabel)
    _plot_curve(axis1, left, plot_config.left_meta_data, plot_type, labels[0], **kwargs)

    # second plot right axis2
    axis2 = axis1.twinx()
    axis2._get_lines.prop_cycler = axis1._get_lines.prop_cycler
    axis2.set_ylabel(right_ylabel)
    _plot_curve(axis2, right, plot_config.right_meta_data, plot_type, labels[1], **kwargs)

    if ylim is not None:
        axis1.set_ylim(ylim)

    _twinx_ticks(axis1, axis2)
    axis2.grid(False)

    figure.legend(loc=legend_loc, bbox_to_anchor=(0.2, 0.2, 0.6, 0.6))

###############################################################################################
# compute twinz ticks so grids align
def _twinx_ticks(axis1, axis2):
    y1_lim = axis1.get_ylim()
    y2_lim = axis2.get_ylim()
    f = lambda x : y2_lim[0] + (x - y1_lim[0])*(y2_lim[1] - y2_lim[0])/(y1_lim[1] - y1_lim[0])
    ticks = f(axis1.get_yticks())
    axis2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

###############################################################################################
# plot curve on specified axis
def _plot_curve(axis, df, meta_data, plot_type, label, **kwargs):
    lw   = get_param_default_if_missing("lw", 2, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)

    x, y = meta_data.get_data(df)

    if npts is None or npts > len(y):
        npts = len(y)

    x = x[:npts]
    y = y[:npts]

    if plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        if label is None:
            axis.loglog(x, y, lw=lw)
        else:
            axis.loglog(x, y, label=label, lw=lw)
    elif plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        if label is None:
            axis.semilogx(x, y, lw=lw)
        else:
            axis.semilogx(x, y, label=label, lw=lw)
    elif plot_type.value == PlotType.YLOG.value:
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
