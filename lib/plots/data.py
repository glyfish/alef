import numpy
from enum import Enum
from matplotlib import pyplot

from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

###############################################################################################
# Configurations used in plots
class DataPlotConfig:
    def __init__(self, df, data_type):
        schema = create_schema(data_type)
        self.meta_data = MetaData.get(df, schema)

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
        params = self.meta_data.params
        formula = self.meta_data.formula
        title = f"{self.meta_data.desc}"
        if formula is not None:
            title = f"{title} {self.meta_data.formula()}"
        if not params:
            return title
        else:
            return f"{title}: {self.meta_data.params_str()}"

###############################################################################################
# Configurations used in plots of data lists
class DataListPlotConfig:
    def __init__(self, dfs, data_type):
        self.nplot = len(dfs)
        schemas = self._create_schemas(data_type)
        self.meta_datas = [MetaData.get(dfs[i], schemas[i]) for i in range(self.nplot)]

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

    def _create_schemas(self, data_type):
        if isinstance(data_type, list):
            m = len(data_type)
            if m < self.nplot:
                data_type.append([data_type[m-1] for i in range(m, self.nplot)])
        else:
            data_type = [data_type for i in range(self.nplot)]

        return [create_schema(type) for type in data_type]

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(df, **kwargs):
    data_type      = get_param_throw_if_missing("data_type", **kwargs)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)

    plot_config    = DataPlotConfig(df, data_type)

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
    data_type      = get_param_throw_if_missing("data_type", **kwargs)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)

    plot_config    = DataListPlotConfig(dfs, data_type)

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
    data_type      = get_param_throw_if_missing("data_type", **kwargs)
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)

    plot_config    = DataListPlotConfig(dfs, data_type)

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
