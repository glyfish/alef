import numpy
from enum import Enum
from matplotlib import pyplot
import matplotlib.ticker

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

# twin plot configurations
class TwinPlotConfig:
    def __init__(self, df, left_data_type, right_data_type):
        left_schema = DataSchema.create(left_data_type)
        right_schema = DataSchema.create(right_data_type)
        source_schema = MetaData.get_source_schema(df)
        self.left_meta_data = MetaData.get(df, left_schema)
        self.right_meta_data = MetaData.get(df, right_schema)
        self.source_meta_data = MetaData.get(df, source_schema)

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
# Plot two curves with different data_types using different y axis scales, same xaxis
# with data in the same DataFrame
def twinx(df, **kwargs):
    left_data_type  = get_param_throw_if_missing("left_data_type", **kwargs)
    right_data_type = get_param_throw_if_missing("right_data_type", **kwargs)

    plot_config     = TwinPlotConfig(df, left_data_type, right_data_type)

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
    _plot_curve(axis1, df, plot_config.left_meta_data, plot_type, labels[0], **kwargs)

    # second plot right axis2
    axis2 = axis1.twinx()
    axis2._get_lines.prop_cycler = axis1._get_lines.prop_cycler
    axis2.set_ylabel(right_ylabel)
    _plot_curve(axis2, df, plot_config.right_meta_data, plot_type, labels[1], **kwargs)

    if ylim is not None:
        axis1.set_ylim(ylim)

    twinx_ticks(axis1, axis2)
    axis2.grid(False)

    figure.legend(loc=legend_loc, bbox_to_anchor=(0.2, 0.2, 0.6, 0.6))

###############################################################################################
# compute twinz ticks so grids align
def twinx_ticks(axis1, axis2):
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

###############################################################################################
# Plot a single curve in a stack of plots that use the same x-axis (Uses PlotDataType config)
def stack(dfs, plot_type, **kwargs):
    title        = get_param_default_if_missing("title", None, **kwargs)
    title_offset = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    labels       = get_param_default_if_missing("labels", None, **kwargs)
    ylim         = get_param_default_if_missing("ylim", None, **kwargs)
    npts         = get_param_default_if_missing("npts", None, **kwargs)

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
        plot_config = create_multi_data_plot_type(plot_type[i])
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
