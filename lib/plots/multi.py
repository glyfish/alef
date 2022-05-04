import numpy
from enum import Enum
from matplotlib import pyplot
import matplotlib.ticker

from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

# Specify CompPlotType
class MultiDataPlotType(Enum):
    ACF_PACF = 2            # ACF-PACF comparison plot

# Plot Configurations
class MultiDataPlotConfig:
    def __init__(self, df, schemas, title, plot_type=PlotType.LINEAR):
        self.plot_type = plot_type
        self.schemas = schemas
        self.title = title

    def __repr__(self):
        return f"MultiDataPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"plot_type=({self.est}), schemas=({self.schemas})"

    @staticmethod
    def _get_func_title(df, schemas):
        meta_datas = [schema.get_meta_data(df) for schema in schemas]
        var_desc  = "-".join([meta_data.desc for meta_data in meta_datas])
        source_meta_data = meta_datas[0].source_schema.get_meta_data(df)
        return f"{source_meta_data.desc} {var_desc} {source_meta_data.params_str()}"

## plot data type takes multiple data types
def create_multi_data_plot_type(plot_type, df):
    if plot_type.value == MultiDataPlotType.ACF_PACF.value:
        schemas = [create_schema(DataType.ACF), create_schema(DataType.PACF)]
        title = MultiDataPlotConfig._get_func_title(df, schemas)
        return MultiDataPlotConfig(df, schemas, title, PlotType.LINEAR)
    else:
        raise Exception(f"Data plot type is invalid: {plot_type}")

###############################################################################################
# Plot two curves with different data_types using different y axis scales, same xaxis
# with data in the same DataFrame
def twinx(df, plot_type, **kwargs):
    title        = get_param_default_if_missing("title", None, **kwargs)
    title_offset = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    legend_loc   = get_param_default_if_missing("legend_loc", "upper right", **kwargs)
    ylim         = get_param_default_if_missing("ylim", None, **kwargs)

    plot_config = create_multi_data_plot_type(plot_type, df)

    if len(plot_config.schemas) != 2:
        raise Exception(f"Must have only two schemas: {plot_type}")

    figure, axis1 = pyplot.subplots(figsize=(13, 10))

    if title is None:
        axis1.set_title(plot_config.title, y=title_offset)
    else:
        axis1.set_title(title, y=title_offset)

    # first plot left axis1
    schema = plot_config.schemas[0]
    meta_data = schema.get_meta_data(df)
    axis1.set_ylabel(meta_data.ylabel)
    axis1.set_xlabel(meta_data.xlabel)
    _plot_curve(axis1, df, schema, plot_config, **kwargs)

    # second plot right axis2
    schema = plot_config.schemas[1]
    meta_data = schema.get_meta_data(df)
    axis2 = axis1.twinx()
    axis2._get_lines.prop_cycler = axis1._get_lines.prop_cycler
    axis2.set_ylabel(meta_data.ylabel)
    _plot_curve(axis2, df, schema, plot_config, **kwargs)

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
def _plot_curve(axis, df, schema, plot_config, **kwargs):
    lw   = get_param_default_if_missing("lw", 2, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)

    x, y = schema.get_data(df)
    meta_data = schema.get_meta_data(df)
    label = meta_data.desc + " (" + meta_data.ylabel + ")"

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
