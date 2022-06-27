import numpy
from enum import Enum
from matplotlib import pyplot
from lib import config

from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

###############################################################################################
## Specify Config for historgram plot
class HistPlotConfig:
    def __init__(self, df):
        self.meta_data = MetaData.get(df)
        self.source_meta_data = MetaData.get_source_meta_data(df)

    def title(self):
        params = self.source_meta_data.params | self.meta_data.params
        return f"{self.source_meta_data.desc} {self.meta_data.desc}: {MetaData.params_to_str(params)}"

    def xlabel(self):
        return self.meta_data.xlabel

    def ylabel(self):
        return self.meta_data.ylabel

    def label(self):
        return self.meta_data.formula

###############################################################################################
## Specify Config for function historgram comparison plot
class FuncHistPlotConfig:
    def __init__(self, data, func):
        self.data_meta_data = MetaData.get(data)
        self.func_meta_data = MetaData.get(func)
        self.source_meta_data = MetaData.get_source_meta_data(data)

    def title(self):
        params = self.func_meta_data.params | self.data_meta_data.params
        if self.source_meta_data is None:
            desc = self.data_meta_data.desc
        else:
            params = params | self.source_meta_data.params
            desc = f"{self.source_meta_data.desc} {self.data_meta_data.desc}"
        if not params:
            return desc
        else:
            return f"{desc}: {MetaData.params_to_str(params)}"

    def xlabel(self):
        return self.data_meta_data.xlabel

    def ylabel(self):
        return self.data_meta_data.ylabel

    def labels(self):
        return [self.data_meta_data.ylabel, self.formula()]

    def formula(self):
        if self.func_meta_data.formula is None:
            return self.func_meta_data.ylabel
        else:
            return self.func_meta_data.ylabel + "=" + self.func_meta_data.formula

###############################################################################################
## Histogram plot (Uses HistPlotType config)
def hist(df, **kwargs):
    title        = kwargs["title"]        if "title"        in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0
    xrange       = kwargs["xrange"]       if "xrange"       in kwargs else None
    ylimit       = kwargs["ylimit"]       if "ylimit"       in kwargs else None
    nbins        = kwargs["nbins"]        if "nbins"        in kwargs else 50
    params       = kwargs["params"]       if "params"       in kwargs else None

    plot_config = create_hist_plot_type(plot_type, params)

    figure, axis = pyplot.subplots(figsize=(12, 8))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_prop_cycle(config.distribution_sample_cycler)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    if plot_config.params is not None:
        x_text = 0.8
        y_text = 0.1
        bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
        axis.text(x_text, y_text, plot_config.params, bbox=bbox, fontsize=18.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type.value == PlotType.LOG.value:
        _, bins, _ = axis.hist(samples, nbins, rwidth=0.8, density=plot_config.denity, log=True)
        axis.yscale('log', nonposy='clip')
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        _, bins, _ = axis.hist(samples, nbins, rwidth=0.8, density=plot_config.denity, log=True)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        _, bins, _ = axis.hist(samples, nbins, rwidth=0.8, density=plot_config.denity)
        axis.yscale('log', nonposy='clip')
    else:
        _, bins, _ = axis.hist(samples, nbins, rwidth=0.8, density=plot_config.denity)

    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)

    axis.set_xlim([xrange[0], xrange[-1]])

    if ylimit is not None:
        axis.set_ylim(ylimit)

###############################################################################################
# bar plot (Uses HistPlotType config)
def bar(df, **kwargs):
    plot_config    = HistPlotConfig(df)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    label          = get_param_default_if_missing("label", plot_config.label(), **kwargs)
    loc            = get_param_default_if_missing("loc", "lup", **kwargs)

    x, y = plot_config.meta_data.get_data(df)
    width = 0.9*(x[1]-x[0])

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_prop_cycle(config.distribution_sample_cycler)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    if label is not None:
        x_text, y_text = _label_loc(loc)
        bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
        axis.text(x_text, y_text, label, bbox=bbox, fontsize=18.0, zorder=10, transform=axis.transAxes)

    axis.bar(x, y, align='center', width=width, zorder=9)

###############################################################################################
# bar plot (Uses HistPlotType config)
def fbar(**kwargs):
    data           = get_param_throw_if_missing("data", **kwargs)
    func           = get_param_throw_if_missing("func", **kwargs)
    plot_config    = FuncHistPlotConfig(data, func)

    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    labels         = get_param_default_if_missing("labels", plot_config.labels(), **kwargs)
    loc            = get_param_default_if_missing("loc", "lup", **kwargs)

    x, y = plot_config.data_meta_data.get_data(data)
    fx, fy = plot_config.func_meta_data.get_data(func)

    width = 0.9*(x[1]-x[0])

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_prop_cycle(config.distribution_sample_cycler)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    axis.bar(x, y, align='center', label=legend_labels[0], width=width)
    axis.plot(fx, fy, label=legend_labels[1], lw=lw)


###############################################################################################
# Helpers
def _label_loc(loc):
    if loc == "lup":
        return 0.7, 0.7
    elif loc == "llow":
        return 0.7, 0.3
    elif loc == "rup":
        return 0.1, 0.7
    else:
        return 0.1, 0.3
