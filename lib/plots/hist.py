import numpy
from enum import Enum
from matplotlib import pyplot
from lib import config

from lib.plots.config import (PlotType, logStyle, logXStyle, logYStyle)

###############################################################################################
## Specify Config for historgram PlotType
class HistPlotType(Enum):
    GENERIC = 1         # Generic plot type
    PDF = 2             # Probability density function
    CDF = 3             # Cummulative density function

###############################################################################################
## Specify Config for historgram PlotType
class HistPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, density=False, params=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.density = density
        self.params = params

###############################################################################################
## plot histogram type
def create_hist_plot_type(plot_type, params=None):
    if plot_type.value == HistPlotType.PDF.value:
        if params is not None:
            plot_params = f"μ={params[1]}\nσ={params[0]}"
        else:
            plot_params = None
        return HistPlotConfig(xlabel=r"$x$", ylabel=r"PDF $p(x)$", plot_type=PlotType.LINEAR, density=True, params=plot_params)
    if plot_type.value == HistPlotType.CDF.value:
        if params is not None:
            plot_params = f"μ={params[1]}\nσ={params[0]}"
        else:
            plot_params = None
        return HistPlotConfig(xlabel=r"$x$", ylabel=r"CDF $P(x)$", plot_type=PlotType.LINEAR, density=True, params=plot_params)
    else:
        return HistPlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

###############################################################################################
## Histogram plot (Uses HistPlotType config)
def hist(samples, **kwargs):
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else HistPlotType.GENERIC
    title = kwargs["title"] if "title" in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 0.0
    xrange = kwargs["xrange"] if "xrange" in kwargs else None
    ylimit = kwargs["ylimit"] if "ylimit" in kwargs else None
    nbins = kwargs["nbins"] if "nbins" in kwargs else 50
    params = kwargs["params"] if "params" in kwargs else None

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
        axis.text(x_text, y_text, plot_config.params, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

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
def bar(x, y, **kwargs):
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else HistPlotType.GENERIC
    title = kwargs["title"] if "title" in kwargs else None
    params = kwargs["params"] if "params" in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    width = 0.9*(x[1]-x[0])

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
        axis.text(x_text, y_text, plot_config.params, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    axis.bar(x, y, align='center', width=width, zorder=10)
