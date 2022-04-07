import numpy
from matplotlib import pyplot
from lib import config
from lib.dist import (DistributionType, DistributionFuncType, HypothesisType, distribution_function)
from lib.plot_config import (create_reg_plot_type, create_data_plot_type, create_func_plot_type, create_dist_plot_type,
                             create_cum_plot_type, create_hist_dist_plot_type,
                             DataPlotType, PlotType, RegPlotType, FuncPlotType, DistPlotType, CumPlotType, HistDistPlotType,
                             logStyle, logXStyle, logYStyle)

###############################################################################################
# Plot a single curve as a function of the dependent variable (Uses DataPlotType config)
def curve(y, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotDataType.GENERIC
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
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotDataType.GENERIC
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
def stack(y, ylim, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotDataType.GENERIC
    labels    = kwargs["labels"]    if "labels"    in kwargs else None

    plot_config = create_data_plot_type(plot_type)

    nplot = len(y)
    if x is None:
        nx = len(y[0])
        x = numpy.tile(numpy.linspace(0, nx-1, nx), (nplot, 1))

    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(13, 10))

    if title is not None:
        axis[0].set_title(title)

    axis[nplot-1].set_xlabel(plot_config.xlabel)

    for i in range(nplot):
        nsample = len(y[i])
        axis[i].set_ylabel(plot_config.ylabel)
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, x[i][-1]])

        if labels is not None:
            text = axis[i].text(0.8*x[i][-1], 0.65*ylim[-1], labels[i], fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

        if plot_config.plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x[i], y[i], lw=1)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x[i], y[i], lw=1)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x[i], y[i], lw=1)
        else:
            axis[i].plot(x[i], y[i], lw=1)

###############################################################################################
# Histogram plot (Uses HistPlotType config)
def hist(samples, **kwargs):
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotDataType.GENERIC
    title = kwargs["title"] if "title" in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 0.0
    xrange = kwargs["xrange"] if "xrange" in kwargs else None
    ylimit = kwargs["ylimit"] if "ylimit" in kwargs else None
    nbins = kwargs["nbins"] if "nbins" in kwargs else 50
    params = kwargs["params"] if "params" in kwargs else None

    plot_config = create_hist_dist_plot_type(plot_type, params)

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
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotDataType.GENERIC
    title = kwargs["title"] if "title" in kwargs else None
    params = kwargs["params"] if "params" in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 0.0

    width = [0.9*(x[i+1]-x[i]) for in range(len(x))]

    plot_config = create_hist_dist_plot_type(plot_type, params)

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


###############################################################################################
# Compare data to the value of a function (Uses PlotFuncType config)
def fcompare(y, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotFuncType.LINEAR
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2
    labels    = kwargs["labels"]    if "labels"    in kwargs else None
    npts      = kwargs["npts"]      if "npts"      in kwargs else 10
    params    = kwargs["params"]    if "params"    in kwargs else []

    plot_config = create_func_plot_type(plot_type, params)

    if x is None:
        nx = len(y)
        if plot_config.plot_type.value == PlotType.XLOG.value or plot_config.plot_type.value == PlotType.LOG.value:
            x = logspace(nx, float(nx-1), 1.0)
        else:
            x = numpy.linspace(0.0, float(nx-1), nx)
    step = int(len(x)/npts)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    if title is not None:
        axis.set_title(title)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.loglog(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogx(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogy(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.plot(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

##################################################################################################################
# Compare the cumulative value of a variable as a function of time with its target value (Uses CumPlotType config)
def cumulative(samples, plot_type, **kwargs):
    title  = kwargs["title"]  if "title"  in kwargs else None
    lw     = kwargs["lw"]     if "lw"     in kwargs else 2
    params = kwargs["params"] if "params" in kwargs else []

    plot_config = create_cum_plot_type(plot_type, params)

    accum = plot_config.f(samples)
    range = max(accum) - min(accum)
    ntime = len(accum)
    time = numpy.linspace(1.0, ntime-1, ntime)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is None:
        axis[0].set_title(title)

    axis.set_ylim([min(accum)-0.25*range, max(accum)+0.25*range])
    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)
    axis.set_title(title)
    axis.set_xlim([1.0, ntime])
    axis.semilogx(time, accum, label=plot_config.legend_labels[0], lw=lw)
    axis.semilogx(time, numpy.full((len(time)), plot_config.target), label=plot_config.legend_labels[1], lw=lw)
    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

###############################################################################################
# Compare the result of a linear regression with teh acutal data (Uses RegPlotType config)
def regression(y, x, results, **kwargs):
    title = kwargs["title"] if "title" in kwargs else None
    plot_type = kwargs["plot_type"]  if "plot_type"  in kwargs else RegressionPlotType.LINEAR

    β = results.params

    if β[1] < 0:
        x_text = 0.1
        y_text = 0.1
        lengend_location = (0.6, 0.65, 0.3, 0.3)
    else:
        x_text = 0.8
        y_text = 0.1
        lengend_location = (0.05, 0.65, 0.3, 0.3)

    plot_config = create_reg_plot_type(plot_type, results, x)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.loglog(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.semilogx(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])

    axis.legend(loc='best', bbox_to_anchor=lengend_location)

###############################################################################################
# Hypothesis test plot (Uses DistPlotType config)
def htest(test_stats, plot_type, **kwargs):
    title        = kwargs["title"]        if "title"       in kwargs else None
    test_type    = kwargs["test_type"]    if "test_type"   in kwargs else HypothesisType.TWO_TAIL
    npts         = kwargs["npts"]         if "npts"        in kwargs else 100
    sig_level    = kwargs["sig_level"]    if "sig_level"   in kwargs else 0.05
    labels       = kwargs["labels"]       if "labels"      in kwargs else None
    dist_params  = kwargs["dist_params"]  if "dist_params" in kwargs else []

    plot_config = create_dist_plot_type(plot_type)
    if plot_config.dist_params is not None:
        dist_params = plot_config.dist_params

    cdf = distribution_function(plot_config.dist_type, DistributionFuncType.CDF, dist_params)
    ppf = distribution_function(plot_config.dist_type, DistributionFuncType.PPF, dist_params)
    x_range = distribution_function(plot_config.dist_type, DistributionFuncType.RANGE, dist_params)

    x_vals = x_range(npts)
    min_stats = min(test_stats)
    max_stats = max(test_stats)
    min_stats = x_vals[0] if x_vals[0] < min_stats else min_stats
    max_stats = x_vals[-1] if x_vals[-1] > max_stats else max_stats
    x_vals = numpy.linspace(min_stats, max_stats, npts)

    y_vals = cdf(x_vals)

    lower_critical_value = None
    lower_label = None
    upper_critical_value = None
    upper_label = None

    if test_type == HypothesisType.TWO_TAIL:
        sig_level_2 = sig_level/2.0
        lower_critical_value = ppf(sig_level_2)
        upper_critical_value = ppf(1.0 - sig_level_2)
        lower_label = f"Lower Tail={format(sig_level_2, '1.3f')}"
        upper_label = f"Upper Tail={format(1.0 - sig_level_2, '1.3f')}"
    elif test_type == HypothesisType.LOWER_TAIL:
        lower_critical_value = ppf(sig_level)
        lower_label = f"Lower Tail={format(sig_level, '1.3f')}"
    elif test_type == HypothesisType.UPPER_TAIL:
        upper_critical_value = ppf(1.0 - sig_level)
        upper_label = f"Upper Tail={format(1.0 - sig_level, '1.3f')}"

    figure, axis = pyplot.subplots(figsize=(12, 8))

    text = axis.text(x_vals[0], 0.05*y_vals[-1], f"Signicance={format(100.0*sig_level, '2.0f')}%", fontsize=18)
    text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

    if title is not None:
        axis.set_title(title)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_ylim([-0.05, 1.05])
    axis.set_xlabel(plot_config.xlabel)

    axis.plot(x_vals, y_vals)
    if lower_critical_value is not None:
        axis.plot([lower_critical_value, lower_critical_value], [0.0, 1.0], color='red', label=lower_label, lw=4)
    if upper_critical_value is not None:
        axis.plot([upper_critical_value, upper_critical_value], [0.0, 1.0], color='black', label=upper_label, lw=4)

    # if z_vals is None:
    for i in range(len(test_stats)):
        if labels is None:
            axis.plot([test_stats[i], test_stats[i]], [0.0, 1.0])
        else:
            axis.plot([test_stats[i], test_stats[i]], [0.0, 1.0], label=labels[i])

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

###############################################################################################
# generate points evenly spaced on a logarithmic axis
def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)
