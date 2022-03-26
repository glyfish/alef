import numpy
from matplotlib import pyplot
from lib import config
from lib.plot_config import (create_regression_plot_type, create_plot_data_type, create_plot_func_type,
                             PlotDataType, PlotType, RegressionPlotType, PlotFuncType,
                             logStyle, logXStyle, logYStyle)

# Plot a single curve as a function of the dependent variable
def curve(y, x=None, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "data_type" in kwargs:
        data_type = kwargs["data_type"]
    else:
        data_type = PlotDataType.GENERIC
    if "lw" in kwargs:
        lw = kwargs["lw"]
    else:
        lw = 2

    plot_config = create_plot_data_type(data_type)

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
        logStyle(axis, x)
        axis.loglog(x, y, lw=lw)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x)
        axis.semilogx(x, y, lw=lw)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x)
        axis.semilogy(x, y, lw=lw)
    else:
        axis.plot(x, y, lw=lw)

# Plot multiple curves using the same axes
def comparison(y, x=None, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "data_type" in kwargs:
        data_type = kwargs["data_type"]
    else:
        data_type = PlotDataType.GENERIC
    if "lw" in kwargs:
        lw = kwargs["lw"]
    else:
        lw = 2
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = None

    plot_config = create_plot_data_type(data_type)
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
        if labels is None:
            label = ""
        else:
            label = labels[i]
        if plot_config.plot_type.value == PlotType.LOG.value:
            logStyle(axis, x[i])
            axis.loglog(x[i], y[i], label=label, lw=lw)
        elif plot_config.plot_type.value == PlotType.XLOG.value:
            logXStyle(axis, x[i])
            axis.semilogx(x[i], y[i], label=label, lw=lw)
        elif plot_config.plot_type.value == PlotType.YLOG.value:
            logYStyle(axis, x[i])
            axis.semilogy(x[i], y[i], label=label, lw=lw)
        else:
            axis.plot(x[i], y[i], label=label, lw=lw)

    if nplot <= 12:
        axis.legend(ncol=ncol, loc='best', bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))

# Compare data to the value of a function
def fcompare(y, x=None, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "func_type" in kwargs:
        func_type = kwargs["func_type"]
    else:
        func_type = PlotFuncType.LINEAR
    if "lw" in kwargs:
        lw = kwargs["lw"]
    else:
        lw = 2
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = None
    if "npts" in kwargs:
        npts = kwargs["npts"]
    else:
        npts = 10
    if "params" in kwargs:
        params = kwargs["params"]
    else:
        params = []

    plot_config = create_plot_func_type(func_type, params)

    if x is None:
        nx = len(y)
        if plot_config.plot_type.value == PlotType.XLOG.value or plot_config.plot_type.value == PlotType.LOG.value:
            x = numpy.linspace(1.0, float(nx-1), nx)
        else:
            x = numpy.linspace(0.0, float(nx-1), nx)

    step = int(len(x)/npts)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    if title is not None:
        axis.set_title(title)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x)
        axis.loglog(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.loglog(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, ps)
        axis.semilogx(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogx(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, ps)
        axis.semilogy(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogy(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.plot(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

# Plot a single curve in a stack of plots that use the same x-axis
def stack(y, ylim, x=None, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "data_type" in kwargs:
        data_type = kwargs["data_type"]
    else:
        data_type = PlotDataType.GENERIC
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = None

    plot_config = create_plot_data_type(data_type)

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

# Compare the cumulative value of a variable as a function of time with its target value
def cumulative(accum, target, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "lw" in kwargs:
        lw = kwargs["lw"]
    else:
        lw = 2
    if "ylabel" in kwargs:
        ylabel = kwargs["ylabel"]
    else:
        ylabel = "y"
    if "label" in kwargs:
        label = kwargs["label"]
    else:
        labels = ylabel

    range = max(accum) - min(accum)
    ntime = len(accum)
    time = numpy.linspace(1.0, ntime-1, ntime)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is None:
        axis[0].set_title(title)

    axis.set_ylim([min(accum)-0.25*range, max(accum)+0.25*range])
    axis.set_ylabel(ylabel)
    axis.set_xlabel("t")
    axis.set_title(title)
    axis.set_xlim([1.0, ntime])
    axis.semilogx(time, accum, label=f"Cumulative "+ ylabel, lw=lw)
    axis.semilogx(time, numpy.full((len(time)), target), label=label, lw=lw)
    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

# Compare the result of a linear regression with teh acutal data
def regression(y, x, results, **kwargs):
    if "title" in kwargs:
        title = kwargs["title"]
    else:
        title = None
    if "type" in kwargs:
        type = kwargs["type"]
    else:
        type = RegressionPlotType.LINEAR

    β = results.params

    if β[1] < 0:
        x_text = 0.1
        y_text = 0.1
        lengend_location = (0.6, 0.65, 0.3, 0.3)
    else:
        x_text = 0.8
        y_text = 0.1
        lengend_location = (0.05, 0.65, 0.3, 0.3)

    plot_config = create_regression_plot_type(type, results, x)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.loglog(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, ps)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.semilogx(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, ps)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])

    axis.legend(loc='best', bbox_to_anchor=lengend_location)

# generate points evenly spaced on a logarithmic axis
def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)
