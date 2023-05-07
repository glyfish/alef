"""
Basic plot components.

Functions
---------
curve
    Plot x and y data on specified axis type.
comparison
    Plot multiple curves on the same x-scale
stack
    Plot a horizontal stack of multiple curves on the same x-scale.
twinx
    Plot two curves with different y scales and the same x scale in the same plot with the scale
    of one curve on the left axis and the other on the right.
"""
import numpy
from matplotlib import pyplot
import matplotlib.ticker

from lib.plots.comp.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import get_param_default_if_missing


def curve(y: numpy.ndarray, x: numpy.ndarray=None, **kwargs):
    """
    Plot x and y data on specified axis type.

    Parameters
    ----------
    x : numpy.ndarray
        data x-axis values.
    y : numpy.ndarray
        data y-axis values.
    plot_type : PlotType
        Axis type.
    title : str
        Plot title. (default None)
    title_offset : str
        Title offset. (default 0)
    xlabel : str
        X-axis label. (default None)
    ylabel : str
        Y-axis label. (default None)
    lw : int
        Line width. (default 2)
    npts : int
        Number of points to plot. (default len(y))
    figsize : (int, int)
        Figure size.
    """
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)
    figsize        = get_param_default_if_missing("figsize", (13, 10), **kwargs)

    if npts is None or npts > len(y):
        npts = len(y)

    if x is None:
        x = numpy.linspace(0, npts-1, npts)

    x = x[:npts]
    y = y[:npts]

    _, axis = pyplot.subplots(figsize=figsize)

    if title is not None:
        axis.set_title(title, y=1.0 + title_offset)

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

def comparison(y: list[numpy.ndarray], x=None, **kwargs):
    """
    Plot multiple curves on the same x-scale.

    Parameters
    ----------
    y : list[numpy.ndarray]
        data y-axis values.
    x : list[numpy.ndarray] or numpy.ndarray
        data x-axis values (default None).
    plot_type : PlotType
        Axis type.
    title : str
        Plot title. (default None)
    title_offset : str
        Title offset. (default 0)
    xlabel : str
        X-axis label. (default None)
    ylabel : str
        Y-axis label. (default None)
    lw : int
        Line width. (default 2)
    npts : int
        Number of points to plot. (default len(y))
    figsize : (int, int)
        Figure size.
    legend_loc : str
        Legend location. (default best)
    """
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel         = get_param_default_if_missing("ylabels", None, **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)
    figsize        = get_param_default_if_missing("figsize", (13, 10), **kwargs)
    legend_loc     = get_param_default_if_missing("legend_loc", "upper right", **kwargs)

    ncol = int(len(y)/6) + 1
    nplot = len(y)

    _, axis = pyplot.subplots(figsize=figsize)

    if title is not None:
        axis.set_title(title, y=1.0 + title_offset)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    for i in range(nplot):
        y_plot = y[i]
        if npts is None or npts > len(y_plot):
            npts = len(y_plot)

        if x_plot is None:
            x_plot = numpy.linspace(0, npts - 1, npts)

        if isinstance(x, list):
            x_plot = x[i]

        if not isinstance(x_plot, numpy.ndarray):
            raise Exception(f"x must be type numpy.ndarray")

        x_plot = x_plot[:npts]
        y_plot = y_plot[:npts]

        label = None
        if labels is not None:
            label = labels[i]

        if plot_type.value == PlotType.LOG.value:
            logStyle(axis, x_plot, y_plot)
            axis.loglog(x_plot, y_plot, label=label, lw=lw)
        elif plot_type.value == PlotType.XLOG.value:
            logXStyle(axis, x_plot, y_plot)
            axis.semilogx(x_plot, y_plot, label=label, lw=lw)
        elif plot_type.value == PlotType.YLOG.value:
            logYStyle(axis, x_plot, y_plot)
            axis.semilogy(x_plot, y_plot, label=label, lw=lw)
        else:
            axis.plot(x_plot, y_plot, label=label, lw=lw)

    if nplot <= 12 and labels is not None:
        axis.legend(ncol=ncol, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))

def stack(y: list[numpy.ndarray], x=None, **kwargs):
    """
    Plot a horizontal stack of multiple curves on the same x-scale.

    Parameters
    ----------
    y : list[numpy.ndarray]
        data y-axis values.
    x : list[numpy.ndarray] or numpy.ndarray
        data x-axis values (default None).
    plot_type : PlotType
        Axis type.
    title : str
        Plot title. (default None)
    title_offset : str
        Title offset. (default 0)
    xlabel : str
        X-axis label. (default None)
    ylabel : str
        Y-axis label. (default None)
    lw : int
        Line width. (default 1)
    npts : int
        Number of points to plot. (default len(y))
    figsize : (int, int)
        Figure size.
    """
    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabels        = get_param_default_if_missing("ylabel",None, **kwargs)
    ylim           = get_param_default_if_missing("ylim", None, **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 1, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)
    figsize        = get_param_default_if_missing("figsize", (13, 10), **kwargs)

    nplot = len(y)
    _, axis = pyplot.subplots(nplot, sharex=True, figsize=figsize)

    axis[nplot-1].set_xlabel(xlabel)

    if title is not None:
        axis[0].set_title(title, y=1.0 + title_offset)

    for i in range(nplot):
        y_plot = y[i]
        if npts is None or npts > len(y_plot):
            npts = len(y_plot)

        if x_plot is None:
            x_plot = numpy.linspace(0, npts - 1, npts)

        if isinstance(x, list):
            x_plot = x[i]

        if not isinstance(x_plot, numpy.ndarray):
            raise Exception(f"x must be type numpy.ndarray")

        x_plot = x_plot[:npts]
        y_plot = y_plot[:npts]

        axis[i].set_ylabel(ylabels[i])

        if ylim is None:
            ylim_plot = [1.1*numpy.amin(y), 1.1*numpy.amax(y)]
        else:
            ylim_plot = ylim

        axis[i].set_ylim(ylim_plot)
        axis[i].set_xlim([x[0], x[npts-1]])

        if labels is not None:
            ypos = 0.8*(ylim_plot[1] - ylim_plot[0]) + ylim_plot[0]
            xpos = 0.8*(x_plot[npts-1] - x_plot[0]) + x_plot[0]
            text = axis[i].text(xpos, ypos, labels[i], fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

        if plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x_plot, y_plot, lw=lw)
        elif plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x_plot, y_plot, lw=lw)
        elif plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x_plot, y_plot, lw=1)
        else:
            axis[i].plot(x_plot, y_plot, lw=1)

def twinx(y_left: numpy.ndarray, y_right: numpy.ndarray, x_left: numpy.ndarray=None, x_right: numpy.ndarray=None, **kwargs):
    """
    Plot two curves with different y scales and the same x scale in the same plot with the scale
    of one curve on the left axis and the other on the right.

    Parameters
    ----------
    y_left : numpy.ndarray
        data left y-axis values.
    y_right : numpy.ndarray
        data right y-axis values.
    x_left : numpy.ndarray
        data left x-axis values.
    x_right : numpy.ndarray
        data right x-axis values.
    plot_type : PlotType
        Axis type.
    title : str
        Plot title. (default None)
    title_offset : str
        Title offset. (default 0)
    xlabel : str
        X-axis label. (default None)
    ylabel : str
        Y-axis label. (default None)
    lw : int
        Line width. (default 1)
    ylim : (int, int)
        Number of points to plot. (default len(y))
    figsize : (int, int)
        Figure size.
    legend_loc : str
        Legend location. (default best)
    """
    plot_type       = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    left_ylabel     = get_param_default_if_missing("left_ylabel", None, **kwargs)
    right_ylabel    = get_param_default_if_missing("right_ylabel", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "upper right", **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)

    figure, axis1 = pyplot.subplots(figsize=(13, 10))

    axis1.set_title(title, y=title_offset)

    # first plot left axis1
    axis1.set_ylabel(left_ylabel)
    axis1.set_xlabel(xlabel)
    label = labels[0] if labels is not None else None
    __plot_curve(axis1, y_left, x_left, plot_type, label, **kwargs)

    # second plot right axis2
    axis2 = axis1.twinx()
    axis2._get_lines.prop_cycler = axis1._get_lines.prop_cycler
    axis1.set_ylabel(right_ylabel)
    label = labels[0] if labels is not None else None
    __plot_curve(axis2, y_right, x_right, plot_type, label, **kwargs)

    if ylim is not None:
        axis1.set_ylim(ylim)

    __twinx_ticks(axis1, axis2)
    axis2.grid(False)

    figure.legend(loc=legend_loc, bbox_to_anchor=(0.2, 0.2, 0.6, 0.6))

def __twinx_ticks(axis1, axis2):
    """
    Compute ticks for right axis for that they align with the right.
    """
    y1_lim = axis1.get_ylim()
    y2_lim = axis2.get_ylim()
    f = lambda x : y2_lim[0] + (x - y1_lim[0])*(y2_lim[1] - y2_lim[0])/(y1_lim[1] - y1_lim[0])
    ticks = f(axis1.get_yticks())
    axis2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

def __plot_curve(axis, y, x, plot_type, label, **kwargs):
    """
    Plot curves for twinx plots.
    """
    lw   = get_param_default_if_missing("lw", 2, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)

    if npts is None or npts > len(y):
        npts = len(y)

    x = x[:npts]
    y = y[:npts]

    if plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=label, lw=lw)
    elif plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=label, lw=lw)
    elif plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=label, lw=lw)
    else:
        axis.plot(x, y, label=label, lw=lw)
