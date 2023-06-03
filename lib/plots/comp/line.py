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
twinx_comparisons
    Compare multiple curves with the same x scale and different y scales where the
    different y scales are on the left and right axis.
"""

import numpy
import pandas
from datetime import datetime, date
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.units as munits

from lib.plots.comp.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import get_param_default_if_missing

def curve(axis, y: numpy.ndarray, x: numpy.ndarray=None, **kwargs):
    """
    Plot a curve.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    y : numpy.ndarray
        Value plotted on y-axis.
    x : numpy.ndarray, optional
        Value plotted on x-axis (default is index values of y)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    lw : int, optional
        Plot line width (default is 2)
    npts : int, optional
        Number of points plotted (default is length of y)
    figsize : (int, int), optional
        Specify the width and height of plot (default is (10,8))
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis are labeled using scientific notation. (default (-3, 3))
    plot_axis_type : PlotAxisType
        The type of axis used in the plot
    """

    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel          = get_param_default_if_missing("ylabel", None, **kwargs)
    lw              = get_param_default_if_missing("lw", 2, **kwargs)
    npts            = get_param_default_if_missing("npts", None, **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    yscilimits      = get_param_default_if_missing("yscilimits", (-3, 3), **kwargs)
    xscilimits      = get_param_default_if_missing("xscilimits", (-3, 3), **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)

    if npts is None or npts > len(y):
        npts = len(y)

    if x is None:
        x = numpy.linspace(0.0, float(npts-1), npts)

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    if not isinstance(x, (numpy.ndarray, numpy.generic)):
        raise Exception("x must be a numpy.array")

    if not isinstance(y, (numpy.ndarray, numpy.generic)):
        raise Exception("y must be a numpy.array")

    x = x[:npts]
    y = y[:npts]

    if title is not None:
        offset = 1.0 + title_offset
        axis.set_title(title, y=offset)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.ticklabel_format(style='sci', axis='y', scilimits=yscilimits, useMathText=True)
    axis.ticklabel_format(style='sci', axis='x', scilimits=xscilimits, useMathText=True)

    if plot_axis_type.value == PlotType.LINEAR.value:
        axis.plot(x, y, lw=lw)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, lw=lw)
    elif plot_axis_type.value == PlotType.XLOG.value:
        logYStyle(axis, x, y)
        axis.semilogx(x, y, lw=lw)
    elif plot_axis_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, lw=lw)
    else:
        raise Exception("Invalid PlotAxisType")


def comparison(axis, y, x=None, **kwargs):
    """
    Plot multiple curves on same scale.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    y : [numpy.ndarray]
        Value plotted on y-axis.
    x : numpy.array or [numpy.ndarray], optional
        Value plotted on x-axis. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    labels : [string], optional
        Curve labels shown in legend.
    ylabel : string, optional
        Plot y-axis label (default is None)
    lw : int, optional
        Plot line width (default is 2)
    npts : int, optional
        Number of points plotted (default is length of y)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis are labeled using scientific notation. (default (-3, 3))
    legend_loc : string
        Specify legend location. (default best)
    legend_title : string
        Specify legend location. (default best)
    plot_axis_type : PlotAxisType
        Axis type. (default PlotAxisType.LINEAR)
    colors : list[str]
        Curve color values (default uses color cycler).
    """
    
    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel          = get_param_default_if_missing("ylabel", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    lw              = get_param_default_if_missing("lw", 2, **kwargs)
    npts            = get_param_default_if_missing("npts", None, **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "best", **kwargs)
    legend_title    = get_param_default_if_missing("legend_title", None, **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)
    colors          = get_param_default_if_missing("colors", None, **kwargs)

    ncurve = len(y)
    if ncurve == 0:
        raise Exception("Length of y must be greater than zero.")

    if labels is not None and ncurve != len(labels):
        raise Exception("Length of labels must equal length of y.")

    if x is None:
        x = []
        for i in range(ncurve):
            ypts = len(y[i])
            x.append(numpy.linspace(0.0, float(ypts-1), ypts))
    elif not isinstance(x, list):
        x_val = x
        x = []
        for i in range(ncurve):
            x.append(x_val)
    elif isinstance(x, list) and len(x) != len(y):
        for i in range(len(x), ncurve):
            ypts = len(y[i])
            x.append(numpy.linspace(0.0, float(ypts-1), ypts))

    if isinstance(x[0][0], pandas.Timestamp) or isinstance(x[0][0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    if title is not None:
        offset = 1.0 + title_offset
        axis.set_title(title, y=offset)

    if xlabel is not None:
        axis.set_xlabel(xlabel)

    if ylabel is not None:
        axis.set_ylabel(ylabel)

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    for i in range(ncurve):
        xplot = x[i]
        yplot = y[i]

        if npts is None or npts > len(yplot):
            npts = len(yplot)
        
        if not isinstance(xplot, (numpy.ndarray, numpy.generic)):
            raise Exception("x must be a numpy.array")

        if not isinstance(yplot, (numpy.ndarray, numpy.generic)):
            raise Exception("y must be a numpy.array")

        label = labels[i] if labels is not None else None

        color = colors[i] if colors is not None else None

        if plot_axis_type.value == PlotType.LINEAR.value:
            axis.plot(xplot[:npts], yplot[:npts], lw=lw, label=label, color=color)
        elif plot_axis_type.value == PlotType.YLOG.value:
            logYStyle(axis, xplot[:npts], yplot[:npts])
            axis.semilogy(xplot[:npts], yplot[:npts], lw=lw, label=label, color=color)
        elif plot_axis_type.value == PlotType.LOG.value:
            axis.loglog(xplot[:npts], yplot[:npts], lw=lw, label=label, color=color)
        else:
            raise Exception("Invalid PlotAxisType")

    if labels is not None:
        axis.legend(loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9), title=legend_title).set_zorder(10)

def stack(axis, y: list[numpy.ndarray], x=None, **kwargs):
    """
    Plot a horizontal stack of multiple curves on the same x-scale.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
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
    """

    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabels        = get_param_default_if_missing("ylabels", None, **kwargs)
    ylim           = get_param_default_if_missing("ylim", None, **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    lw             = get_param_default_if_missing("lw", 1, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    nplot = len(y)

    if xlabel is not None:
        axis[nplot-1].set_xlabel(xlabel)

    if title is not None:
        axis[0].set_title(title, y=1.0 + title_offset)

    if x is None:
        x = []
        for i in range(nplot):
            ypts = len(y[i])
            x.append(numpy.linspace(0.0, float(ypts-1), ypts))
    elif isinstance(x, numpy.ndarray):
        x = numpy.tile(x, (nplot, 1))

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    for i in range(nplot):
        y_plot = y[i]
        x_plot = x[i]

        if npts is None or npts > len(y_plot):
            npts = len(y_plot)

        x_plot = x_plot[:npts]
        y_plot = y_plot[:npts]

        if isinstance(ylabels, list):
            axis[i].set_ylabel(ylabels[i])
        elif isinstance(ylabels, str):
            axis[i].set_ylabel(ylabels)

        if ylim is None:
            ylim_plot = [1.1*numpy.amin(y), 1.1*numpy.amax(y)]
        else:
            ylim_plot = ylim

        axis[i].set_ylim(ylim_plot)
        axis[i].set_xlim([x_plot[0], x_plot[-1]])

        if labels is not None:
            ypos = 0.8*(ylim_plot[1] - ylim_plot[0]) + ylim_plot[0]
            xpos = 0.8*(x_plot[npts-1] - x_plot[0]) + x_plot[0]
            text = axis[i].text(xpos, ypos, labels[i])
            text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

        if plot_type.value == PlotType.LOG.value:
            axis[i].loglog(x_plot, y_plot, lw=lw)
        elif plot_type.value == PlotType.XLOG.value:
            axis[i].semilogx(x_plot, y_plot, lw=lw)
        elif plot_type.value == PlotType.YLOG.value:
            axis[i].semilogy(x_plot, y_plot, lw=1)
        else:
            axis[i].plot(x_plot, y_plot, lw=1)

def twinx(axis, left: numpy.ndarray, right: numpy.ndarray, x: numpy.ndarray=None, **kwargs):
    """
    Plot two curves with different scales on the y-axis that use the same scale on the
    x-axis.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    left : numpy.ndarray
        Value plotted on left y-axis.
    right : numpy.ndarray
        Value plotted on right y-axis.
    x : numpy.array or numpy.ndarray, optional
        Value plotted on x-axis. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    left_ylabel : string, optional
        Plot left y-axis label (default is None)
    right_ylabel : string, optional
        Plot left y-axis label (default is None)
    labels : [string], optional
        Curve labels shown in legend. Must have length of 2.
    lw : int, optional
        Plot line width (default is 2)
    npts : int, optional
        Number of points plotted (default is length of y)
    left_ylim : (float, float)
        Specify the limits for the left y axis. (default None)
    right_ylim : (float, float)
        Specify the limits for the right y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis is labeled using scientific notation. (default (-3, 3))
    legend_loc : string
        Specify legend location. (default best)
    plot_axis_type : PlotAxisType
        Axis type. (default PlotAxisType.LINEAR)
 """

    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    left_ylabel     = get_param_default_if_missing("left_ylabel", None, **kwargs)
    right_ylabel    = get_param_default_if_missing("right_ylabel", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    left_ylim       = get_param_default_if_missing("left_ylim", None, **kwargs)
    right_ylim      = get_param_default_if_missing("right_ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "best", **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)
    npts            = get_param_default_if_missing("npts", None, **kwargs)

    if npts is not None and (npts > len(left) or npts > len(right)):
        npts = min(len(left), len(right))

    if x is not None and (isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime)):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)
    if left_ylabel is not None:
        axis.set_ylabel(left_ylabel)
    if xlabel is not None:        
        axis.set_xlabel(xlabel)        
    list1 = plot_curve(axis, x, left, npts, 0, **kwargs)

    axis2 = axis.twinx()
    axis2.grid(False)
    axis2._get_lines.prop_cycler = axis._get_lines.prop_cycler
    if right_ylabel is not None:
        axis2.set_ylabel(right_ylabel, rotation=-90, labelpad=15)
    list2 = plot_curve(axis2, x, right, npts, 1, **kwargs)

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if left_ylim is not None:
        axis.set_ylim(left_ylim)

    if right_ylim is not None:
        axis2.set_ylim(right_ylim)

    if xlim is not None:
        axis.set_xlim(xlim)

    twinx_ticks(axis, axis2)

    if labels is not None:
        list = list1 + list2
        labs = [l.get_label() for l in list]
        axis.legend(list, labs, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9)).set_zorder(10)

def twinx_comparison(axis, left: list[numpy.ndarray], right: list[numpy.ndarray], x=None, **kwargs):
    """
    Plot two curves with different scales on the y-axis that use the same scale on the
    x-axis.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    left : list[numpy.ndarray]
        Value plotted on left y-axis.
    right : list[numpy.ndarray]
        Value plotted on right y-axis.
    x : numpy.array or numpy.ndarray, optional
        Value plotted on x-axis. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    left_ylabel : string, optional
        Plot left y-axis label (default is None)
    right_ylabel : string, optional
        Plot left y-axis label (default is None)
    labels : [string], optional
        Curve labels shown in legend. Must have length of 2.
    lw : int, optional
        Plot line width (default is 2)
    npts : int, optional
        Number of points plotted (default is length of y)
    left_ylim : (float, float)
        Specify the limits for the left y axis. (default None)
    right_ylim : (float, float)
        Specify the limits for the right y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis is labeled using scientific notation. (default (-3, 3))
    legend_loc : string
        Specify legend location. (default best)
    plot_axis_type : PlotAxisType
        Axis type. (default PlotAxisType.LINEAR)
    """

    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    left_ylabel     = get_param_default_if_missing("left_ylabel", None, **kwargs)
    right_ylabel    = get_param_default_if_missing("right_ylabel", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    left_ylim       = get_param_default_if_missing("left_ylim", None, **kwargs)
    right_ylim      = get_param_default_if_missing("right_ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "best", **kwargs)
    npts            = get_param_default_if_missing("npts", None, **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)

    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)
    if left_ylabel is not None:
        axis.set_ylabel(left_ylabel)
    if xlabel is not None:        
        axis.set_xlabel(xlabel)

    n_left = min([len(l) for l in left])
    n_right = min([len(r) for r in right])

    if npts is not None and (npts > n_left or npts > n_right):
        npts = min(n_left, n_right)

    nplots = len(left) + len(right)
    if isinstance(x, numpy.ndarray):
        x = numpy.tile(x, (nplots, 1))

    if x is not None and (isinstance(x[0][0], pandas.Timestamp) or isinstance(x[0][0], datetime)):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    list1 = [plot_curve(axis, x[i], left[i], npts, i, **kwargs) for i in range(len(left))]

    axis2 = axis.twinx()
    axis2.grid(False)
    axis2._get_lines.prop_cycler = axis._get_lines.prop_cycler
    if right_ylabel is not None:
        axis2.set_ylabel(right_ylabel, rotation=-90, labelpad=15)

    list2 = [plot_curve(axis2, x[i], right[i], npts, len(left) + i, **kwargs)  for i in range(len(right))]

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if left_ylim is not None:
        axis.set_ylim(left_ylim)

    if right_ylim is not None:
        axis2.set_ylim(right_ylim)

    if xlim is not None:
        axis.set_xlim(xlim)

    twinx_ticks(axis, axis2)

    if labels is not None:
        list = [item for sublist in list1 + list2 for item in sublist]
        labs = [l.get_label() for l in list]
        axis.legend(list, labs, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9))

# Compute twinx ticks
def twinx_ticks(axis1, axis2):
    y1_lim = axis1.get_ylim()
    y2_lim = axis2.get_ylim()
    f = lambda x : y2_lim[0] + (x - y1_lim[0])*(y2_lim[1] - y2_lim[0])/(y1_lim[1] - y1_lim[0])
    ticks = f(axis1.get_yticks())
    axis2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

# Plot twinx curve
def plot_curve(axis, x, y, npts, n, **kwargs):
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    colors         = get_param_default_if_missing("colors", None, **kwargs)
    plot_axis_type = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)

    cycler = axis._get_lines.prop_cycler
    color = colors[n] if colors is not None else next(cycler)['color']

    if x is None:
        ny = len(y)
        x = numpy.linspace(0.0, float(ny-1), ny)
    else:
        npts = min(len(y), len(x))    

    if npts is not None:
        x = x[:npts]
        y = y[:npts]

    label = labels[n] if labels is not None and n < len(labels) else None

    if plot_axis_type.value == PlotType.LINEAR.value:
        return axis.plot(x, y, lw=lw, label=label, color=color, zorder=10)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        return axis.semilogy(x, y, lw=lw, label=label, color=color, zorder=10)
    elif plot_axis_type.value == PlotType.XLOG.value:
        logYStyle(axis, x, y)
        return axis.semilogx(x, y, lw=lw, label=label, color=color, zorder=10)
    elif plot_axis_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        return axis.loglog(x, y, lw=lw, label=label, color=color, zorder=10)
    else:
        raise Exception("Invalid PlotAxisType")
