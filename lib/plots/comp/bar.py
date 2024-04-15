import numpy
import pandas
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib import pyplot
from datetime import datetime, date

from lib.plots.comp.line import (__plot_curve, __twinx_ticks)

from lib.utils import get_param_default_if_missing
from lib import config

def bar(axis: pyplot.axis, y, x=None, **kwargs):
    """
    Plot samples in a bar chart.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    y : numpy.ndarray
        Value plotted on y-axis.
    x : numpy.ndarray
        Value plotted in x axis (default use y index)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    alpha : float
        Bar alpha (default 0.5)
    border_width : float
        Bar border width (default)
    bar_width : float
        Bar width ras faction of x delta.
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    """

    title          = get_param_default_if_missing("title", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", "x", **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", "y", **kwargs)
    xlim           = get_param_default_if_missing("xlim", None, **kwargs)
    ylim           = get_param_default_if_missing("ylim", None, **kwargs)

    if x is None:
        x = numpy.linspace(0, len(y) - 1, len(y))

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    
    
    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    __plot_bar(axis, x, y, 0, **kwargs)

def twinx_bar(axis: pyplot.axis, left: numpy.ndarray, right: numpy.ndarray, x_left: numpy.ndarray=None, x_right: numpy.ndarray=None, **kwargs):
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
    x_left : numpy.array or numpy.ndarray, optional
        Value plotted for the left on the x-axis. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    x_right : numpy.array or numpy.ndarray, optional
        Value plotted for the right on the x-axis. If property is an list each x is plotted with y of 
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
    alpha : float
        Bar alpha (default 0.5)
    border_width : float
        Bar border width (default)
    bar_width : float
        Bar width ras faction of x delta.
    colors : list[float]
        Colors. Default uses color cycler
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

    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)
    if left_ylabel is not None:
        axis.set_ylabel(left_ylabel)
    if xlabel is not None:        
        axis.set_xlabel(xlabel)

    if x_left is None and x_right is not None:
        x_left = x_right
    if x_left is not None and x_right is None:
        x_right = x_left
    if x_left is None and x_right is None:
        x_right = numpy.linspace(0, len(right) - 1, len(right))
        x_left = numpy.linspace(0, len(left) - 1, len(left))

    if (isinstance(x_left[0], pandas.Timestamp) and isinstance(x_right[0], pandas.Timestamp)) or  (isinstance(x_left[0], datetime) and isinstance(x_right[0], datetime)):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter  

    list1 = __plot_bar(axis, x_left, left, 0, **kwargs)

    axis2 = axis.twinx()
    axis2._get_lines.prop_cycler = axis._get_lines.prop_cycler
    if right_ylabel is not None:
        axis2.set_ylabel(right_ylabel, rotation=-90, labelpad=15)
    list2 = __plot_bar(axis2, x_right, right, 1, **kwargs)
    
    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if left_ylim is not None:
        axis.set_ylim(left_ylim)

    if right_ylim is not None:
        axis2.set_ylim(right_ylim)

    if xlim is not None:
        axis.set_xlim(xlim)

    __twinx_ticks(axis, axis2)
    axis2.grid(False)

    if labels is not None:
        list = [list1,  list2]
        labs = [l.get_label() for l in list]
        axis2.legend(list, labs, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9)).set_zorder(10)

def twinx_bar_line(axis: pyplot.axis, y_bar: numpy.ndarray, y_line: numpy.ndarray, x_bar: numpy.ndarray=None, x_line: numpy.ndarray=None, **kwargs):
    """
    Plot two curves with different scales on the y-axis that use the same scale on the
    x-axis.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    y_bar : numpy.ndarray
        Bar y axis plot data.
    y_line : numpy.ndarray
        Line y axis plot data.
    x_bar : numpy.array or numpy.ndarray, optional
        Value plotted on x-axis for bar plot. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    x_line : numpy.array or numpy.ndarray, optional
        Value plotted on x-axis for bar plot. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    bar_ylabel : string, optional
        Bar plot y-axis label (default is None)
    line_ylabel : string, optional
        Line plot left y-axis label (default is None)
    labels : [string], optional
        Curve labels shown in legend. Must have length of 2.
    lw : int, optional
        Plot line width (default is 2)
    npts : int, optional
        Number of points plotted (default is length of y)
    bar_ylim : (float, float)
        Specify the limits for the bar y axis. (default None)
    line_ylim : (float, float)
        Specify the limits for the right y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis is labeled using scientific notation. (default (-3, 3))
    legend_loc : string
        Specify legend location. (default best)
    prec : int
        Precision shown for y axis ticks.
    alpha : float
        Bar alpha (default 0.5)
    border_width : float
        Bar border width (default)
    bar_width : float
        Bar width ras faction of x delta.
    colors : list[float]
        Colors. Default uses color cycler
    """

    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    bar_ylabel      = get_param_default_if_missing("bar_ylabel", None, **kwargs)
    line_ylabel     = get_param_default_if_missing("line_ylabel", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    bar_ylim       = get_param_default_if_missing("bar_ylim", None, **kwargs)
    line_ylim      = get_param_default_if_missing("line_ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "best", **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)

    if x_bar is None:
        x_bar = numpy.linspace(0, len(y_bar) - 1, len(y_bar))

    if x_line is None:
        x_line = numpy.linspace(0, len(y_line) - 1, len(y_line))

    if isinstance(x_bar[0], pandas.Timestamp) or isinstance(x_bar[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)
    if bar_ylabel is not None:
        axis.set_ylabel(bar_ylabel)
    if xlabel is not None:        
        axis.set_xlabel(xlabel)        
    list1 = __plot_bar(axis, x_bar, y_bar, 0, 10, **kwargs)

    axis2 = axis.twinx()
    axis2._get_lines.prop_cycler = axis._get_lines.prop_cycler
    if line_ylabel is not None:
        axis2.set_ylabel(line_ylabel, rotation=-90, labelpad=15)
    list2 = __plot_curve(axis2, x_line, y_line, 1, **kwargs)
    
    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if bar_ylim is not None:
        axis.set_ylim(bar_ylim)

    if line_ylim is not None:
        axis2.set_ylim(line_ylim)

    if xlim is not None:
        axis.set_xlim(xlim)

    __twinx_ticks(axis, axis2)
    axis2.grid(False)

    if labels is not None:
        list = [list1] + list2
        labs = [l.get_label() for l in list]
        axis.legend(list, labs, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9))

def twinx_bar_comparison(axis: pyplot.axis, left: list[numpy.ndarray], right: list[numpy.ndarray], x_left: list[numpy.ndarray]=None, x_right: list[numpy.ndarray]=None, **kwargs):
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
    x_left : numpy.array or numpy.ndarray, optional
        Value plotted for the left on the x-axis. If property is an list each x is plotted with y of 
        same index (default is index values of y)
    x_right : numpy.array or numpy.ndarray, optional
        Value plotted for the right on the x-axis. If property is an list each x is plotted with y of 
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
    alpha : float
        Bar alpha (default 0.5)
    border_width : float
        Bar border width (default)
    bar_width : float
        Bar width ras faction of x delta.
    colors : list[float]
        Colors. Default uses color cycler
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

    if title is not None:
        axis.set_title(title, y=title_offset + 1.0)
    if left_ylabel is not None:
        axis.set_ylabel(left_ylabel)
    if xlabel is not None:        
        axis.set_xlabel(xlabel)

    if x_left is None and x_right is not None:
        x_left = x_right
    if x_left is not None and x_right is None:
        x_right = x_left
    if x_left is None and x_right is None:
        x_right = numpy.linspace(0, len(right) - 1, len(right))
        x_left = numpy.linspace(0, len(left) - 1, len(left))

    if (isinstance(x_left[0], pandas.Timestamp) and isinstance(x_right[0], pandas.Timestamp)) or  (isinstance(x_left[0], datetime) and isinstance(x_right[0], datetime)):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    
    list1 = __plot_bar(axis, x_left, left, 0, 1, **kwargs)

    axis2 = axis.twinx()
    axis2._get_lines.prop_cycler = axis._get_lines.prop_cycler
    if right_ylabel is not None:
        axis2.set_ylabel(right_ylabel, rotation=-90, labelpad=15)
    list2 = __plot_bar(axis2, x_right, right, 1, 2, **kwargs)
    
    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if left_ylim is not None:
        axis.set_ylim(left_ylim)

    if right_ylim is not None:
        axis2.set_ylim(right_ylim)

    if xlim is not None:
        axis.set_xlim(xlim)

    __twinx_ticks(axis, axis2)
    axis2.grid(False)

    if labels is not None:
        list = [list1,  list2]
        labs = [l.get_label() for l in list]
        axis.legend(list, labs, loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9))

def hist(axis: pyplot.axis, samples: numpy.ndarray, fx=None, **kwargs):
    """
    Plot samples in histogram and compare with given function.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    samples : numpy.ndarray
        Value plotted on y-axis.
    fx : function of x
        Comparison function (default is None)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    lw : int, optional
        Plot line width if fx is present (default is 2)
    nbins : int, optional
        Number of histogram bins (default is 50)
    density : int, optional
        Normalize histogram to represent a probability density (dealt is True)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    labels : [string], optional
        Curve labels shown in legend. The first is for histogram and second is f(x) if provided
        and the labels are only shown of fx is not None (default None).
    legend_loc : string
        Specify legend location. (default best)
    """

    title           = get_param_default_if_missing("title", None, **kwargs)
    title_offset    = get_param_default_if_missing("title_offset", 0.0, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel          = get_param_default_if_missing("ylabel", None, **kwargs)
    lw              = get_param_default_if_missing("lw", 2, **kwargs)
    nbins           = get_param_default_if_missing("nbins", None, **kwargs)
    density         = get_param_default_if_missing("density", True, **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    legend_loc      = get_param_default_if_missing("legend_loc", "best", **kwargs)

    if title is not None:
        axis.set_title(title, y=title_offset)
    if xlabel is not None:
        axis.set_ylabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)

    axis.set_prop_cycle(config.distribution_sample_cycler)

    if labels is not None:
        hist_label = labels[0] 
        fx_label = labels[1]

    _, bins, _ = axis.hist(samples, nbins, rwidth=0.8, density=density, label=hist_label, zorder=5)

    delta = (bins[-1] - bins[0]) / 500.0
    x = numpy.arange(bins[0], bins[-1], delta)

    if fx is not None:
        axis.plot(x, fx(x), lw=lw, zorder=6, label=fx_label)

    if ylim is not None:
        axis.set_ylim(ylim)

    if xlim is None:
        xlim = (x[0], x[-1])
    else:
        xlim = (bins[0], bins[-1])
    axis.set_xlim(xlim)

    if labels is not None:
        axis.legend(loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9))

def __plot_bar(axis, x, y, n, zorder=10, **kwargs):
    alpha        = get_param_default_if_missing("alpha", 0.5, **kwargs)
    border_width = get_param_default_if_missing("border_width", 1, **kwargs)
    bar_width    = get_param_default_if_missing("bar_width", 1.0, **kwargs)
    labels       = get_param_default_if_missing("labels", None, **kwargs)
    colors       = get_param_default_if_missing("colors", None, **kwargs)

    alpha_value = alpha[n] if isinstance(alpha, list) else alpha
        
    cycler = axis._get_lines.prop_cycler
    color = colors[n] if colors is not None else next(cycler)['color']

    width = bar_width*(x[1]-x[0])

    if labels is None:
        return axis.bar(x, y, align='center', width=width, zorder=zorder, alpha=alpha_value, linewidth=border_width, color=color)
    else:
        return axis.bar(x, y, align='center', width=width, zorder=zorder, alpha=alpha_value, linewidth=border_width, label=labels[n], color=color)

