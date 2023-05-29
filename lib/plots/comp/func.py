"""
Basic plot components for comparing data to functions.

Functions
---------
fpoints
    Plot x and y data on specified axis type.
fcurve
    Plot multiple curves on the same x-scale
"""

import numpy
import pandas
from datetime import datetime, date
import matplotlib.dates as mdates
import matplotlib.units as munits

from lib.plots.comp.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

def fpoints(axis, data: numpy.ndarray[float], func: numpy.ndarray[float], x: numpy.ndarray=None, fx: numpy.ndarray=None, **kwargs):
    """"
    Compare data to a function by plotting the data as a curve
    and the function as points.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : numpy.ndarray
        Data compared to function.
    func : numpy.ndarray
        Function data plotted as points
    x : numpy.ndarray, optional
        Value plotted on x-axis (default is index values of data)
    fx : numpy.ndarray, optional
        Value plotted on x-axis for function (default is index values of data)
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
    figsize : (int, int), optional
        Specify the width and height of plot (default is (10,8))
    labels : [string], optional
        Curve labels shown in legend.
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
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    yscilimits      = get_param_default_if_missing("yscilimits", (-3, 3), **kwargs)
    xscilimits      = get_param_default_if_missing("xscilimits", (-3, 3), **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)

    if x is None:
        npts = len(data)
        x = numpy.linspace(0.0, float(npts-1), npts)

    if fx is None:
        npts = len(func)
        fx = numpy.linspace(0.0, float(npts-1), npts)

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, y=1.0 + title_offset)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.ticklabel_format(style='sci', axis='y', scilimits=yscilimits, useMathText=True)
    axis.ticklabel_format(style='sci', axis='x', scilimits=xscilimits, useMathText=True)

    data_label = None
    func_label = None
    if labels is not None and len(labels) == 2:
        data_label = labels[0]
        func_label = labels[1]

    if plot_axis_type.value == PlotType.LOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        if fx[0] == 0.0:
            fx = fx[1:]
            func = func[1:]
        logStyle(axis, x, data)
        axis.loglog(x, data, label=data_label, lw=lw)
        axis.loglog(fx, func, label=func_label, marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_axis_type.value == PlotType.XLOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        if fx[0] == 0.0:
            fx = fx[1:]
            func = func[1:]
        logXStyle(axis, x, data)
        axis.semilogx(x, data, label=data_label, lw=lw)
        axis.semilogx(fx, func, label=func_label, marker='o', linestyle="None", markeredgewidth=1.0, markersize=10.0)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, data)
        axis.semilogy(x, data, label=data_label, lw=lw)
        axis.semilogy(fx, func, label=func_label, marker='o', linestyle="None", markeredgewidth=1.0, markersize=10.0)
    else:
        axis.plot(x, data, label=data_label, lw=lw)
        axis.plot(fx, func, label=func_label, marker='o', linestyle="None", markeredgewidth=1.0, markersize=10.0)

    if labels is not None:
        axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

def fcurve(axis, data: numpy.ndarray[float], func: numpy.ndarray[float], x: numpy.ndarray=None, fx: numpy.ndarray=None, **kwargs):
    """"
    Compare data to a function by plotting by plotting both as curves.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : numpy.ndarray
        Data compared to function.
    func : numpy.ndarray
        Function data plotted as points
    x : numpy.ndarray, optional
        Value plotted on x-axis (default is index values of data)
    fx : numpy.ndarray, optional
        Value plotted on x-axis for function (default is index values of data)
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
    figsize : (int, int), optional
        Specify the width and height of plot (default is (10,8))
    labels : [string], optional
        Curve labels shown in legend.
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
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    yscilimits      = get_param_default_if_missing("yscilimits", (-3, 3), **kwargs)
    xscilimits      = get_param_default_if_missing("xscilimits", (-3, 3), **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)

    if x is None:
        npts = len(data)
        x = numpy.linspace(0.0, float(npts-1), npts)

    if fx is None:
        npts = len(func)
        fx = numpy.linspace(0.0, float(npts-1), npts)

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter    

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, y=1.0 + title_offset)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    axis.ticklabel_format(style='sci', axis='y', scilimits=yscilimits, useMathText=True)
    axis.ticklabel_format(style='sci', axis='x', scilimits=xscilimits, useMathText=True)

    data_label = None
    func_label = None
    if labels is not None and len(labels) == 2:
        data_label = labels[0]
        func_label = labels[1]

    if plot_axis_type.value == PlotType.LOG.value:
        if x[0] == 0.0:
            x = x[1:]
            data = data[1:]
        if fx[0] == 0.0:
            fx = fx[1:]
            func = func[1:]
        logStyle(axis, x, data)
        axis.loglog(x, data, label=data_label, lw=lw)
        axis.loglog(fx, func, label=func_label, lw=lw)
    elif plot_axis_type.value == PlotType.XLOG.value:
        if x[0] == 0.0:
            x = x[1:]
            data = data[1:]
        if fx[0] == 0.0:
            fx = fx[1:]
            func = func[1:]
        logXStyle(axis, x, data)
        axis.semilogx(x, data, label=data_label, lw=lw)
        axis.semilogx(fx, func, label=func_label, lw=lw)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, data)
        axis.semilogy(x, data, label=data_label, lw=lw)
        axis.semilogy(fx, func, label=func_label, lw=lw)
    else:
        axis.plot(x, data, label=data_label, lw=lw)
        axis.plot(fx, func, label=func_label, lw=lw)

    if labels is not None:
        axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
