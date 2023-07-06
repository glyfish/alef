from matplotlib import pyplot
import numpy

from lib.utils import get_param_default_if_missing
from lib.plots import comp
from typing import Callable
from lib.plots.comp.axis import PlotType

def periodogram(data: numpy.ndarray[float], func: Callable[[float], float], results: str, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results of an FBM periodogram analysis used to estimate the Hurst parameter.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    x : numpy.ndarray[float], optional
        Value plotted on x-axis (default is index values of data)
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    lw : int, optional
        Plot line width (default is 2)
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
    legend_loc : string
        Specify legend location. (default best)
    legend_title : string
        Specify legend title. (default None) 
    figsize : (int, int), optional
        Specify the width and height of plot (default is (10,8))
   """
   
    _, axis = pyplot.subplots(figsize=(13, 10))

    x_text = 0.1
    y_text = 0.1
    legend_loc = "upper right"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, results, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, plot_axis_type=PlotType.LOG, **kwargs)


