import numpy
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib import config

def contour(axis: pyplot.axis, 
            f: numpy.ndarray[float, float],
            x: numpy.ndarray[float, float], 
            y: numpy.ndarray[float, float], 
            values: numpy.ndarray[float],
            **kwargs):
    """
    Contour plot for f(x,y)

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    y : numpy.ndarray[float, float]
        Value plotted on y-axis.
    x : numpy.ndarray[float, float]
        Value plotted in x axis
    f : numpy.ndarray[float, float]
        Function contoured.
    values : numpy.ndarray[float]
        Values of contours plotted.
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
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

    if xlim is None:
        xlim = (numpy.min(x), numpy.max(x))
    if ylim is None:
        ylim = (numpy.min(y), numpy.max(y))
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    axis.set_title(title, y=1.0 + title_offset)

    contour = axis.contour(x, y, f, values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
