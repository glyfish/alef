import numpy
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib.plots import comp

def contour(f: numpy.ndarray[float, float], x: numpy.ndarray[float, float], y: numpy.ndarray[float, float], values: numpy.ndarray[float], **kwargs):
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
    figsize : (int, int)
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)

    _, axis = pyplot.subplots(figsize=figsize)
    comp.contour(axis, f, x, y, values, **kwargs)