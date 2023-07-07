import numpy
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib.plots import comp
from lib.plots.comp.axis import PlotType
from lib.data import OLSSingleVarResult

def periodogram(data: numpy.ndarray[float], results: OLSSingleVarResult, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results of an FBM periodogram analysis used to estimate the Hurst parameter.

    Parameters
    ----------
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    results : OLSSingleVarResult
        OLS results.
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
   
    figsize = get_param_default_if_missing("figsize", (10, 6), **kwargs)
    title = get_param_default_if_missing("title", None, **kwargs)

    _, axis = pyplot.subplots(figsize=figsize)

    if title is None:
        kwargs["title"] = f"FBM Periodogram"

    transform = results.transform
    const = transform.const.est
    H = transform.param.est
    func = lambda x: const*x**(1.0 - 2.0*H)

    x_text = 0.1 if H > 0.5 else 0.8
    y_text = 0.1

    legend_loc = "upper right" if H > 0.5 else "upper left"

    labels = ["Power Spectrum", transform.model]
    xlabel = r"$\omega$"
    ylabel = r"$\rho_\omega$"

    estimates = f"{transform.param.est_label}={format(transform.param.est, '1.2f')}\n" + \
                f"{transform.param.err_label}={format(transform.param.err, '1.2e')}\n" + \
                f"{transform.const.est_label}={format(transform.const.est, '1.2e')}\n" + \
                f"{transform.const.err_label}={format(transform.const.err, '1.2e')}\n" + \
                f"$R^2$={format(results.r2, '1.2f')}"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, estimates, bbox=bbox, fontsize=12.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, plot_axis_type=PlotType.LOG, labels=labels, 
                  ylabel=ylabel, xlabel=xlabel, **kwargs)
    
def variance_agg(data: numpy.ndarray[float], results: OLSSingleVarResult, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results of an FBM variance aggregation analysis used to estimate the Hurst parameter.

    Parameters
    ----------
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    results : OLSSingleVarResult
        OLS results.
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
   
    figsize = get_param_default_if_missing("figsize", (10, 6), **kwargs)
    title = get_param_default_if_missing("title", None, **kwargs)

    _, axis = pyplot.subplots(figsize=figsize)

    if title is None:
        kwargs["title"] = f"FBM Aggregated Variance"

    transform = results.transform
    const = transform.const.est
    H = transform.param.est
    func = lambda x: const*x**(2.0*(H - 1.0))

    x_text = 0.1
    y_text = 0.1

    legend_loc = "upper right"

    labels = ["Data", transform.model]
    xlabel = r"$m$"
    ylabel = r"VAR$\left(X^m\right)$"

    estimates = f"{transform.param.est_label}={format(transform.param.est, '1.2f')}\n" + \
                f"{transform.param.err_label}={format(transform.param.err, '1.2e')}\n" + \
                f"{transform.const.est_label}={format(transform.const.est, '1.2e')}\n" + \
                f"{transform.const.err_label}={format(transform.const.err, '1.2e')}\n" + \
                f"$R^2$={format(results.r2, '1.2f')}"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, estimates, bbox=bbox, fontsize=12.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, plot_axis_type=PlotType.LOG, labels=labels, 
                  ylabel=ylabel, xlabel=xlabel, **kwargs)


