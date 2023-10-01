import numpy
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib.plots import comp
from lib.plots.comp.axis import PlotType
from lib.data import OLSResult

def periodogram(data: numpy.ndarray[float], results: OLSResult, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results of an FBM periodogram analysis used to estimate the Hurst parameter.

    Parameters
    ----------
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    results : OLSResult
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

    transforms = results.param_transforms
    const = results.const_transform.param
    H = transforms[0].param.est
    func = lambda x: const.est*x**(1.0 - 2.0*H)

    x_text = 0.1 if H > 0.5 else 0.8
    y_text = 0.1

    legend_loc = "upper right" if H > 0.5 else "upper left"

    labels = ["Power Spectrum", results.model]
    xlabel = r"$\omega$"
    ylabel = r"$\rho_\omega$"

    estimates = f"{transforms[0].param.est_label}={format(transforms[0].param.est, '1.2f')}\n" + \
                f"{transforms[0].param.err_label}={format(transforms[0].param.err, '1.2e')}\n" + \
                f"{const.est_label}={format(const.est, '1.2e')}\n" + \
                f"{const.err_label}={format(const.err, '1.2e')}\n" + \
                f"$R^2$={format(results.r2, '1.2f')}"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, estimates, bbox=bbox, fontsize=12.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, plot_axis_type=PlotType.LOG, labels=labels, 
                  ylabel=ylabel, xlabel=xlabel, **kwargs)
    

def variance_agg(data: numpy.ndarray[float], results: OLSResult, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results of an FBM variance aggregation analysis used to estimate the Hurst parameter.

    Parameters
    ----------
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    results : OLSResult
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

    transforms = results.param_transforms
    const = results.const_transform.param
    H = transforms[0].param.est
    func = lambda x:  const.est*x**(2.0*(H - 1.0))

    x_text = 0.1
    y_text = 0.1

    legend_loc = "upper right"

    labels = ["Data", results.model]
    xlabel = r"$m$"
    ylabel = r"VAR$\left(X^m\right)$"

    estimates = f"{transforms[0].param.est_label}={format(transforms[0].param.est, '1.2f')}\n" + \
                f"{transforms[0].param.err_label}={format(transforms[0].param.err, '1.2e')}\n" + \
                f"{const.est_label}={format(const.est, '1.2e')}\n" + \
                f"{const.err_label}={format(const.err, '1.2e')}\n" + \
                f"$R^2$={format(results.r2, '1.2f')}"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, estimates, bbox=bbox, fontsize=12.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, plot_axis_type=PlotType.LOG, labels=labels, 
                  ylabel=ylabel, xlabel=xlabel, **kwargs)


def ecm_beta(data: numpy.ndarray[float], results: OLSResult, x: numpy.ndarray[float]=None, **kwargs):
    """"
    Plot the results OLS analysis that computes ECM beta parameter.

    Parameters
    ----------
    data : numpy.ndarray
        Data compared to function.
    func : Callable[[float], float]
        Function plotted as a function of x.
    results : OLSResult
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
   
    figsize = get_param_default_if_missing("figsize", (8, 6), **kwargs)
    title = get_param_default_if_missing("title", None, **kwargs)

    _, axis = pyplot.subplots(figsize=figsize)

    if title is None:
        kwargs["title"] = r"ECM $\beta$ Estimation"

    transforms = results.param_transforms
    const = results.const_transform.param
    beta = transforms[0].param.est
    func = lambda x:  const.est + beta * x

    x_text = 0.1
    if beta >= 0.0:
        y_text = 0.75
        legend_loc = "lower right"
    else:
        y_text = 0.1
        legend_loc = "upper right"

    labels = ["Data", results.model]
    xlabel = r"$x_t$"
    ylabel = r"$y_t$"

    estimates = f"{transforms[0].param.est_label}={format(transforms[0].param.est, '1.2f')}, " + \
                f"{transforms[0].param.err_label}={format(transforms[0].param.err, '1.2e')}\n" + \
                f"{const.est_label}={format(const.est, '1.2e')}, " + \
                f"{const.err_label}={format(const.err, '1.2e')}\n" + \
                f"$R^2$={format(results.r2, '1.2f')}"

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, estimates, bbox=bbox, fontsize=12.0, zorder=7, transform=axis.transAxes)

    comp.fscatter(axis, data, func, x, legend_loc=legend_loc, labels=labels, 
                  ylabel=ylabel, xlabel=xlabel, **kwargs)

