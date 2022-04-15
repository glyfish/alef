import numpy
from enum import Enum
from matplotlib import pyplot

from lib import stats

from lib.plots.config import (PlotType, logStyle, logXStyle, logYStyle)

###############################################################################################
# Specify PlotConfig for regression plot
class RegPlotType(Enum):
    LINEAR = 1          # Default
    FBM_AGG_VAR = 2     # FBM variance aggregation
    FBM_PSPEC = 3       # FBM Power Spectrum

###############################################################################################
# Create regression PlotConfig
class RegPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, results_text=None, y_fit=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.results_text = results_text
        self.y_fit = y_fit

###############################################################################################
# Create regression plot configuartion
def create_reg_plot_type(plot_type, results, x):
    β = results.params
    σ = results.bse[1]/2
    r2 = results.rsquared

    if plot_type.value == RegPlotType.FBM_AGG_VAR.value:
        h = float(1.0 + β[1]/2.0)
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{\sigma}^2=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return RegPlotConfig(xlabel=r"$\omega$",
                             ylabel=r"$Var(X^{m})$",
                             plot_type=PlotType.LOG,
                             results_text=results_text,
                             legend_labels=["Data", r"$Var(X^{m})=\sigma^2 m^{2H-2}$"],
                             y_fit=10**β[0]*x**β[1])
    elif plot_type.value == RegPlotType.FBM_PSPEC.value:
        h = float(1.0 - β[1])/2.0
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{C}=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return RegPlotConfig(xlabel=r"$m$",
                             ylabel=r"$\hat{\rho}^H_\omega$",
                             plot_type=PlotType.LOG,
                             results_text=results_text,
                             legend_labels=["Data", r"$\hat{\rho}^H_\omega = C | \omega |^{1 - 2H}$"],
                             y_fit=10**β[0]*x**β[1])
    else:
        results_text = r"$\alpha=$" + f"{format(β[1], '2.2f')}\n" + \
                       r"$\beta=$" + f"{format(β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return RegPlotConfig(xlabel="x",
                             ylabel="y",
                             plot_type=PlotType.LINEAR,
                             results_text=results_text,
                             legend_labels=["Data", r"$y=\beta + \alpha x$"],
                             y_fit=β[0]+x*β[1])

###############################################################################################
# Compare the result of a linear regression with teh acutal data (Uses RegPlotType config)
def reg(y, x, results, **kwargs):
    title = kwargs["title"] if "title" in kwargs else None
    plot_type = kwargs["plot_type"]  if "plot_type"  in kwargs else RegressionPlotType.LINEAR

    β = results.params

    if β[1] < 0:
        x_text = 0.1
        y_text = 0.1
        lengend_location = (0.6, 0.65, 0.3, 0.3)
    else:
        x_text = 0.8
        y_text = 0.1
        lengend_location = (0.05, 0.65, 0.3, 0.3)

    plot_config = create_reg_plot_type(plot_type, results, x)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.loglog(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.semilogx(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])

    axis.legend(loc='best', bbox_to_anchor=lengend_location)
