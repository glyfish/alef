import numpy
from enum import Enum

from lib import fbm
from lib import arima
from lib import stats
from lib.dist import (DistributionType, DistributionFuncType, HypothesisType, distribution_function)

# Specify PlotConfig for regression plot
class RegPlotType(Enum):
    LINEAR = 1          # Default
    FBM_AGG_VAR = 2     # FBM variance aggregation
    FBM_PSPEC = 3       # FBM Power Spectrum

## Specify PlotConfig for distributions plot
class DistPlotType(Enum):
    VR_TEST = 1         # Variance ration test used to detect brownian motion

class RegPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, results_text=None, y_fit=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.results_text = results_text
        self.y_fit = y_fit

class DistPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, dist_type=None, dist_params=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.dist_type = dist_type
        self.dist_params = dist_params

# Regression plot configuartion
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

# Create distribution plot type
def create_dist_plot_type(plot_type):
    if plot_type.value == DistPlotType.VR_TEST.value:
        return DistPlotConfig(xlabel=r"$Z(s)$",
                              ylabel=r"Normal(CDF)",
                              plot_type=PlotType.LINEAR,
                              dist_type=DistributionType.NORMAL,
                              dist_params = [1.0, 0.0])
    else:
        raise Exception(f"Distribution plot type is invalid: {plot_type}")
