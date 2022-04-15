import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import fbm
from lib.models import arima
from lib import stats

from lib.plots.config import (PlotType, logStyle, logXStyle, logYStyle)

##################################################################################################################
## Specify PlotConfig for fcompare plot
class FuncPlotType(Enum):
    LINEAR = 1                # Linear Model
    FBM_MEAN = 2              # FBM model mean with data
    FBM_STD = 3               # FBM model standard deviation with data
    FBM_ACF = 4               # FBM model autocorrelation with data
    BM_MEAN = 5               # BM model mean with data
    BM_DRIFT_MEAN = 6         # BM model mean with data
    BM_STD = 7                # BM model standard deviation with data
    GBM_MEAN = 8              # GBM model mean with data
    GBM_STD = 9               # GBM model standard deviation with data
    AR1_ACF = 10              # AR1 model ACF autocorrelation function with data
    MAQ_ACF = 11              # MA(q) model ACF autocorrelation function with data
    LAGG_VAR = 12             # Lagged variance computed from a time
    VR = 13                   # Vraiance ratio use in test for brownian motion

##################################################################################################################
## Function compare plot config
class FuncPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, f=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.f = f

##################################################################################################################
## plot function type
def create_func_plot_type(plot_type, params):
    if plot_type.value == FuncPlotType.FBM_MEAN.value:
        f = lambda t : numpy.full(len(t), 0.0)
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\mu=0$"],
                              f=f)
    elif plot_type.value == FuncPlotType.FBM_STD.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ*numpy.sqrt(fbm.var(H, t))
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma t^H$"],
                              f=f)
    elif plot_type.value == FuncPlotType.FBM_ACF.value:
        H = params[0]
        f = lambda t : fbm.acf(H, t)
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\frac{1}{2}[(\tau-1)^{2H} + (\tau+1)^{2H} - 2\tau^{2H})]$"],
                              f=f)
    elif plot_type.value == FuncPlotType.BM_MEAN.value:
        μ = params[0]
        f = lambda t : numpy.full(len(t), μ)
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", f"μ={μ}"],
                              f=f)
    elif plot_type.value == FuncPlotType.BM_DRIFT_MEAN.value:
        μ = params[0]
        f = lambda t : μ*t
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$μ_t=μt$"],
                              f=f)
    elif plot_type.value == FuncPlotType.BM_STD.value:
        σ = params[0]
        f = lambda t : σ*numpy.sqrt(t)
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma_t = \sigma \sqrt{t}$"],
                              f=f)
    elif plot_type.value == FuncPlotType.GBM_MEAN.value:
        S0 = params[0]
        μ = params[1]
        f = lambda t : S0*numpy.exp(μ*t)
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\mu_t = S_0 e^{\mu t}$"],
                              f=f)
    elif plot_type.value == FuncPlotType.GBM_STD.value:
        S0 = params[0]
        μ = params[1]
        σ = params[2]
        f = lambda t : numpy.sqrt(S0**2*numpy.exp(2*μ*t)*(numpy.exp(t*σ**2)-1))
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma_t=S_0 e^{\mu t}\sqrt{e^{\sigma^2 t} - 1}$"],
                              f=f)
    elif plot_type.value == FuncPlotType.AR1_ACF.value:
        φ = params[0]
        f = lambda t : φ**t
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\phi^\tau$"],
                              f=f)
    elif plot_type.value == FuncPlotType.MAQ_ACF.value:
        θ = params[0]
        σ = params[1]
        f = lambda t : arima.maq_acf(θ, σ, len(t))
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\rho_\tau = \left( \sum_{i=i}^{q-n} \vartheta_i \vartheta_{i+n} + \vartheta_n \right)$"],
                              f=f)
    elif plot_type.value == FuncPlotType.LAGG_VAR.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ**2*fbm.var(H, t)
        return FuncPlotConfig(xlabel=r"$s$",
                              ylabel=r"$\sigma^2(s)$",
                              plot_type=PlotType.LOG,
                              legend_labels=[r"$\sigma^2(s)$", r"$\sigma^2 t^{2H}$"],
                              f=f)
    elif plot_type.value == FuncPlotType.VR.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ**2*t**(2*H - 1.0)
        return FuncPlotConfig(xlabel=r"$s$",
                              ylabel=r"VR(s)",
                              plot_type=PlotType.LOG,
                              legend_labels=[r"VR(s)", r"$\sigma^2 t^{2H-1}$"],
                              f=f)
    else:
        f = lambda t : t
        return FuncPlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR, legend_labels=["Data", "f(x)"], f=f)

###############################################################################################
# Compare data to the value of a function (Uses PlotFuncType config)
def fcompare(y, x=None, **kwargs):
    title     = kwargs["title"]     if "title"     in kwargs else None
    plot_type = kwargs["plot_type"] if "plot_type" in kwargs else PlotFuncType.LINEAR
    lw        = kwargs["lw"]        if "lw"        in kwargs else 2
    labels    = kwargs["labels"]    if "labels"    in kwargs else None
    npts      = kwargs["npts"]      if "npts"      in kwargs else 10
    params    = kwargs["params"]    if "params"    in kwargs else []

    plot_config = create_func_plot_type(plot_type, params)

    if x is None:
        nx = len(y)
        if plot_config.plot_type.value == PlotType.XLOG.value or plot_config.plot_type.value == PlotType.LOG.value:
            x = logspace(nx, float(nx-1), 1.0)
        else:
            x = numpy.linspace(0.0, float(nx-1), nx)
    step = int(len(x)/npts)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    if title is not None:
        axis.set_title(title)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.loglog(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogx(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogy(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.plot(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
