import numpy
from enum import Enum
from lib import fbm
from lib import arima
from lib import stats
from lib.dist import (DistributionType, DistributionFuncType, HypothesisType, distribution_function)

# Supported plot types supported
class PlotType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

# Specify PlotConfig for regression plot
class RegPlotType(Enum):
    LINEAR = 1          # Default
    FBM_AGG_VAR = 2     # FBM variance aggregation
    FBM_PSPEC = 3       # FBM Power Spectrum

# Specify PlotConfig for curve, comparison and stack plots
class DataPlotType(Enum):
    GENERIC = 1         # Unknown data type
    TIME_SERIES = 2     # Time Series
    PSPEC = 3           # Power Spectrum
    ACF = 4             # Autocorrelation function
    VR_STAT = 5         # FBM variance ratio test statistic

# Specify PlotConfig for fcompare plot
class FuncPlotType(Enum):
    LINEAR = 1          # Linear Model
    FBM_MEAN = 2        # FBM model mean with data
    FBM_STD = 3         # FBM model standard deviation with data
    FBM_ACF = 4         # FBM model autocorrelation with data
    BM_MEAN = 5         # BM model mean with data
    BM_DRIFT_MEAN = 6   # BM model mean with data
    BM_STD = 7          # BM model standard deviation with data
    GBM_MEAN = 8        # GBM model mean with data
    GBM_STD = 9         # GBM model standard deviation with data
    AR1_ACF = 10        # AR1 model ACF autocorrelation function with data
    MAQ_ACF = 11        # MA(q) model ACF autocorrelation function with data
    LAGG_VAR = 12       # Lagged variance computed from a time
    VR = 13             # Vraiance ratio use in test for brownian motion

## Specify PlotConfig for distributions plot
class DistPlotType(Enum):
    VR_TEST = 1         # Variance ration test used to detect brownian motion

# Specify PlotConfig for cumulative plot
class CumPlotType(Enum):
    AR1_MEAN = 1        # Accumulation mean for AR(1)
    AR1_STD = 2         # Accumulation standard deviation for AR(1)
    MAQ_MEAN = 3        # Accumulation mean for MA(q)
    MAQ_STD = 4         # Accumulation standard deviation for MA(q)

# Specify Config for historgram PlotType
class HistDistPlotType(Enum):
    PDF = 1             # Probability density function
    CDF = 2             # Cummulative density function

# Configurations used in plots
class PlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels

class RegPlotConfig(PlotConfig):
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, results_text=None, y_fit=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.results_text = results_text
        self.y_fit = y_fit

class FuncPlotConfig(PlotConfig):
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, f=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.f = f

class CumPlotConfig(FuncPlotConfig):
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, f=None, target=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.f = f
        self.target = target

class DistPlotConfig(PlotConfig):
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, dist_type=None, dist_params=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.dist_type = dist_type
        self.dist_params = dist_params

class HistDistPlotType:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, density=False, params=None, f=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.density = density
        self.params = params
        self.f = f

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

## plot data type
def create_data_plot_type(plot_type):
    if plot_type.value == DataPlotType.TIME_SERIES.value:
        return PlotConfig(xlabel=r"$t$", ylabel=r"$X_t$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.PSPEC.value:
        return PlotConfig(xlabel=r"$\omega$", ylabel=r"$\rho_\omega$", plot_type=PlotType.LOG)
    elif plot_type.value == DataPlotType.ACF.value:
        return PlotConfig(xlabel=r"$\tau$", ylabel=r"$\rho_\tau$", plot_type=PlotType.LINEAR)
    elif plot_type.value == DataPlotType.VR_STAT.value:
        return PlotConfig(xlabel=r"$s$", ylabel=r"$Z(s)$", plot_type=PlotType.LINEAR)
    else:
        return PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

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

# Create Cumlative plot type
def create_cum_plot_type(plot_type, params):
    if plot_type.value == CumPlotType.AR1_MEAN.value:
        f = lambda t : stats.cummean(t)
        return CumPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\mu_t$",
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             f=f,
                             target = 0.0)
    if plot_type.value == CumPlotType.AR1_STD.value:
        φ = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : stats.cumsigma(t)
        return CumPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\sigma_t$",
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             f=f,
                             target=arima.ar1_sigma(φ, σ))
    if plot_type.value == CumPlotType.MAQ_MEAN.value:
        f = lambda t : stats.cummean(t)
        return CumPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\mu_t$",
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             f=f,
                             target = 0.0)
    if plot_type.value == CumPlotType.MAQ_STD.value:
        θ = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : stats.cumsigma(t)
        return CumPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\sigma_t$",
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             f=f,
                             target=arima.maq_sigma(θ, σ))
    else:
        raise Exception(f"Cumulative plot type is invalid: {plot_type}")

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

## plot histogram type
def create_hist_dist_plot_type(plot_type, params=None):
    if plot_type.value == HistPlotType.PDF.value:
        plot_params = f"μ={params[1]}\nσ={params[0]}"
        return PlotConfig(xlabel=r"$x$", ylabel=r"$p(x)$", plot_type=PlotType.LINEAR, density=True, params=plot_params)
    if plot_type.value == HistPlotType.CDF.value:
        plot_params = f"μ={params[1]}\nσ={params[0]}"
        return PlotConfig(xlabel=r"$x$", ylabel=r"$P(x)$", plot_type=PlotType.LINEAR, density=True, params=plot_params)
    else:
        return PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

# Add axes for log plots for 1 to 3 decades
def logStyle(axis, x, y):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logXStyle(axis, x, y):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logYStyle(axis, x, y):
    if numpy.log10(max(y)/min(y)) < 4:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['left'].set_color("#b0b0b0")
