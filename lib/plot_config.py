import numpy
from enum import Enum
from lib import fbm
from lib import arima

# Specify PlotConfig for regression plot
class RegressionPlotType(Enum):
    LINEAR = 1          # Default
    FBM_AGG_VAR = 2     # FBM variance aggregation
    FBM_PSPEC = 3       # FBM Power Spectrum

# Supported plot types supported
class PlotType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

# Specify plot config which specifies configuarble plot parameters
class PlotDataType(Enum):
    GENERIC = 1         # Unknown data type
    TIME_SERIES = 2     # Time Series
    PSPEC = 3           # Power Spectrum
    ACF = 4             # Autocorrelation function

# Specify plot config which specifies configuarble plot parameters
class PlotFuncType(Enum):
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

class HypothesisTestType(Enum):
    VR = 1              # Variance ration test

# Config used in plots
class PlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, results_text=None, legend_labels=None, y_fit=None, f=None, dist_type=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.results_text = results_text
        self.legend_labels = legend_labels
        self.y_fit = y_fit
        self.f = f
        self.dist_type = dist_type

# Regression plot configuartion
def create_regression_plot_type(plot_type, results, x):
    β = results.params
    σ = results.bse[1]/2
    r2 = results.rsquared

    if plot_type.value == RegressionPlotType.FBM_AGG_VAR.value:
        h = float(1.0 + β[1]/2.0)
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{\sigma}^2=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return PlotConfig(xlabel=r"$\omega$",
                           ylabel=r"$Var(X^{m})$",
                           plot_type=PlotType.LOG,
                           results_text=results_text,
                           legend_labels=["Data", r"$Var(X^{m})=\sigma^2 m^{2H-2}$"],
                           y_fit=10**β[0]*x**β[1])
    elif plot_type.value == RegressionPlotType.FBM_PSPEC.value:
        h = float(1.0 - β[1])/2.0
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{C}=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return PlotConfig(xlabel=r"$m$",
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
        return PlotConfig(xlabel="x",
                           ylabel="y",
                           plot_type=PlotType.LINEAR,
                           results_text=results_text,
                           legend_labels=["Data", r"$y=\beta + \alpha x$"],
                           y_fit=β[0]+x*β[1])

## plot data type
def create_plot_data_type(plot_type):
    if plot_type.value == PlotDataType.TIME_SERIES.value:
        return PlotConfig(xlabel=r"$t$", ylabel=r"$X_t$", plot_type=PlotType.LINEAR)
    elif plot_type.value == PlotDataType.PSPEC.value:
        return PlotConfig(xlabel=r"$\omega$", ylabel=r"$\rho_\omega$", plot_type=PlotType.LOG)
    elif plot_type.value == PlotDataType.ACF.value:
        return PlotConfig(xlabel=r"$\tau$", ylabel=r"$\rho_\tau$", plot_type=PlotType.LINEAR)
    elif plot_type.value == PlotDataType.VR.value:
        return PlotConfig(xlabel=r"$s$", ylabel=r"$VR(s)$", plot_type=PlotType.LOG)
    else:
        return PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

## plot function type
def create_plot_func_type(plot_type, params):
    if plot_type.value == PlotFuncType.FBM_MEAN.value:
        f = lambda t : numpy.full(len(t), 0.0)
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\mu_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\mu=0$"],
                          f=f)
    elif plot_type.value == PlotFuncType.FBM_STD.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ*numpy.sqrt(fbm.var(H, t))
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\sigma_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\sigma t^H$"],
                          f=f)
    elif plot_type.value == PlotFuncType.FBM_ACF.value:
        H = params[0]
        f = lambda t : fbm.acf(H, t)
        return PlotConfig(xlabel=r"$\tau$",
                          ylabel=r"$\rho_\tau$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\frac{1}{2}[(\tau-1)^{2H} + (\tau+1)^{2H} - 2\tau^{2H})]$"],
                          f=f)
    elif plot_type.value == PlotFuncType.BM_MEAN.value:
        μ = params[0]
        f = lambda t : numpy.full(len(t), μ)
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\mu_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", f"μ={μ}"],
                          f=f)
    elif plot_type.value == PlotFuncType.BM_DRIFT_MEAN.value:
        μ = params[0]
        f = lambda t : μ*t
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\mu_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$μ_t=μt$"],
                          f=f)
    elif plot_type.value == PlotFuncType.BM_STD.value:
        σ = params[0]
        f = lambda t : σ*numpy.sqrt(t)
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\sigma_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Average", r"$\sigma_t = \sigma \sqrt{t}$"],
                           f=f)
    elif plot_type.value == PlotFuncType.GBM_MEAN.value:
        S0 = params[0]
        μ = params[1]
        f = lambda t : S0*numpy.exp(μ*t)
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\mu_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\mu_t = S_0 e^{\mu t}$"],
                          f=f)
    elif plot_type.value == PlotFuncType.GBM_STD.value:
        S0 = params[0]
        μ = params[1]
        σ = params[2]
        f = lambda t : numpy.sqrt(S0**2*numpy.exp(2*μ*t)*(numpy.exp(t*σ**2)-1))
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$\sigma_t$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\sigma_t=S_0 e^{\mu t}\sqrt{e^{\sigma^2 t} - 1}$"],
                          f=f)
    elif plot_type.value == PlotFuncType.AR1_ACF.value:
        φ = params[0]
        f = lambda t : φ**t
        return PlotConfig(xlabel=r"$\tau$",
                          ylabel=r"$\rho_\tau$",
                          plot_type=PlotType.LINEAR,
                          legend_labels=["Average", r"$\phi^\tau$"],
                          f=f)
    elif plot_type.value == PlotFuncType.MAQ_ACF.value:
        θ = params[0]
        σ = params[1]
        f = lambda t : arima.maq_acf(θ, σ, len(t))
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\rho_\tau$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Average", r"$\rho_\tau = \left( \sum_{i=i}^{q-n} \vartheta_i \vartheta_{i+n} + \vartheta_n \right)$"],
                           f=f)
    elif plot_type.value == PlotFuncType.LAGG_VAR.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ**2*fbm.var(H, t)
        return PlotConfig(xlabel=r"$s$",
                          ylabel=r"$\sigma^2(s)$",
                          plot_type=PlotType.LOG,
                          legend_labels=[r"$\sigma^2(s)$", r"$\sigma^2 t^{2H}$"],
                          f=f)
    elif plot_type.value == PlotFuncType.VR.value:
        H = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : σ**2*t**(2*H - 1.0)
        return PlotConfig(xlabel=r"$s$",
                          ylabel=r"VR(s)",
                          plot_type=PlotType.LOG,
                          legend_labels=[r"VR(s)", r"$\sigma^2 t^{2H-1}$"],
                          f=f)
    else:
        f = lambda t : t
        return PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR, legend_labels=["Data", "f(x)"], f=f)

## Hypothesis Test Type
def create_hypothesis_test_plot_type(plot_type):
    if plot_type.value == HypothesisTestType.VR.value:
        return PlotConfig(xlabel=r"$t$",
                          ylabel=r"$X_t$",
                          plot_type=PlotType.LINEAR)
    else:
        raise Exception(f"Hypothesis test type is invalid: {plot_type}")

# Add axes for log plots for 1 to 3 decades
def logStyle(axis, x, y):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])
        if numpy.log10(max(y)/min(y)) < 1:
            axis.set_ylim([min(y)/5.0, 5.0*max(y)])


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
        if numpy.log10(max(y)/min(y)) < 1:
            axis.set_ylim([min(y)/1.5, 1.5*max(y)])
