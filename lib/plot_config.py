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
    TIME_SERIES = 1     # Time Series
    PSPEC = 2           # Power spectrum
    FBM_MEAN = 3        # Compare FBM model mean with data
    FBM_STD = 4         # Compare FBM model standard deviation with data
    FBM_ACF = 5         # Compare FBM model autocorrelation with data
    ENSEMBLE = 6        # Data ensemble
    BM_MEAN = 7         # Compare BM model mean with data
    BM_DRIFT_MEAN = 8   # Compare BM model mean with data
    BM_STD = 9          # Compare BM model standard deviation with data
    GBM_MEAN = 10       # Compare GBM model mean with data
    GBM_STD = 11        # Compare GBM model standard deviation with data
    AR1_ACF = 12        # Compare AR1 model ACF autocorrelation function with data
    MAQ_ACF = 13        # Compare MA(q) model ACF autocorrelation function with data
    ACF_PACF = 14       # Compare ACF and PACF for an ARIMA process

# Config used in plots
class PlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, results_text=None, legend_labels=None, y_fit=None, f=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.results_text = results_text
        self.legend_labels = legend_labels
        self.y_fit = y_fit
        self.f = f

# Regression plot configuartion
def create_regression_plot_type(type, results, x):
    β = results.params
    σ = results.bse[1]/2
    r2 = results.rsquared

    if type.value == RegressionPlotType.FBM_AGG_VAR.value:
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
    elif type.value == RegressionPlotType.FBM_PSPEC.value:
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
def create_plot_data_type(data_type, params=[]):
    if data_type.value == PlotDataType.TIME_SERIES.value:
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$X_t$",
                           plot_type=PlotType.LINEAR)
    elif data_type.value == PlotDataType.PSPEC.value:
        plot_type = PlotType.LOG
        return PlotConfig(xlabel=r"$\omega$",
                           ylabel=r"$\rho_\omega$",
                           plot_type=PlotType.LOG)
    elif data_type.value == PlotDataType.FBM_MEAN.value:
        f = lambda t : numpy.full(len(t), 0.0)
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\mu_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\mu=0$"],
                           f=f)
    elif data_type.value == PlotDataType.FBM_STD.value:
        H = params[0]
        f = lambda t : t**H
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\sigma_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$t^H$"],
                           f=f)
    elif data_type.value == PlotDataType.FBM_ACF.value:
        H = params[0]
        f = lambda t : fbm.acf(H, t)
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\rho_\tau$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\frac{1}{2}[(\tau-1)^{2H} + (\tau+1)^{2H} - 2\tau^{2H})]$"],
                           f=f)
    elif data_type.value == PlotDataType.ENSEMBLE.value:
        return PlotConfig(xlabel=r"$t$", ylabel=r"$X_t$", plot_type=PlotType.LINEAR)
    elif data_type.value == PlotDataType.BM_MEAN.value:
        μ = params[0]
        f = lambda t : numpy.full(len(t), μ)
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\mu_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", f"μ={μ}"],
                           f=f)
    elif data_type.value == PlotDataType.BM_DRIFT_MEAN.value:
        μ = params[0]
        f = lambda t : μ*t
        return PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\mu_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$μ_t=μt$"],
                           f=f)
    elif data_type.value == PlotDataType.BM_STD.value:
        σ = params[0]
        f = lambda t : σ*numpy.sqrt(t)
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\sigma_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\sigma_t = \sigma \sqrt{t}$"],
                           f=f)
    elif data_type.value == PlotDataType.GBM_MEAN.value:
        S0 = params[0]
        μ = params[1]
        f = lambda t : S0*numpy.exp(μ*t)
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\mu_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\mu_t = S_0 e^{\mu t}$"],
                           f=f)
    elif data_type.value == PlotDataType.GBM_STD.value:
        S0 = params[0]
        μ = params[1]
        σ = params[2]
        f = lambda t : numpy.sqrt(S0**2*numpy.exp(2*μ*t)*(numpy.exp(t*σ**2)-1))
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\sigma_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\sigma_t=S_0 e^{\mu t}\sqrt{e^{\sigma^2 t} - 1}$"],
                           f=f)
    elif data_type.value == PlotDataType.AR1_ACF.value:
        φ = params[0]
        f = lambda t : φ**t
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\rho_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\phi^\tau$"],
                           f=f)
    elif data_type.value == PlotDataType.MAQ_ACF.value:
        θ = params[0]
        σ = params[1]
        f = lambda t : arima.maq_acf(θ, σ, len(t))
        return PlotConfig(xlabel=r"$\tau$",
                           ylabel=r"$\rho_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\rho_\tau = \left( \sum_{i=i}^{q-n} \vartheta_i \vartheta_{i+n} + \vartheta_n \right)$"],
                           f=f)
    else:
        plot_type = PlotType.LINEAR
        xlabel = "y"
        ylabel = "x"
        f = lambda t : t
        return PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR, legend_labels=["Data", "f(x)"], f=f)

# Add axes for log plots for 1 to 3 decades
def logStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def logXStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def logYStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['left'].set_color("#b0b0b0")
