import numpy
from matplotlib import pyplot
from lib import config
from lib import fbm
from enum import Enum

# Specify PlotConfig for regression plot
class RegressionPlotType(Enum):
    LINEAR = 1          # Default
    FBM_AGG_VAR = 2     # FBM variance aggregation
    FBM_PSPEC = 3       # FBM Power Spectrum

# Supported plot types supported for all plots
class PlotType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

# Specify plot config for plots
class PlotDataType(Enum):
    TIME_SERIES = 1     # Time Series
    PSPEC = 2           # Power spectrum
    MEAN = 3            # Compare model mean with data
    STD = 4             # Compare model standard deviation with data
    AUTO_COR = 5        # Compare model autocorrelation
    ENSEMBLE = 6        # Data ensemble

# Plot a single curve as a function of the dependent variable
def curve(x, y, title, data_type=PlotDataType.TIME_SERIES):
    plot_config = _create_plot_data_type(data_type)

    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_title(title)
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    _plot(lot_config.plot_type)

# Plot multiple curves using the same axes
def comparison(x, y, labels, title, lengend_location="upper left", data_type=PlotDataType.TIME_SERIES):
    plot_config = _create_plot_data_type(data_type)
    nplot = len(y)
    ncol = len(labels)/6

    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)
    axis.set_title(title)

    for i in range(nplot):
        _plot(lot_config.plot_type)

    axis.legend(ncol=ncol, loc=lengend_location)

# Compare data to the value of a function
def fcompare(y, x, title, params=[], npts=10, lengend_location="upper left", data_type=PlotDataType.TIME_SERIES):
    plot_config = _create_plot_data_type(data_type, params)
    step = int(len(x)/npts)

    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)
    axis.set_title(title)

    if plot_config.plot_type == PlotType.LOG:
        _logStyle(axis, x)
        axis.loglog(x, y, label=plot_config.legend_labels[0])
        axis.loglog(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type == PlotType.XLOG:
        _logXStyle(axis, ps)
        axis.semilogx(x, y, label=plot_config.legend_labels[0])
        axis.semilogx(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type == PlotType.YLOG:
        _logYStyle(axis, ps)
        axis.semilogy(x, y, label=plot_config.legend_labels[0])
        axis.semilogy(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=plot_config.legend_labels[0])
        axis.plot(x[::step], plot_config.f(x[::step]), label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc=lengend_location)

# Plot a single curve in a stack of plots that use the same x-axis
def stack(y, ylim, labels, title, x=None, data_type=PlotDataType.TIME_SERIES):
    plot_config = _create_plot_data_type(data_type)

    nplot = len(y)
    if x is None:
        nx = len(y[0])
        x = numpy.tile(numpy.linspace(0, nx-1, nx), (nplot,1))

    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(15, 12))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(plot_config.xlabel)

    for i in range(nplot):
        nsample = len(y[i])
        axis[i].set_ylabel(plot_config.ylabel)
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, x[i][-1]])
        text = axis[i].text(x[i][int(0.9*nsample)], 0.65*ylim[-1], labels[i], fontsize=18)
        text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
        _plot(lot_config.plot_type, logStyle=False)

# Compare the cumulative value of a variable as a function of time with its target value
def cumulative(accum, target, title, label, lengend_location="upper left"):
    range = max(accum) - min(accum)
    nsample = len(accum)
    time = numpy.linspace(1.0, nsample, nsample)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_ylim([min(accum)-0.25*range, max(accum)+0.25*range])
    axis.set_xlabel("Time")
    axis.set_ylabel(label)
    axis.set_title(title)
    axis.set_xlim([1.0, nsample])
    axis.semilogx(time, accum, label=f"Cumulative "+label)
    axis.semilogx(time, numpy.full((len(time)), target), label="Target "+label)
    axis.legend(loc=lengend_location)

# Plot the autocorrelation function and the partial autocorrelation function of a random process
def acf_pacf(title, acf, pacf, max_lag):
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag])
    axis.set_ylim([-1.1, 1.1])
    axis.plot(range(max_lag+1), acf, label="ACF")
    axis.plot(range(1, max_lag+1), pacf, label="PACF")
    axis.legend(fontsize=16)

# Compare the result of a linear regression with teh acutal data
def regression(y, x, results, title, type=RegressionPlotType.LINEAR):
    β = results.params

    if β[1] < 0:
        x_text = 0.1
        y_text = 0.1
        legend_loc = 'upper right'
    else:
        x_text = 0.8
        y_text = 0.1
        legend_loc = 'upper left'

    plot_config = _create_regression_plot_type(type, results, x)

    figure, axis = pyplot.subplots(figsize=(15, 12))

    axis.set_title(title)
    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type == PlotType.LOG:
        _logStyle(axis, x)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=legend_labels[0])
        axis.loglog(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_type == PlotType.XLOG:
        _logXStyle(axis, ps)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=legend_labels[0])
        axis.semilogx(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_type == PlotType.YLOG:
        _logYStyle(axis, ps)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])

    axis.legend(loc=legend_loc)

# generate points evenly spaced on a logarithmic axis
def logspace(npts, max, min=10.0):
    return numpy.logspace(numpy.log10(min), numpy.log10(max/min), npts)

### -- Private ---
# Config used in plots
class _PlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, results_text=None, legend_labels=None, y_fit=None, f=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.results_text = results_text
        self.legend_labels = legend_labels
        self.y_fit = y_fit
        self.f = f

# Regression plot configuartion
def _create_regression_plot_type(type, results, x):
    β = results.params
    σ = results.bse[1]/2
    r2 = results.rsquared

    if type == RegressionPlotType.FBM_AGG_VAR:
        h = float(1.0 + β[1]/2.0)
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{\sigma}^2=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return _PlotConfig(xlabel=r"$\omega$",
                           ylabel=r"$Var(X^{m})$",
                           plot_type=PlotType.LOG,
                           results_text=results_text,
                           legend_labels=["Data", r"$Var(X^{m})=\sigma^2 m^{2H-2}$"],
                           y_fit=10**β[0]*x**β[1])
    elif type == RegressionPlotType.FBM_PSPEC:
        h = float(1.0 - β[1])/2.0
        results_text = r"$\hat{Η}=$" + f"{format(h, '2.2f')}\n" + \
                       r"$\hat{C}=$" + f"{format(10**β[0], '2.2f')}\n" + \
                       r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.2f')}\n" + \
                       r"$R^2=$" + f"{format(r2, '2.2f')}"
        return _PlotConfig(xlabel=r"$m$",
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
        return _PlotConfig(xlabel="x",
                           ylabel="y",
                           plot_type=PlotType.LINEAR,
                           results_text=results_text,
                           legend_labels=["Data", r"$y=\beta + \alpha x$"],
                           y_fit=β[0]+x*β[1])

## plot data type
def _create_plot_data_type(data_type, params=[]):
    if data_type == PlotDataType.TIME_SERIES:
        return _PlotConfig(xlabel=r"$t$",
                           ylabel=r"$X_t$",
                           plot_type=PlotType.LINEAR)
    elif data_type == PlotDataType.PSPEC:
        plot_type = PlotType.LOG
        return _PlotConfig(xlabel=r"$\omega$",
                           ylabel=r"$\rho_\omega$",
                           plot_type=PlotType.LOG)
    elif data_type == PlotDataType.MEAN:
        f = lambda t : numpy.full(len(t), 0.0)
        return _PlotConfig(xlabel=r"$t$",
                         ylabel=r"$\mu_t$",
                         plot_type=PlotType.LINEAR,
                         legend_labels=["Ensemble Average", r"$\mu=0$"],
                         f=f)
    elif data_type == PlotDataType.STD:
        H = params[0]
        f = lambda t : t**H
        return _PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\sigma_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$t^H$"],
                           f=f)
    elif data_type == PlotDataType.AUTO_COR:
        H = params[0]
        f= lambda t : fbm.autocorrelation(H, t)
        return _PlotConfig(xlabel=r"$t$",
                           ylabel=r"$\rho_t$",
                           plot_type=PlotType.LINEAR,
                           legend_labels=["Ensemble Average", r"$\frac{1}{2}[(t-1)^{2H} + (t+1)^{2H} - 2t^{2H})]$"],
                           f=f)
    elif data_type == PlotDataType.ENSEMBLE:
        return _PlotConfig(xlabel=r"$t$", ylabel=r"$X_t$", plot_type=PlotType.LINEAR)

    else:
        plot_type = PlotType.LINEAR
        xlabel = "y"
        ylabel = "x"
        return _PlotConfig(xlabel="x", ylabel="y", plot_type=PlotType.LINEAR)

# plot using PlotType
def _plot(plot_type, logStyle=True, lw=1):
    if plot_type == PlotType.LOG:
        if logStyle:
            _logXStyle(axis, ps)
        axis.loglog(x, y[i], lw=lw, label=labels[i])
    elif plot_type == PlotType.XLOG:
        if logStyle:
            _logXStyle(axis, ps)
        axis.semilogx(x, y[i], lw=lw, label=labels[i])
    elif plot_type == PlotType.YLOG:
        if logStyle:
            _logYStyle(axis, ps)
        axis.semilogy(x, y[i], lw=lw, label=labels[i])
    else:
        axis.plot(x, y[i], lw=lw, label=labels[i])

# Add axes for log plots for 1 to 3 decades
def _logStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def _logXStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.set_xlim([min(x), max(x)])

def _logYStyle(axis, x):
    if numpy.log10(max(x)/min(x)) < 4:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['left'].set_color("#b0b0b0")
