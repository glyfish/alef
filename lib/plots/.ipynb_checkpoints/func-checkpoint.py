import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import fbm
from lib.models import arima
from lib import stats

from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)

##################################################################################################################
## Specify PlotConfig for fcompare plot
class FuncPlotType(Enum):
    LINEAR = 1                # Linear Model
    FBM_MEAN = 2              # FBM model mean with data
    FBM_SD = 3                # FBM model standard deviation with data
    FBM_ACF = 4               # FBM model autocorrelation with data
    BM_MEAN = 5               # BM model mean with data
    BM_DRIFT_MEAN = 6         # BM model mean with data
    BM_SD = 7                 # BM model standard deviation with data
    GBM_MEAN = 8              # GBM model mean with data
    AR1_ACF = 10              # AR1 model ACF autocorrelation function with data
    MAQ_ACF = 11              # MA(q) model ACF autocorrelation function with data
    LAGG_VAR = 12             # Lagged variance computed from a time
    VR = 13                   # Vraiance ratio use in test for brownian motion

##################################################################################################################
## Function compare plot config
class FuncPlotConfig:
    def __init__(self, xlabel, ylabel, data_type, func_type, plot_type=PlotType.LINEAR, legend_labels=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data_schema = create_schema(data_type)
        self.func_schema = create_schema(func_type)
        self.plot_type = plot_type
        self.legend_labels = legend_labels

##################################################################################################################
## plot function type
def create_func_plot_type(plot_type):
    if plot_type.value == FuncPlotType.FBM_MEAN.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              data_type=DataType.MEAN,
                              func_type=DataType.FBM_MEAN,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\mu=0$"])
    elif plot_type.value == FuncPlotType.FBM_SD.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              data_type=DataType.SD,
                              func_type=DataType.FBM_SD,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma t^H$"])
    elif plot_type.value == FuncPlotType.FBM_ACF.value:
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              data_type=DataType.ACF,
                              func_type=DataType.FBM_ACF,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\frac{1}{2}[(\tau-1)^{2H} + (\tau+1)^{2H} - 2\tau^{2H})]$"])
    elif plot_type.value == FuncPlotType.BM_MEAN.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              data_type=DataType.MEAN,
                              func_type=DataType.BM_MEAN,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", f"μ={μ}"])
    elif plot_type.value == FuncPlotType.BM_DRIFT_MEAN.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              data_type=DataType.MEAN,
                              func_type=DataType.BM_DRIFT_MEAN,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$μ_t=μt$"])
    elif plot_type.value == FuncPlotType.BM_SD.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              data_type=DataType.SD,
                              func_type=DataType.BM_SD,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma_t = \sigma \sqrt{t}$"])
    elif plot_type.value == FuncPlotType.GBM_MEAN.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\mu_t$",
                              data_type=DataType.MEAN,
                              func_type=DataType.GBM_MEAN,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\mu_t = S_0 e^{\mu t}$"])
    elif plot_type.value == FuncPlotType.GBM_SD.value:
        return FuncPlotConfig(xlabel=r"$t$",
                              ylabel=r"$\sigma_t$",
                              data_type=DataType.SD,
                              func_type=DataType.GBM_SD,
                              plot_type=PlotType.LINEAR,
                              legend_labels=["Average", r"$\sigma_t=S_0 e^{\mu t}\sqrt{e^{\sigma^2 t} - 1}$"])
    elif plot_type.value == FuncPlotType.AR1_ACF.value:
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              data_type=DataType.ACF,
                              func_type=DataType.AR1_ACF,
                              plot_type=DataType.LINEAR,
                              legend_labels=["Ensemble Average", r"$\varphi^\tau$"])
    elif plot_type.value == FuncPlotType.MAQ_ACF.value:
        return FuncPlotConfig(xlabel=r"$\tau$",
                              ylabel=r"$\rho_\tau$",
                              data_type=DataType.ACF,
                              func_type=DataType.MAQ_ACF,
                              plot_type=DataType.LINEAR,
                              legend_labels=["Ensemble Average", r"$\rho_\tau = \left( \sum_{i=i}^{q-n} \vartheta_i \vartheta_{i+n} + \vartheta_n \right)$"])
    elif plot_type.value == FuncPlotType.AGG_VAR.value:
        return FuncPlotConfig(xlabel=r"$s$",
                              ylabel=r"$\sigma^2(s)$",
                              data_type=DataType.TIME_SERIES,
                              func_type=DataType.AGG_VAR,
                              plot_type=PlotType.LOG,
                              legend_labels=[r"$\sigma^2(s)$", r"$\sigma^2 t^{2H}$"])
    elif plot_type.value == FuncPlotType.VR.value:
        return FuncPlotConfig(xlabel=r"$s$",
                              ylabel=r"VR(s)",
                              data_type=DataType.TIME_SERIES,
                              func_type=DataType.VR,
                              plot_type=PlotType.LOG,
                              legend_labels=[r"VR(s)", r"$\sigma^2 t^{2H-1}$"])
    else:
        raise Exception(f"FuncPlotType type is invalid: {data_type}")

###############################################################################################
# Compare data to the value of a function (Uses PlotFuncType config)
def fcompare(df, plot_type, **kwargs):
    title        = kwargs["title"]        if "title"        in kwargs else None
    lw           = kwargs["lw"]           if "lw"           in kwargs else 2
    labels       = kwargs["labels"]       if "labels"       in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    plot_config = create_func_plot_type(plot_type)
    x, y = plot_config.data_schema.get_data(df)
    fx, fy = plot_config.func_schema.get_data(df)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(plot_config.xlabel)
    axis.set_ylabel(plot_config.ylabel)

    if title is not None:
        axis.set_title(title)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.loglog(fx, fy, label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogx(fx, fy, label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogy(fx, fy, label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=plot_config.legend_labels[0], lw=lw)
        axis.plot(fx, fy, label=plot_config.legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
