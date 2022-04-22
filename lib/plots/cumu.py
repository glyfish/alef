import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import arima
from lib import stats

from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.schema import (DataType, create_data_type)

##################################################################################################################
## Specify PlotConfig for cumulative plot
class CumuPlotType(Enum):
    AR1_MEAN = 1             # Accumulation mean for AR(1)
    AR1_STD = 2              # Accumulation standard deviation for AR(1)
    MAQ_MEAN = 3             # Accumulation mean for MA(q)
    MAQ_STD = 4              # Accumulation standard deviation for MA(q)
    AR1_OFFSET_MEAN = 5      # AR1 with Offset model mean
    AR1_OFFSET_STD = 6       # AR1 with Offset model standard deviation

##################################################################################################################
## Plot Configuration
class CumuPlotConfig:
    def __init__(self, xlabel, ylabel, data_type, plot_type=PlotType.LINEAR, legend_labels=None, target=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data_type = data_type
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.target = target

##################################################################################################################
# Create Cumlative plot type
def create_cumu_plot_type(plot_type):
    if plot_type.value == CumuPlotType.AR1_MEAN.value:
        f = lambda t : stats.cummean(t)
        data_type=create_data_type(DataType.CUM_MEAN)
        return CumuPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\mu_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target = 0.0)
    if plot_type.value == CumuPlotType.AR1_STD.value:
        φ = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : stats.cumsigma(t)
        data_type=create_data_type(DataType.CUM_STD)
        return CumuPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\sigma_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target=arima.ar1_sigma(φ, σ))
    if plot_type.value == CumuPlotType.MAQ_MEAN.value:
        data_type=create_data_type(DataType.CUM_MEAN)
        return CumuPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\mu_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target = 0.0)
    if plot_type.value == CumuPlotType.MAQ_STD.value:
        θ = params[0]
        if len(params) > 1:
            σ = params[1]
        else:
            σ = 1.0
        f = lambda t : stats.cumsigma(t)
        data_type=create_data_type(DataType.CUM_STD)
        return CumuPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\sigma_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target=arima.maq_sigma(θ, σ))
    elif plot_type.value == CumuPlotType.AR1_OFFSET_MEAN.value:
        φ = params[0]
        μ = params[1]
        f = lambda t : stats.cummean(t)
        data_type=create_data_type(DataType.CUM_MEAN)
        return CumPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\mu_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target=arima.ar1_offset_mean(φ, μ))
    elif plot_type.value == CumuPlotType.AR1_OFFSET_STD.value:
        φ = params[0]
        σ = params[1]
        f = lambda t : stats.cumsigma(t)
        data_type=create_data_type(DataType.CUM_STD)
        return CumuPlotConfig(xlabel=r"$t$",
                             ylabel=r"$\sigma_t$",
                             data_type=data_type,
                             plot_type=PlotType.XLOG,
                             legend_labels=["Accumulation", "Target"],
                             target=arima.ar1_offset_sigma(φ, σ))
    else:
        raise Exception(f"Cumulative plot type is invalid: {plot_type}")

##################################################################################################################
## Compare the cumulative value of a variable as a function of time with its target value (Uses CumPlotType config)
def cumulative(df, plot_type, **kwargs):
    title  = kwargs["title"]  if "title"  in kwargs else None
    lw     = kwargs["lw"]     if "lw"     in kwargs else 2
    params = kwargs["params"] if "params" in kwargs else []

    plot_config = create_cumu_plot_type(plot_type, params)
    accum = plot_config.f(samples)
    time, accum = plot_config.data_type.get_data(df)

    range = max(1.0, (max(accum) - min(accum)))
    max_time = len(time)
    min_time = max(1.0, min(time))

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is None:
        axis[0].set_title(title)

    axis.set_ylim([min(accum)-range, max(accum)+range])
    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)
    axis.set_title(title)
    axis.set_xlim([min_time, max_time])
    axis.semilogx(time, accum, label=plot_config.legend_labels[0], lw=lw)
    axis.semilogx(time, numpy.full((len(time)), plot_config.target), label=plot_config.legend_labels[1], lw=lw)
    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
