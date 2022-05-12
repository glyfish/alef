import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import arima
from lib import stats

from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.schema import (DataType, create_schema)
from lib.data.func import (create_data_func)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

##################################################################################################################
## Specify PlotConfig for cumulative plot
class CumuPlotType(Enum):
    AR1_MEAN = 1             # Accumulation mean for AR(1)
    AR1_SD = 2               # Accumulation standard deviation for AR(1)
    MAQ_MEAN = 3             # Accumulation mean for MA(q)
    MAQ_SD = 4               # Accumulation standard deviation for MA(q)
    AR1_OFFSET_MEAN = 5      # AR1 with Offset model mean
    AR1_OFFSET_SD = 6        # AR1 with Offset model standard deviation

##################################################################################################################
## Plot Configuration
class CumuPlotConfig:
    def __init__(self, df, data_type, target_type, plot_type=PlotType.LINEAR, xlabel=None, ylabel=None, legend_labels=None):
        self.schema = create_schema(data_type)
        self.target_schema = create_schema(target_type)
        self.plot_type = plot_type
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._legend_labels = legend_labels
        self._meta_data = MetaData.get(df, self.schema)
        self._target_meta_data = MetaData.get(df, self.target_schema)

    def __repr__(self):
        return f"CumuPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"schema=({self.schema}), target_schema=({self.target_schema}), plot_type=({self.plot_type}), xlabel=({self.xlabel()}), ylabel=({self.ylabel()}), legend_labels=({self.legend_labels()}), title=({self.title()})"

    def xlabel(self):
        if self._xlabel is None:
            return self._meta_data.xlabel
        else:
            return self._xlabel

    def ylabel(self):
        if self._xlabel is None:
            return self._meta_data.xlabel
        else:
            return self._xlabel

    def legend_labels(self):
        if self._legend_labels is None:
            return [self._meta_data.ylabel, self._target_meta_data.ylabel]
        else:
            return self._legend_labels

    def title(self):
        return f"{self._meta_data.desc} {self._target_meta_data.desc} {self._meta_data.params_str()}"

##################################################################################################################
# Create Cumlative plot type
def create_cumu_plot_config(plot_type, df):
    if plot_type.value == CumuPlotType.AR1_MEAN.value:
        return CumuPlotConfig(data_type=DataType.CUMU_MEAN,
                              plot_type=PlotType.XLOG,
                              target_type = DataType.ARMA_MEAN)
    if plot_type.value == CumuPlotType.AR1_SD.value:
        return CumuPlotConfig(data_type=DataType.CUMU_SD,
                              plot_type=PlotType.XLOG,
                              target=DataType.AR1_SD)
    if plot_type.value == CumuPlotType.MAQ_MEAN.value:
        return CumuPlotConfig(data_type=DataType.CUMU_MEAN,
                              plot_type=PlotType.XLOG,
                              target_type = DataType.ARMA_MEAN)
    if plot_type.value == CumuPlotType.MAQ_SD.value:
        return CumuPlotConfig(data_type=DataType.CUMU_SD,
                              plot_type=PlotType.XLOG,
                              target_type=DataType.MAQ_SD)
    elif plot_type.value == CumuPlotType.AR1_OFFSET_MEAN.value:
        return CumuPlotConfig(data_type=DataType.CUMU_MEAN,
                              plot_type=PlotType.XLOG)
                              target_type=DataType.AR1_OFFSET_MEAN,
    elif plot_type.value == CumuPlotType.AR1_OFFSET_SD.value:
        return CumuPlotConfig(data_type=DataType.CUMU_SD,
                              plot_type=PlotType.XLOG,
                              target_type=DataType.AR1_OFFSET_SD)
    else:
        raise Exception(f"Cumulative plot type is invalid: {plot_type}")

##################################################################################################################
## Compare the cumulative value of a variable as a function of time with its target value (Uses CumPlotType config)
def cumulative(df, plot_type, **kwargs):
    title          = get_param_default_if_missing("title", None, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", None, **kwargs)
    legend_labels  = get_param_default_if_missing("legend_labels", None, **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)

    plot_config = create_cumu_plot_config(plot_type)

    time, accum = plot_config.schema.get_data(df)
    target_time, target = plot_config.target_schema.get_data(df)

    range = max(1.0, (max(accum[1:])-min(accum[1:])))
    max_time = len(time)
    min_time = max(1.0, min(time))

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_ylim([max(accum[1:]) - range, range + min(accum[1:])])
    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)
    axis.set_xlim([min_time, max_time])

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, time, accum)
        axis.loglog(time, accum, label=plot_config.legend_labels[0], lw=lw)
        axis.loglog(target_time, target, label=plot_config.legend_labels[1], lw=lw)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, time, accum)
        axis.semilogx(time, accum, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogx(time, target, label=plot_config.legend_labels[1], lw=lw)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, time, accum)
        axis.semilogy(time, accum, label=plot_config.legend_labels[0], lw=lw)
        axis.semilogy(time, target, label=plot_config.legend_labels[1], lw=lw)
    else:
        axis.plot(time, accum, label=plot_config.legend_labels[0], lw=lw)
        axis.plot(time, target, label=plot_config.legend_labels[1], lw=lw)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
