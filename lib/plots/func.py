import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import fbm
from lib.models import arima
from lib import stats

from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

##################################################################################################################
## Specify PlotConfig for fcompare plot
class FuncPlotType(Enum):
    LINEAR = "LINEAR"                               # Linear Model
    FBM_MEAN = "FBM_MEAN"                           # FBM model mean with data
    FBM_SD = "FBM_SD"                               # FBM model standard deviation with data
    FBM_ACF = "FBM_ACF"                             # FBM model autocorrelation with data
    BM_MEAN = "BM_MEAN"                             # BM model mean with data
    BM_DRIFT_MEAN = "BM_DRIFT_MEAN"                 # BM model mean with data
    BM_SD = "BM_SD"                                 # BM model standard deviation with data
    GBM_MEAN = "GBM_MEAN"                           # GBM model mean with data
    GBM_SD = "GBM_SD"                               # GBM model mean with data
    AR1_ACF = "AR1_ACF"                             # AR1 model ACF autocorrelation function with data
    MAQ_ACF = "MAQ_ACF"                             # MA(q) model ACF autocorrelation function with data
    LAG_VAR = "LAG_VAR"                             # Lagged variance computed from a time series
    VR = "VR"                                       # Vraiance ratio use in test for brownian motion
    AR1_CUMU_MEAN = "AR1_CUMU_MEAN"                 # Cumulative mean for AR(1)
    AR1_CUMU_SD = "AR1_CUMU_SD"                     # Cumulative standard deviation for AR(1)
    MAQ_CUMU_MEAN = "MAQ_CUMU_MEAN"                 # Cumulative mean for MA(q)
    MAQ_CUMU_SD = "MAQ_CUMU_SD"                     # Cumulative standard deviation for MA(q)
    AR1_CUMU_OFFSET_MEAN = "AR1_CUMU_OFFSET_MEAN"   # AR1 with Offset model mean
    AR1_CUMU_OFFSET_SD = "AR1_CUMU_OFFSET_SD"       # AR1 with Offset model standard deviation

##################################################################################################################
## Function compare plot config
class FuncPlotConfig:
    def __init__(self, df, data_type, func_type, plot_type=PlotType.LINEAR, xlabel=None, ylabel=None, legend_labels=None, title=None):
        self.data_schema = create_schema(data_type)
        self.func_schema = create_schema(func_type)
        self.plot_type = plot_type
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._legend_labels = legend_labels
        self._title = title
        self._data_meta_data = MetaData.get(df, self.data_schema)
        self._func_meta_data = MetaData.get(df, self.func_schema)

        if self._data_meta_data.source_schema is not None:
            self._source_meta_data = MetaData.get(df, self._data_meta_data.source_schema)
        else:
            self._source_meta_data = None

    def __repr__(self):
        return f"CumuPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"data_schema=({self.data_schema}), " \
               f"func_schema=({self.func_schema}), " \
               f"plot_type=({self.plot_type}), " \
               f"xlabel=({self.xlabel()}), " \
               f"ylabel=({self.ylabel()}), " \
               f"legend_labels=({self.legend_labels()}), " \
               f"title=({self.title()})"

    def xlabel(self):
        if self._xlabel is None:
            return self._data_meta_data.xlabel
        else:
            return self._xlabel

    def ylabel(self):
        if self._ylabel is None:
            return self._data_meta_data.ylabel
        else:
            return self._ylabel

    def legend_labels(self):
        if self._legend_labels is None:
            return [self._data_meta_data.ylabel, self.formula()]
        else:
            return self._legend_labels

    def formula(self):
        if self._func_meta_data.formula is None:
            return self._func_meta_data.ylabel
        else:
            return self._func_meta_data.ylabel + "=" + self._func_meta_data.formula

    def title(self):
        if self._title is None:
            return self._build_title()
        else:
            return self._title

    def _build_title(self):
        params = self._data_meta_data.params | self._func_meta_data.params
        if self._source_meta_data is None:
            desc = self._data_meta_data.desc
        else:
            params = params | self._source_meta_data.params
            desc = f"{self._source_meta_data.desc} {self._data_meta_data.desc}"

        if not params:
            return desc
        else:
            return f"{desc}: {MetaData.params_to_str(params)}"

##################################################################################################################
## plot function type
def create_func_plot_type(plot_type, df):
    if plot_type.value == FuncPlotType.FBM_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.MEAN,
                              func_type=DataType.FBM_MEAN,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.FBM_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.SD,
                              func_type=DataType.FBM_SD,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.FBM_ACF.value:
        return FuncPlotConfig(df,
                              data_type=DataType.ACF,
                              func_type=DataType.FBM_ACF,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.BM_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.MEAN,
                              func_type=DataType.BM_MEAN,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.BM_DRIFT_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.MEAN,
                              func_type=DataType.BM_DRIFT_MEAN,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.BM_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.SD,
                              func_type=DataType.BM_SD,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.GBM_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.MEAN,
                              func_type=DataType.GBM_MEAN,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.GBM_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.SD,
                              func_type=DataType.GBM_SD,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.AR1_ACF.value:
        return FuncPlotConfig(df,
                              data_type=DataType.ACF,
                              func_type=DataType.AR1_ACF,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.MAQ_ACF.value:
        return FuncPlotConfig(df,
                              data_type=DataType.ACF,
                              func_type=DataType.MAQ_ACF,
                              plot_type=PlotType.LINEAR)
    elif plot_type.value == FuncPlotType.LAG_VAR.value:
        return FuncPlotConfig(df,
                              data_type=DataType.TIME_SERIES,
                              func_type=DataType.LAG_VAR,
                              plot_type=PlotType.LOG)
    elif plot_type.value == FuncPlotType.VR.value:
        return FuncPlotConfig(df,
                              data_type=DataType.TIME_SERIES,
                              func_type=DataType.VR,
                              plot_type=PlotType.LOG)
    elif plot_type.value == FuncPlotType.AR1_CUMU_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_MEAN,
                              func_type=DataType.ARMA_MEAN,
                              plot_type=PlotType.XLOG)
    elif plot_type.value == FuncPlotType.AR1_CUMU_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_SD,
                              func_type=DataType.AR1_SD,
                              plot_type=PlotType.XLOG)
    elif plot_type.value == FuncPlotType.MAQ_CUMU_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_MEAN,
                              func_type = DataType.ARMA_MEAN,
                              plot_type=PlotType.XLOG)
    elif plot_type.value == FuncPlotType.MAQ_CUMU_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_SD,
                              func_type=DataType.MAQ_SD,
                              plot_type=PlotType.XLOG)
    elif plot_type.value == FuncPlotType.AR1_CUMU_OFFSET_MEAN.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_MEAN,
                              func_type=DataType.AR1_OFFSET_MEAN,
                              plot_type=PlotType.XLOG)
    elif plot_type.value == FuncPlotType.AR1_CUMU_OFFSET_SD.value:
        return FuncPlotConfig(df,
                              data_type=DataType.CUMU_SD,
                              func_type=DataType.AR1_OFFSET_SD,
                              plot_type=PlotType.XLOG)
    else:
        raise Exception(f"FuncPlotType type is invalid: {data_type}")

###############################################################################################
# Compare data to the value of a function at a specified number of points
def fpoints(df, plot_type, **kwargs):
    plot_config    = create_func_plot_type(plot_type, df)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    legend_labels  = get_param_default_if_missing("legend_labels", plot_config.legend_labels(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)

    x, y = plot_config.data_schema.get_data(df)
    fx, fy = plot_config.func_schema.get_data(df)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    axis.set_title(title)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=legend_labels[0], lw=lw)
        axis.loglog(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=legend_labels[0], lw=lw)
        axis.semilogx(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=legend_labels[0], lw=lw)
        axis.semilogy(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=legend_labels[0], lw=lw)
        axis.plot(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

##################################################################################################################
## Compare the function with all data points
def fcurve(df, plot_type, **kwargs):
    plot_config    = create_func_plot_type(plot_type, df)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    legend_labels  = get_param_default_if_missing("legend_labels", plot_config.legend_labels(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)

    x, y = plot_config.data_schema.get_data(df)
    fx, fy = plot_config.func_schema.get_data(df)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    axis.set_title(title, y=title_offset)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, y=title_offset)

    if plot_config.plot_type.value == PlotType.LOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        logStyle(axis, x, y)
        axis.loglog(x, y, label=legend_labels[0], lw=lw)
        axis.loglog(fx, fy, label=legend_labels[1], lw=lw)
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=legend_labels[0], lw=lw)
        axis.semilogx(fx, fy, label=legend_labels[1], lw=lw)
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=legend_labels[0], lw=lw)
        axis.semilogy(fx, fy, label=legend_labels[1], lw=lw)
    else:
        axis.plot(x, y, label=legend_labels[0], lw=lw)
        axis.plot(fx, fy, label=legend_labels[1], lw=lw)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
