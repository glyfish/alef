import numpy
from enum import Enum
from matplotlib import pyplot

from lib.models import fbm
from lib.models import arima
from lib import stats

from lib.data.schema import (DataType)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.data.meta_data import (MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

##################################################################################################################
## Function compare plot config
class FuncPlotConfig:
    def __init__(self, data, func):
        self.data_meta_data = MetaData.get(data)
        self.func_meta_data = MetaData.get(func)
        self.source_meta_data = MetaData.get_source_meta_data(data)

    def __repr__(self):
        return f"CumuPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"data_meta_data=({self.data_meta_data}), func_meta_data=({self.func_meta_data})"

    def xlabel(self):
        return self.data_meta_data.xlabel

    def ylabel(self):
        return self.data_meta_data.ylabel

    def legend_labels(self):
        return [self.data_meta_data.ylabel, self.formula()]

    def formula(self):
        if self.func_meta_data.formula is None:
            return self.func_meta_data.ylabel
        else:
            return self.func_meta_data.ylabel + "=" + self.func_meta_data.formula

    def title(self):
        params = self.func_meta_data.params | self.data_meta_data.params
        if self.source_meta_data is None:
            desc = self.data_meta_data.desc
        else:
            params = params | self.source_meta_data.params
            desc = f"{self.source_meta_data.desc} {self.data_meta_data.desc}"
        if not params:
            return desc
        else:
            return f"{desc}: {MetaData.params_to_str(params)}"

###############################################################################################
# Compare data to the value of a function at a specified number of points
def fpoints(**kwargs):
    data           = get_param_throw_if_missing("data", **kwargs)
    func           = get_param_throw_if_missing("func", **kwargs)
    plot_config    = FuncPlotConfig(data, func)

    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    legend_labels  = get_param_default_if_missing("legend_labels", plot_config.legend_labels(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)

    x, y = plot_config.data_meta_data.get_data(data)
    fx, fy = plot_config.func_meta_data.get_data(func)

    figure, axis = pyplot.subplots(figsize=(13, 10))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    axis.set_title(title, y=title_offset)

    if plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, label=legend_labels[0], lw=lw)
        axis.loglog(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=legend_labels[0], lw=lw)
        axis.semilogx(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    elif plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=legend_labels[0], lw=lw)
        axis.semilogy(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    else:
        axis.plot(x, y, label=legend_labels[0], lw=lw)
        axis.plot(fx, fy, label=legend_labels[1], marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))

##################################################################################################################
## Compare the function with all data points
def fcurve(**kwargs):
    data           = get_param_throw_if_missing("data", **kwargs)
    func           = get_param_throw_if_missing("func", **kwargs)
    plot_config    = FuncPlotConfig(data, func)

    plot_type      = get_param_default_if_missing("plot_type", PlotType.LINEAR, **kwargs)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    legend_labels  = get_param_default_if_missing("legend_labels", plot_config.legend_labels(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)

    x, y = plot_config.data_meta_data.get_data(data)
    fx, fy = plot_config.func_meta_data.get_data(func)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    axis.set_title(title, y=title_offset)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    if plot_type.value == PlotType.LOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        logStyle(axis, x, y)
        axis.loglog(x, y, label=legend_labels[0], lw=lw)
        axis.loglog(fx, fy, label=legend_labels[1], lw=lw)
    elif plot_type.value == PlotType.XLOG.value:
        if x[0] == 0.0:
            x = x[1:]
            y = y[1:]
        logXStyle(axis, x, y)
        axis.semilogx(x, y, label=legend_labels[0], lw=lw)
        axis.semilogx(fx, fy, label=legend_labels[1], lw=lw)
    elif plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, label=legend_labels[0], lw=lw)
        axis.semilogy(fx, fy, label=legend_labels[1], lw=lw)
    else:
        axis.plot(x, y, label=legend_labels[0], lw=lw)
        axis.plot(fx, fy, label=legend_labels[1], lw=lw)

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
