import numpy
from enum import Enum
from matplotlib import pyplot

from lib import stats

from lib.data.meta_data import (MetaData)
from lib.data.est import (EstType)
from lib.data.schema import (DataType, DataSchema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

###############################################################################################
# Create single variable regression PlotConfig
class SingleVarPlotConfig:
    def __init__(self, df, est_type):
        schema = DataSchema.get_schema(df)
        source_schema = MetaData.get_source_schema(df)
        self.meta_data = MetaData.get(df, schema)
        self.source_meta_data = MetaData.get(df, source_schema)
        self.est = self.meta_data.get_estimate(est_type.ols_key())

    def __repr__(self):
        return f"DataPlotConfig({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"meta_data=({self.meta_data}), " \
               f"source_meta_data=({self.source_meta_data})"

    def xlabel(self):
        return self.meta_data.xlabel

    def ylabel(self):
        return self.meta_data.ylabel

    def title(self):
        params = self.source_meta_data.params | self.meta_data.params
        desc = f"{self.source_meta_data.desc} {self.meta_data.desc}"
        return f"{desc} : {MetaData.params_to_str(params)}"

    def labels(self):
        return ["Data", f"{self.ylabel()}={self.est.formula()}"]

    def yfit(self, x):
        return self.est.get_yfit()(x)

    def slope_is_negative(self):
        return self.est.param.est < 0.0

    def results(self):
        param = self.est.trans_param()
        const = self.est.trans_const()
        r2 = self.est.r2
        param_est = f"{param.est_label}={format(param.est, '2.2f')}"
        paream_err = f"{param.err_label}={format(param.err, '2.2f')}"
        const_est = f"{const.est_label}={format(const.est, '2.2f')}"
        const_err = f"{const.err_label}={format(const.err, '2.2f')}"
        r2_row = f"$R^2$ = {format(r2, '2.2f')}"
        return f"{param_est}\n{paream_err}\n{const_est}\n{const_err}\n{r2_row}"

    def plot_type(self):
        if self.est.reg_type.value == stats.RegType.LOG.value:
            return PlotType.LOG
        elif self.reg_type.value == stats.RegType.LINEAR.value:
            return PlotType.LINEAR
        elif self.reg_type.value == stats.RegType.XLOG.value:
            return PlotType.XLOG
        elif self.reg_type.value == stats.RegType.YLOG.value:
            return PlotType.YLOG
        else:
            raise Exception(f"Regression type is invalid: {self.reg_type}")

###############################################################################################
# Compare the result of a linear regression with teh acutal data (Uses RegPlotType config)
def single_var(df, **kwargs):
    data_type      = get_param_throw_if_missing("data_type", **kwargs)
    est_type       = get_param_throw_if_missing("est_type", **kwargs)

    plot_config    = SingleVarPlotConfig(df, data_type, est_type)

    plot_type      = get_param_default_if_missing("plot_type", plot_config.plot_type(), **kwargs)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    labels         = get_param_default_if_missing("labels", plot_config.labels(), **kwargs)

    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    x, y = plot_config.meta_data.get_data(df)

    if plot_config.slope_is_negative():
        x_text = 0.1
        y_text = 0.1
        legend_loc = "upper right"
    else:
        x_text = 0.8
        y_text = 0.1
        legend_loc = "upper left"

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results(), bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type().value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=labels[0])
        axis.loglog(x, plot_config.yfit(x), zorder=10, label=labels[1])
    elif plot_config.plot_type().value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=labels[0])
        axis.semilogx(x, plot_config.yfit(x), zorder=10, label=labels[1])
    elif plot_config.plot_type().value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=labels[0])
        axis.plot(x, plot_config.yfit(x), zorder=10, label=labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=labels[0])
        axis.plot(x, plot_config.yfit(x), zorder=10, label=labels[1])

    axis.legend(loc=legend_loc, bbox_to_anchor=(0.1, 0.1, 0.85, 0.85))
