import numpy
from enum import Enum
from matplotlib import pyplot

from lib import stats

from lib.data.meta_data import (MetaData)
from lib.data.est import (EstType)
from lib.data.schema import (DataType, create_schema)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

###############################################################################################
# Create single variable regression PlotConfig
class SingleVarPlotConfig:
    def __init__(self, df, data_type, est_type):
        schema = create_schema(data_type)
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

    def results(self):
        param = self.param.trans_est()
        const = self.est.trans_const()
        r2 = self.est.r2
        param_est_row = f"{param.est_label}=f{format(param.est, '2.2f')}"
        paream_err_row = f"{param.err_label}=f{format(param.err, '2.2f')}"
        const_est_row = f"{const.est_label}=f{format(const.est, '2.2f')}"
        const_err_row = f"{const.err_label}=f{format(const.err, '2.2f')}"
        r2_row = f"$R^2$={format(r2, '2.2f')}"

    def plot_type(self):
        if self.reg_type.value == stats.RegType.LOG.value:
            return PlotType.LOG
        elif self.reg_type.value == stats.RegType.LINEAR.value:
            return PlotType.LINEAR
        elif self.reg_type.value == stats.RegType.XLOG.value:
            return PlotType.XLOG
        elif self.reg_type.value == stats.RegType.YLOG.value:
            return PlotType.YLOG
        else:
            raise Exception(f"Regression type is invalid: {self.reg_type}")

    def _get_est(self, est_type):
        return se

###############################################################################################
# Create regression plot configuartion
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

###############################################################################################
# Compare the result of a linear regression with teh acutal data (Uses RegPlotType config)
def single_var(df, data_type, est_type, **kwargs):
    data_type      = get_param_throw_if_missing("data_type", **kwargs)
    est_type      = get_param_throw_if_missing("data_type", **kwargs)

    plot_config    = SingleVarPlotConfig(df, data_type)

    plot_type      = get_param_default_if_missing("plot_type", plot_config.plot_type(), **kwargs)
    title          = get_param_default_if_missing("title", plot_config.title(), **kwargs)
    title_offset   = get_param_default_if_missing("title_offset", 1.0, **kwargs)
    xlabel         = get_param_default_if_missing("xlabel", plot_config.xlabel(), **kwargs)
    ylabel         = get_param_default_if_missing("ylabel", plot_config.ylabel(), **kwargs)
    lw             = get_param_default_if_missing("lw", 2, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    β = results.params

    if β[1] < 0:
        x_text = 0.1
        y_text = 0.1
        lengend_location = (0.6, 0.65, 0.3, 0.3)
    else:
        x_text = 0.8
        y_text = 0.1
        lengend_location = (0.05, 0.65, 0.3, 0.3)

    plot_config = create_reg_plot_type(plot_type, results, x)

    figure, axis = pyplot.subplots(figsize=(13, 10))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_xlabel(plot_config.xlabel)

    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='white')
    axis.text(x_text, y_text, plot_config.results_text, bbox=bbox, fontsize=16.0, zorder=7, transform=axis.transAxes)

    if plot_config.plot_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.loglog(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.semilogx(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    elif plot_config.plot_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])
    else:
        axis.plot(x, y, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label=plot_config.legend_labels[0])
        axis.plot(x, plot_config.y_fit, zorder=10, label=plot_config.legend_labels[1])

    axis.legend(loc='best', bbox_to_anchor=lengend_location)
