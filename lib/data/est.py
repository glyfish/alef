import numpy
from enum import Enum
from pandas import (DataFrame)

from lib.models import arima
from lib import stats
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)
from lib.data.schema import (DataType, DataSchema)

##################################################################################################################
# Parameter Estimates
class EstType(Enum):
    AR = "AR"                     # Autoregressive model parameters
    AR_OFFSET = "AR_OFFSET"       # Autoregressive model with constant offset parameters
    MA = "MA"                     # Moving average model parameters
    MA_OFFSET = "MA_OFFSET"       # Moving average model  with constant offset parameters
    PERGRAM = "PERGRAM"           # Periodogram esimate of FBM Hurst parameter using OLS
    VAR_AGG = "VAR_AGG"           # Variance Aggregation esimate of FBM Hurst parameter using OLS
    LINEAR = "LINEAR"             # Simple single variable linear regression
    LOG = "LOG"                   # Log log single variable linear regression

    def ols_key(self):
        return self.value

##################################################################################################################
# Estimated parameter
class ParamEst:
    def __init__(self, est, err, est_label=None, err_label=None):
            self.est = est
            self.err = err
            self.est_label = est_label
            self.err_label = err_label
            self._set_data()

    def set_labels(self, est_label, err_label):
        self.est_label = est_label
        self.err_label = err_label
        self._set_data()

    def __repr__(self):
        return f"ParamEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est=({self.est}), " \
               f"err=({self.err}, " \
               f"est_label=({self.est_label}), "\
               f"err_label=({self.err_label})"

    def _set_data(self):
        self.data = {"Estimate": self.est,
                     "Error": self.err,
                     "Estimate Label": self.est_label,
                     "Error Label": self.err_label}

    @staticmethod
    def from_array(meta_data):
        return ParamEst(meta_data[0], meta_data[1])

##################################################################################################################
# ARMA estimated parameters
class ARMAEst:
    def __init__(self, type, const, sigma2, params):
        self.type = type
        self.const = const
        self.order = len(params)
        self.params = params
        self.sigma2 = sigma2
        self._set_const_labels()
        self._set_params_labels()
        self._set_sigma2_labels()
        self.data = {"Type": type,
                     "Const": const.data,
                     "Parameters": [p.data for p in params],
                     "Sigma2": sigma2.data}

    def __repr__(self):
        return f"ARMAEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), " \
               f"const=({self.const}), " \
               f"params=({self.params}), " \
               f"sigma2=({self.sigma2})"

    def key(self):
        return f"{self.type.value}({self.order})"

    def get_formula():
        if self.type.value == EstType.AR.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.type.value == EstType.AR_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        elif self.type.value == EstType.MA.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.type.value == EstType.MA_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _set_const_labels(self):
        self.const.set_labels(est_label=r"$\hat{\mu^*}$",
                              err_label=r"$\sigma_{\hat{\mu^*}}$")

    def _set_params_labels(self):
        for i in range(len(self.params)):
            self._set_param_labels(i)

    def _set_param_labels(self, i):
        param = self.params[i]
        if self.type.value == EstType.AR.value or self.type.value == EstType.AR_OFFSET.value:
            param.set_labels(est_label=f"$\hat{{\phi_{{{i}}}}}$",
                             err_label=f"$\sigma_{{$\hat{{\phi_{{{i}}}}}}}$")
        elif self.type.value == EstType.MA.value or self.type.value == EstType.MA_OFFSET.value:
            param.set_labels(est_label=f"$\hat{{\\theta_{{{i}}}}}$",
                             err_label=f"$\sigma_{{$\hat{{\theta_{{{i}}}}}}}$")
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _set_sigma2_labels(self):
            self.sigma2.set_labels(est_label=r"$\hat{\sigma^2}$",
                                   err_label=r"$\sigma_{\hat{\sigma^2}}$")

    @staticmethod
    def from_dict(meta_data):
        return ARMAEst(
            type=meta_data["Type"],
            const=ParamEst.from_array(meta_data["Const"]),
            sigma2=ParamEst.from_array(meta_data["Sigma2"]),
            params=[ParamEst.from_array(est) for est in  meta_data["Parameters"]]
        )

##################################################################################################################
# Single variable OLS estimated parameters
class OLSSingleVarEst:
    def __init__(self, type, reg_type, const, param, r2):
        self.type = type
        self.reg_type = reg_type
        self.const = const
        self.param = param
        self.r2 = r2
        self.const.set_labels(est_label=r"$\hat{\alpha}$",
                              err_label=r"$\sigma_{\hat{\alpha}}$")
        self.param.set_labels(est_label=r"$\hat{\beta}$",
                              err_label=r"$\sigma_{\hat{beta}}$")
        self.data = {"Type": type,
                     "Regression Type": reg_type,
                     "Constant": const.data,
                     "Parameter": param.data,
                     "R2": r2}
        self.trans = _create_ols_single_var_trans(type, self.const, self.param)

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), " \
               f"reg_type=({self.reg_type}), " \
               f"const=({self.const}), " \
               f"params=({self.param}, "\
               f"r2=({self.r2})"

    def key(self):
        return self.type.value

    def formula(self):
        return self.trans.formula

    def trans_est(self):
        return self.trans.est

    def trans_const(self):
        return self.trans.const

    def get_yfit(self):
        if self.reg_type.value == stats.RegType.LOG.value:
            return self._log_fit()
        elif self.reg_type.value == stats.RegType.LINEAR.value:
            return self._linear_fit()
        elif self.reg_type.value == stats.RegType.XLOG.value:
            raise Exception(f"Regression type not supported: {self.reg_type}")
        elif self.reg_type.value == stats.RegType.YLOG.value:
            raise Exception(f"Regression type not supported: {self.reg_type}")
        else:
            raise Exception(f"Regression type is invalid: {self.reg_type}")

    def _log_fit(self):
        return lambda x : 10**self.const[0] * x**self.param[0]

    def _linear_fit(self):
        return lambda x : self.const[0] + x*self.param[0]

    @staticmethod
    def from_dict(meta_data):
        return OLSSingleVarEst(type=meta_data["Type"],
                               reg_type=meta_data["Regression Type"],
                               const=ParamEst.from_array(meta_data["Constant"]),
                               param=ParamEst.from_array(meta_data["Parameter"]),
                               r2=meta_data["R2"])

##################################################################################################################
# transformed parameters
class OLSSinlgeVarTrans:
    def __init__(self, formula, const, param):
        self.formula = formula
        self.param = param
        self.const = const

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"formula=({self.formula}), " \
               f"param=({self.param}), " \
               f"const=({self.const})"


##################################################################################################################
def _create_ols_single_var_trans(est_type, param, const):
    if est_type.value == EstType.VAR_AGG.value:
        return _create_var_agg_trans(param, const)
    elif est_type.value == EstType.PERGRAM.value:
        return _create_pergram_trans(param, const)
    elif est_type.value == EstType.LINEAR.value:
        return _create_linear_trans(param, const)
    elif est_type.value == EstType.LOG.value:
        return _create_log_trans(param, const)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

# EstType.VAR_AGG
def _create_var_agg_trans(param, const):
    formula = r"$\sigma^2 m^{2\left(H-1\right)}$"
    param = ParamEst(est=1.0 + param.est/2.0,
                     err=param.est_err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$")
    c = 10.0**const.est
    const = ParamEst(est=c,
                     err= c*const.est_err,
                     est_label=r"$\hat{\sigma}^2$",
                     err_label=r"$\sigma^2_{\hat{\sigma}^2}$")
    return OLSSinlgeVarTrans(formula, param, const)

# EstType.PERGRAM
def _create_pergram_trans(param, const):
    formula = r"$C\omega^{1 - 2H}$"
    param = ParamEst(est=(1.0 - param.est)/2.0,
                     err=param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$")
    c = 10.0**const.est
    const = ParamEst(est=c,
                     err=c*const.err,
                     est_label=r"$\hat{C}$",
                     err_label=r"$\sigma_{\hat{C}}$")
    return OLSSinlgeVarTrans(formula, param, const)

# EstType.LINEAR
def _create_linear_trans(param, const):
    return OLSSinlgeVarTrans(r"$\alpha + \beta x$", param, const)

# EstType.LOG
def _create_log_trans(param, const):
    return OLSSinlgeVarTrans(r"$10^\alpha x^\beta$", param, const)

##################################################################################################################
# Create estimates
def create_estimates_from_dict(dict):
    result = {}
    for key in dict.keys():
        result[key] = create_estimate_from_dict(dict[key])
    return result

def create_estimate_from_dict(dict):
    est_type = dict["Type"]
    if est_type.value == EstType.AR.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.AR_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.MA.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.MA_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.PERGRAM.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == EstType.VAR_AGG.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == EstType.LINEAR.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == EstType.LOG.value:
        return OLSSingleVarEst.from_dict(dict)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

def create_dict_from_estimates(ests):
    result = {}
    for key in ests.keys():
        result[key] = ests[key].data
    return result

##################################################################################################################
# Perform esimate for specified esimate types
def perform_est_for_type(df, est_type, **kwargs):
    data_type = get_param_default_if_missing("data_type", DataType.TIME_SERIES, **kwargs)
    x, y = DataSchema.get_data_type(df, data_type)
    if est_type.value == EstType.AR.value:
        return _ar_estimate(y, **kwargs)
    elif est_type.value == EstType.AR_OFFSET.value:
        return _ar_offset_estimate(y, **kwargs)
    elif est_type.value == EstType.MA.value:
        return _ma_estimate(y, **kwargs)
    elif est_type.value == EstType.MA_OFFSET.value:
        return _ma_offset_estimate(y, **kwargs)
    elif est_type.value == EstType.PERGRAM.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    elif est_type.value == EstType.VAR_AGG.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    elif est_type.value == EstType.LINEAR.value:
        return _ols_estimate(x, y, stats.RegType.LINEAR, est_type, **kwargs)
    elif est_type.value == EstType.LOG.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

def _arma_estimate_from_result(result, type):
    nparams = len(result.params)
    params = []
    for i in range(1, nparams-1):
        params.append(ParamEst.from_array([result.params.iloc[i], result.bse.iloc[i]]))
    const = ParamEst.from_array([result.params.iloc[0], result.bse.iloc[0]])
    sigma2 = ParamEst.from_array([result.params.iloc[nparams-1], result.bse.iloc[nparams-1]])
    return ARMAEst(type, const, sigma2, params)

def _ols_estimate_from_result(x, y, reg_type, est_type, result):
    const = ParamEst.from_array([result.params[0], result.bse[0]])
    param = ParamEst.from_array([result.params[1], result.bse[1]])
    r2 = result.rsquared
    return OLSSingleVarEst(est_type, reg_type, const, param, r2)

# EstType.AR
def _ar_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_fit(samples, order)
    return result, _arma_estimate_from_result(result, EstType.AR)

# EstType.AR_OFFSET
def _ar_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, EstType.AR_OFFSET)

# EstType.MA
def _ma_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_fit(samples, order)
    return result, _arma_estimate_from_result(result, EstType.MA)

# EstType.MA_OFFSET
def _ma_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, EstType.MA_OFFSET)

# EstType.PERGRAM, EstType.VAR_AGG, EstType.LINEAR, EstType.LOG
def _ols_estimate(x, y, reg_type, est_type, **kwargs):
    result = stats.OLS_fit(y, x, reg_type)
    return result, _ols_estimate_from_result(x, y, reg_type, est_type, result)
