import numpy
from enum import Enum
from pandas import (DataFrame)

from lib.models import arima
from lib import stats

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

##################################################################################################################
# Estimated parameter
class ParamEst:
    def __init__(self, est, err):
            self.est = est
            self.err = err
            self.est_label = None
            self.err_label = None
            self.data = {"Estimate": self.est,
                         "Error": self.err,
                         "Estimate Label": self.est_label,
                         "Error Label": self.err_label}


    def __repr__(self):
        return f"ParamEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est=({self.est}), err=({self.err}), data=({self.data})"

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
        self.formula = self._set_formula()
        self.data = {"Type": type,
                     "Const": const.data,
                     "Parameters": [p.data for p in params],
                     "Sigma2": sigma2.data}
        self._set_const_labels()
        self._set_params_labels()
        self._set_params_labels()

    def __repr__(self):
        return f"ARMAEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), " \
               f"const=({self.const}), " \
               f"order=({self.order}), " \
               f"params=({self.params}), " \
               f"sigma2=({self.sigma2}), " \
               f"formula=({self.formula})"

    def key(self):
        return f"{self.type.value}({self.order})"

    def _set_const_labels(self):
        self.const.est_label = r"$\hat{\mu^*}$"
        self.const.err_label = r"$\sigma_{\hat{\mu^*}}$"

    def _set_params_labels(self):
        for i in range(self.params):
            self._set_param_labels(params[i], i)

    def _set_param_labels(self, param, i):
        if self.type.value == EstType.AR.value or self.type.value == EstType.AR_OFFSET.value:
            param.est_label = f"$\hat{{\varphi_{{{i}}}}}$"
            param.est_label = f"$\sigma_{{$\hat{{\varphi_{{{i}}}}}}}$"
        elif self.type.value == EstType.MA.value or self.type.value == EstType.MA_OFFSET.value:
            param.est_label = f"$\hat{{\vartheta_{{{i}}}}}$"
            param.est_label = f"$\sigma_{{$\hat{{\varphi_{{{i}}}}}}}$"
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _set_sigma2(self):
        self.const.est_label = r"$\hat{\sigma^2}$"
        self.const.err_label = r"$\sigma_{\hat{\sigma^2}}$"

    def _set_formula():
        if self.type.value == EstType.AR.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.type.value == EstType.AR_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        elif self.type.value == EstType.MA.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$""
        elif self.type.value == EstType.MA_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$""
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

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
    def __init__(self, type, const, param, r2):
        self.type = type
        self.const = self._get_const(const)
        self.param = self._get_param(param)
        self.r2 = r2
        self.data = {"Const": const.data, "Parameters": params.data}
        self.formula = self._set_formula()
        self.results_text = self._set_results_text()
        self.plot_type = self._set_plotType()
        self._set_const_labels()
        self._set_param_labels()

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), const=({self.const}), params=({self.param}), data=({self.data})"

    def key(self):
        return self.type.value

    def _get_param(self, param):
        if self.type.value == EstType.VAR_AGG.value:
            return ParamEst(est=1.0 + param[0]/2.0, err=param[1]/2.0)
        elif self.type.value == EstType.PERGRAM.value:
            return ParamEst(est=1.0 - param[0]/2.0, err=param[1]/2.0)
        elif est_type.value == EstType.LINEAR.value:
            return param
        elif est_type.value == EstType.LOG.value:
            return param
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _get_const(self, const):
        if self.type.value == EstType.VAR_AGG.value:
            return self._get_log_const(const)
        elif self.type.value == EstType.PERGRAM.value:
            return self._get_log_const(const)
        elif est_type.value == EstType.LINEAR.value:
            return param
        elif est_type.value == EstType.LOG.value:
            return self._get_log_const(const)
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _get_log_const(self, const):
        c = 10.0**const[0]
        return ParamEst(est=c, err=c*const[1])

    def _set_const_labels(self):
        if self.type.value == EstType.PERGRAM.value:
        elif self.type.value == EstType.PERGRAM.value:
        elif est_type.value == EstType.LINEAR.value:
        elif est_type.value == EstType.LOG.value:
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    def _set_params_labels(self):
        if self.type.value == EstType.PERGRAM.value:
        elif self.type.value == EstType.PERGRAM.value:
        elif est_type.value == EstType.LINEAR.value:
        elif est_type.value == EstType.LOG.value:
        else:
            raise Exception(f"Esitmate type is invalid: {est_type}")

    @staticmethod
    def from_dict(meta_data):
        return OLSSingleVarEst(
            type=meta_data["Type"],
            const = ParamEst.from_array(meta_data["Const"]),
            param = ParamEst.from_array(meta_data["Parameter"])
            r2 = meta_data["R2"]
        )

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
        return _ols_estimate(x, y, RegType.LOG, **kwargs)
    elif est_type.value == EstType.VAR_AGG.value:
        return _ols_estimate(x, y, RegType.LOG, **kwargs)
    elif est_type.value == EstType.LINEAR.value:
        return _ols_estimate(x, y, RegType.LINEAR, **kwargs)
    elif est_type.value == EstType.LOG.value:
        return _ols_estimate(x, y, RegType.LOG, **kwargs)
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

def _ols_estimate_from_result(x, y, result, type):
    const = ParamEst.from_array([result.param[0], result.bse[0]])
    param = ParamEst.from_array([result.param[1], result.bse[1]])
    r2 = results.rsquared
    return OLSSingleVarEst(type, const, param, r2)

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
    results = stats.OLS_fit(y, x, reg_type)
    return _ols_estimate_from_result(x, y, est_type, result)
