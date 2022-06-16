import numpy
from enum import Enum
from pandas import (DataFrame)

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima
from lib.models import (TestHypothesis, Dist)

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)
from lib.data.schema import (DataType, DataSchema)

##################################################################################################################
# Meta Data Schema
##################################################################################################################
class MetaData:
    def __init__(self, npts, data_type, params, desc, xlabel, ylabel, ests={}, tests={}, source_schema=None, formula=None):
        self.npts = npts
        self.schema = data_type.schema()
        self.params = params
        self.desc = desc
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.source_schema = source_schema
        self.formula = formula
        self.ests = ests
        self.tests = tests
        self.dict = {
          "npts": npts,
          "DataType": data_type,
          "Parameters": params,
          "Description": desc,
          "ylabel": ylabel,
          "xlabel": xlabel,
          "SourceSchema": source_schema,
          "Formula": formula,
          "Estimates": _create_dict_from_estimates(ests),
          "Tests": _create_dict_from_tests(tests)
        }

    def __repr__(self):
        return f"MetaData({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"npts=({self.npts}), " \
               f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"desc=({self.desc}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"ests=({self.ests}), " \
               f"tests=({self.tests}), " \
               f"source_schema=({self.source_schema}) " \
               f"formula=({self.formula}) "

    def _create_dict_from_estimates(ests):
        result = {}
        for key in ests.keys():
            result[key] = ests[key].dict
        return result

    def insert_estimate(self, est):
        self.ests[est.key()] = est
        self.dict["Estimates"][est.key()] = est.dict

    def insert_test(self, test):
        self.tests[test.key()] = test
        self.dict["Tests"][test.key()] = test.dict

    def params_str(self):
        return MetaData.params_to_str(self.params)

    def get_data(self, df):
        return self.schema.get_data(df)

    def get_estimate(self, est_key):
        return self.ests[est_key]

    @classmethod
    def get_schema_data(cls, df):
        return cls.get(df).get_data(df)

    @classmethod
    def get(cls, df):
        type = DataSchema.get_type(df)
        return MetaData.from_dict(df.attrs[type.value])

    @classmethod
    def get_source_meta_data(cls, df):
        type = DataSchema.get_source_type(df)
        if type is not None:
            return MetaData.from_dict(df.attrs[type.value])
        else:
            return None

    @classmethod
    def set(cls, df, meta_data):
        type = DataSchema.get_type(df)
        df.attrs[type.value] = meta_data.dict

    @classmethod
    def add_estimate(cls, df, est):
        meta_data = MetaData.get(df)
        meta_data.insert_estimate(est)
        MetaData.set(df, meta_data)

    @classmethod
    def add_test(cls, df, test):
        meta_data = MetaData.get(df)
        meta_data.insert_test(test)
        MetaData.set(df, meta_data)

    @staticmethod
    def from_dict(meta_data):
        source_schema =  meta_data["SourceSchema"] if "SourceSchema" in meta_data else None
        formula =  meta_data["Formula"] if "Formula" in meta_data else None
        ests = _create_estimates_from_dict(meta_data["Estimates"]) if "Estimates" in meta_data else {}
        tests = _create_tests_from_dict(meta_data["Tests"]) if "Tests" in meta_data else {}

        return MetaData(
            npts=meta_data["npts"],
            data_type=meta_data["DataType"],
            params=meta_data["Parameters"],
            desc=meta_data["Description"],
            xlabel=meta_data["xlabel"],
            ylabel=meta_data["ylabel"],
            ests=ests,
            tests=tests,
            source_schema=source_schema,
            formula=formula
        )

    @staticmethod
    def params_to_str(params):
        set_param_keys = []
        for key in params.keys():
            value = params[key]
            if isinstance(value, list) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, numpy.ndarray) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, int) and value != 0:
                set_param_keys.append(key)
            if isinstance(value, float) and value != 0.0:
                set_param_keys.append(key)
            if isinstance(value, str) and value != "":
                set_param_keys.append(key)
        if len(set_param_keys) == 0:
            return ""
        params_strs = []
        for key in set_param_keys:
            value = params[key]
            if value is None:
                continue
            if isinstance(value, numpy.ndarray):
                value_str = numpy.array2string(value, precision=2, separator=',', suppress_small=True)
            else:
                value_str = f"{value}"
            params_strs.append(f"{key}={value_str}")
        return ", ".join(params_strs)

##################################################################################################################
# Parameter Estimates
##################################################################################################################
class Est(Enum):
    AR = "AR"                     # Autoregressive model parameters
    AR_OFFSET = "AR_OFFSET"       # Autoregressive model with constant offset parameters
    MA = "MA"                     # Moving average model parameters
    MA_OFFSET = "MA_OFFSET"       # Moving average model  with constant offset parameters
    PERGRAM = "PERGRAM"           # Periodogram esimate of FBM Hurst parameter using OLS
    AGG_VAR = "AGG_VAR"           # Variance Aggregation esimate of FBM Hurst parameter using OLS
    LINEAR = "LINEAR"             # Simple single variable linear regression
    LOG = "LOG"                   # Log log single variable linear regression

    def ols_key(self):
        return self.value

    def arma_key(self, order):
        return f"{self.value}({order})"

    def perform(self, df, **kwargs):
        x, y = DataSchema.get_schema_data(df)
        result, est = _perform_est_for_type(x, y, self, **kwargs)
        MetaData.add_estimate(df, est)
        return result

##################################################################################################################
# Estimated parameter
class ParamEst:
    def __init__(self, est, err, est_label=None, err_label=None):
            self.est = est
            self.err = err
            self.est_label = est_label
            self.err_label = err_label
            self._set_dict()

    def set_labels(self, est_label, err_label):
        self.est_label = est_label
        self.err_label = err_label
        self._set_dict()

    def __repr__(self):
        return f"ParamEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est=({self.est}), " \
               f"err=({self.err}, " \
               f"est_label=({self.est_label}), "\
               f"err_label=({self.err_label})"

    def _set_dict(self):
        self.dict = {"Estimate": self.est,
                     "Error": self.err,
                     "Estimate Label": self.est_label,
                     "Error Label": self.err_label}

    @staticmethod
    def from_dict(meta_data):
        if "Estimate Label" in meta_data:
            est_label = meta_data["Estimate Label"]
        else:
            est_label = None
        if "Error Label" in meta_data:
            err_label = meta_data["Error Label"]
        else:
            err_label = None

        return ParamEst(meta_data["Estimate"],
                        meta_data["Error"],
                        est_label,
                        err_label)

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
        self.dict = {"Type": type,
                     "Const": const.dict,
                     "Parameters": [p.dict for p in params],
                     "Sigma2": sigma2.dict}

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
        return self.type.arma_key(self.order)

    def get_formula():
        if self.type.value == Est.AR.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.type.value == Est.AR_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        elif self.type.value == Est.MA.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.type.value == Est.MA_OFFSET.value:
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
        if self.type.value == Est.AR.value or self.type.value == Est.AR_OFFSET.value:
            param.set_labels(est_label=f"$\hat{{\phi_{{{i}}}}}$",
                             err_label=f"$\sigma_{{$\hat{{\phi_{{{i}}}}}}}$")
        elif self.type.value == Est.MA.value or self.type.value == Est.MA_OFFSET.value:
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
            const=ParamEst.from_dict(meta_data["Const"]),
            sigma2=ParamEst.from_dict(meta_data["Sigma2"]),
            params=[ParamEst.from_dict(est) for est in  meta_data["Parameters"]]
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
        self.dict = {"Type": type,
                     "Regression Type": reg_type,
                     "Constant": const.dict,
                     "Parameter": param.dict,
                     "R2": r2}
        self.trans = _create_ols_single_var_trans(type, self.param, self.const)

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
        return self.type.ols_key()

    def formula(self):
        return self.trans.formula

    def trans_param(self):
        return self.trans.param

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
        return lambda x : 10**self.const.est * x**self.param.est

    def _linear_fit(self):
        return lambda x : self.const.est + x*self.param.est

    @staticmethod
    def from_dict(meta_data):
        return OLSSingleVarEst(type=meta_data["Type"],
                               reg_type=meta_data["Regression Type"],
                               const=ParamEst.from_dict(meta_data["Constant"]),
                               param=ParamEst.from_dict(meta_data["Parameter"]),
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
    if est_type.value == Est.AGG_VAR.value:
        return _create_agg_var_trans(param, const)
    elif est_type.value == Est.PERGRAM.value:
        return _create_pergram_trans(param, const)
    elif est_type.value == Est.LINEAR.value:
        return _create_linear_trans(param, const)
    elif est_type.value == Est.LOG.value:
        return _create_log_trans(param, const)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

# Est.AGG_VAR
def _create_agg_var_trans(param, const):
    formula = r"$\sigma^2 m^{2\left(H-1\right)}$"
    param = ParamEst(est=1.0 + param.est/2.0,
                     err=param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$")
    c = 10.0**const.est
    const = ParamEst(est=c,
                     err= c*const.err,
                     est_label=r"$\hat{\sigma}^2$",
                     err_label=r"$\sigma^2_{\hat{\sigma}^2}$")
    return OLSSinlgeVarTrans(formula, const, param)

# Est.PERGRAM
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
    return OLSSinlgeVarTrans(formula, const, param)

# Est.LINEAR
def _create_linear_trans(param, const):
    return OLSSinlgeVarTrans(r"$\alpha + \beta x$", const, param)

# Est.LOG
def _create_log_trans(param, const):
    return OLSSinlgeVarTrans(r"$10^\alpha x^\beta$", const, param)

##################################################################################################################
# Create estimates
def _create_estimates_from_dict(dict):
    result = {}
    for key in dict.keys():
        result[key] = _create_estimate_from_dict(dict[key])
    return result

def _create_estimate_from_dict(dict):
    est_type = dict["Type"]
    if est_type.value == Est.AR.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == Est.AR_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == Est.MA.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == Est.MA_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == Est.PERGRAM.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == Est.AGG_VAR.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == Est.LINEAR.value:
        return OLSSingleVarEst.from_dict(dict)
    elif est_type.value == Est.LOG.value:
        return OLSSingleVarEst.from_dict(dict)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

def _create_dict_from_estimates(ests):
    result = {}
    for key in ests.keys():
        result[key] = ests[key].dict
    return result

##################################################################################################################
# Perform esimate for specified estimate types
def _perform_est_for_type(x, y, est_type, **kwargs):
    if est_type.value == Est.AR.value:
        return _ar_estimate(y, **kwargs)
    elif est_type.value == Est.AR_OFFSET.value:
        return _ar_offset_estimate(y, **kwargs)
    elif est_type.value == Est.MA.value:
        return _ma_estimate(y, **kwargs)
    elif est_type.value == Est.MA_OFFSET.value:
        return _ma_offset_estimate(y, **kwargs)
    elif est_type.value == Est.PERGRAM.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    elif est_type.value == Est.AGG_VAR.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    elif est_type.value == Est.LINEAR.value:
        return _ols_estimate(x, y, stats.RegType.LINEAR, est_type, **kwargs)
    elif est_type.value == Est.LOG.value:
        return _ols_estimate(x, y, stats.RegType.LOG, est_type, **kwargs)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

# Est.AR
def _ar_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_fit(samples, order)
    return result, _arma_estimate_from_result(result, Est.AR)

# Est.AR_OFFSET
def _ar_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, Est.AR_OFFSET)

# Est.MA
def _ma_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_fit(samples, order)
    return result, _arma_estimate_from_result(result, Est.MA)

# Est.MA_OFFSET
def _ma_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, Est.MA_OFFSET)

# Est.PERGRAM, Est.AGG_VAR, Est.LINEAR, Est.LOG
def _ols_estimate(x, y, reg_type, est_type, **kwargs):
    result = stats.OLS_fit(y, x, reg_type)
    return result, _ols_estimate_from_result(x, y, reg_type, est_type, result)

##################################################################################################################
# Construct estimate objects from result object
def _arma_estimate_from_result(result, type):
    nparams = len(result.params)
    params = []
    for i in range(1, nparams-1):
        params.append(ParamEst.from_dict({"Estimate": result.params.iloc[i],
                                          "Error": result.bse.iloc[i]}))
    const = ParamEst.from_dict({"Estimate": result.params.iloc[0],
                                "Error": result.bse.iloc[0]})
    sigma2 = ParamEst.from_dict({"Estimate": result.params.iloc[nparams-1],
                                 "Error": result.bse.iloc[nparams-1]})
    return ARMAEst(type, const, sigma2, params)

def _ols_estimate_from_result(x, y, reg_type, est_type, result):
    const = ParamEst.from_dict({"Estimate": result.params[0],
                                "Error": result.bse[0]})
    param = ParamEst.from_dict({"Estimate": result.params[1],
                                "Error": result.bse[1]})
    r2 = result.rsquared
    return OLSSingleVarEst(est_type, reg_type, const, param, r2)

##################################################################################################################
# Test
##################################################################################################################
class Test(Enum):
    STATIONARITY = "STATIONARITY"                  # Test for stationarity
    STATIONARITY_OFFSET = "STATIONARITY_OFFSET"    # Test for stationarity
    STATIONARITY_DRIFT = "STATIONARITY_DRIFT"      # Test for stationarity
    BM = "BM"                                      # Test for brownian motion
    AUTO_CORR = "AUTO_CORR"                        # Test for autocorrelated fractional brownian motion
    NEG_AUTO_CORR = "NEG_AUTO_CORR"                # Test for negative autocorrelated fractional brownian motion

    def perform(self, df, **kwargs):
        impl = self._impl()
        return impl.perform(df, self, **kwargs)

    def status(self, status):
        if self.value == Test.STATIONARITY.value:
            return not status[2]
        if self.value == Test.STATIONARITY_OFFSET.value:
            return not status[2]
        if self.value == Test.STATIONARITY_DRIFT.value:
            return not status[2]
        elif self.value == Test.BM.value:
            npass = 0
            for stat in status:
                if stat:
                    npass += 1
            return npass >= len(status)/2
        elif self.value == Test.AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return True
            return False
        elif self.value == Test.NEG_AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return True
            return False
        else:
            raise Exception(f"Test type is invalid: {self}")

    def _desc(self):
        if self.value == Test.STATIONARITY.value:
            return "Stationarity Test"
        if self.value == Test.STATIONARITY_OFFSET.value:
            return "Stationarity Test"
        if self.value == Test.STATIONARITY_DRIFT.value:
            return "Stationarity Test"
        elif self.value == Test.BM.value:
            return "Brownian Motion Test"
        elif self.value == Test.AUTO_CORR.value:
            return "Autocorrelation Test"
        elif self.value == Test.NEG_AUTO_CORR.value:
            return "Negative Autocorrelation Test"
        else:
            raise Exception(f"Test type is invalid: {self}")

    def _impl(self):
        if self.value == Test.STATIONARITY.value:
            return _TestImpl.ADF
        if self.value == Test.STATIONARITY_OFFSET.value:
            return _TestImpl.ADF_OFFSET
        if self.value == Test.STATIONARITY_DRIFT.value:
            return _TestImpl.ADF_DRIFT
        elif self.value == Test.BM.value:
            return _TestImpl.VR_TWO_TAILED
        elif self.value == Test.AUTO_CORR.value:
            return _TestImpl.VR_UPPER_TAIL
        elif self.value == Test.NEG_AUTO_CORR.value:
            return _TestImpl.VR_LOWER_TAIL
        else:
            raise Exception(f"Test type is invalid: {self}")

##################################################################################################################
# Test Implementation
##################################################################################################################
class _TestImpl(Enum):
    ADF = "ADF"                       # Augmented Dickey Fuller test for staionarity
    ADF_OFFSET = "ADF_OFFSET"         # Augmented Dickey Fuller with off set test test for staionarity
    ADF_DRIFT = "ADF_DRIFT"           # Augmented Dickey Fuller with drift test test for staionarity
    VR_TWO_TAILED = "VR_TWO_TAILED"   # Variance ratio test for brownian motion
    VR_LOWER_TAIL = "VR_LOWER_TAIL"   # Variance ratio test for anti-autocorrelated fractional brownian motion
    VR_UPPER_TAIL = "VR_UPPER_TAIL"   # Variance ratio test for autocorrelated fractional brownian motion

    def perform(self, df, test_type, **kwargs):
        x, y = DataSchema.get_schema_data(df)
        result, test = _perform_test_for_impl(x, y, test_type, self, **kwargs)
        MetaData.add_test(df, test)
        return result

##################################################################################################################
# Perform test forspecified implementaion
def _perform_test_for_impl(x, y, test_type, impl_type, **kwargs):
    if impl_type.value == _TestImpl.ADF.value:
        return _adf_test(y, test_type, impl_type, **kwargs)
    elif impl_type.value == _TestImpl.ADF_OFFSET.value:
        return _adf_offset_test(y, test_type, impl_type, **kwargs)
    elif impl_type.value == _TestImpl.ADF_DRIFT.value:
        return _adf_drift_test(y, test_type, impl_type, **kwargs)
    elif impl_type.value == _TestImpl.VR_TWO_TAILED.value:
        return _vr_test(y, TestHypothesis.TWO_TAIL, test_type, impl_type, **kwargs)
    elif impl_type.value == _TestImpl.VR_LOWER_TAIL.value:
        return _vr_test(y, TestHypothesis.LOWER_TAIL, test_type, impl_type, **kwargs)
    elif impl_type.value == _TestImpl.VR_UPPER_TAIL.value:
        return _vr_test(y, TestHypothesis.UPPER_TAIL, test_type, impl_type, **kwargs)
    else:
        raise Exception(f"Test type is invalid: {self}")

# _TestImpl.ADF
def _adf_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test(y)
    return result, _adf_report_from_result(result, test_type, impl_type)

# _TestImpl.ADF_OFFSET
def _adf_offset_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test_offset(y)
    return result, _adf_report_from_result(result, test_type, impl_type)

# _TestImpl.ADF_DRIFT
def _adf_drift_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test_drift(y)
    return result, _adf_report_from_result(result, test_type, impl_type)

# _TestImpl.VR_TWO_TAILED
def _vr_test(y, hypo_type, test_type, impl_type, **kwargs):
    sig_level = get_param_default_if_missing("sig_level", 0.1, **kwargs)
    s = get_param_default_if_missing("s", [4, 6, 10, 16, 24], **kwargs)
    verify_type(s, list)
    result = fbm.vr_test(y, s, sig_level, hypo_type)
    return result, _vr_report_from_result(result, test_type, impl_type)

##################################################################################################################
# Test Parameter
class TestParam:
    def __init__(self, label, value):
        self.label = label
        self.value = value
        self.dict = {"Value": value,
                     "Label": label}

    def __repr__(self):
        return f"TestParam({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"label=({self.label}), " \
               f"value=({self.value})"

    @staticmethod
    def from_dict(meta_data):
        return TestParam(label=meta_data["Label"],
                         value=meta_data["Value"])

##################################################################################################################
# Test Data
class TestData:
    def __init__(self, status, stat, pval, params, sig, lower, upper):
        self.status = status
        self.stat = stat
        self.pval = pval
        self.params = params
        self.sig = sig
        self.lower = lower
        self.upper = upper
        self.dict = {"Status": status,
                     "Statistic": stat.dict,
                     "PValue": pval.dict,
                     "Parameters": [param.dict for param in params],
                     "Significance": sig.dict,
                     "Lower Critical Value": lower.dict if lower is not None else None,
                     "Upper Critical Value": upper.dict if upper is not None else None}

    def __repr__(self):
        return f"TestData({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status=({self.status}), " \
               f"stat=({self.stat}), " \
               f"pval=({self.pval}, " \
               f"params=({self.params}), " \
               f"sig=({self.sig}), " \
               f"lower=({self.lower}), " \
               f"upper=({self.upper})"

    @staticmethod
    def from_dict(meta_data):
        lower = meta_data["Lower Critical Value"]
        upper = meta_data["Upper Critical Value"]
        return TestData(status=meta_data["Status"],
                        stat=TestParam.from_dict(meta_data["Statistic"]),
                        pval=TestParam.from_dict(meta_data["PValue"]),
                        params=[TestParam.from_dict(param) for param in meta_data["Parameters"]],
                        sig=TestParam.from_dict(meta_data["Significance"]),
                        lower=TestParam.from_dict(lower) if lower is not None else None,
                        upper=TestParam.from_dict(upper) if upper is not None else None)

##################################################################################################################
# Test Report
class TestReport:
    def __init__(self, status, hyp_type, test_type, impl_type, test_data, dist, **dist_params):
        self.status = status
        self.hyp_type = hyp_type
        self.test_type = test_type
        self.impl_type = impl_type
        self.test_data = test_data
        self.desc = test_type._desc()
        self.dist = dist
        self.dict = {"Status": status,
                     "TestHypothesis": hyp_type,
                     "TestType": test_type,
                     "ImplType": impl_type,
                     "Description": self.desc,
                     "Distribution": dist,
                     "Distribution Params": dist_params,
                     "TestData": [data.dict for data in test_data]}

    def __repr__(self):
        return f"TestReport({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status=({self.status}), " \
               f"hyp_type=({self.hyp_type}), " \
               f"test_type=({self.test_type}), " \
               f"impl_type=({self.impl_type}, " \
               f"desc=({self.desc}, " \
               f"test_data=({self.test_data})"

    def key(self):
        return self.test_type.value

    @staticmethod
    def from_dict(meta_data):
        return TestReport(status=meta_data["Status"],
                          hyp_type=meta_data["TestHypothesis"],
                          test_type=meta_data["TestType"],
                          impl_type=meta_data["ImplType"],
                          desc=meta_data["Description"],
                          dist=meta_data["Distribution"],
                          dist_params=meta_data["Distribution Params"],
                          test_data=[TestData.from_dict(data) for data in meta_data["TestData"]])

##################################################################################################################
# Create tests
def _create_dict_from_tests(tests):
    result = {}
    for key in tests.keys():
        result[key] = tests[key].dict
    return result

def _create_tests_from_dict(meta_data):
    result = {}
    for key in meta_data.keys():
        result[key] = _create_test_from_dict(meta_data[key])
    return result

def _create_test_from_dict(meta_data):
    return TestReport.from_dict(meta_data)

##################################################################################################################
# Constrct test report from result object
def _adf_report_from_result(result, test_type, impl_type):
    sigs = [TestParam(label=result.sig_str[i], value=result.sig[i]) for i in range(3)]
    stat = TestParam(label=r"$t$", value=result.stat)
    pval = TestParam(label=r"$p-value$", value = result.pval)
    lower_vals = [TestParam(label=r"$t_L$", value=val) for val in result.critical_vals]
    test_data = []
    for i in range(3):
        data = TestData(status=result.status_vals[i],
                        stat=stat,
                        pval=pval,
                        params=[],
                        sig=sigs[i],
                        lower=lower_vals[i],
                        upper=None)
        test_data.append(data)
    return TestReport(status=test_type.status(result.status_vals),
                      hyp_type=TestHypothesis.LOWER_TAIL,
                      test_type=test_type,
                      impl_type=impl_type,
                      test_data=test_data,
                      dist=None)

def _vr_report_from_result(result, test_type, impl_type):
    sig = TestParam(label=f"{int(100.0*result.sig_level)}%", value=result.sig_level)
    s_vals = [TestParam(label=r"$s$", value=s) for s in result.s_vals]
    stats = [TestParam(label=r"$Z(s)$", value=stat) for stat in result.stats]
    pvals = [TestParam(label=r"$p-value$", value=pval) for pval in result.p_vals]
    lower = result.critical_values[0]
    if lower is not None:
        lower = TestParam(label=r"$Z_L(s)$", value=lower)
    upper = result.critical_values[1]
    if upper is not None:
        upper = TestParam(label=r"$Z_U(s)$", value=upper)
    test_data = []
    for i in range(len(s_vals)):
        data = TestData(status=result.status_vals[i],
                        stat=stats[i],
                        pval=pvals[i],
                        params=[s_vals[i]],
                        sig=sig,
                        lower=lower,
                        upper=upper)
        test_data.append(data)
    return TestReport(status=test_type.status(result.status_vals),
                      hyp_type=result.hyp_type,
                      test_type=test_type,
                      impl_type=impl_type,
                      test_data=test_data,
                      dist=Dist.NORMAL,
                      loc=0.0,
                      scale=1.0)
