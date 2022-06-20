from enum import Enum
import uuid
import numpy

from lib.models import arima

from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.source import (DataSource, SourceBase)
from lib.data.schema import (DataType)
from lib.data.meta_data import (EstBase, TestBase, TestImplBase,
                                TestParam, TestData, TestReport,
                                ParamEst, ARMAEst)
from lib.models import (TestHypothesis)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create ARIMA Functions
class ARIMA:
    # Funcs
    class Func(FuncBase):
        PACF = "PACF"                          # Partial Autocorrelation function
        MEAN = "ARMA_MEAN"                     # ARMA(p,q) MEAN
        AR1_ACF = "AR1_ACF"                    # AR(1) Autocorrelation function
        MAQ_ACF = "MAQ_ACF"                    # MA(q) Autocorrelation function
        AR1_SD = "AR1_SD"                      # AR(1) standard seviation
        MAQ_SD = "MAQ_SD"                      # MA(q) standard deviation
        AR1_OFFSET_MEAN = "AR1_OFFSET_MEAN"    # AR(1) with constant offset mean
        AR1_OFFSET_SD = "AR1_OFFSET_SD"        # AR(1) with offset standard deviation

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Sources
    class Source(SourceBase):
        AR = "AR"                                      # AR(p) simulation
        AR_DRIFT = "AR_DRIFT"                          # AR(p) with drift
        AR_OFFSET = "AR_OFFSET"                        # AR(p) with offset
        MA = "MA"                                      # MA(q) simulation
        ARMA = "ARMA"                                  # ARMA(p, q) simulation
        ARIMA = "ARIMA"                                # ARIMA(p, d, q) simulation
        ARIMA_FROM_ARMA = "ARIMA_FROM_ARMA"            # ARIMA(p, d, q) simulation created from ARMA(p,q)

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

    # Test
    class Test(TestBase):
        STATIONARITY = "STATIONARITY"                  # Test for stationarity
        STATIONARITY_OFFSET = "STATIONARITY_OFFSET"    # Test for stationarity
        STATIONARITY_DRIFT = "STATIONARITY_DRIFT"      # Test for stationarity

        def status(self, status):
            if self.value == ARIMA.Test.STATIONARITY.value:
                return not status[2]
            if self.value == ARIMA.Test.STATIONARITY_OFFSET.value:
                return not status[2]
            if self.value == ARIMA.Test.STATIONARITY_DRIFT.value:
                return not status[2]
            else:
                raise Exception(f"Test type is invalid: {self}")

        def _desc(self):
            return "Stationarity Test"

        def _impl(self):
            if self.value == ARIMA.Test.STATIONARITY.value:
                return ARIMA._TestImpl.ADF
            if self.value == ARIMA.Test.STATIONARITY_OFFSET.value:
                return ARIMA._TestImpl.ADF_OFFSET
            if self.value == ARIMA.Test.STATIONARITY_DRIFT.value:
                return ARIMA._TestImpl.ADF_DRIFT
            else:
                raise Exception(f"Test type is invalid: {self}")

    # _TestImpl
    class _TestImpl(TestImplBase):
        ADF = "ADF"                          # Augmented Dickey Fuller test for staionarity
        ADF_OFFSET = "ADF_OFFSET"            # Augmented Dickey Fuller with off set test test for staionarity
        ADF_DRIFT = "ADF_DRIFT"              # Augmented Dickey Fuller with drift test test for staionarity

        def _perform_test_for_impl(self, x, y, test_type, **kwargs):
            return _perform_test_for_impl(x, y, test_type, self, **kwargs)

    # Est
    class Est(EstBase):
        AR = "AR"                     # Autoregressive model parameters
        AR_OFFSET = "AR_OFFSET"       # Autoregressive model with constant offset parameters
        MA = "MA"                     # Moving average model parameters
        MA_OFFSET = "MA_OFFSET"       # Moving average model  with constant offset parameters

        def _perform_est_for_type(self, x, y, **kwargs):
            if self.value == ARIMA.Est.AR.value:
                return _ar_estimate(y, **kwargs)
            elif self.value == ARIMA.Est.AR_OFFSET.value:
                return _ar_offset_estimate(y, **kwargs)
            elif self.value == ARIMA.Est.MA.value:
                return _ma_estimate(y, **kwargs)
            elif self.value == ARIMA.Est.MA_OFFSET.value:
                return _ma_offset_estimate(y, **kwargs)
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

        def _formula(self):
            if self.type.value == ARIMA.Est.AR.value:
                return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
            elif self.type.value == ARIMA.Est.AR_OFFSET.value:
                return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
            elif self.type.value == ARIMA.Est.MA.value:
                return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
            elif self.type.value == ARIMA.Est.MA_OFFSET.value:
                return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

        def _set_param_labels(self, param, i):
            if self.value == ARIMA.Est.AR.value or self.value == ARIMA.Est.AR_OFFSET.value:
                param.set_labels(est_label=f"$\hat{{\phi_{{{i}}}}}$",
                                 err_label=f"$\sigma_{{$\hat{{\phi_{{{i}}}}}}}$")
            elif self.value == ARIMA.Est.MA.value or self.value == ARIMA.Est.MA_OFFSET.value:
                param.set_labels(est_label=f"$\hat{{\\theta_{{{i}}}}}$",
                                 err_label=f"$\sigma_{{$\hat{{\theta_{{{i}}}}}}}$")
            else:
                raise Exception(f"Esitmate type is invalid: {est_type}")

###################################################################################################
## Create function definition for data type
###################################################################################################
def _create_func(func_type, **kwargs):
    if func_type.value == ARIMA.Func.PACF.value:
        return _create_pacf(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.MEAN.value:
        return _create_arma_mean(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.AR1_ACF.value:
        return _create_ar1_acf(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.MAQ_ACF.value:
        return _create_maq_acf(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.AR1_SD.value:
        return _create_ar1_sd(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.MAQ_SD.value:
        return _create_maq_sd(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.AR1_OFFSET_MEAN.value:
        return _create_ar1_offset_mean(func_type, **kwargs)
    elif func_type.value == ARIMA.Func.AR1_OFFSET_SD.value:
        return _create_ar1_offset_sd(func_type, **kwargs)
    else:
        raise Exception(f"func_type is invalid: {func_type}")

###################################################################################################
# Func.PACF
def _create_pacf(func_type, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[1:nlags+1]
    fy = lambda x, y : arima.yw(y, nlags)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.TIME_SERIES,
                    params={"nlags": nlags},
                    ylabel=r"$\varphi_\tau$",
                    xlabel=r"$\tau$",
                    desc="PACF",
                    fy=fy,
                    fx=fx)

# Func.AR1_ACF
def _create_ar1_acf(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[:nlags +1 ]
    fy = lambda x, y : φ**x
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"φ": φ},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="AR(1) ACF",
                    fy=fy,
                    fx=fx,
                    formula=r"$\varphi^\tau$")

# Func.MAQ_ACF
def _create_maq_acf(func_type, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)
    fx = lambda x : x[:nlags]
    fy = lambda x, y : arima.maq_acf(θ, σ, len(x))
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"θ": θ, "σ": σ},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc=f"MA({len(θ)}) ACF",
                    fy=fy,
                    fx=fx,
                    formula=r"$\sigma^2 \left( \sum_{i=i}^{q-\tau} \vartheta_i \vartheta_{i+\tau} + \vartheta_\tau \right)$")

# Func.ARMA_MEAN
def _create_arma_mean(func_type, **kwargs):
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\mu_\infty$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc="ARMA(p,q) Mean",
                    fy=fy)

# Func.AR1_SD
def _create_ar1_sd(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_sigma(φ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, "σ": σ},
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\sigma^2}{1-\varphi^2}$",
                    desc="AR(1) SD",
                    fy=fy)

# Func.MAQ_SD
def _create_maq_sd(func_type, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.maq_sigma(θ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"θ": θ, "σ": σ},
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\sigma^2 \left( \sum_{i=1}^q \vartheta_i^2 + 1 \right)$",
                    desc="MA(q) SD",
                    fy=fy)

# Func.AR1_OFFSET_MEAN
def _create_ar1_offset_mean(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_offset_mean(φ, μ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, r"$μ^*$": μ},
                    ylabel=r"$\mu_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\mu^*}{1 - \varphi}$",
                    desc="AR(1) with Offset Mean",
                    fy=fy)

# Func.AR1_OFFSET_SD
def _create_ar1_offset_sd(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_offset_sigma(φ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, "σ": σ},
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\sqrt{\frac{\sigma^2}{1 - \varphi^2}}$",
                    desc="AR(1) with Offset Mean",
                    fy=fy)

###################################################################################################
## Create data source for specified type
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == ARIMA.Source.AR.value:
        return _create_ar_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.AR_DRIFT.value:
        return _create_ar_drift_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.AR_OFFSET.value:
        return _create_ar_offset_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.MA.value:
        return _create_ma_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.ARMA.value:
        return _create_arma_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.ARIMA.value:
        return _create_arima_source(source_type, x, **kwargs)
    elif source_type.value == ARIMA.Source.ARIMA_FROM_ARMA.value:
        return _create_arima_from_arma_source(source_type, x, **kwargs)
    else:
        raise Exception(f"source_type is invalid: {source_type}")

###################################################################################################
# Source.AR
def _create_ar_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp(numpy.array(φ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# Source.AR_DRIFT
def _create_ar_drift_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_drift(numpy.array(φ), μ, γ, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-DRIFT-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "μ": μ, "γ": γ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# Source.AR_OFFSET
def _create_ar_offset_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_offset(numpy.array(φ), μ, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-Offset-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "μ": μ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# Source.MA
def _create_ma_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)
    f = lambda x : arima.maq(numpy.array(θ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"MA({len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"MA({len(θ)})",
                      f=f,
                      x=x)

# Source.ARMA
def _create_arma_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arma(numpy.array(φ), numpy.array(θ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARMA({len(φ)}, {len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "φ": φ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARMA({len(φ)},{len(θ)})",
                      f=f,
                      x=x)

# Source.ARIMA
def _create_arima_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arima(numpy.array(φ), numpy.array(θ), d, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARIMA({len(φ)}, {d}, {len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "φ": φ, "σ": σ, "d": d},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARIMA({len(φ)},{d},{len(θ)})",
                      f=f,
                      x=x)

# Source.ARIMA_FROM_ARMA
def _create_arima_from_arma_source(source_type, x, **kwargs):
    samples_df = get_param_throw_if_missing("arma", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    samples_schema = DataType.TIME_SERIES.schema()
    _, samples = samples_schema.get_data(samples_df)
    f = lambda x : arima.arima_from_arma(samples, d)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARIMA(p, {d}, q)-Simulation-{str(uuid.uuid4())}",
                      params={"d": d},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARIMA({d}) from ARMA",
                      f=f,
                      x=x)

##################################################################################################################
# Perform test for specified implementaion
def _perform_test_for_impl(x, y, test_type, impl_type, **kwargs):
    if impl_type.value == ARIMA._TestImpl.ADF.value:
        return _adf_test(y, test_type, impl_type, **kwargs)
    elif impl_type.value == ARIMA._TestImpl.ADF_OFFSET.value:
        return _adf_offset_test(y, test_type, impl_type, **kwargs)
    elif impl_type.value == ARIMA._TestImpl.ADF_DRIFT.value:
        return _adf_drift_test(y, test_type, impl_type, **kwargs)
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

##################################################################################################################
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

##################################################################################################################
# Perform estimate for specified estimate types
##################################################################################################################
# Est.AR
def _ar_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_fit(samples, order)
    return result, _arma_estimate_from_result(result, ARIMA.Est.AR)

# Est.AR_OFFSET
def _ar_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, ARIMA.Est.AR_OFFSET)

# Est.MA
def _ma_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_fit(samples, order)
    return result, _arma_estimate_from_result(result, ARIMA.Est.MA)

# Est.MA_OFFSET
def _ma_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_offset_fit(samples, order)
    return result, _arma_estimate_from_result(result, ARIMA.Est.MA_OFFSET)

##################################################################################################################
# Construct estimate objects from result object
def _arma_estimate_from_result(result, est_type):
    nparams = len(result.params)
    params = []
    for i in range(1, nparams-1):
        params.append(ParamEst.from_dict({"Estimate": result.params.iloc[i],
                                          "Error": result.bse.iloc[i]}))
    const = ParamEst.from_dict({"Estimate": result.params.iloc[0],
                                "Error": result.bse.iloc[0]})
    sigma2 = ParamEst.from_dict({"Estimate": result.params.iloc[nparams-1],
                                 "Error": result.bse.iloc[nparams-1]})
    return ARMAEst(est_type, const, sigma2, params)
