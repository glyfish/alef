from enum import Enum
from pandas import DataFrame
from datetime import datetime
import uuid
import numpy

from lib import stats
from lib.models import arima

from lib.data.meta_data import (MetaData)
from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create ARIMA Functions
class ARIMA:
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

###################################################################################################
## create function definition for data type
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
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
# arima
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
