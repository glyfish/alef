from enum import Enum
from pandas import DataFrame
from datetime import datetime
import uuid
import numpy

from lib.models import bm
from lib.data.func import (DataFunc, FuncBase)

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create Fractional Brownian Motion
# Func.PSPEC
class BM:
    class Func(FuncBase):
        MEAN = "BM_MEAN"                    # Brownian Motion mean
        DRIFT_MEAN = "BM_DRIFT_MEAN"        # Brownian Motion model mean with data
        SD = "BM_SD"                        # Brownian Motion model standard deviation
        GBM_MEAN = "BM_GBM_MEAN"            # Geometric Brownian Motion model mean
        GBM_SD = "BM_GBM_SD"                # Geometric Brownian Motion model standard deviation
        BM = "BM_BM"                        # Brownian motion computed from brownian motion noise

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

###################################################################################################
## create function definition for data type
def _create_func(func_type, **kwargs):
    if func_type.value == BM.Func.MEAN.value:
        return _create_bm_mean(func_type, **kwargs)
    elif func_type.value == BM.Func.DRIFT_MEAN.value:
        return _create_bm_drift_mean(func_type, **kwargs)
    elif func_type.value == BM.Func.SD.value:
        return _create_bm_sd(func_type, **kwargs)
    elif func_type.value == BM.Func.GBM_MEAN.value:
        return _create_gbm_mean(func_type, **kwargs)
    elif func_type.value == BM.Func.GBM_SD.value:
        return _create_gbm_sd(func_type, **kwargs)
    elif func_type.value == BM.Func.BM.value:
        return _create_bm(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
# bm
# Func.BM_MEAN
def _create_bm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.full(len(x), μ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM Mean",
                    formula=r"$0$",
                    fy=fy,
                    fx=fx)

# Func.BM_DRIFT_MEAN
def _create_bm_drift_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : μ*x
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    formula=r"$\mu t$",
                    fy=fy,
                    fx=fx)

# Func.BM_SD
def _create_bm_sd(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : σ*numpy.sqrt(x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ},
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    formula=r"$\sqrt{t}$",
                    fy=fy,
                    fx=fx)

# Func.GBM_MEAN
def _create_gbm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : S0*numpy.exp(μ*x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ, "S0": S0},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM Mean",
                    formula=r"$S_0 e^{\mu t}$",
                    fy=fy,
                    fx=fx)

# Func.GBM_SD
def _create_gbm_sd(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.sqrt(S0**2*numpy.exp(2*μ*x)*(numpy.exp(x*σ**2)-1))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ, "μ": μ, "S0": S0},
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM SD",
                    formula=r"$S_0^2 e^{2\mu t}\left( e^{\sigma^2 t} - 1 \right)$",
                    fy=fy,
                    fx=fx)

# Func.BM
def _create_bm(func_type, **kwargs):
    fy = lambda x, y : stats.from_noise(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$S_t$",
                    xlabel=r"$t$",
                    desc="BM",
                    fy=fy)
