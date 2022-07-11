from enum import Enum
import uuid
import numpy

from lib.models import bm

from lib.data.func import (DataFunc, FuncBase)
from lib.data.source import (DataSource, SourceBase)
from lib.data.schema import (DataType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# BM Funcs and Sources
class BM:
    # Funcs
    class Func(FuncBase):
        MEAN = "BM_MEAN"                    # Brownian Motion mean
        DRIFT_MEAN = "BM_DRIFT_MEAN"        # Brownian Motion model mean with data
        SD = "BM_SD"                        # Brownian Motion model standard deviation
        GBM_MEAN = "BM_GBM_MEAN"            # Geometric Brownian Motion model mean
        GBM_SD = "BM_GBM_SD"                # Geometric Brownian Motion model standard deviation
        BM = "BM_BM"                        # Brownian motion computed from brownian motion noise

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Sources
    class Source(SourceBase):
        NOISE = "BM_NOISE"                # Brownian Motion noise simulation
        MOTION = "BM_MOTION"              # Brownian Motion simulation
        DRIFT_MOTION= "BM_DRIFT_MOTION"   # Brownoan Motion with drift simulation
        GEO_MOTION = "BM_GEO_MOTION"      # Geometric Brownian motion simulation

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

###################################################################################################
## Create DataFunc object for func type
###################################################################################################
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
        raise Exception(f"func_type is invalid: {func_type}")

###################################################################################################
# Func.MEAN
def _create_bm_mean(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
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

# Func.DRIFT_MEAN
def _create_bm_drift_mean(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
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

# Func.SD
def _create_bm_sd(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
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
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
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
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
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

###################################################################################################
# Create DataSource objects for specified DataType
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == BM.Source.NOISE.value:
        return _create_bm_noise_source(source_type, x, **kwargs)
    elif source_type.value == BM.Source.MOTION.value:
        return _create_bm_source(source_type, x, **kwargs)
    elif source_type.value == BM.Source.DRIFT_MOTION.value:
        return _create_bm_drift_source(source_type, x, **kwargs)
    elif source_type.value == BM.Source.GEO_MOTION.value:
        return _create_bm_geo_source(source_type, x, **kwargs)
    else:
        raise Exception(f"source_type is invalid: {source_type}")

###################################################################################################
# Source.NOISE
def _create_bm_noise_source(source_type, x, **kwargs):
    f = lambda x : bm.noise(len(x))
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={},
                      ylabel=r"$\Delta S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Noise",
                      f=f,
                      x=x)

# Source.MOTION
def _create_bm_source(source_type, x, **kwargs):
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm(len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      params={"Δx": Δx},
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion",
                      f=f,
                      x=x)

# Source.DRIFT_MOTION
def _create_bm_drift_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_with_drift(μ, σ, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion With Drift",
                      f=f,
                      x=x)

# Source.GEO_MOTION
def _create_bm_geo_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_geometric(μ, σ, S0, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Geometric-BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx, "S0": S0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Geometric Brownian Motion",
                      f=f,
                      x=x)
