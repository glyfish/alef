from enum import Enum
import numpy

from lib import stats
from lib.data.func import (DataFunc, FuncBase)
from lib.data.schema import (DataType)
from lib.data.source import (DataSource, SourceBase)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create Ornstien-Uhlenbeck Process Functions
class OU:
    # Funcs
    class Func(FuncBase):
        MEAN = "OU_MEAN"                    # Ornstein-Uhelenbeck process mean
        VAR = "OU_VAR"                      # Ornstein-Uhelenbeck process variance
        COV = "OU_COV"                      # Ornstein-Uhelenbeck process covariance
        PDF = "OU_PDF"                      # Ornstein-Uhelenbeck process PDF
        CDF = "OU_CDF"                      # Ornstein-Uhelenbeck process CDF
        PDF_LIMIT = "OU_PDF"                # Ornstein-Uhelenbeck process PDF
        CDF_LIMIT = "OU_CDF"                # Ornstein-Uhelenbeck process CDF

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Sources
    class Source(SourceBase):
        PROC = "OU_PROC"                     # Ornstein-Uhlenbeck process simulation
        XT = "OU_XT"                         # Ornstein-Uhlenbeck process solution

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

###################################################################################################
## create DataFunc for func_type
###################################################################################################
def _create_func(func_type, **kwargs):
    if source_type.value == OU.MEAN.value:
        _create_ou_mean(source_type, x, **kwargs)
    else:
        Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Source.MEAN
def _create_ou_mean(source_type, x, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    f = lambda x : ou.ou(μ, λ, Δt, len(x), σ, x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="Ensemble ACF",
                    fy=fy,
                    fx=fx)

###################################################################################################
## create DataSource object for source_type
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == OU.Source.XT.value:
        return _create_xt_source(source_type, x, **kwargs)
    elif source_type.value == OU.Source.PROC.value:
        return _create_proc_source(source_type, x, **kwargs)
    else:
        raise Exception(f"Source type is invalid: {source_type}")

###################################################################################################
# Source.XT
def _create_xt_source(source_type, x, **kwargs):
    t = get_param_throw_if_missing("t", **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    f = lambda x : ou.xt(μ, λ, t, σ, x0, len(x))
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Ornstein-Uhlenbeck-Simulation-{str(uuid.uuid4())}",
                      params={"μ": μ, "λ": λ, "t": t, "X0": x0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Ornstein-Uhlenbeck Simulation",
                      f=f,
                      x=x)

# Source.PROC
def _create_proc_source(source_type, x, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δx", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    f = lambda x : ou.ou(μ, λ, Δt, len(x), σ, x0)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Ornstein-Uhlenbeck-Simulation-{str(uuid.uuid4())}",
                      params={"μ": μ, "λ": λ, "Δt": Δt, "X0": x0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Ornstein-Uhlenbeck Simulation",
                      f=f,
                      x=x)
