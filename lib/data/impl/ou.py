from enum import Enum
import uuid
import numpy

from lib.models import ou

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
        MEAN = "OU_MEAN"                      # Ornstein-Uhelenbeck process mean
        VAR = "OU_VAR"                        # Ornstein-Uhelenbeck process variance
        COV = "OU_COV"                        # Ornstein-Uhelenbeck process covariance
        PDF = "OU_PDF"                        # Ornstein-Uhelenbeck process PDF
        CDF = "OU_CDF"                        # Ornstein-Uhelenbeck process CDF
        PDF_LIMIT = "OU_PDF"                  # Ornstein-Uhelenbeck process PDF t->infty limit
        CDF_LIMIT = "OU_CDF"                  # Ornstein-Uhelenbeck process CDF t->infty limit
        MEAN_HALF_LIFE = "OU_MEAN_HALF_LIFE"  # Ornstein-Uhelenbeck process halflife

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
        _create_ou_mean(source_type, **kwargs)
    elif source_type.value == OU.VAR.value:
        _create_ou_var(source_type, **kwargs)
    elif source_type.value == OU.COV.value:
        _create_ou_cov(source_type, **kwargs)
    elif source_type.value == OU.PDF.value:
        _create_ou_pdf(source_type, **kwargs)
    elif source_type.value == OU.CDF.value:
        _create_ou_cdf(source_type, **kwargs)
    elif source_type.value == OU.PDF_LIMIT.value:
        _create_ou_pdf_limit(source_type, **kwargs)
    elif source_type.value == OU.CDF_LIMIT.value:
        _create_ou_cdf_limit(source_type, **kwargs)
    else:
        Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Func.MEAN
def _create_ou_mean(func_type, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x : ou.mean(μ, λ, x, x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=source_type,
                    params={"μ": μ, "λ": λ, "x0": x0},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    formula=r"$X_0 e^{-\lambda t} + \mu \left( 1 - e^{-\lambda t} \right)\hspace{89pt}$",
                    desc="Ornstein-Uhlenbeck Mean",
                    fy=fy)

# Func.VAR
def _create_ou_var(func_type, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    fy = lambda x : ou.var(λ, x, σ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ},
                    ylabel=r"$\sigma^2_t$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\sigma^2}{2\lambda} \left( 1 - e^{-2\lambda t} \right)$",
                    desc="Ornstein-Uhlenbeck Var",
                    fy=fy)

# Func.COV
def _create_ou_cov(source_type, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    s = get_param_default_if_missing("s", 1.0, **kwargs)
    fx = lambda x : x[s:]
    fy = lambda x : ou.cov(λ, s, x, σ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$\Cov(S_s, S_t)$",
                    xlabel=r"$t$",
                    formula=r"$ \frac{\sigma^2}{2\lambda} \left[ e^{-\lambda \left( t-s \right)} - e^{-\lambda \left( t+s \right)} \right]$",
                    desc="Ornstein-Uhlenbeck Covariance",
                    fy=fy,
                    fx=fx)

# Func.PDF
def _create_ou_pdf(source_type, **kwargs):
    t = get_param_throw_if_missing("t", **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x : ou.pdf(x, μ, λ, t, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$p(x)$",
                    xlabel=r"$x$",
                    desc="Ornstein-Uhlenbeck PDF",
                    fy=fy)

# Func.CDF
def _create_ou_cdf(source_type, **kwargs):
    t = get_param_throw_if_missing("t", **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x : ou.cdf(x, μ, λ, t, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$P(x)$",
                    xlabel=r"$x$",
                    desc="Ornstein-Uhlenbeck CDF",
                    fy=fy)

# Func.PDF_LIMIT
def _create_ou_pdf_limit(source_type, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x : ou.pdf_limit(x, μ, λ, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$p(x)$",
                    xlabel=r"$x$",
                    desc=r"Ornstein-Uhlenbeck $t\to \infty$ PDF",
                    fy=fy)

# Func.CDF_LIMIT
def _create_ou_cdf_limit(source_type, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x : ou.cdf_limit(x, μ, λ, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$P(x)$",
                    xlabel=r"$x$",
                    desc=r"Ornstein-Uhlenbeck $t\to \infty$ CDF",
                    fy=fy)

# Func.MEAN_HALF_LIFE
def _create_ou_mean_half_life(source_type, **kwargs):
    fy = lambda x : ou.mean_halflife(x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=source_type,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$t_H(\lambda)$",
                    xlabel=r"$\lambda$",
                    desc=r"Ornstein-Uhlenbeck $t\to \infty$ CDF",
                    fy=fy)

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
                      desc=f"Ornstein-Uhlenbeck Solution",
                      f=f,
                      x=x)

# Source.PROC
def _create_proc_source(source_type, x, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    f = lambda x : ou.ou(μ, λ, Δt, len(x), σ, x0)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Ornstein-Uhlenbeck-Simulation-{str(uuid.uuid4())}",
                      params={"μ": μ, "λ": λ, "Δt": Δt, "X0": x0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Ornstein-Uhlenbeck Process",
                      f=f,
                      x=x)
