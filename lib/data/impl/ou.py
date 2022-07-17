from enum import Enum
import uuid
import numpy

from lib.models import ou

from lib.data.func import (DataFunc, FuncBase)
from lib.data.schema import (DataType)
from lib.data.source import (DataSource, SourceBase)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)
from lib.data.meta_data import (EstBase, ARMAEst)

###################################################################################################
# Create Ornstien-Uhlenbeck Process Functions
class OU:
    # Funcs
    class Func(FuncBase):
        MEAN = "OU_MEAN"                      # Ornstein-Uhelenbeck process mean
        VAR = "OU_VAR"                        # Ornstein-Uhelenbeck process variance
        COV = "OU_COV"                        # Ornstein-Uhelenbeck process covariance
        MEAN_LIMIT = "OU_MEAN_LIMIT"          # Ornstein-Uhelenbeck process mean t -> infty
        VAR_LIMIT = "OU_VAR_LIMIT"            # Ornstein-Uhelenbeck process var t -> infty
        COV_LIMIT = "OU_COV_LIMIT"            # Ornstein-Uhelenbeck process covariance
        PDF = "OU_PDF"                        # Ornstein-Uhelenbeck process PDF
        CDF = "OU_CDF"                        # Ornstein-Uhelenbeck process CDF
        PDF_LIMIT = "OU_PDF_LIMIT"            # Ornstein-Uhelenbeck process PDF t->infty limit
        CDF_LIMIT = "OU_CDF_LIMIT"            # Ornstein-Uhelenbeck process CDF t->infty limit
        MEAN_HALF_LIFE = "OU_MEAN_HALF_LIFE"  # Ornstein-Uhelenbeck process halflife

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Sources
    class Source(SourceBase):
        PROC = "OU_PROC"                     # Ornstein-Uhlenbeck process simulation
        XT = "OU_XT"                         # Ornstein-Uhlenbeck process solution

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

    # Est
    class Est(EstBase):
        AR = "OU_AR"                        # Use Autoregressive parameter estimation

        def _perform_est_for_type(self, x, y, **kwargs):
            if self.value == OU.Est.AR.value:
                return _ar_estimate(y, **kwargs)
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

        def _formula(self):
            if self.value == OU.Est.AR.value:
                return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

        def _set_param_labels(self, param, i):
            if self.value == OU.Est.AR.value:
                param.set_labels(est_label=f"$\hat{{\phi_{{{i}}}}}$",
                                 err_label=f"$\sigma_{{$\hat{{\phi_{{{i}}}}}}}$")
            else:
                raise Exception(f"Esitmate type is invalid: {est_type}")

###################################################################################################
## create DataFunc for func_type
###################################################################################################
def _create_func(func_type, **kwargs):
    if func_type.value == OU.Func.MEAN.value:
        return _create_ou_mean(func_type, **kwargs)
    elif func_type.value == OU.Func.MEAN_LIMIT.value:
        return _create_ou_mean_limit(func_type, **kwargs)
    elif func_type.value == OU.Func.VAR.value:
        return _create_ou_var(func_type, **kwargs)
    elif func_type.value == OU.Func.VAR_LIMIT.value:
        return _create_ou_var_limit(func_type, **kwargs)
    elif func_type.value == OU.Func.COV.value:
        return _create_ou_cov(func_type, **kwargs)
    elif func_type.value == OU.Func.COV_LIMIT.value:
        return _create_ou_cov_limit(func_type, **kwargs)
    elif func_type.value == OU.Func.PDF.value:
        return _create_ou_pdf(func_type, **kwargs)
    elif func_type.value == OU.Func.CDF.value:
        return _create_ou_cdf(func_type, **kwargs)
    elif func_type.value == OU.Func.PDF_LIMIT.value:
        return _create_ou_pdf_limit(func_type, **kwargs)
    elif func_type.value == OU.Func.CDF_LIMIT.value:
        return _create_ou_cdf_limit(func_type, **kwargs)
    else:
        Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Func.MEAN
def _create_ou_mean(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
    fy = lambda x, y : ou.mean(μ, λ, x, x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ, "λ": λ, "x0": x0},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    formula=r"$X_0 e^{-\lambda t} + \mu \left( 1 - e^{-\lambda t} \right)$",
                    desc=r"Ornstein-Uhlenbeck $\mu_t$",
                    fy=fy,
                    fx=fx)

# Func.MEAN_LIMIT
def _create_ou_mean_limit(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
    fy = lambda x, y : numpy.full(len(x), μ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ},
                    ylabel=r"$\lim_{t \to \infty} \mu_t$",
                    xlabel=r"$t$",
                    formula=r"$\mu$",
                    desc=r"Ornstein-Uhlenbeck $\lim_{t \to \infty}{\mu_t}$",
                    fy=fy,
                    fx=fx)

# Func.VAR
def _create_ou_var(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
    fy = lambda x, y : ou.var(λ, x, σ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ, "λ": λ},
                    ylabel=r"$\sigma^2_t$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\sigma^2}{2\lambda} \left( 1 - e^{-2\lambda t} \right)$",
                    desc=r"Ornstein-Uhlenbeck $\sigma^2_t$",
                    fy=fy,
                    fx=fx)

# Func.VAR_LIMIT
def _create_ou_var_limit(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(npts - 1))]
    fy = lambda x, y : numpy.full(len(x), ou.var_limit(λ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ, "λ": λ},
                    ylabel=r"$\lim_{t \to \infty} \sigma^2_t$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\sigma^2}{2\lambda}$",
                    desc=r"Ornstein-Uhlenbeck $\limit_{t \to \infty} \sigma^2_t$",
                    fy=fy,
                    fx=fx)

# Func.COV
def _create_ou_cov(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    s = get_param_default_if_missing("s", 1.0, **kwargs)
    x0 = int(s/Δt)
    step = lambda x : 1 if len(x)-x0 < npts - 1 else int((len(x)-x0)/(npts-1))
    fx = lambda x : x[x0::step(x)]
    fy = lambda x, y : ou.cov(λ, s, x, σ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ, "λ": λ, "s": s},
                    ylabel=r"$Cov(S_s, S_t)$",
                    xlabel=r"$t$",
                    formula=r"$ \frac{\sigma^2}{2\lambda} \left[ e^{-\lambda \left( t-s \right)} - e^{-\lambda \left( t+s \right)} \right]$",
                    desc=r"Ornstein-Uhlenbeck Covariance",
                    fy=fy,
                    fx=fx)

# Func.COV_LIMIT
def _create_ou_cov_limit(func_type, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    s = get_param_default_if_missing("s", 1.0, **kwargs)
    step = lambda x : 1 if len(x)-x0 < npts - 1 else int((len(x)-x0)/(npts-1))
    fx = lambda x : x[x0::step(x)]
    fy = lambda x, y : numpy.full(npts, 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"s": s},
                    ylabel=r"$\lim_{t \to \infty} Cov(S_s, S_t)$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc=r"Ornstein-Uhlenbeck $t \to \infty$ Covariance",
                    fy=fy,
                    fx=fx)

# Func.PDF
def _create_ou_pdf(func_type, **kwargs):
    t = get_param_throw_if_missing("t", **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x, y : ou.pdf(x, μ, λ, t, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=None,
                    params={"σ": σ, "λ": λ, "t": t, "μ": μ, "x0": x0},
                    ylabel=r"$p(x)$",
                    xlabel=r"$x$",
                    formula=r"$Normal(\mu_t, \sigma_t)$",
                    desc="Ornstein-Uhlenbeck PDF",
                    fy=fy)

# Func.CDF
def _create_ou_cdf(func_type, **kwargs):
    t = get_param_throw_if_missing("t", **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x, y : ou.cdf(x, μ, λ, t, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=None,
                    params={"σ": σ, "λ": λ, "t": t, "μ": μ, "x0": x0},
                    ylabel=r"$P(x)$",
                    xlabel=r"$x$",
                    formula=r"$Normal(\mu_t, \sigma_t)$",
                    desc="Ornstein-Uhlenbeck CDF",
                    fy=fy)

# Func.PDF_LIMIT
def _create_ou_pdf_limit(func_type, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x, y : ou.pdf_limit(x, μ, λ, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=None,
                    params={"σ": σ, "λ": λ, "μ": μ, "x0": x0},
                    ylabel=r"$p(x)$",
                    xlabel=r"$x$",
                    formula=r"$Normal(\mu_t, \sigma_t)$",
                    desc=r"Ornstein-Uhlenbeck $t\to \infty$ PDF",
                    fy=fy)

# Func.CDF_LIMIT
def _create_ou_cdf_limit(func_type, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    fy = lambda x, y : ou.cdf_limit(x, μ, λ, σ=σ, x0=x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=v,
                    params={"σ": σ, "λ": λ, "μ": μ, "x0": x0},
                    ylabel=r"$P(x)$",
                    xlabel=r"$x$",
                    formula=r"$Normal(\mu_t, \sigma_t)$",
                    desc=r"Ornstein-Uhlenbeck $t\to \infty$ CDF",
                    fy=fy)

# Func.MEAN_HALF_LIFE
def _create_ou_mean_half_life(func_type, **kwargs):
    fy = lambda x, y : ou.mean_halflife(x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$t_H(\lambda)$",
                    xlabel=r"$\lambda$",
                    desc=r"Ornstein-Uhlenbeck Half-Life of Mean Decay",
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
                      params={"μ": μ, "λ": λ, "t": t, "x0": x0},
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
                      params={"μ": μ, "λ": λ, "Δt": Δt, "x0": x0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Ornstein-Uhlenbeck Process",
                      f=f,
                      x=x)

##################################################################################################################
# Perform estimate for specified estimate types
##################################################################################################################
# Est.AR
def _ar_estimate(samples, **kwargs):
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    result = ou.ou_fit(samples, Δt, x0)
    return result, _arma_estimate_from_result(result, ARIMA.Est.AR)

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
