from enum import (Enum, EnumMeta)
import numpy

from lib import stats
from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.schema import (DataType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create Stats Functions
class Stats:
    class Func(FuncBase):
        PSPEC = "PSPEC"                        # Power Spectrum
        ACF = "ACF"                            # Auto Correlation Function  (has ensemble and time series implementations)
        LAG_VAR = "LAG_VAR"                    # Lagged varinace
        DIFF = "DIFF"                          # Time series difference
        CUMU_MEAN = "CUMU_MEAN"                # Cumulative mean
        CUMU_SD = "CUMU_SD"                    # Cumulative standard deviation
        AGG_VAR = "AGG_VAR"                    # Aggregated time series variance
        AGG = "AGG"                            # Aggregated time series
        PDF_HIST = "PDF_HIST"                  # Compute PDF histogram from simulation
        CDF_HIST = "CDF_HIST"                  # Compute CDF histogram from simulation
        MEAN = "MEAN"                          # Mean as a function of time (Ensemble Func)
        SD = "SD"                              # Standard deviation as a function of time (Ensemble Func)

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

        def _create_ensemble_data_func(self, **kwargs):
            return _create_ensemble_data_func(self, **kwargs)

###################################################################################################
## create function definition for data type
def _create_func(func_type, **kwargs):
    if func_type.value == Stats.Func.PSPEC.value:
        return _create_pspec(func_type, **kwargs)
    elif func_type.value == Stats.Func.ACF.value:
        return _create_acf(func_type, **kwargs)
    elif func_type.value == Stats.Func.LAG_VAR.value:
        return _create_lag_var(func_type, **kwargs)
    elif func_type.value == Stats.Func.DIFF.value:
        return _create_diff(func_type, **kwargs)
    elif func_type.value == Stats.Func.CUMU_MEAN.value:
        return _create_cumu_mean(func_type, **kwargs)
    elif func_type.value == Stats.Func.CUMU_SD.value:
        return  _create_cumu_sd(func_type, **kwargs)
    elif func_type.value == Stats.Func.AGG_VAR.value:
        return _create_agg_var(func_type, **kwargs)
    elif func_type.value == Stats.Func.AGG.value:
        return _create_agg(func_type, **kwargs)
    elif func_type.value == Stats.Func.PDF_HIST.value:
        return _create_pdf_hist(func_type, **kwargs)
    elif func_type.value == Stats.Func.CDF_HIST.value:
        return _create_cdf_hist(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Create DataFunc objects for specified Func
# Func.PSPEC
def _create_pspec(func_type, **kwargs):
    fx = lambda x : x[1:]
    fy = lambda x, y : stats.pspec(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.FOURIER_TRANS,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\rho_\omega$",
                    xlabel=r"$\omega$",
                    desc="Power Spectrum",
                    fy=fy,
                    fx=fx)

# Func.ACF
def _create_acf(func_type, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    fx = lambda x : x[:nlags+1]
    fy = lambda x, y : stats.acf(y, nlags)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="ACF",
                    fy=fy,
                    fx=fx)

# Func.DIFF
def _create_diff(func_type, **kwargs):
    ndiff = get_param_default_if_missing("ndiff", 1, **kwargs)
    fx = lambda x : x[:-ndiff]
    fy = lambda x, y : stats.ndiff(y, ndiff)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"ndiff": ndiff},
                    ylabel=f"$\Delta^{{{ndiff}}} S_t$",
                    xlabel=r"$t$",
                    desc="Difference",
                    fy=fy,
                    fx=fx)

# Func.CUMU_MEAN
def _create_cumu_mean(func_type, **kwargs):
    fy = lambda x, y : stats.cumu_mean(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Cumulative Mean",
                    fy=fy)

# Func.CUMU_SD
def _create_cumu_sd(func_type, **kwargs):
        fy = lambda x, y : stats.cumu_sd(y)
        return DataFunc(func_type=func_type,
                        data_type=DataType.TIME_SERIES,
                        source_type=DataType.TIME_SERIES,
                        params={},
                        ylabel=r"$\sigma_t$",
                        xlabel=r"$t$",
                        desc="Cumulative SD",
                        fy=fy)

# Func.AGG_VAR
def _create_agg_var(func_type, **kwargs):
    npts = get_param_throw_if_missing("npts", **kwargs)
    m_max = get_param_throw_if_missing("m_max", **kwargs)
    m_min = get_param_default_if_missing("m_min", 1.0, **kwargs)
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    fx = lambda x : create_logspace(npts=npts, xmax=m_max, xmin=m_min)
    fy = lambda x, y : stats.agg_var(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={"npts": npts, "m_max": m_max, "m_min": m_min},
                    ylabel=r"$Var(X^m)$",
                    xlabel=r"$m$",
                    desc="Aggregated Variance",
                    fy=fy,
                    fx=fx)
# Func.AGG
def _create_agg(func_type, **kwargs):
    m = get_param_throw_if_missing("m", **kwargs)
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    fx = lambda x : stats.agg_time(x, m)
    fy = lambda x, y : stats.agg(y, m)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={"m": m},
                    ylabel=f"$X^{{{m}}}$",
                    xlabel=r"$t$",
                    desc=f"Aggregation",
                    fy=fy,
                    fx=fx)

# Func.PDF_HIST
def _create_pdf_hist(func_type, **kwargs):
    xmin = get_param_throw_if_missing("xmin", **kwargs)
    xmax = get_param_throw_if_missing("xmax", **kwargs)
    nbins = get_param_default_if_missing("nbins", 50, **kwargs)
    fx = lambda x : x[:-1]
    fyx = lambda x, y : stats.pdf_hist(y, [xmin, xmax], nbins)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$p(x)$",
                    xlabel=r"$x$",
                    formula=r"$x \sim \frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$",
                    desc="PDF",
                    fx=fx,
                    fyx=fyx)

# Func.CDF_HIST
def _create_cdf_hist(func_type, **kwargs):
    fx = lambda x : x[:-1]
    fy = lambda x, y : stats.cdf_hist(x, y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.DIST,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$P(x)$",
                    xlabel=r"$x$",
                    formula=r"$x \sim \frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$",
                    desc="CDF",
                    fx=fx,
                    fy=fy)

# Func.LAG_VAR
def _create_lag_var(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : stats.lag_var_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$\sigma^2(s)$",
                    xlabel=r"$s$",
                    desc="Lagged Variance",
                    formula=r"$\frac{1}{m} \sum_{i=s}^t \left( X_t - X_{t-s} - s\mu \right)^2$",
                    fy=fy,
                    fx=fx)

###################################################################################################
## create function definition applied to lists of data frames for data type
def _create_ensemble_data_func(func_type, **kwargs):
    if func_type.value == Stats.Func.MEAN.value:
        return _create_ensemble_mean(func_type, **kwargs)
    elif func_type.value == Stats.Func.SD.value:
        return _create_ensemble_sd(func_type, **kwargs)
    elif func_type.value == Stats.Func.ACF.value:
        return _create_ensemble_acf(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
## Create dataFunc objects for specified data type
# Func.MEAN
def _create_ensemble_mean(func_type, **kwargs):
    fy = lambda x, y : stats.ensemble_mean(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Ensemble Mean",
                    fy=fy)

# Func.SD
def _create_ensemble_sd(func_type, **kwargs):
    fy = lambda x, y : stats.ensemble_sd(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="Ensemble SD",
                    fy=fy)

# Func.ACF
def _create_ensemble_acf(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    nlags = get_param_default_if_missing("nlags", None, **kwargs)
    fy = lambda x, y : stats.ensemble_acf(y, nlags)
    fx = lambda x : x[:nlags]
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="Ensemble ACF",
                    fy=fy,
                    fx=fx)
