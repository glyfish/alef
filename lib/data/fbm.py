from enum import (Enum, EnumMeta)
import numpy

from lib.models import fbm
from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.schema import (DataType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Fractional Brownian Motion Fuctions
class FBM:
    class Func(FuncBase):
        MEAN = "FBM_MEAN"                      # Fractional Brownian Motion mean
        VAR = "FBM_VAR"                        # Fractional Brownian Motion variance
        SD = "FBM_SD"                          # Fractional Brownian Motion standard deviation
        ACF = "FBM_ACF"                        # Fractional Brownian Motion autocorrelation function
        COV = "FBM_COV"                        # Fractional Brownian Motion covariance function
        VR = "FBM_VR"                          # Fractional Brownian Motion Variance Ratio
        VR_STAT = "FBM_VR_STAT"                # Variance ratio test statistic
        VR_HOMO_STAT = "FBM_VR_HOMO_STAT"      # Homoscedastic variance ratio test statistic
        VR_HETERO_STAT = "FBM_VR_HETERO_STAT"  # Heteroscedastic variance ratio test statistic

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

###################################################################################################
## create function definition for data type
def _create_func(func_type, **kwargs):
    if func_type.value == FBM.Func.MEAN.value:
        return _create_fbm_mean(func_type, **kwargs)
    elif func_type.value == FBM.Func.VAR.value:
        return _create_fbm_var(func_type, **kwargs)
    elif func_type.value == FBM.Func.SD.value:
        return _create_fbm_sd(func_type, **kwargs)
    elif func_type.value == FBM.Func.ACF.value:
        return _create_fbm_acf(func_type, **kwargs)
    elif func_type.value == FBM.Func.COV.value:
        return _create_fbm_cov(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR.value:
        return _create_fbm_vr(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_STAT.value:
        return _create_vr_stat(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_HOMO_STAT.value:
        return _create_vr_homo_stat(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_HETERO_STAT.value:
        return _create_vr_hetero_stat(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Create DataFunc objects for specified Func
# Func.FBM_MEAN
def _create_fbm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc="FBM Mean",
                    fy=fy,
                    fx=fx)

# Func.FBM_SD
def _create_fbm_sd(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1., **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : Δt*numpy.sqrt(fbm.var(H, x))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H, "Δt": Δt},
                    ylabel=r"$\sigma^H_t$",
                    xlabel=r"$t$",
                    desc="FBM SD",
                    formula=r"$(Δt) t^H$",
                    fy=fy,
                    fx=fx)

# Func.FBM_VAR
def _create_fbm_var(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1., **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : Δt**2*fbm.var(H, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H, "Δt": Δt},
                    ylabel=r"$\sigma^2$",
                    xlabel=r"$t$",
                    desc="FBM VAR",
                    formula=r"$Δt^2 t^{2H}$",
                    fy=fy,
                    fx=fx)

# Func.FBM_ACF
def _create_fbm_acf(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : fbm.acf(H, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"H": H},
                    ylabel=r"$\rho^H_n$",
                    xlabel=r"$n$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[(n-1)^{2H} + (n+1)^{2H} - 2n^{2H}]$",
                    fy=fy,
                    fx=fx)

# Func.FBM_VR_STAT
def _create_fbm_vr(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y :  x**(2*H - 1.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H},
                    ylabel=r"$VR(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    formula=r"$s^{2H-1}$",
                    fy=fy,
                    fx=fx)

# DataType.FBM_COV
def _create_fbm_cov(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    s = get_param_throw_if_missing("s", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : fbm.cov(H, s, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"H": H, "s": s},
                    ylabel=r"$R^H(t,s)$",
                    xlabel=r"$t$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[t^{2H}+s^{2H}-(t-s)^{2H}]$",
                    fy=fy,
                    fx=fx)

# Func.VR_STAT
def _create_vr_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$VR(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    formula=r"$\frac{\sigma^2(s)}{\sigma_B^2(s)}$",
                    fy=fy,
                    fx=fx)

# Func.VR_HOMO_STAT
def _create_vr_homo_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_stat_homo_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Homoscedastic Variance Ratio Statistic",
                    formula=r"$\frac{VR(s) - 1}{\sqrt{\theta^\ast (s)}}$",
                    fy=fy,
                    fx=fx)

# Func.VR_HETERO_STAT
def _create_vr_hetero_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_stat_hetero_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Heteroscedastic Variance Ratio Statistic",
                    formula=r"$\frac{VR(s) - 1}{\sqrt{\theta^\ast (s)}}$",
                    fy=fy,
                    fx=fx)
