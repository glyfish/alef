from enum import Enum
from pandas import DataFrame

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima

from lib.data.schema import (DataType, DataSchema, create_schema, MetaData)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

###################################################################################################
# Data Function consist of the input schema and function used to compute resulting data columns
#
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
# fy: Function used to compute ycol from source DataType, fy is assumed to have the form
#     fy(fx(x),y) -> ycol
# fx: Function used to compute xcol from source DataType xcol, fx is assumed to have the form
#     fx(x) -> xcol
# source: DataType input into f used to compute xcol and ycol
#
class DataFunc:
    def __init__(self, schema, source_data_type, params, fy, ylabel, xlabel, desc, fx=None):
        self.schema=schema
        self.fy=fy
        self.params=params
        self.source_schema=create_schema(source_data_type)
        self.ylabel=ylabel
        self.xlabel=xlabel
        self.desc=desc
        if fx is None:
            self.fx=lambda x: x
        else:
            self.fx=fx

    def __repr__(self):
        return f"DataFunc({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"schema=({self.schema}), fy=({self.fy}), fx=({self.fx}), params=({self.params}), xlabel=({self.xlabel}), ylabel=({self.ylabel}), desc=({self.desc}), source_schema=({self.source_schema})"

    def apply(self, df):
        x, y = self.source_schema.get_data(df)
        x_result = self.fx(x)
        y_result = self.fy(x_result, y)
        df_result = self.create_data_frame(x_result, y_result)
        return DataSchema.concatinate(df, df_result)

    def meta_data(self, x, y):
        return MetaData(
            npts=len(y),
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            source_schema = self.source_schema
        )

    def create_data_frame(self, x, y):
        return DataSchema.create_data_frame(x, y, self.meta_data(x, y))

    @staticmethod
    def apply_func_type(df, func_type, **kwargs):
        data_func = create_data_func(func_type, **kwargs)
        return data_func.apply(df)

## create definition for data type
def create_data_func(data_type, **kwargs):
    schema = create_schema(data_type)
    if data_type.value == DataType.PSPEC.value:
        return _create_pspec(schema, **kwargs)
    elif data_type.value == DataType.ACF.value:
        return _create_acf(schema, **kwargs)
    elif data_type.value == DataType.PACF.value:
        return _create_pacf(schema, **kwargs)
    elif data_type.value == DataType.VR_STAT.value:
        return _create_vr_stat(schema, **kwargs)
    elif data_type.value == DataType.DIFF_1.value:
        return _create_diff_1(schema, **kwargs)
    elif data_type.value == DataType.DIFF_2.value:
        return _create_diff_2(schema, **kwargs)
    elif data_type.value == DataType.CUMU_MEAN.value:
        return _create_cumu_mean(schema, **kwargs)
    elif data_type.value == DataType.CUMU_SD.value:
        return  _create_cumu_sd(schema, **kwargs)
    elif data_type.value == DataType.AR1_ACF.value:
        return _create_ar1_acf(schema, **kwargs)
    elif data_type.value == DataType.MAQ_ACF.value:
        return _create_maq_acf(schema, **kwargs)
    elif data_type.value == DataType.FBM_MEAN.value:
        return _create_fbm_mean(schema, **kwargs)
    elif data_type.value == DataType.FBM_SD.value:
        return _create_fbm_sd(schema, **kwargs)
    elif data_type.value == DataType.FBM_ACF.value:
        return _create_fbm_acf(schema, **kwargs)
    elif data_type.value == DataType.BM_MEAN.value:
        return _create_bm_mean(schema, **kwargs)
    elif data_type.value == DataType.BM_DRIFT_MEAN.value:
        return _create_bm_drift_mean(schema, **kwargs)
    elif data_type.value == DataType.BM_SD.value:
        return _create_bm_sd(schema, **kwargs)
    elif data_type.value == DataType.GBM_MEAN.value:
        return _create_gbm_mean(schema, **kwargs)
    elif data_type.value == DataType.GBM_SD.value:
        return _create_gbm_sd(schema, **kwargs)
    elif data_type.value == DataType.AGG_VAR.value:
        return _create_agg_var(schema, **kwargs)
    elif data_type.value == DataType.VR.value:
        return _create_vr(schema, **kwargs)
    elif data_type.value == DataType.BM_NOISE.value:
        return _create_bm_noise(schema, **kwargs)
    elif data_type.value == DataType.BM.value:
        return _create_bm(schema, **kwargs)
    else:
        raise Exception(f"DataType is invalid: {data_type}")

###################################################################################################
# Create DataFunc objects for specified DataType
# DataType.PSPEC
def _create_pspec(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\rho_\omega$",
                    xlabel=r"$\omega$",
                    desc="Power Spectrum")

# DataType.ACF
def _create_acf(schema, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[:nlags+1]
    fy = lambda x, y : stats.acf(y, nlags)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={"nlags": nlags},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="ACF",
                    fx=fx)

# DataType.PACF
def _create_pacf(schema, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[1:nlags+1]
    fy = lambda x, y : arima.yw(y, nlags)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={"nlags": nlags},
                    fy=fy,
                    ylabel=r"$\varphi_\tau$",
                    xlabel=r"$\tau$",
                    desc="PACF",
                    fx=fx)
# DataType.VR_STAT
def _create_vr_stat(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio Statistic",
                    fy=fy)

# DataType.DIFF_1
def _create_diff_1(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\Delta^1_t$",
                    desc="First Difference",
                    xlabel=r"$t$")

# DataType.DIFF_2
def _create_diff_2(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\Delta^2_t$",
                    xlabel=r"$t$",
                    desc="Second Difference")


# DataType.CUMU_MEAN
def _create_cumu_mean(schema, **kwargs):
    fy = lambda x, y : stats.cumu_mean(y)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Cumulative Mean")

# DataType.CUMU_SD
def _create_cumu_sd(schema, **kwargs):
        fy = lambda x, y : stats.cumu_sd(y)
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        params={},
                        fy=fy,
                        ylabel=r"$\sigma_t$",
                        xlabel=r"$t$",
                        desc="Cumulative SD")


# DataType.AR1_ACF
def _create_ar1_acf(schema, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[:nlags+1]
    fy = lambda x, y : φ**x
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    params={"φ": φ},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="AR(1) ACF",
                    fx=fx)

# DataType.MAQ_ACF
def _create_maq_acf(schema, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[:nlags]
    fy = lambda x, y : arima.maq_acf(θ, σ, len(x))
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    params={"θ": θ, "σ": σ},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="MA(q) ACF",
                    fx=fx)

# DataType.FBM_MEAN
def _create_fbm_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
                    params={"npts": npts},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="FBM Mean",
                    fx=fx)

# DataType.FBM_SD
def _create_fbm_sd(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : σ*numpy.sqrt(fbm.var(H, x))
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    params={"npts": npts, "H": H, "σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="FBM SD",
                    fx=fx)

# DataType.FBM_ACF
def _create_fbm_acf(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : fbm.acf(H, x)
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    params={"npts": npts, "H": H},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="FBM ACF",
                    fx=fx)

# DataType.BM_MEAN
def _create_bm_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : numpy.full(len(x), μ)
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
                    params={"npts": npts, "μ": μ},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM Mean",
                    fx=fx)

# DataType.BM_DRIFT_MEAN
def _create_bm_drift_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : μ*x
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
                    params={"npts": npts, "μ": μ},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    fx=fx)

# DataType.BM_SD
def _create_bm_sd(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : σ*numpy.sqrt(x)
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    params={"npts": npts, "σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    fx=fx)

# DataType.GBM_MEAN
def _create_gbm_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : S0*numpy.exp(μ*x)
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
                    params={"npts": npts, "μ": μ, "S0": S0},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM Mean",
                    fx=fx)

# DataType.GBM_SD
def _create_gbm_sd(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : numpy.sqrt(S0**2*numpy.exp(2*μ*x)*(numpy.exp(x*σ**2)-1))
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    params={"npts": npts, "σ": σ, "μ": μ, "S0": S0},
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM SD",
                    fx=fx)

# DataType.AGG_VAR
def _create_agg_var(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : σ**2*fbm.var(H, t)
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    params={"npts": npts, "H": H, "σ": σ},
                    fy=fy,
                    ylabel=r"$\text{Var}(X^m)$",
                    xlabel=r"$m$",
                    desc="Aggregated Variance",
                    fx=fx)

# DataType.VR
def _create_vr(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y :  σ**2*t**(2*H - 1.0)
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    params={"npts": npts, "H": H, "σ": σ},
                    fy=fy,
                    ylabel=r"$\text{VR}(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    fx=fx)

# DataType.BM_NOISE
def _create_bm_noise(schema, **kwargs):
    fx = lambda x : x[1:]
    fy = lambda x, y : bm.to_noise(y)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\Delta S_t$",
                    xlabel=r"$t$",
                    desc="BM Noise",
                    fx=fx)

# DataType.BM
def _create_bm(schema, **kwargs):
    fy = lambda x, y : bm.to_noise(y)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$S_t$",
                    xlabel=r"$t$",
                    desc="BM",
                    fx=fx)
