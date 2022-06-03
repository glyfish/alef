from enum import Enum
from pandas import DataFrame
from datetime import datetime
import uuid
import numpy

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

##################################################################################################################
# Specify Funcs used in analysis
class Func(Enum):
    PSPEC = "PSPEC"                        # Power Spectrum
    ACF = "ACF"                            # Autocorrelation function
    DIFF = "DIFF"                          # Time series difference
    DIFF_ACF = "DIFF_ACF"                  # ACF Difference
    CUMU_MEAN = "CUMU_MEAN"                # Cumulative mean
    CUMU_SD = "CUMU_SD"                    # Cumulative standard deviation
    MEAN = "MEAN"                          # Mean as a function of time
    SD = "SD"                              # Standard deviation as a function of time
    AR1_ACF = "AR1_ACF"                    # AR(1) Autocorrelation function
    MAQ_ACF = "MAQ_ACF"                    # MA(q) Autocorrelation function
    FBM_MEAN = "FBM_MEAN"                  # Fractional Brownian Motion mean
    FBM_SD = "FBM_SD"                      # Fractional Brownian Motion standard deviation
    FBM_ACF = "FBM_ACF"                    # Fractional Brownian Motion autocorrelation function
    FBM_COV = "FBM_COV"                    # Fractional Brownian Motion covariance function
    BM_MEAN = "BM_MEAN"                    # Brownian Motion mean
    BM_DRIFT_MEAN = "BM_DRIFT_MEAN"        # Brownian Motion model mean with data
    BM_SD = "BM_SD"                        # Brownian Motion model standard deviation
    GBM_MEAN = "GBM_MEAN"                  # Geometric Brownian Motion model mean
    GBM_SD = "GBM_SD"                      # Geometric Brownian Motion model standard deviation
    AGG_VAR = "AGG_VAR"                    # Aggregated variance
    AGG = "AGG"                            # Aggregated time series
    VR = "VR"                              # Variance Ratio use in test for brownian motion
    VR_STAT = "VR_STAT"                    # FBM variance ratio test statistic
    PACF = "PACF"                          # Partial Autocorrelation function
    BM = "BM"                              # Brownian motion computed from brownian motion noise
    ARMA_MEAN = "ARMA_MEAN"                # ARMA(p,q) MEAN
    AR1_SD = "AR1_SD"                      # AR(1) standard seviation
    MAQ_SD = "MAQ_SD"                      # MA(q) standard deviation
    AR1_OFFSET_MEAN = "AR1_OFFSET_MEAN"    # AR(1) with constant offset mean
    AR1_OFFSET_SD = "AR1_OFFSET_SD"        # AR(1) with offset standard deviation

    def create(self, **kwargs):
        return _create_func_type(self, **kwargs)

    def apply(self, df, **kwargs):
        return _apply_func_type(self, df, **kwargs)

    def apply_to_ensemble(self, dfs, **kwargs):
        return _apply_func_type_to_ensemble(self, dfs, **kwargs)

    def apply_to_list(self, dfs, **kwargs):
        return _apply_func_type_to_list(self, dfs, **kwargs)

    def apply_to_parameter_scan(self, df, *args):
        return _apply_func_type_to_parameter_scan(self, df, *args)

###################################################################################################
# DataFunc consist of the input schema and function used to compute resulting data columns
#
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
#
# fy: Function used to compute ycol from source DataType, fy is assumed to have the form
#     fy(fx(x),y) -> ycol
# fx: Function used to compute xcol from source DataType xcol, fx is assumed to have the form
#     fx(x) -> xcol
# source: DataType input into f used to compute xcol and ycol
#
class DataFunc:
    def __init__(self, func_type, data_type, source_type, params, fy, ylabel, xlabel, desc, formula=None, fx=None):
        self.func_type = func_type
        self.schema = data_type.schema()
        self.params = params
        self.source_schema = source_type.schema()
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.desc = desc
        self.fy = fy
        self.formula = formula
        if fx is None:
            self.fx = lambda x: x
        else:
            self.fx = fx

    def __repr__(self):
        return f"DataFunc({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"func_type={self.func_type},  " \
               f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"source_schema=({self.source_schema}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"desc=({self.desc}), " \
               f"formula=({self.formula})"

    def _name(self):
        return f"{self.schema.data_type.value}-{str(uuid.uuid4())}"

    def _set_df_meta_data(self, df, source_type, source_name):
        DataSchema.set_source_type(df, source_type)
        DataSchema.set_source_name(df, source_name)
        DataSchema.set_source_schema(df, self.source_schema)
        DataSchema.set_date(df)
        DataSchema.set_type(df, self.func_type)
        DataSchema.set_name(df, self._name())
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_iterations(df, None)

    def apply(self, df):
        x, y = self.source_schema.get_data(df)
        x_result = self.fx(x)
        y_result = self.fy(x_result, y)
        df_result = self.create_data_frame(x_result, y_result)
        source_name = DataSchema.get_name(df)
        source_type = DataSchema.get_type(df)
        self._set_df_meta_data(df_result, source_type, source_name)
        MetaData.set(df_result, self.meta_data(len(y_result)))
        df_result.attrs = df.attrs | df_result.attrs
        return df_result

    def apply_ensemble(self, dfs):
        if len(dfs) == 0:
            Exception(f"No DataFrames provided")
        x, y = self.source_schema.get_data_from_list(dfs)
        x_result = self.fx(x)
        y_result = self.fy(x_result, y)
        df = self.create_data_frame(x_result, y_result)
        source_name = DataSchema.get_name(dfs[0])
        source_type = DataSchema.get_type(dfs[0])
        self._set_df_meta_data(df, source_type, source_name)
        MetaData.set(df, self.meta_data(len(y_result)))
        df.attrs = dfs[0].attrs | df.attrs
        return df

    def apply_list(self, dfs):
        return [self.apply(df) for df in dfs]

    def meta_data(self, npts):
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            source_schema = self.source_schema,
            formula = self.formula
        )

    def ensemble_meta_data(self, npts, df):
        source_meta_data = MetaData.get(df)
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=source_meta_data.params | self.params,
            desc=f"{source_meta_data.desc} {self.desc}",
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            source_schema = None,
            formula = self.formula
        )

    def create(self, x):
        y = self.fy(x, None)
        df = self.create_data_frame(x, y)
        DataSchema.set_date(df)
        DataSchema.set_source_schema(df, self.source_schema)
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_name(df, self.desc)
        return df

    def create_data_frame(self, x, y):
        return self.schema.create_data_frame(x, y)

###################################################################################################
## create function definition for data type
def _apply_func_type(func_type, df,**kwargs):
    data_func = _create_func(func_type, **kwargs)
    return data_func.apply(df)

###################################################################################################
## create function definition for data type
def _apply_func_type_to_ensemble(func_type, dfs, **kwargs):
    data_func = _create_ensemble_data_func(func_type, **kwargs)
    return data_func.apply_ensemble(dfs)

###################################################################################################
## create function definition for data type
def _apply_func_type_to_list(func_type, dfs, **kwargs):
    data_func = _create_func(func_type, **kwargs)
    return data_func.apply_list(dfs)

###################################################################################################
## create function definition for data type
def _apply_func_type_to_parameter_scan(func_type, df, *args):
    dfs = []
    for kwargs in args:
        dfs.append(_apply_func_type(df, func_type, **kwargs))
    return dfs

###################################################################################################
## create function definition for data type
def _create_func_type(func_type, **kwargs):
    x = get_param_default_if_missing("x", None, **kwargs)
    if x is None:
        x = create_space(**kwargs)
    kwargs["npts"] = len(x)
    data_func = _create_func(func_type, **kwargs, npolt=len(x))
    return data_func.create(x)

###################################################################################################
## create function definition for data type
def _create_func(func_type, **kwargs):
    if func_type.value == Func.PSPEC.value:
        return _create_pspec(func_type, **kwargs)
    elif func_type.value == Func.ACF.value:
        return _create_acf(func_type, **kwargs)
    elif func_type.value == Func.PACF.value:
        return _create_pacf(func_type, **kwargs)
    elif func_type.value == Func.VR_STAT.value:
        return _create_vr_stat(func_type, **kwargs)
    elif func_type.value == Func.DIFF.value:
        return _create_diff(func_type, **kwargs)
    elif func_type.value == Func.CUMU_MEAN.value:
        return _create_cumu_mean(func_type, **kwargs)
    elif func_type.value == Func.CUMU_SD.value:
        return  _create_cumu_sd(func_type, **kwargs)
    elif func_type.value == Func.AR1_ACF.value:
        return _create_ar1_acf(func_type, **kwargs)
    elif func_type.value == Func.MAQ_ACF.value:
        return _create_maq_acf(func_type, **kwargs)
    elif func_type.value == Func.FBM_MEAN.value:
        return _create_fbm_mean(func_type, **kwargs)
    elif func_type.value == Func.FBM_SD.value:
        return _create_fbm_sd(func_type, **kwargs)
    elif func_type.value == Func.FBM_ACF.value:
        return _create_fbm_acf(func_type, **kwargs)
    elif func_type.value == Func.FBM_COV.value:
        return _create_fbm_cov(func_type, **kwargs)
    elif func_type.value == Func.BM_MEAN.value:
        return _create_bm_mean(func_type, **kwargs)
    elif func_type.value == Func.BM_DRIFT_MEAN.value:
        return _create_bm_drift_mean(func_type, **kwargs)
    elif func_type.value == Func.BM_SD.value:
        return _create_bm_sd(func_type, **kwargs)
    elif func_type.value == Func.GBM_MEAN.value:
        return _create_gbm_mean(func_type, **kwargs)
    elif func_type.value == Func.GBM_SD.value:
        return _create_gbm_sd(func_type, **kwargs)
    elif func_type.value == Func.AGG_VAR.value:
        return _create_agg_var(func_type, **kwargs)
    elif func_type.value == Func.AGG.value:
        return _create_agg(func_type, **kwargs)
    elif func_type.value == Func.VR.value:
        return _create_vr(func_type, **kwargs)
    elif func_type.value == Func.BM.value:
        return _create_bm(func_type, **kwargs)
    elif func_type.value == Func.ARMA_MEAN.value:
        return _create_arma_mean(func_type, **kwargs)
    elif func_type.value == Func.AR1_SD.value:
        return _create_ar1_sd(func_type, **kwargs)
    elif func_type.value == Func.MAQ_SD.value:
        return _create_maq_sd(func_type, **kwargs)
    elif func_type.value == Func.AR1_OFFSET_MEAN.value:
        return _create_ar1_offset_mean(func_type, **kwargs)
    elif func_type.value == Func.AR1_OFFSET_SD.value:
        return _create_ar1_offset_sd(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Create DataFunc objects for specified DataType
# DataType.PSPEC
def _create_pspec(func_type, **kwargs):
    fx = lambda x : x[1:]
    fy = lambda x, y : stats.pspec(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.FOURIER_TRANS,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\rho_\omega$",
                    xlabel=r"$\omega$",
                    desc="Power Spectrum",
                    fx=fx)

# DataType.ACF
def _create_acf(func_type, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    fx = lambda x : x[:nlags+1]
    fy = lambda x, y : stats.acf(y, nlags)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="ACF",
                    fx=fx)

# DataType.PACF
def _create_pacf(func_type, **kwargs):
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[1:nlags+1]
    fy = lambda x, y : arima.yw(y, nlags)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.TIME_SERIES,
                    params={"nlags": nlags},
                    fy=fy,
                    ylabel=r"$\varphi_\tau$",
                    xlabel=r"$\tau$",
                    desc="PACF",
                    fx=fx)

# DataType.VR_STAT
def _create_vr_stat(func_type, **kwargs):
    fy = lambda x, y : y
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio Statistic",
                    formula=r"$\frac{VR(s) - 1}{\sqrt{\theta^\ast (s)}}$",
                    fy=fy)

# DataType.DIFF
def _create_diff(func_type, **kwargs):
    ndiff = get_param_default_if_missing("ndiff", 1, **kwargs)
    fx = lambda x : x[:-1]
    fy = lambda x, y : stats.ndiff(y, ndiff)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"ndiff": ndiff},
                    fy=fy,
                    ylabel=r"$\Delta S_t$",
                    xlabel=r"$t$",
                    desc="Difference",
                    fx=fx)

# DataType.CUMU_MEAN
def _create_cumu_mean(func_type, **kwargs):
    fy = lambda x, y : stats.cumu_mean(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Cumulative Mean")

# DataType.CUMU_SD
def _create_cumu_sd(func_type, **kwargs):
        fy = lambda x, y : stats.cumu_sd(y)
        return DataFunc(func_type=func_type,
                        data_type=DataType.TIME_SERIES,
                        source_type=DataType.TIME_SERIES,
                        params={},
                        fy=fy,
                        ylabel=r"$\sigma_t$",
                        xlabel=r"$t$",
                        desc="Cumulative SD")


# DataType.AR1_ACF
def _create_ar1_acf(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    fx = lambda x : x[:nlags +1 ]
    fy = lambda x, y : φ**x
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"φ": φ},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="AR(1) ACF",
                    fx=fx,
                    formula=r"$\varphi^\tau$")

# DataType.MAQ_ACF
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
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc=f"MA({len(θ)}) ACF",
                    fx=fx,
                    formula=r"$\sigma^2 \left( \sum_{i=i}^{q-\tau} \vartheta_i \vartheta_{i+\tau} + \vartheta_\tau \right)$")

# DataType.FBM_MEAN
def _create_fbm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.MEAN,
                    params={},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc="FBM Mean",
                    fx=fx)

# DataType.FBM_SD
def _create_fbm_sd(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.sqrt(fbm.var(H, x))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H},
                    fy=fy,
                    ylabel=r"$\sigma^H_t$",
                    xlabel=r"$t$",
                    desc="FBM SD",
                    formula=r"$t^H$",
                    fx=fx)

# DataType.FBM_ACF
def _create_fbm_acf(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : fbm.acf(H, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"H": H},
                    fy=fy,
                    ylabel=r"$\rho^H_n$",
                    xlabel=r"$n$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[(n-1)^{2H} + (n+1)^{2H} - 2n^{2H}]$",
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
                    fy=fy,
                    ylabel=r"$R^H(t,s)$",
                    xlabel=r"$t$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[t^{2H}+s^{2H}-(t-s)^{2H}]$",
                    fx=fx)

# DataType.BM_MEAN
def _create_bm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.full(len(x), μ)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM Mean",
                    formula=r"$0$",
                    fx=fx)

# DataType.BM_DRIFT_MEAN
def _create_bm_drift_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : μ*x
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"μ": μ},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    formula=r"$\mu t$",
                    fx=fx)

# DataType.BM_SD
def _create_bm_sd(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : σ*numpy.sqrt(x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="BM with Drift Mean",
                    formula=r"$\sqrt{t}$",
                    fx=fx)

# DataType.GBM_MEAN
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
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM Mean",
                    formula=r"$S_0 e^{\mu t}$",
                    fx=fx)

# DataType.GBM_SD
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
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="Geometric BM SD",
                    formula=r"$S_0^2 e^{2\mu t}\left( e^{\sigma^2 t} - 1 \right)$",
                    fx=fx)

# DataType.AGG_VAR
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
                    fy=fy,
                    ylabel=r"$Var(X^m)$",
                    xlabel=r"$m$",
                    desc="Aggregated Variance",
                    fx=fx)
# DataType.AGG
def _create_agg(func_type, **kwargs):
    m = get_param_throw_if_missing("m", **kwargs)
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    fx = lambda x : stats.agg_time(x, m)
    fy = lambda x, y : stats.agg(y, m)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={"m": m},
                    fy=fy,
                    ylabel=f"$X^{{{m}}}$",
                    xlabel=r"$t$",
                    desc=f"Aggregation",
                    fx=fx)

# DataType.VR
def _create_vr(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y :  σ**2*t**(2*H - 1.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H, "σ": σ},
                    fy=fy,
                    ylabel=r"$VR(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    formula=r"$\frac{\sigma^2(s)}{\sigma_B^2(s)}$",
                    fx=fx)

# DataType.BM
def _create_bm(func_type, **kwargs):
    fy = lambda x, y : stats.from_noise(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$S_t$",
                    xlabel=r"$t$",
                    desc="BM")

# DataType.ARMA_MEAN
def _create_arma_mean(func_type, **kwargs):
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\mu_\infty$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc="ARMA(p,q) Mean")

# DataType.AR1_SD
def _create_ar1_sd(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_sigma(φ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, "σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\sigma^2}{1-\varphi^2}$",
                    desc="AR(1) SD")

# DataType.MAQ_SD
def _create_maq_sd(func_type, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.maq_sigma(θ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"θ": θ, "σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\sigma^2 \left( \sum_{i=1}^q \vartheta_i^2 + 1 \right)$",
                    desc="MA(q) SD")

# DataType.AR1_OFFSET_MEAN
def _create_ar1_offset_mean(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_offset_mean(φ, μ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, r"$μ^*$": μ},
                    fy=fy,
                    ylabel=r"$\mu_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\frac{\mu^*}{1 - \varphi}$",
                    desc="AR(1) with Offset Mean")

# DataType.AR1_OFFSET_SD
def _create_ar1_offset_sd(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : numpy.full(len(x), arima.ar1_offset_sigma(φ, σ))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"φ": φ, "σ": σ},
                    fy=fy,
                    ylabel=r"$\sigma_\infty$",
                    xlabel=r"$t$",
                    formula=r"$\sqrt{\frac{\sigma^2}{1 - \varphi^2}}$",
                    desc="AR(1) with Offset Mean")

###################################################################################################
## create function definition applied to lists of data frames for data type
def _create_ensemble_data_func(func_type, **kwargs):
    if func_type.value == Func.MEAN.value:
        return _create_ensemble_mean(func_type, **kwargs)
    elif func_type.value == Func.SD.value:
        return _create_ensemble_sd(func_type, **kwargs)
    elif func_type.value == Func.ACF.value:
        return _create_ensemble_acf(func_type, **kwargs)
    else:
        raise Exception(f"Func is invalid: {func_type}")

###################################################################################################
## Create dataFunc objects for specified data type
# DataType.MEAN
def _create_ensemble_mean(func_type, **kwargs):
    fy = lambda x, y : stats.ensemble_mean(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    desc="Ensemble Mean")

# DataType.SD
def _create_ensemble_sd(func_type, **kwargs):
    fy = lambda x, y : stats.ensemble_sd(y)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy,
                    ylabel=r"$\sigma_t$",
                    xlabel=r"$t$",
                    desc="Ensemble SD")

# DataType.ACF
def _create_ensemble_acf(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    nlags = get_param_default_if_missing("nlags", None, **kwargs)
    fy = lambda x, y : stats.ensemble_acf(y, nlags)
    fx = lambda x : x[:nlags]
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    fy=fy,
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="Ensemble ACF",
                    fx=fx)
