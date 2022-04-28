from enum import Enum
from pandas import DataFrame

from lib import stats
from lib.models import fbm
from lib.models import arima

from lib.data.schema import (DataType, DataSchema, create_schema)
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
    def __init__(self, schema, source_data_type, params, fy, fx=None):
        self.schema = schema
        self.fy = fy
        self.params = params
        self.fx = lambda x: x if fx is None else fx
        self.source_schema = create_schema(source_data_type)

    def apply(self, df):
        x, y = self.source_schema.get_data(df)
        x_result = self.fx(x)
        y_result = self.fy(x_result, y)
        df_result = self.data_frame(x_result, y_result)
        return DataSchema.concatinate(df, df_result)

    def data_frame(self, x, y):
        df = DataFrame({
            self.schema.xcol: x,
            self.schema.ycol: y
        })
        meta_data = {
            self.schema.ycol: {"npts": len(y),
                               "DataType": self.schema.data_type,
                               "Parameters": self.params}
        }
        df.attrs = meta_data
        return df

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
                    fy=fy)

# DataType.ACF
def _create_acf(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy)

# DataType.VR_STAT
def _create_vr_stat(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy)

# DataType.DIFF_1
def _create_diff_1(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy)

# DataType.DIFF_2
def _create_diff_2(schema, **kwargs):
    fy = lambda x, y : y
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy)

# DataType.CUMU_MEAN
def _create_cumu_mean(schema, **kwargs):
    fy = lambda x, y : stats.cumu_mean(y)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    params={},
                    fy=fy)

# DataType.CUMU_SD
def _create_cumu_sd(schema, **kwargs):
        fy = lambda x, y : stats.cumu_sd(y)
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        params={},
                        fy=fy)

# DataType.AR1_ACF
def _create_ar1_acf(schema, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    fy = lambda x, y : φ**x
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    params={"φ": φ},
                    fy=fy)

# DataType.MAQ_ACF
def _create_maq_acf(schema, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : arima.maq_acf(θ, σ, len(x))
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    params={"θ": θ, "σ": σ},
                    fy=fy)

# DataType.FBM_MEAN
def _create_fbm_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
                    params={"npts": npts},
                    fy=fy,
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
                    fx=fx)
