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
#     fy(fx(x),y) -> DataFrame
# fx: Function used to compute xcol from source DataType xcol, fx is assumed to have the form
#     fx(x) -> xcol
# source: DataType input into f used to compute xcol and ycol
#
class DataFunc:
    def __init__(self, schema, source_data_type, fy, fx=None):
        self.schema = schema
        self.fy = fy
        self.fx = lambda x: x if fx is None else fx
        self.source_data_type = source_data_type

    def apply(self, df):
        schema = create_schema(self.source_data_type)
        x, y = schema.get_data(df)
        result = self.fy(self.fx(x), y)
        return DataSchema.concatinate(df, result)

    @staticmethod
    def apply_data_type(df, data_type, **kwargs):
        data_func = create_data_func(data_type, **kwargs)
        return data_func.apply(df)

    @staticmethod
    def create_data_frame(x, y, data_type):
        schema = create_schema(data_type)
        df = DataFrame({
            schema.xcol: x,
            schema.ycol: y
        })
        meta_data = {
            schema.ycol: {"npts": len(y), "DataType": data_type}
        }
        df.attrs = meta_data
        return df

## create definition for data type
def create_data_func(data_type, **kwargs):
    schema = create_schema(data_type)
    if data_type.value == DataType.PSPEC.value:
        fy = lambda x, y : y
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)
    elif data_type.value == DataType.ACF.value:
        fy = lambda x, y : y
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)
    elif data_type.value == DataType.VR_STAT.value:
        fy = lambda x, y : y
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)
    elif data_type.value == DataType.DIFF_1.value:
        fy = lambda x, y : y
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)
    elif data_type.value == DataType.DIFF_2.value:
        fy = lambda x, y : y
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)
    elif data_type.value == DataType.CUM_MEAN.value:
        return _create_cum_mean(schema, **kwargs)
    elif data_type.value == DataType.CUM_SD.value:
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
        npts = get_param_default_if_missing("npts", 10, **kwargs)
        μ = get_param_default_if_missing("μ", 0.0, **kwargs)
        S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
        fx = lambda x : x[::int(len(x)/npts)]
        fy = lambda x, y : S0*numpy.exp(μ*x)
        return DataFunc(schema=schema,
                        source_data_type=DataType.MEAN,
                        fy=fy,
                        fx=fx)
    elif data_type.value == DataType.GBM_STD.value:
        npts = get_param_default_if_missing("npts", 10, **kwargs)
        σ = get_param_default_if_missing("σ", 1.0, **kwargs)
        μ = get_param_default_if_missing("μ", 0.0, **kwargs)
        S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
        fx = lambda x : x[::int(len(x)/npts)]
        fy = lambda x, y : numpy.sqrt(S0**2*numpy.exp(2*μ*x)*(numpy.exp(x*σ**2)-1))
        return DataFunc(schema=schema,
                        source_data_type=DataType.SD,
                        fy=fy,
                        fx=fx)
    else:
        raise Exception(f"Data type is invalid: {data_type}")


###################################################################################################
# Create DataFunc objects for specified DataType

# DataType.CUM_MEAN
def _create_cumu_mean(schema, **kwargs):
    fy = lambda x, y : stats.cumu_mean(x, y)
    return DataFunc(schema=schema,
                    source_data_type=DataType.TIME_SERIES,
                    fy=fy)
# DataType.CUM_SD
def _create_cumu_sd(schema, **kwargs):
        fy = lambda x, y : stats.cumu_std(x, y)
        return DataFunc(schema=schema,
                        source_data_type=DataType.TIME_SERIES,
                        fy=fy)

# DataType.AR1_ACF
def _create_ar1_acf(schema, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    fy = lambda x, y : φ**x
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    fy=fy)

# DataType.MAQ_ACF
def _create_maq_acf(schema, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fy = lambda x, y : arima.maq_acf(θ, σ, len(x))
    return DataFunc(schema=schema,
                    source_data_type=DataType.ACF,
                    fy=fy)

# DataType.FBM_MEAN
def _create_fbm_mean(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(schema=schema,
                    source_data_type=DataType.MEAN,
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
                    fy=fy,
                    fx=fx)

# DataType.BM_STD
def _create_bm_sd(schema, **kwargs):
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    fx = lambda x : x[::int(len(x)/npts)]
    fy = lambda x, y : σ*numpy.sqrt(x)
    return DataFunc(schema=schema,
                    source_data_type=DataType.SD,
                    fy=fy,
                    fx=fx)
