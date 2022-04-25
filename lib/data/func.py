from enum import Enum
from lib import stats
from pandas import DataFrame
from lib.data.schema import (DataType, DataSchema, create_schema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing)

##################################################################################################################
# Data Function consist of the input schema and function used to compute resulting data columns
##
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
# f: Function used to compute xcol and ycol from source DataType,F is assumed to have the format
#    f(x,y) -> DataFrame
# source: DataType input into f used to compute xcol and ycol
#
class DataFunc:
    def __init__(self, schema, f=None, source_data_type=DataType.TIME_SERIES):
        self.schema = schema
        self.f = f
        self.source_data_type = source_data_type

    def apply(self, df):
        if self.f is None:
            return df
        schema = create_schema(self.source_data_type)
        x, y = schema.get_data(df)
        result = self.f(x, y)
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
    source_data_type = get_param_default_if_missing("source_data_type", DataType.TIME_SERIES, **kwargs)
    if data_type.value == DataType.PSPEC.value:
        f = lambda x, y : y
    elif data_type.value == DataType.ACF.value:
        f = lambda x, y : y
    elif data_type.value == DataType.VR_STAT.value:
        f = lambda x, y : y
    elif data_type.value == DataType.DIFF_1.value:
        f = lambda x, y : y
    elif data_type.value == DataType.DIFF_2.value:
        f = lambda x, y : y
    elif data_type.value == DataType.CUM_MEAN.value:
        f = lambda x, y : stats.cumu_mean(x, y)
    elif data_type.value == DataType.CUM_STD.value:
        f = lambda x, y : stats.cumu_std(x, y)
    elif data_type.value == DataType.AR1_ACF.value:
        φ = get_param_throw_if_missing("φ", **kwargs)
        f = lambda x, y : φ**x
    elif data_type.value == DataType.MAQ_ACF.value:
        θ = get_param_throw_if_missing("θ", **kwargs)
        σ = get_param_default_if_missing("σ", 1.0, **kwargs)
        f = lambda x, y : arima.maq_acf(θ, σ, len(x))
    elif data_type.value == DataType.FBM_MEAN.value:
        npts = get_param_throw_if_missing("npts", **kwargs)
        f = lambda x, y : numpy.full(len(x[::int(len(x)/npts)]), 0.0)
    elif data_type.value == DataType.FBM_SD.value:

    elif data_type.value == DataType.FBM_ACF.value:

    else:
        raise Exception(f"Data type is invalid: {data_type}")

    schema = create_schema(data_type)
    return DataFunc(schema, f, source_data_type)
