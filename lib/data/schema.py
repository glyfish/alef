import numpy
from datetime import datetime
from enum import Enum
from pandas import (DataFrame, concat)

##################################################################################################################
# Specify DataTypes used in analysis
class DataType(Enum):
    TIME_SERIES = "TIME_SERIES"              # Time Series
    FOURIER_TRANS = "FOURIER_TRANS"          # Fourier transform
    ACF = "ACF"                              # Autocorrelation Functiom

    def schema(self):
        return _create_schema(self)

##################################################################################################################
## create shema for data type: The schema consists of the DataFrame columns used by the
## DataType
class DataSchema:
    def __init__(self, xcol, ycol, data_type):
        self.xcol = xcol
        self.ycol = ycol
        self.data_type = data_type

    def __repr__(self):
        return f"DataSchema({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return  f"xcol=({self.xcol}), ycol=({self.ycol}), data_type=({self.data_type})"

    def get_data(self, df):
        if not self._is_in(df):
            raise Exception(f"DataFrame does not contain schema={self}")
        meta_data = df.attrs
        xcol = self.xcol
        ycol = self.ycol
        return df[xcol], df[ycol]

    def get_data_from_list(self, dfs):
        data = []
        for df in dfs:
            x, y = self.get_data(df)
            data.append(y)
        return x, data

    def _is_in(self, df):
        cols = df.columns
        return (self.xcol in cols) and (self.ycol in cols)

    def create_data_frame(self, x, y):
        df = DataFrame({
            self.xcol: x,
            self.ycol: y
        })
        return df

    @classmethod
    def get_schema_data(cls, df):
        schema = cls.get_schema(df)
        return schema.get_data(df)

    @classmethod
    def get_type(cls, df):
        return df.attrs["Type"]

    @classmethod
    def set_type(cls, df, type):
        df.attrs["Type"] = type

    @classmethod
    def get_schema(cls, df):
        return df.attrs["Schema"]

    @classmethod
    def set_schema(cls, df, schema):
        df.attrs["Schema"] = schema

    @classmethod
    def get_source_schema(cls, df):
        return df.attrs["Schema"]

    @classmethod
    def set_source_schema(cls, df, schema):
        df.attrs["Schema"] = schema

    @classmethod
    def get_source_type(cls, df):
        return df.attrs["SourceType"]

    @classmethod
    def set_source_type(cls, df, schema):
        df.attrs["SourceType"] = schema

    @classmethod
    def get_source_name(cls, df):
        return df.attrs["SourceName"]

    @classmethod
    def set_source_name(cls, df, name):
        df.attrs["SourceName"] = name

    @classmethod
    def get_name(cls, df):
        return df.attrs["Name"]

    @classmethod
    def set_name(cls, df, name):
        df.attrs["Name"] = name

    @classmethod
    def get_date(cls, df):
        return df.attrs["Date"]

    @classmethod
    def set_date(cls, df):
        df.attrs["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    @classmethod
    def get_iterations(cls, df):
        return df.attrs["SchemaIterations"]

    @classmethod
    def set_iterations(cls, df, iter):
        df.attrs["SchemaIterations"] = iter

##################################################################################################################
## create shema for data type
def _create_schema(data_type):
    if data_type.value == DataType.TIME_SERIES.value:
        return DataSchema("t", "S(t)", data_type)
    elif data_type.value == DataType.FOURIER_TRANS.value:
        return DataSchema("ω", "s(ω)", data_type)
    elif data_type.value == DataType.ACF.value:
        return DataSchema("τ", "ρ(τ)", data_type)
    else:
        raise Exception(f"Data type is invalid: {data_type}")
