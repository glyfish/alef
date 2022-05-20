import numpy
from enum import Enum
from pandas import (DataFrame, concat)

##################################################################################################################
# Specify DataTypes used in analysis
class DataType(Enum):
    GENERIC = "GENERIC"                    # Unknown data type
    TIME_SERIES = "TIME_SERIES"            # Time Series
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
    VR = "VR"                              # Variance Ratio use in test for brownian motion
    VR_STAT = "VR_STAT"                    # FBM variance ratio test statistic
    PACF = "PACF"                          # Partial Autocorrelation function
    BM = "BM"                              # Brownian motion computed from brownian motion noise
    ARMA_MEAN = "ARMA_MEAN"                # ARMA(p,q) MEAN
    AR1_SD = "AR1_SD"                      # AR(1) standard seviation
    MAQ_SD = "MAQ_SD"                      # MA(q) standard deviation
    AR1_OFFSET_MEAN = "AR1_OFFSET_MEAN"    # AR(1) with constant offset mean
    AR1_OFFSET_SD = "AR1_OFFSET_SD"        # AR(1) with offset standard deviation

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
        if not self.is_in(df):
            raise Exception(f"DataFrame does not contain schema={self}")

        meta_data = df.attrs
        xcol = self.xcol
        ycol = self.ycol
        if ycol in meta_data.keys():
            npts = meta_data[ycol]["npts"]
        else:
            y = df[ycol]
            npts = len(y[~numpy.isnan(y)])
        return df[xcol][:npts], df[ycol][:npts]

    def get_data_from_list(self, dfs):
        data = []
        for df in dfs:
            x, y = self.get_data(df)
            data.append(y)
        return x, data

    def is_in(self, df):
        cols = df.columns
        return (self.xcol in cols) and (self.ycol in cols)

    @staticmethod
    def concatinate(df1, df2):
        df = concat([df1, df2], axis=1)
        df.attrs = df1.attrs | df2.attrs
        return df.loc[:,~df.columns.duplicated()]

    @staticmethod
    def create_data_frame(x, y, meta_data):
        schema = meta_data.schema
        df = DataFrame({
            schema.xcol: x,
            schema.ycol: y
        })
        df.attrs[schema.ycol] = meta_data.data
        return df

    @staticmethod
    def get_data_type(df, data_type):
        schema = create_schema(data_type)
        return schema.get_data(df)

    @staticmethod
    def get_data_type_from_list(dfs, data_type):
        schema = create_schema(data_type)
        return schema.get_data_from_list(df)

    @staticmethod
    def create(data_type):
        return create_schema(data_type)

##################################################################################################################
## create shema for data type
def create_schema(data_type):
    if data_type.value == DataType.GENERIC.value:
        return DataSchema("x", "y", data_type)
    elif data_type.value == DataType.TIME_SERIES.value:
        return DataSchema("Time", "S(t)", data_type)
    elif data_type.value == DataType.PSPEC.value:
        return DataSchema("Frequency", "Power Spectrum", data_type)
    elif data_type.value == DataType.ACF.value:
        return DataSchema("ACF Lag", "ACF", data_type)
    elif data_type.value == DataType.PACF.value:
        return DataSchema("PACF Lag", "PACF", data_type)
    elif data_type.value == DataType.VR_STAT.value:
        return DataSchema("Lag", "Variance Ratio", data_type)
    elif data_type.value == DataType.DIFF.value:
        return DataSchema("Difference Time", "Difference", data_type)
    elif data_type.value == DataType.CUMU_MEAN.value:
        return DataSchema("Time", "Cumulative Mean", data_type)
    elif data_type.value == DataType.CUMU_SD.value:
        return DataSchema("Time", "Cumulative SD", data_type)
    elif data_type.value == DataType.MEAN.value:
        return DataSchema("Time", "Mean", data_type)
    elif data_type.value == DataType.SD.value:
        return DataSchema("Time", "SD", data_type)
    elif data_type.value == DataType.AR1_ACF.value:
        return DataSchema("AR(1) Lag", "AR(1) ACF", data_type)
    elif data_type.value == DataType.MAQ_ACF.value:
        return DataSchema("MA(q) Lag", "MA(q) ACF", data_type)
    elif data_type.value == DataType.FBM_MEAN.value:
        return DataSchema("FBM Mean Time", "FBM Mean", data_type)
    elif data_type.value == DataType.FBM_SD.value:
        return DataSchema("FBM SD Time", "FBM SD", data_type)
    elif data_type.value == DataType.FBM_ACF.value:
        return DataSchema("FBM ACF Time", "FBM ACF", data_type)
    elif data_type.value == DataType.FBM_COV.value:
        return DataSchema("FBM COV Time", "FBM COV", data_type)
    elif data_type.value == DataType.BM_MEAN.value:
        return DataSchema("BM Mean Time", "BM Mean", data_type)
    elif data_type.value == DataType.BM_DRIFT_MEAN.value:
        return DataSchema("BM Drift Mean Time", "BM Drift Mean", data_type)
    elif data_type.value == DataType.BM_SD.value:
        return DataSchema("BM SD Time", "BM SD", data_type)
    elif data_type.value == DataType.GBM_MEAN.value:
        return DataSchema("GBM Mean Time", "GBM Mean", data_type)
    elif data_type.value == DataType.GBM_SD.value:
        return DataSchema("GBM STD Time", "GBM SD", data_type)
    elif data_type.value == DataType.AGG_VAR.value:
        return DataSchema("Agg Time", "Aggregated Variance", data_type)
    elif data_type.value == DataType.VR.value:
        return DataSchema("VR Time", "Variance Ratio", data_type)
    elif data_type.value == DataType.BM.value:
        return DataSchema("Time", "BM", data_type)
    elif data_type.value == DataType.ARMA_MEAN.value:
        return DataSchema("Time", "ARMA(p,q) Mean", data_type)
    elif data_type.value == DataType.AR1_SD.value:
        return DataSchema("Time", "AR(1) SD", data_type)
    elif data_type.value == DataType.MAQ_SD.value:
        return DataSchema("Time", "MA(q) SD", data_type)
    elif data_type.value == DataType.AR1_OFFSET_MEAN.value:
        return DataSchema("Time", "AR(1) Offset Mean", data_type)
    elif data_type.value == DataType.AR1_OFFSET_SD.value:
        return DataSchema("Time", "AR(1) Offset SD", data_type)
    else:
        raise Exception(f"Data type is invalid: {data_type}")
