from enum import Enum
from pandas import (DataFrame, concat)

##################################################################################################################
# Meta Data Schema
class MetaData:
    def __init__(self, npts, data_type, params, desc, xlabel, ylabel):
        self.npts = npts
        self.schema = create_schema(data_type)
        self.params = params
        self.desc = desc
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data =  {
          "npts": npts,
          "DataType": data_type,
          "Parameters": params,
          "Description": desc,
          "ylabel": ylabel,
          "xlabel": xlabel
        }

    @staticmethod
    def from_dict(meta_data):
        return MetaData(meta_data["npts"],
                        meta_data["DataType"],
                        meta_data["Parameters"],
                        meta_data["Description"],
                        meta_data["xlabel"],
                        meta_data["ylabel"])

    @staticmethod
    def get(df, data_type):
        schema = create_schema(data_type)
        return MetaData.from_dict(df.attrs[schema.ycol])

##################################################################################################################
# Parameter Estimates
class ParamEst:
    def __init__(self, est, err):
            self.est = est
            self.err = err
            self.data = [est, err]

class ARMAEst:
    def __init__(self, type, const, sigma2, params):
        self.type = type
        self.const = const
        self.params = params
        self.data = {"Type": type, "Const": const, "Parameters": params, "Sigma2": sigma2 }

class OLSEst:
    def __init__(self, const, params):
        aelf.const = const
        self.params = params

##################################################################################################################
# Specify DataTypes used in analysis
class DataType(Enum):
    GENERIC = 1             # Unknown data type
    TIME_SERIES = 2         # Time Series
    PSPEC = 3               # Power Spectrum
    ACF = 4                 # Autocorrelation function
    DIFF_1 = 5              # First time series difference
    DIFF_2 = 6              # Second time series difference
    CUMU_MEAN = 7           # Cumulative mean
    CUMU_SD = 8             # Cumulative standard deviation
    MEAN = 9                # Mean as a function of time
    SD = 10                 # Standard deviation as a function of time
    AR1_ACF = 11            # AR(1) Autocorrelation function
    MAQ_ACF = 12            # MA(q) Autocorrelation function
    FBM_MEAN = 13           # Fractional Brownian Motion mean
    FBM_SD = 14             # Fractional Brownian Motion standard deviation
    FBM_ACF = 15            # Fractional Brownian Motion autocorrelation function
    BM_MEAN = 16            # Brownian Motion mean
    BM_DRIFT_MEAN = 17      # Brownian Motion model mean with data
    BM_SD = 18              # Brownian Motion model standard deviation
    GBM_MEAN = 19           # Geometric Brownian Motion model mean
    GBM_SD = 20             # Geometric Brownian Motion model standard deviation
    AGG_VAR = 21            # Aggregated variance
    VR = 22                 # Variance Ratio use in test for brownian motion
    VR_STAT = 23            # FBM variance ratio test statistic
    PACF = 24

##################################################################################################################
## create shema for data type: The schema consists of the DataFrame columns used by the
## DataType
class DataSchema:
    def __init__(self, xcol, ycol, data_type):
        self.xcol = xcol
        self.ycol = ycol
        self.data_type = data_type

    def get_data(self, df):
        meta_data = df.attrs
        xcol = self.xcol
        ycol = self.ycol
        if ycol in meta_data.keys():
            npts = meta_data[ycol]["npts"]
        else:
            y = df[ycol]
            npts = len(y[~numpy.isnan(y)])
        return df[xcol][:npts], df[ycol][:npts]

    def get_meta_data(self, df):
        meta_data = df.attrs
        return MetaData.from_dict(meta_data[self.ycol])

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
    elif data_type.value == DataType.DIFF_1.value:
        return DataSchema("Time", "Difference 1", data_type)
    elif data_type.value == DataType.DIFF_2.value:
        return DataSchema("Time", "Difference 2", data_type)
    elif data_type.value == DataType.CUMU_MEAN.value:
        return DataSchema("Time", "Cumulative Mean", data_type)
    elif data_type.value == DataType.CUMU_SD.value:
        return DataSchema("Time", "Cumulative Standard Deviation", data_type)
    elif data_type.value == DataType.MEAN.value:
        return DataSchema("Time", "Mean", data_type)
    elif data_type.value == DataType.SD.value:
        return DataSchema("Time", "Standard Deviation", data_type)
    elif data_type.value == DataType.AR1_ACF.value:
        return DataSchema("AR(1) Lag", "AR(1) Autocorrelation", data_type)
    elif data_type.value == DataType.MAQ_ACF.value:
        return DataSchema("MA(q) Lag", "MA(q) Autocorrelation", data_type)
    elif data_type.value == DataType.FBM_MEAN.value:
        return DataSchema("FBM Mean Time", "FBM Mean", data_type)
    elif data_type.value == DataType.FBM_SD.value:
        return DataSchema("FBM SD Time", "FBM Standard Deviation", data_type)
    elif data_type.value == DataType.FBM_ACF.value:
        return DataSchema("FBM ACF Time", "FBM ACF", data_type)
    elif data_type.value == DataType.BM_MEAN.value:
        return DataSchema("BM Mean Time", "BM Mean", data_type)
    elif data_type.value == DataType.BM_DRIFT_MEAN.value:
        return DataSchema("BM Drift Mean Time", "BM Drift Mean", data_type)
    elif data_type.value == DataType.BM_SD.value:
        return DataSchema("BM SD Time", "BM Standard Deviation", data_type)
    elif data_type.value == DataType.GBM_MEAN.value:
        return DataSchema("GBM Mean Time", "GBM Mean", data_type)
    elif data_type.value == DataType.GBM_SD.value:
        return DataSchema("GBM STD Time", "GBM Standard Deviation", data_type)
    elif data_type.value == DataType.AGG_VAR.value:
        return DataSchema("Agg Time", "Aggregated Variance", data_type)
    elif data_type.value == DataType.VR.value:
        return DataSchema("VR Time", "Variance Ratio", data_type)
    else:
        raise Exception(f"Data type is invalid: {data_type}")

##################################################################################################################
# Specify meta_data schema
