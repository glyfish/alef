from enum import Enum
from pandas import concat

##################################################################################################################
# Specify DataTypes used in analysis
class DataType(Enum):
    GENERIC = 1             # Unknown data type
    TIME_SERIES = 2         # Time Series
    PSPEC = 3               # Power Spectrum
    ACF = 4                 # Autocorrelation function
    VR_STAT = 5             # FBM variance ratio test statistic
    DIFF_1 = 6              # First time series difference
    DIFF_2 = 7              # Second time series difference
    CUMU_MEAN = 8           # Cumulative mean
    CUMU_SD = 9             # Cumulative standard deviation
    MEAN = 10               # Mean as a function of time
    SD = 11                 # Standard deviation as a function of time
    AR1_ACF = 12            # AR(1) Autocorrelation function
    MAQ_ACF = 13            # MA(q) Autocorrelation function
    FBM_MEAN = 14           # Fractional Brownian Motion mean
    FBM_SD = 15             # Fractional Brownian Motion standard deviation
    FBM_ACF = 16            # Fractional Brownian Motion autocorrelation function
    BM_MEAN = 17            # Brownian Motion mean
    BM_DRIFT_MEAN = 18      # Brownian Motion model mean with data
    BM_SD = 19              # Brownian Motion model standard deviation with data
    GBM_MEAN = 20           # Geometric Brownian Motion model mean with data
    GBM_SD = 21             # Geometric Brownian Motion model standard deviation with data
    LAGG_VAR = 22           # Lagged variance computed
    VR = 23                 # Variance Ratio use in test for brownian motion


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

    def is_in(self, df):
        cols = df.columns
        return (self.xcol in cols) and (self.ycol in cols)

    @staticmethod
    def concatinate(df1, df2):
        df = concat([df1, df2], axis=1)
        df.attrs = df1.attrs | df2.attrs
        return df.loc[:,~df.columns.duplicated()]

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
        return DataSchema("Lag", "ACF", data_type)
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
        return DataSchema("Lag", "AR(1) Autocorrelation", data_type)
    elif data_type.value == DataType.MAQ_ACF.value:
        return DataSchema("Lag", "MA(q) Autocorrelation", data_type)
    elif data_type.value == DataType.FBM_MEAN.value:
        return DataSchema("FBM Mean Time", "FBM Mean", data_type)
    elif data_type.value == DataType.FBM_SD.value:
        return DataSchema("FBM SD Time", "FBM Standard Deviation", data_type)
    elif data_type.value == DataType.FBM_ACF.value:
        return DataSchema("Time", "FBM ACF", data_type)
    else:
        raise Exception(f"Data type is invalid: {data_type}")
