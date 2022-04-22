from enum import Enum
from lib import stats

##################################################################################################################
# Specify DataTypes used in analysis
class DataType(Enum):
    GENERIC = 1         # Unknown data type
    TIME_SERIES = 2     # Time Series
    PSPEC = 3           # Power Spectrum
    ACF = 4             # Autocorrelation function
    VR_STAT = 5         # FBM variance ratio test statistic
    DIFF_1 = 6          # First time series difference
    DIFF_2 = 7          # Second time series difference
    VAR = 8             # Variance
    COV = 9             # Covariance
    MEAN = 10           # Mean
    STD = 11            # Standard deviation
    CUM_MEAN = 12       # Cumulative mean
    CUM_STD = 13        # Cumulative standard deviation

##################################################################################################################
# Data definition consist of the x and y column and function used to compute data columns
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
# f: Function used to compute xcol and ycol from source DataType
# source: DataType input into f used to compute xcol and ycol
#
class DataConfig:
    def __init__(self, xcol, ycol, f=None, source=DataType.TIME_SERIES):
        self.xcol = xcol
        self.ycol = ycol
        self.f = f
        self.source = source

    def apply(self, df):
        if self.f is None:
            return df
        config = create_data_config(self.source)
        x, y = config.get_data(df)
        range = self.f(x, y)
        return DataConfig.concat(df, range)

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
    def concat(df1, df2):
        df = pandas.concat([df1, df2], axis=1)
        df.attrs = df1.attrs | df2.attrs
        return df.loc[:,~df.columns.duplicated()]

## create definition for data type
def create_data_config(data_type, **kwrags):
    if data_type.value == DataType.TIME_SERIES.value:
        return DataConfig(xcol="Time", ycol="S(t)")
    elif data_type.value == DataType.PSPEC.value:
        return DataConfig(xcol="Frequency", ycol="Power Spectrum")
    elif data_type.value == DataType.ACF.value:
        return DataConfig(xcol="Lag", ycol="Autocorrelation")
    elif data_type.value == DataType.VR_STAT.value:
        return DataConfig(xcol="Lag", ycol="Variance Ratio")
    elif data_type.value == DataType.DIFF_1.value:
        return DataConfig(xcol="Time", ycol="Difference 1")
    elif data_type.value == DataType.DIFF_2.value:
        return DataConfig(xcol="Time", ycol="Difference 2")
    elif data_type.value == DataType.VAR.value:
        return DataConfig(xcol="Time", ycol="Variance")
    elif data_type.value == DataType.COV.value:
        return DataConfig(xcol="Time", ycol="Covariance")
    elif data_type.value == DataType.MEAN.value:
        return DataConfig(xcol="Time", ycol="Mean")
    elif data_type.value == DataType.STD.value:
        return DataConfig(xcol="Time", ycol="STD")
    elif data_type.value == DataType.CUM_MEAN.value:
        f = lambda x, y : stats.cummean(y)
        return DataConfig(xcol="Time", ycol="Cumulative Mean", f=f)
    elif data_type.value == DataType.CUM_STD.value:
        return DataConfig(xcol="Time", ycol="Cumulative Standard Deviation")
    elif data_type.value == DataType.GENERIC.value:
        return DataConfig(xcol="x", ycol="y")
    else:
        raise Exception(f"Data type is invalid: {data_type}")
