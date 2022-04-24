from enum import Enum
from pandas import concat

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
    CUM_MEAN = 8        # Cumulative mean
    CUM_STD = 9         # Cumulative standard deviation
    MEAN = 10           # Mean as a function of time
    STD = 11            # Standard deviation as a function of time
    AR1_ACF = 12        # AR(1) Autocorrelation function
    MAQ_ACF = 13        # MA(q) Autocorrelation function

##################################################################################################################
## create shema for data type: The schema consists of the DataFrame columns used by the
## DataType
class DataSchema:
    def __init__(self, xcol, ycol):
        self.xcol = xcol
        self.ycol = ycol

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
    if data_type.value == DataType.TIME_SERIES.value:
        return DataSchema(xcol="Time", ycol="S(t)")
    elif data_type.value == DataType.PSPEC.value:
        return DataSchema(xcol="Frequency", ycol="Power Spectrum")
    elif data_type.value == DataType.ACF.value:
        return DataSchema(xcol="Lag", ycol="Autocorrelation")
    elif data_type.value == DataType.VR_STAT.value:
        return DataSchema(xcol="Lag", ycol="Variance Ratio")
    elif data_type.value == DataType.DIFF_1.value:
        return DataSchema(xcol="Time", ycol="Difference 1")
    elif data_type.value == DataType.DIFF_2.value:
        return DataSchema(xcol="Time", ycol="Difference 2")
    elif data_type.value == DataType.CUM_MEAN.value:
        return DataSchema(xcol="Time", ycol="Cumulative Mean")
    elif data_type.value == DataType.CUM_STD.value:
        return DataSchema(xcol="Time", ycol="Cumulative Standard Deviation")
    elif data_type.value == DataType.MEAN.value:
        return DataSchema(xcol="Time", ycol="Mean")
    elif data_type.value == DataType.STD.value:
        return DataSchema(xcol="Time", ycol="Standard Deviation")
    elif data_type.value == DataType.GENERIC.value:
        return DataSchema(xcol="x", ycol="y")
    else:
        raise Exception(f"Data type is invalid: {data_type}")
