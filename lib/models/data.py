from enum import Enum

# Specify PlotConfig for curve, comparison and stack plots
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

# Configurations used in plots
class DataConfig:
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
            y_total = df[ycol]
            npts = len(y_total[~numpy.isnan(y_total)])

        return df[xcol][:npts], df[ycol][:npts]

## plot data type
def create_data_type(data_type):
    if data_type.value == DataType.TIME_SERIES.value:
        return DataConfig(xcol="Time", ycol="Xt")
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
    elif data_type.value == DataType.GENERIC.value:
        return DataConfig(xcol="x", ycol="y")
    else:
        raise Exception(f"Data type is invalid: {data_type}")
