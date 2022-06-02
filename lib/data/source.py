from enum import Enum
from pandas import DataFrame
import uuid
import numpy

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space)

##################################################################################################################
# Specify Data Source Types used in analysis
#
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
#
# f: Function used to compute ycol from xcol, fy is assumed to have the form
#     f(x) -> ycol
#
class SourceType(Enum):
    AR = "AR"                            # AR(p) simulation
    AR_DRIFT = "AR_DRIFT"                # AR(p) with drift
    AR_OFFSET = "AR_OFFSET"              # AR(p) with offset
    MA = "MA"                            # MA(q) simulation
    ARMA = "ARMA"                        # ARMA(p, q) simulation
    ARIMA = "ARIMA"                      # ARIMA(p, d, q) simulation
    ARIMA_FROM_ARMA = "ARIMA_FROM_ARMA"  # ARIMA(p, d, q) simulation created from ARMA(p,q)
    BM_NOISE = "BM_NOISE"                # Brownian Motion noise simulation
    BM = "BM"                            # Brownian Motion simulation
    BM_DRIFT = "BM_DRIFT"                # Brownoan Motion with drift simulation
    BM_GEO = "BM_GEO"                    # Geometric Brownian motion simulation
    FBM_NOISE_CHOL = "FBM_NOISE_CHOL"    # FBM Noise simulation implemented using the Cholesky method
    FBM_NOISE_FFT = "FBM_NOISE_FFT"      # FBM Noise simulation implemented using the FFT method
    FBM_CHOL = "FBM_CHOL"                # FBM simulation implemented using the Cholesky method
    FBM_FFT = "FBM_FFT"                  # FBM Noise simulation implemented using the FFT method

    def create(self, **kwargs):
        return _create_data_source(self, **kwargs)

    def create_parameter_scan(self, *args):
        return _create_parameter_scan(self, *args)

    def create_ensemble(self, nsim, **kwargs):
        return _create_ensemble(self, nsim, **kwargs)

###################################################################################################
# DataSource is used to create input data types. DataTypes can be model simulations
# or real data. The model properies define the meta data of the source and a Function
# of the form f(x) used to create the source. The create() method returns a DataFrame()
# conmtining the DataType.
#
class DataSource:
    def __init__(self, source_type, schema, name, params, ylabel, xlabel, desc, f, x=None):
        self.source_type = source_type
        self.schema = schema
        self.name = name
        self.params = params
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.desc = desc
        self.f = f
        self.x = x

    def __repr__(self):
        return f"DataSource({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"desc=({self.desc})"

    def meta_data(self, npts):
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )

    def create(self):
        y = self.f(self.x)
        df = self.schema.create_data_frame(self.x, y)
        DataSchema.set_source_type(df, None)
        DataSchema.set_source_name(df, None)
        DataSchema.set_source_schema(df, None)
        DataSchema.set_date(df)
        DataSchema.set_type(df, self.source_type)
        DataSchema.set_name(df, self.name)
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_iterations(df, None)
        MetaData.set(df, self.meta_data(len(y)))
        return df

###################################################################################################
# Create an ensemble of specified DataSource with specified paramters
def _create_ensemble(source_type, nsim, **kwargs):
    ensemble = []
    for i in range(nsim):
        ensemble.append(_create_data_source(source_type, **kwargs))
    return ensemble

###################################################################################################
# Create a parameter scan of specified DataSource with specied parameters
def _create_parameter_scan(source_type, *args):
    dfs = []
    for kwargs in args:
        dfs.append(_create_data_source(source_type, **kwargs))
    return dfs

###################################################################################################
# Create specified DataSource with specified paramters
def _create_data_source(source_type, **kwargs):
    x = get_param_default_if_missing("x", None, **kwargs)
    if x is None:
        x = create_space(**kwargs)
    source = _get_data_source(source_type, x, **kwargs)
    return source.create()

def _get_data_source(source_type, x, **kwargs):
    if source_type.value == SourceType.AR.value:
        return _create_ar_source(source_type, x, **kwargs)
    if source_type.value == SourceType.AR_DRIFT.value:
        return _create_ar_drift_source(source_type, x, **kwargs)
    if source_type.value == SourceType.AR_OFFSET.value:
        return _create_ar_offset_source(source_type, x, **kwargs)
    if source_type.value == SourceType.MA.value:
        return _create_ma_source(source_type, x, **kwargs)
    if source_type.value == SourceType.ARMA.value:
        return _create_arma_source(source_type, x, **kwargs)
    if source_type.value == SourceType.ARIMA.value:
        return _create_arima_source(source_type, x, **kwargs)
    if source_type.value == SourceType.ARIMA_FROM_ARMA.value:
        return _create_arima_from_arms_source(source_type, x, **kwargs)
    if source_type.value == SourceType.BM_NOISE.value:
        return _create_bm_noise_source(source_type, x, **kwargs)
    if source_type.value == SourceType.BM.value:
        return _create_bm_source(source_type, x, **kwargs)
    if source_type.value == SourceType.BM_DRIFT.value:
        return _create_bm_drift_source(source_type, x, **kwargs)
    if source_type.value == SourceType.BM_GEO.value:
        return _create_bm_geo_source(source_type, x, **kwargs)
    if source_type.value == SourceType.FBM_NOISE_CHOL.value:
        return _create_fbm_noise_chol_source(source_type, x, **kwargs)
    if source_type.value == SourceType.FBM_NOISE_FFT.value:
        return _create_fbm_noise_fft_source(source_type, x, **kwargs)
    if source_type.value == SourceType.FBM_CHOL.value:
        return _create_fbm_chol_source(source_type, x, **kwargs)
    if source_type.value == SourceType.FBM_FFT.value:
        return _create_fbm_fft_source(source_type, x, **kwargs)
    else:
        raise Exception(f"Source type is invalid: {data_type}")

###################################################################################################
# Create DataSource objects for specified DataType
# SourceType.AR
def _create_ar_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp(numpy.array(φ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# SourceType.AR_DRIFT
def _create_ar_drift_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_drift(numpy.array(φ), μ, γ, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "μ": μ, "γ": γ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# SourceType.AR_OFFSET
def _create_ar_offset_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_offset(numpy.array(φ), μ, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"AR({len(φ)})-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "μ": μ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)

# SourceType.MA
def _create_ma_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)
    f = lambda x : arima.maq(numpy.array(θ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"MA({len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"MA({len(θ)})",
                      f=f,
                      x=x)

# SourceType.ARMA
def _create_arma_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arma(numpy.array(φ), numpy.array(θ), len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARMA({len(φ)}, {len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "φ": φ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARMA({len(φ)},{len(θ)})",
                      f=f,
                      x=x)

# SourceType.ARIMA
def _create_arima_source(source_type, x, **kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arima(numpy.array(φ), numpy.array(θ), d, len(x), σ)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARIMA({len(φ)}, {d}, {len(θ)})-Simulation-{str(uuid.uuid4())}",
                      params={"θ": θ, "φ": φ, "σ": σ, "d": d},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARIMA({len(φ)},{d},{len(θ)})",
                      f=f,
                      x=x)

# SourceType.ARIMA_FROM_ARMA
def _create_arima_from_arma_source(source_type, x, **kwargs):
    samples_df = get_param_throw_if_missing("samples", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    samples_schema = DataType.TIME_SERIES.schema()
    _, samples = samples_schema.get_data(samples_df)
    f = lambda x : arima.arima_from_arma(samples, d)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"ARIMA(p, {d}, q)-Simulation-{str(uuid.uuid4())}",
                      params={"d": d},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"ARIMA({d}) from ARMA",
                      f=f,
                      x=x)

# SourceType.BM_NOISE
def _create_bm_noise_source(source_type, x, **kwargs):
    f = lambda x : bm.noise(len(x))
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={},
                      ylabel=r"$\Delta S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Noise",
                      f=f,
                      x=x)

# SourceType.BM
def _create_bm_source(source_type, x, **kwargs):
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm(len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      params={"Δx": Δx},
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion",
                      f=f,
                      x=x)

# SourceType.BM_WITH_DRIFT
def _create_bm_drift_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_with_drift(μ, σ, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion",
                      f=f,
                      x=x)

# SourceType.BM_GEO
def _create_bm_geo_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_geometric(μ, σ, S0, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Geometric-BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx, "S0": S0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion",
                      f=f,
                      x=x)

# SourceType.FBM_NOISE_CHOL
def _create_fbm_noise_chol_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB, DataType.TIME_SERIES)
    else:
        ΔB = None
    f = lambda x : fbm.cholesky_noise(H, len(x[:-1]), Δx, ΔB, L)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Cholesky-FBM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Cholesky FBM Noise",
                      f=f,
                      x=x)

# SourceType.FBM_NOISE_FFT
def _create_fbm_noise_fft_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB, DataType.TIME_SERIES)
    else:
        ΔB = None
    f = lambda x : fbm.fft_noise(H, len(x), Δx, ΔB)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"FFT-FBM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"FFT FBM Noise",
                      f=f,
                      x=x)

# SourceType.FBM_CHOL
def _create_fbm_chol_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB, DataType.TIME_SERIES)
    else:
        ΔB = None
    f = lambda x : fbm.generate_cholesky(H, len(x[:-1]), Δx, ΔB, L)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Cholesky-FBM-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Cholesky FBM",
                      f=f,
                      x=x)

# SourceType.FBM_FFT
def _create_fbm_fft_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB, DataType.TIME_SERIES)
    else:
        ΔB = None
    f = lambda x : fbm.generate_fft(H, len(x), Δx, ΔB)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"FFT-FBM-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"FFT FBM",
                      f=f,
                      x=x)
