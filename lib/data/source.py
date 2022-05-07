from enum import Enum
from pandas import DataFrame
from datetime import datetime
import uuid

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema, create_schema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

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

class DataSource:
    def __init__(self, data_type, params, ylabel, xlabel, desc, f, x=None):
        self.schema=create_schema(data_type)
        self.params=params
        self.ylabel=ylabel
        self.xlabel=xlabel
        self.desc=desc
        self.f=f
        self.x = x

    def __repr__(self):
        return f"DataSource({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"schema=({self.schema}), params=({self.params}), xlabel=({self.xlabel}), ylabel=({self.ylabel}), desc=({self.desc})"

    def meta_data(self, x, y):
        return MetaData(
            npts=len(y),
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )

    def create(self):
        y = self.f(self.x)
        df = DataSchema.create_data_frame(self.x, y, self.meta_data(self.x, y))
        attrs = df.attrs
        attrs["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        attrs["Name"] = f"ARMA-Simulation-{str(uuid.uuid4())}"
        df.attrs = attrs
    return df

    @staticmethod
    def ensemble(source_type, nsim):
        data_source = create_data_source(source_type, **kwargs)
        series = []
        for i in range(nsim):
            series.append(data_source.create())
        return series

    @staticmethod
    def create_source(source_type, **kwargs):
        x = get_param_default_if_missing("x", None, **kwargs)
        data_source = create_data_source(source_type, **kwargs)
        if data_source.x is None and x is None:
            raise Exception(f"x must be specified")
        if x is None:
            x = data_source.x
        return data_source.create(x)

    @staticmethod
    def create_sequence(xmax, x0=0.0, Δx=1.0)
        npts = (xmax - x0) / Δx
        return numpy.linspace(x0, xmax, npts)

###################################################################################################
# Create specified DataSource
def create_data_source(source_type, **kwargs):
    if source_type.value == SourceType.AR.value:
        return _create_ar_source(**kwargs)
    if source_type.value == SourceType.AR_DRIFT.value:
        return _create_ar_drift_source(**kwargs)
    if source_type.value == SourceType.AR_OFFSET.value:
        return _create_ar_offset_source(**kwargs)
    if source_type.value == SourceType.MA.value:
        return _create_ma_source(**kwargs)
    if source_type.value == SourceType.ARMA.value:
        return _create_arma_source(**kwargs)
    if source_type.value == SourceType.ARIMA.value:
        return _create_arima_source(**kwargs)
    if source_type.value == SourceType.ARIMA_FROM_ARMA.value:
        return _create_arima_from_arms_source(**kwargs)
    if source_type.value == SourceType.BM_NOISE.value:
        return _create_bm_noise_source(**kwargs)
    if source_type.value == SourceType.BM.value:
        return _create_bm_source(**kwargs)
    if source_type.value == SourceType.BM_WITH_DRIFT.value:
        return _create_bm_with_drift_source(**kwargs)
    if source_type.value == SourceType.BM_GEO.value:
        return _create_bm_geo_source(**kwargs)
    if source_type.value == SourceType.FBM_NOISE_CHOL.value:
        return _create_fbm_noise_chol_source(**kwargs)
    if source_type.value == SourceType.FBM_NOISE_FFT.value:
        return _create_fbm_noise_fft_source(**kwargs)
    if source_type.value == SourceType.FBM_CHOL.value:
        return _create_fbm_chol_source(**kwargs)
    if source_type.value == SourceType.FBM_FFT.value:
        return _create_fbm_fft_source(**kwargs)
    else:
        raise Exception(f"Source type is invalid: {data_type}")

###################################################################################################
# Create DataSource objects for specified DataType
# SourceType.AR
def _create_ar_source(**kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp(φ, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"φ": φ, "σ": σ, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"AR({len(φ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

def _create_ar_drift_source(**kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_drift(φ, μ, γ, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"φ": φ, "μ": μ, "γ": γ, "σ": σ, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"AR({len(φ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

def _create_ar_offset_source(**kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(φ, list)
    f = lambda x : arima.arp_offset(φ, μ, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"φ": φ, "μ": μ, "σ": σ, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"AR({len(φ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.MA
def _create_ma_source(**kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(θ, list)
    f = lambda x : arima.maq(θ, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"θ": θ, "σ": σ, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"MA({len(θ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.ARMA
def _create_arma_source(**kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arma(φ, θ, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"θ": θ, "φ": φ, "σ": σ, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"ARMA({len(φ)},{len(θ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.ARIMA
def _create_arima_source(**kwargs):
    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    f = lambda x : arima.arima(φ, θ, d, npts, σ)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"θ": θ, "φ": φ, "σ": σ, "d": d, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"ARIMA({len(φ)},{d},{len(θ)})",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.ARIMA_FROM_ARMA
def _create_arima_from_arma_source(**kwargs):
    samples_df = get_param_throw_if_missing("samples", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    xmax = get_param_default_if_missing("xmax", npts, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)
    samples_schema = create_schema(DataType.TIME_SERIES)
    _, samples = samples_schema.get_data(samples_df)
    f = lambda x : arima.arima_from_arma(samples, d)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"d": d, "npts": npts},
               ylabel=r"$S_t$",
               xlabel=r"$t$",
               desc=f"ARIMA({d}) from ARMA",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.BM_NOISE
def _create_bm_noise_source(**kwargs):
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    f = lambda x : bm.noise(npts)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"npts": npts},
               ylabel=r"$\Delta S_t$",
               xlabel=r"$t$",
               desc=f"Brownian Noise",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.BM
def _create_bm_source(**kwargs):
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    f = lambda x : bm.bm(npts, Δt)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"npts": npts, "Δt": Δt},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"Brownian Motion",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.BM_WITH_DRIFT
def _create_bm_with_drift_source(**kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    f = lambda x : bm.bm_with_drift(μ, σ, npts, Δt)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"σ": σ, "μ": μ, "Δt": Δt, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"Brownian Motion",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.BM_GEO
def _create_bm_geo_source(**kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    f = lambda x : bm.bm_geometric(μ, σ, S0, npts, Δt)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"σ": σ, "μ": μ, "Δt": Δt, "S0": S0, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"Brownian Motion",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.FBM_NOISE_CHOL
def _create_fbm_noise_chol_source(**kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    f = lambda x : fbm.cholesky_noise(H, npts, Δt, dB, L)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"H": H, "Δt": Δt, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"Cholesky FBM Noise",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.FBM_NOISE_FFT
def _create_fbm_noise_fft_source(**kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    f = lambda x : fbm.fft_noise(H, npts, Δt, dB)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"H": H, "Δt": Δt, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"FFT FBM Noise",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.FBM_CHOL
def _create_fbm_chol_source(**kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    f = lambda x : fbm.generate_cholesky(H, npts, Δt, dB, L)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"H": H, "Δt": Δt, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"Cholesky FBM",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))

# SourceType.FBM_FFT
def _create_fbm_fft_source(**kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    f = lambda x : fbm.generate_fft(H, npts, Δt, dB)
    DataSource(data_type=DataType.TIME_SERIES,
               params={"H": H, "Δt": Δt, "npts": npts},
               ylabel=r"$\S_t$",
               xlabel=r"$t$",
               desc=f"FFT FBM",
               f=f,
               x=SourceType.create_sequence(x0, xmax, Δx))
