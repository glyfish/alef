###############################################################################################
## Brownian motion simulators
import numpy
from datetime import datetime
import uuid

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema, create_schema)

def to_noise(samples):
    nsim, npts = samples.shape
    noise = numpy.zeros((nsim, npts-1))
    for i in range(nsim):
        for j in range(npts-1):
            noise[i,j] = samples[i,j+1] - samples[i,j]
    return noise

def from_noise(dB):
    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i-1] + dB[i]
    return B

def noise(n):
    bm_noise = numpy.random.normal(0.0, 1.0, n)
    return _create_bm_simulation_data_frame(bm_noise, n, 0.0, 1.0, 0.0, "BM Noise")

def bm(n, Δt=1):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return _create_bm_simulation_data_frame(samples, n, 0.0, σ, 0.0, "BM")

def bm_with_drift(μ, σ, n, Δt=1):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ*Δ*numpy.sqrt(Δt)) + (μ*Δt)
    return _create_bm_simulation_data_frame(samples, n, μ, σ*numpy.sqrt(Δt), 0.0, "BM With Drift")

def bm_geometric(μ, σ, s0, n, Δt=1):
    gbm_drift = μ - 0.5*σ**2
    df = bm_with_drift(gbm_drift, σ, n, Δt)
    schema = create_schema(DataType.TIME_SERIES)
    _, samples = schema.get_data(df)
    gbm = s0*numpy.exp(samples)
    return _create_bm_simulation_data_frame(samples, n, μ, σ*numpy.sqrt(Δt), S0, "Geometric BM")

def ensemble(nsim, npts, Δt = 1.0):
    samples = []
    for i in range(nsim):
        samples.append(bm(npts, Δt))
    return samples

def ensemble_with_drift(nsim, npts, μ, σ):
    samples = []
    for i in range(nsim):
        samples.append(bm.bm_with_drift(μ, σ, npts, Δt))
    return samples

def ensemble_geometric(nsim, npts, μ, σ, s0):
    samples = []
    for i in range(nsim):
        samples.append(bm.bm_geometric(μ, σ, s0, npts, Δt))
    return samples

def _create_bm_simulation_data_frame(xt, n, μ, σ, S0, desc):
    t = numpy.linspace(0, n-1, n)
    schema = create_schema(DataType.TIME_SERIES)
    meta_data = {
        "npts": n,
        "DataType": DataType.TIME_SERIES,
        "Parameters": {"σ": σ, "μ": μ, "S0": S0},
        "Description": desc,
        "xlabel": r"$t$",
        "ylabel": r"$S_t$"
    }
    df = DataSchema.create_data_frame(t, xt, MetaData.from_dict(meta_data))
    attrs = df.attrs
    attrs["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    attrs["Name"] = f"BM-Simulation-{str(uuid.uuid4())}"
    df.attrs = attrs
    return df
