###############################################################################################
## Brownian motion simulators
import numpy
from datetime import datetime
import uuid

def noise(n):
    return numpy.random.normal(0.0, 1.0, n)

def bm(n, Δt=1):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return samples

def bm_with_drift(μ, σ, n, Δt=1):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ*Δ*numpy.sqrt(Δt)) + (μ*Δt)
    return samples

def bm_geometric(μ, σ, s0, n, Δt=1):
    gbm_drift = μ - 0.5*σ**2
    samples = bm_with_drift(gbm_drift, σ, n, Δt)
    return s0*numpy.exp(samples)
