###############################################################################################
## Brownian motion simulators
import numpy

def noise(n):
    return numpy.random.normal(0.0, 1.0, n)

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
        B[i] = B[i - 1] + dB[i]
    return B

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
        samples[i] = samples[i-1] + (σ * Δ * numpy.sqrt(Δt)) + (μ * Δt)
    return samples

def bm_geometric(μ, σ, s0, n, Δt=1):
    gbm_drift = μ - 0.5*σ**2
    samples = bm_with_drift(gbm_drift, σ, n, Δt)
    return s0*numpy.exp(samples)
