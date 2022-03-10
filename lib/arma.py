import numpy
import statsmodels.api as sm

from lib import bm

def ar(φ, σ, n):
    q = len(φ)
    samples = numpy.zeros(n)
    ε = σ*bm.noise(n)
    for i in range(q, n):
        samples[i] = ε[i]
        for j in range(0, q):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ar1(φ, n, σ):
    return arp(numpy.array([φ]), n, σ)

def arp(φ, n, σ=1):
    φ = numpy.r_[1, -φ]
    δ = numpy.array([1.0])
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

def maq(δ, n, σ=1.0):
    φ = numpy.array(1.0)
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

def generate(φ, δ, n, σ=1):
    φ = numpy.r_[1, -φ]
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)
