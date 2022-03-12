import numpy
import statsmodels.api as sm

from lib import bm

def ar(φ, n, σ):
    p = len(φ)
    samples = numpy.zeros(n)
    ε = σ*bm.noise(n)
    for i in range(p, n):
        samples[i] = ε[i]
        for j in range(0, p):
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

def yw_acf(x, max_lag):
    pacf, _ = sm.regression.yule_walker(x, order=max_lag, method='mle')
    return pacf

def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

def pacf(samples, nlags):
    return sm.tsa.stattools.pacf(samples, nlags=nlags, method="ywunbiased")
