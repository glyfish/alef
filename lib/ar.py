import numpy
from lib import bm

def arq_series(q, φ, σ, n):
    samples = numpy.zeros(n)
    ε = σ*bm.noise(n)
    for i in range(q, n):
        samples[i] = ε[i]
        for j in range(0, q):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples
