import numpy
import statsmodels.api as sm
from enum import Enum

class RegType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

def ensemble_mean(samples):
    nsim, npts = samples.shape
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j,i] / float(nsim)
    return mean

def ensemble_std(samples):
    mean = ensemble_mean(samples)
    nsim, npts = samples.shape
    std = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            std[i] += (samples[j,i] - mean[i])**2 / float(nsim)
    return numpy.sqrt(std)

def ensemble_acf(samples, nlags=None):
    nsim, npts = samples.shape
    if nlags is None:
        nlags = npts
    ac_avg = numpy.zeros(npts)
    for j in range(nsim):
        ac = acf(samples[j], nlags).real
        for i in range(npts):
            ac_avg[i] += ac[i]
    return ac_avg / float(nsim)

def cummean(samples):
    nsample = len(samples)
    mean = numpy.zeros(nsample)
    mean[0] = samples[0]
    for i in range(1, nsample):
        mean[i] = (float(i)*mean[i-1]+samples[i])/float(i+1)
    return mean

def cumsigma(samples):
    nsample = len(samples)
    mean = cummean(samples)
    var = numpy.zeros(nsample)
    var[0] = samples[0]**2
    for i in range(1, nsample):
        var[i] = (float(i)*var[i-1]+samples[i]**2)/float(i+1)
    return numpy.sqrt(var-mean**2)

def cumcov(x, y):
    nsample = min(len(x), len(y))
    cov = numpy.zeros(nsample)
    meanx = cummean(x)
    meany = cummean(y)
    cov[0] = x[0]*y[0]
    for i in range(1, nsample):
        cov[i] = (float(i)*cov[i-1]+x[i]*y[i])/float(i+1)
    return cov-meanx*meany

def cov(x, y):
    nsample = len(x)
    meanx = numpy.mean(x)
    meany = numpy.mean(y)
    c = 0.0
    for i in range(nsample):
        c += x[i]*y[i]
    return c/nsample-meanx*meany

def pspec(x):
    n = len(x)
    μ = x.mean()
    x_shifted = x - μ
    energy = numpy.sum(x_shifted**2)
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    power = numpy.conj(x_fft)*x_fft
    return power[1:n].real/(n*energy)

def pdf_hist(samples, range, nbins=50):
    return numpy.histogram(samples, bins=nbins, range=range, density=True)

def cdf_hist(x, pdf):
    npoints = len(pdf)
    cdf = numpy.zeros(npoints)
    for i in range(npoints):
        dx = x[i+1] - x[i]
        cdf[i] = numpy.sum(pdf[:i])*dx
    return cdf

def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

## OLS
def OLS(y, x, type=RegType.LINEAR):
    if type == RegType.LOG:
        x = numpy.log10(x)
        y = numpy.log10(y)
    x = sm.add_constant(x)
    return sm.OLS(y, x)

def OLS_fit(y, x, type=RegType.LINEAR):
    model = OLS(y, x, type=type)
    results = model.fit()
    results.summary()
    return results
