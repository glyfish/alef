import numpy
from pandas import DataFrame
import statsmodels.api as sm
from enum import Enum

class RegType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

###############################################################################################
# Transformations
def to_noise(samples):
    npts = len(samples)
    noise = numpy.zeros(npts-1)
    for j in range(npts-1):
        noise[j] = samples[j+1] - samples[j]
    return noise

def from_noise(dB):
    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i-1] + dB[i]
    return B

def to_geometric(samples):
    return numpy.exp(samples)

def from_geometric(samples):
    return numpy.log(samples/s0)

def diff(samples):
    n = len(samples)
    d = numpy.zeros(n-1)
    for i in range(n-1):
        d[i] = samples[i+1] - samples[i]
    return d

###############################################################################################
# Ensemble averages
def ensemble_mean(samples):
    if len(samples) == 0:
        raise Exception(f"no data")
    nsim = len(samples)
    npts = len(samples[0])
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j][i]/float(nsim)
    return mean

def ensemble_sd(samples):
    if len(samples) == 0:
        raise Exception(f"no data")
    nsim = len(samples)
    mean = ensemble_mean(samples)
    npts = len(samples[0])
    var = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            var[i] += (samples[j][i] - mean[i])**2/float(nsim)
    return numpy.sqrt(var)

def ensemble_acf(samples, nlags=None):
    if len(samples) == 0:
        raise Exception(f"no data")
    nsim = len(samples)
    if nlags is None or nlags > len(samples):
        nlags = len(samples[0])
    ac_avg = numpy.zeros(nlags)
    for j in range(nsim):
        ac = acf(samples[j], nlags).real
        for i in range(nlags):
            ac_avg[i] += ac[i]
    return ac_avg/float(nsim)

###############################################################################################
# Cumulative
def cumu_mean(y):
    ny = len(y)
    mean = numpy.zeros(ny)
    mean[0] = y[0]
    for i in range(1, ny):
        mean[i] = (float(i)*mean[i-1]+y[i])/float(i+1)
    return mean

def cumu_sd(y):
    mean = cumu_mean(y)
    ny = len(y)
    var = numpy.zeros(ny)
    var[0] = y[0]**2
    for i in range(1, ny):
        var[i] = (float(i)*var[i-1]+y[i]**2)/float(i+1)
    return numpy.sqrt(var-mean**2)

def cumu_cov(x, y):
    nsample = min(len(x), len(y))
    cov = numpy.zeros(nsample)
    meanx = cumu_mean(x)
    meany = cumu_mean(y)
    cov[0] = x[0]*y[0]
    for i in range(1, nsample):
        cov[i] = (float(i)*cov[i-1]+x[i]*y[i])/float(i+1)
    return cov-meanx*meany

###############################################################################################
# Covaraince and auto covariance implementations
def cov(x, y):
    nsample = len(x)
    meanx = numpy.mean(x)
    meany = numpy.mean(y)
    c = 0.0
    for i in range(nsample):
        c += x[i]*y[i]
    return c/nsample-meanx*meany

def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True, missing="drop")

###############################################################################################
# Power spec
def pspec(x):
    n = len(x)
    μ = x.mean()
    x_shifted = x - μ
    energy = numpy.sum(x_shifted**2)
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    power = numpy.conj(x_fft)*x_fft
    return power[1:n].real/(n*energy)

###############################################################################################
# PDF and CDF histograms
def pdf_hist(samples, range, nbins=50):
    return numpy.histogram(samples, bins=nbins, range=range, density=True)

def cdf_hist(x, pdf):
    npoints = len(pdf)
    cdf = numpy.zeros(npoints)
    for i in range(npoints):
        dx = x[i+1] - x[i]
        cdf[i] = numpy.sum(pdf[:i])*dx
    return cdf

###############################################################################################
## Aggregation
def agg(samples, m):
    n = len(samples)
    d = int(n/m)
    agg = numpy.zeros(d)
    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k]/m
    return agg

def agg_var(samples, m_vals):
    npts = len(m_vals)
    var = numpy.zeros(npts)
    for i in range(npts):
        m = int(m_vals[i])
        vals = agg(samples, m)
        mean = numpy.mean(vals)
        d = len(vals)
        for k in range(d):
            var[i] += (vals[k] - mean)**2/(d - 1)
    return var

def agg_series(samples, m):
    series = []
    for i in range(len(m)):
        series.append(agg(samples, m[i]))
    return series

def agg_time(samples, m):
    n = len(samples)
    times = []
    for i in range(len(m)):
        d = int(n/m[i])
        times.append(numpy.linspace(0, n-1, d))
    return times

###############################################################################################
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
