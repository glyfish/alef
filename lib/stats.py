"""
stats.py

Useful statistical functions.

"""

import numpy
from copy import deepcopy
from pandas import DataFrame
import statsmodels.api as sm
from enum import Enum

class RegType(Enum):
    """
    Specify regression model to use for linear repression models.

        Values
    ------
    LINEAR : 1
        Assume a linear relation between regression variables.
            y = a*x + b
        where a and b be are regression constants.
    LOG : 2
        Assume power law relation between the regression variables.
            y = b*x**a
        where a and b be are regression constants.
    XLOG : 3
        Assume an exponential relationship between the regression variables.
            y = b*exp(a*x)
        where a and b be are regression constants.
    YLOG : 4
        Assume a logarithmic relation between the regression variables.
            y = b*ln(a*x)
        where a and b be are regression constants.
    """

    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

def to_noise(samples: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Difference the given samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Differenced data
    """

    return diff(samples)

def from_noise(dB: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Integrate the given samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Integrate data.
    """

    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i-1] + dB[i]
    return B

def to_geometric(samples: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Take the exponential of the given samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Exponential of sampled data.
    """

    return numpy.exp(samples)

def from_geometric(samples: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Take the log of the given samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Logarithm of sampled data.
    """

    return numpy.log(samples)

def ndiff(samples: numpy.ndarray[float], ndiff: int) -> numpy.ndarray[float]:
    """
    Take the specified number of differences of the samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Samples differenced n times.
    """

    result = deepcopy(samples)
    i = 0
    while i < ndiff:
        result = diff(result)
        i += 1
    return result

def diff(samples: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Difference the given samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Differenced data
    """

    n = len(samples)
    d = numpy.zeros(n-1)
    for i in range(n-1):
        d[i] = samples[i+1] - samples[i]
    return d

def ensemble_mean(samples: numpy.ndarray[float]) -> numpy.ndarray[float]:
    """
    Compute the time varying mean of the sampled ensemble.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    numpy.ndarray[float]
        Differenced data

    Raises
    ______
    Exception
        Samples are not a two dimensional array that containsdata.
    """

    if len(samples) == 0:
        raise Exception(f"no data")
    if len(samples.shape) == 2:
        raise Exception(f"Input must be a two dimensional array.")

    nsim = len(samples)
    npts = len(samples[0])
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j][i]/float(nsim)
    return mean

def ensemble_var(samples, Δt=1.0):
    if len(samples) == 0:
        raise Exception(f"no data")
    nsim = len(samples)
    mean = ensemble_mean(samples)
    npts = len(samples[0])
    var = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            var[i] += (samples[j][i] - mean[i])**2/float(nsim)
    return var/Δt

def ensemble_sd(samples, Δt=1.0):
    return numpy.sqrt(ensemble_var(samples, Δt))

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

def cumu_var(y, Δt=1.0):
    mean = cumu_mean(y)
    ny = len(y)
    var = numpy.zeros(ny)
    var[0] = y[0]**2
    for i in range(1, ny):
        var[i] = (float(i)*var[i-1]+y[i]**2)/float(i+1)
    return (var-mean**2)/Δt

def cumu_sd(y, Δt=1.0):
    return numpy.sqrt(cumu_var(y, Δt))

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

def cov_fft(x, y):
    n = len(x)
    x_shifted = x - x.mean()
    y_shifted = y - y.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    y_padded = numpy.concatenate((y_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    y_fft = numpy.fft.fft(y_padded)
    h_fft = numpy.conj(x_fft)*y_fft
    cc = numpy.fft.ifft(h_fft)
    return cc[0:n] / float(n)

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
    npoints = len(x)
    cdf = numpy.zeros(npoints)
    dx = x[1] - x[0]
    for i in range(npoints):
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

def agg_time(x, m):
    n = len(x)
    d = int(n/m)
    return numpy.linspace(x[0], x[n-1], d)

###############################################################################################
## Lag variance
def lag_var(samples, s):
    t = len(samples) - 1
    μ = (samples[t] - samples[0]) / t
    m = (t - s + 1.0)*(1.0 - s/t)
    σ = 0.0
    for i in range(int(s), t+1):
        σ += (samples[i] - samples[i-s] - μ*s)**2
    return σ/m

def lag_var_scan(samples, s_vals):
    return [lag_var(samples, s) for s in s_vals]

###############################################################################################
## OLS
def OLS(y, x, type=RegType.LINEAR):
    if type == RegType.LOG:
        x = numpy.log10(x)
        y = numpy.log10(y)
    x = sm.add_constant(x)
    return sm.OLS(y, x, missing='drop')

def OLS_fit(y, x, type=RegType.LINEAR):
    model = OLS(y, x, type=type)
    results = model.fit()
    return results


###############################################################################################
## Multivariate normal random variable
def multivariate_normal(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)
