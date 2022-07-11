###############################################################################################
## Ornstein-Uhlenbeck process, mean, variance, covariance, distribution and simulators
import numpy
import math
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels.tsa as tsa

from lib.models.dist import Dist

###############################################################################################
## Ornstein-Uhlenbeck mean, variance, covariance, PDF, halflife
def mean(μ, λ, t, x0=0):
    return x0*numpy.exp(-λ*t) + μ*(1.0 - numpy.exp(-λ*t))

def var(λ, t, σ=1.0):
    return (σ**2/(2.0*λ))*(1.0 - numpy.exp(-2.0*λ*t))

def cov(λ, s, t, σ=1.0):
    c = numpy.exp(-λ*(t - s)) - numpy.exp(-λ*(t + s))
    return σ**2*c/(2.0*λ)

def var_limit(λ, σ=1.0):
    return  σ**2/(2.0*λ)

def pdf(x, μ, λ, t, σ=1.0, x0=0):
    μt = mean(μ, λ, t, x0)
    σt = numpy.sqrt(var(λ, t, σ))
    dist = Dist.NORMAL.create(loc=μt, scale=σt)
    return dist.pdf(x)

def cdf(x, μ, λ, t, σ=1.0, x0=0):
    μt = mean(μ, λ, t, x0)
    σt = numpy.sqrt(var(λ, t, σ))
    dist = Dist.NORMAL.create(loc=μt, scale=σt)
    return dist.cdf(x)

def pdf_limit(x, μ, λ, σ=1.0, x0=0):
    σl = numpy.sqrt(var_limit(λ, σ))
    dist = Dist.NORMAL.create(loc=μ, scale=σl)
    return dist.pdf(x)

def cdf_limit(x, μ, λ, σ=1.0, x0=0):
    σl = numpy.sqrt(var_limit(λ, σ))
    dist = Dist.NORMAL.create(loc=μ, scale=σl)
    return dist.cdf(x)

def mean_halflife(λ):
    return numpy.log(2)/λ

###############################################################################################
## generate n samples of x_t for a specified t
def xt(μ, λ, t, σ=1.0, x0=0, n=1):
    μt = mean(μ, λ, t, x0)
    σt = numpy.sqrt(var(λ, t, σ))
    ε = numpy.random.normal(0.0, 1.0, n)
    return μt + σt*ε

###############################################################################################
## genenerate a time series with n steps and step size Δt
def ou(μ, λ, Δt, n, σ=1.0, x0=0):
    x = numpy.zeros(n)
    ε = numpy.random.normal(0.0, 1.0, n)
    x[0] = x0
    for i in range(0, n-1):
        x[i+1] = x[i] + λ*(μ - x[i])*Δt + σ*Δt*ε[i]
    return x

###############################################################################################
## Estimate model parameters
def ou_model(samples, Δt=1.0):
    return tsa.arima.model.ARIMA(samples, order=(1, 0, 0), trend='c')

def ou_fit(samples, Δt=1.0):
    results = ou_model(samples).fit()
