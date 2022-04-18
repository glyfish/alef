###############################################################################################
## Ornstein-Uhlenbeck process, mean, variance, covariance, distribution and simulators
import numpy
import math
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels.tsa as tsa

###############################################################################################
## Ornstein-Uhlenbeck mean, variance, covariance, PDF, halflife
def mean(μ, λ, t, x0=0):
    return x0*numpy.exp(-λ*t) + μ*(1.0 - numpy.exp(-λ*t))

def std(λ, t, σ=1.0):
    var = σ**2*(1.0 - numpy.exp(-2.0*λ*t))/(2.0*λ)
    return numpy.sqtt(var)

def cov(λ, s, t, σ=1.0):
    c = numpy.exp(-λ*(t - s)) - numpy.exp(-λ*(t + s))
    return σ**2*c/(2.0*λ)

def std_limit(λ, σ=1.0):
    return σ/numpy.sqrt(2.0*λ)

def pdf(x, μ, λ, t, σ=1.0, x0=0):
    μt = mean(μ, λ, t, x0)
    σt = std(λ, t, σ)
    return numpy.exp(((x - μt)/(2.0*σt**2))**2)/(σt*numpy(2.0*math.pi))

def pdf(x, μ, λ, t, σ=1.0, x0=0):
    μt = mean(μ, λ, t, x0)
    σt = std(λ, t, σ)
    return numpy.exp(((x - μt)/(2.0*σt**2))**2)/(σt*numpy(2.0*math.pi))

def pdf_limit(x, μ, λ, σ=1.0, x0=0):
    σl = std_limit(λ, σ)
    return

def mean_halflife(λ):
    return numpy.log(2)/λ

###############################################################################################
## generate n samples of xt for a specified t
def xt(μ, λ, t, σ=1.0, x0=0, n=1):
    μt = mean(μ, λ, t, x0)
    σt = std(λ, t, σ)
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
def ou_model(samples, Δt=1.0, report=False):
    return tsa.arima.model.ARIMA(samples, order=(1, 0, 0), trend='c')

def ou_fit(samples, Δt=1.0):
    results = ou_model(samples).fit()

class OrnsteinUhlenbeckResults:
    def __init__(self, results, Δt):
        conf_int = results.conf_int()
        self.delta_t = Δt
        self._offset_est = (conf_int[0][1] - conf_int[0][0])/2.0
        self._offset_error = self.offset_est - conf_int[0][0]
        self._coeff_est = (conf_int[1][1] - conf_int[1][0])/2.0
        self._coeff_error = self.coeff_est - conf_int[1][0]
        self._sigma2_est = (conf_int[2][1] - conf_int[2][0])/2.0
        self._sigma2_error = self.sigma2_est - conf_int[2][0]

    def mu_est(self):
        return self._offset_est

    def lambda_est(self):
        return numpy.log(self._coeff_est)/self.delta_t

    def lambda_error(self):
        return -self._coeff_error/(self._coeff_est*self.delta_t)

    def sigma2(self):
        return 2.0*self.lambda_est()*self._sigma2_est/(1.0 - self._coeff_est**2)
