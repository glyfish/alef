###############################################################################################
## Simulator of Dickey-Fuller distribution
import numpy
from matplotlib import pyplot
from lib import config
from scipy import special

###############################################################################################
## brownian motion simulator
def scaled_brownian_noise(n):
    return numpy.random.normal(0.0, 1.0/numpy.sqrt(n), n)

def brownian_motion(bn, t):
    return sum(bn[:t])

def modified_chi_squared(x):
    return 2.0*numpy.exp(-(2.0*x+1.0)/2.0) / numpy.sqrt(2.0*numpy.pi*(2.0*x+1.0))

###############################################################################################
# stochastic integral simulation
# \int_0^1{B(s)dB(s)}
def stochastic_integral_ensemble_1(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_1(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_1(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n):
        val += brownian_motion(bn, i-1)*bn[i]
    return val

###############################################################################################
# Analytic Solution of integral 1
# \frac{1}{2}[B^2(1) - 1]
def stochastic_integral_solution_1(n):
    return 0.5*(numpy.random.normal(0.0, 1.0, n)**2 - 1.0)

###############################################################################################
# stochastic integral simulation
# \int_0^1{B^2(s)ds}
def stochastic_integral_ensemble_2(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_2(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_2(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return val/n

###############################################################################################
# stochastic integral simulation
# \sqrt{\int_0^1{B^2(s)ds}}
def stochastic_integral_ensemble_3(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_3(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return numpy.sqrt(val/n)

###############################################################################################
# Dickey-Fuller Statistic distribution
# \frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_0^1{B^2(s)ds}}
def dist_ensemble(n, nsim):
    vals = numpy.zeros(nsim)
    numerator = stochastic_integral_solution_1(nsim)
    for i in range(nsim):
        vals[i] = numerator[i] / stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

###############################################################################################
# Dickey-Fuller statistic
def statistic(samples, σ=1.0):
    nsample = len(samples)
    delta_numerator = 0.0
    var = 0.0
    for i in range(1, nsample):
        delta = samples[i] - samples[i-1]
        delta_numerator += samples[i-1] * delta
        var += samples[i-1]**2
    return delta_numerator / (numpy.sqrt(var)*σ**2)
