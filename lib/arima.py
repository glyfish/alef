import numpy
import statsmodels.api as sm
import statsmodels.tsa as tsa
from tabulate import tabulate

from lib import bm

## MA(q) standard deviation amd ACF
def maq_sigma(θ, σ):
    q = len(θ)
    v = 0
    for u in θ:
        v += u**2
    return σ * numpy.sqrt(v + 1)

def maq_cov(θ, σ):
    q = len(θ)
    c = numpy.zeros(q)
    s = numpy.zeros(q)
    for n in range(1,q):
        for i in range(q-n):
            c[n] += θ[i]*θ[i+n]
    for n in range(q):
        s[n] = θ[n]
    return σ**2 * (c + s)

def maq_acf(θ, σ, max_lag):
    ac = maq_cov(θ, σ) / maq_sigma(θ, σ)**2
    ac_eq = numpy.zeros(max_lag)
    ac_eq[0] = 1
    for i in range(len(ac)):
        ac_eq[i+1] = ac[i]
    return ac_eq

## AR1 standard deviation and autocorrelation
def ar1_sigma(φ, σ):
    return numpy.sqrt(σ**2/(1.0-φ**2))

def ar1_acf(φ, nvals):
    return [φ**n for n in range(nvals)]

## AR(p) simulators
def ar(φ, x0, n, σ):
    p = len(φ)
    samples = numpy.zeros(n)
    for i in range(0, q):
        samples[i] = x0[i]
    ε = σ*bm.noise(n)
    for i in range(p, n):
        samples[i] = ε[i]
        for j in range(0, p):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ar_drift(φ, μ, γ, n, σ):
    p = len(φ)
    samples = numpy.zeros(n)
    ε = σ*bm.noise(n)
    for i in range(p, n):
        samples[i] = ε[i] + γ*i + μ
        for j in range(0, p):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ar1(φ, n, σ=1.0):
    return arp(numpy.array([φ]), n, σ)

def arp(φ, n, σ=1.0):
    φ = numpy.r_[1, -φ]
    δ = numpy.array([1.0])
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

## MA(q) simulator
def maq(δ, n, σ=1.0):
    φ = numpy.array([1.0])
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

## ARMA(p,q) simulator
def arma(φ, δ, n, σ=1):
    φ = numpy.r_[1, -φ]
    δ = numpy.r_[1, δ]
    return sm.tsa.arma_generate_sample(φ, δ, n, σ)

### ARIMA(p,d,q) simulator
def diff(samples):
    n = len(samples)
    d = numpy.zeros(n-1)
    for i in range(n-1):
        d[i] = samples[i+1] - samples[i]
    return d

def arima(φ, δ, d, n, σ=1.0):
    assert d <= 2, "d must equal 1 or 2"
    samples = arma_generate_sample(φ, δ, n, σ)
    if d == 1:
        return numpy.cumsum(samples)
    else:
        for i in range(2, n):
            samples[i] = samples[i] + 2.0*samples[i-1] - samples[i-2]
        return samples

## Yule-Walker ACF and PACF
def yw(x, max_lag):
    pacf, _ = sm.regression.yule_walker(x, order=max_lag, method='mle')
    return pacf

def pacf(samples, nlags):
    return sm.tsa.stattools.pacf(samples, nlags=nlags, method="ywunbiased")

## ARIMA parameter estimation
def ar_estimate(samples, order):
    return tsa.arima.model.ARIMA(samples, order=(order, 0, 0)).fit()

def ma_estimate(samples, order):
    return tsa.arima.model.ARIMA(samples, order=(0, 0, order)).fit()

## ADF Test
def adf_test(samples, report=False, tablefmt="fancy_grid"):
    return _adfuller_test(samples, 'nc', report, tablefmt)

def adf_test_offset(samples, report=False, tablefmt="fancy_grid"):
    return _adfuller_test(samples, 'c', report, tablefmt)

def adf_test_drift(samples, report=False, tablefmt="fancy_grid"):
    return _adfuller_test(samples, 'ct', report, tablefmt)

def _adfuller_test(samples, test_type, report, tablefmt):
    results = sm.tsa.stattools.adfuller(samples, regression=test_type)
    _adfuller_report(results, report, tablefmt)
    stat = results[0]
    status = stat >= results[4]["10%"]
    return status

def _adfuller_report(results, report, tablefmt):
    if not report:
        return
    stat = results[0]
    header = [["Test Statistic", stat],
              ["pvalue", results[1]],
              ["Lags", results[2]],
              ["Number Obs", results[3]]]
    status = ["Passed" if stat >= results[4][sig] else "Failed" for sig in ["1%", "5%", "10%"]]
    results = [["1%", results[4]["1%"], status[0]],
               ["5%", results[4]["5%"], status[1]],
               ["10%", results[4]["10%"], status[2]]]
    headers = ["Signisficance", "Critical Value", "Result"]
    print(tabulate(header, tablefmt=tablefmt))
    print(tabulate(results, tablefmt=tablefmt, headers=headers))
