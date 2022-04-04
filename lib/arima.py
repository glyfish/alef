import numpy
import statsmodels.api as sm
import statsmodels.tsa as tsa

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
def ar(φ, n, σ):
    p = len(φ)
    samples = numpy.zeros(n)
    ε = σ*bm.noise(n)
    for i in range(p, n):
        samples[i] = ε[i]
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
def df_test(samples):
    return adfuller_report(samples, 'n')

def adf_report(samples):
    return adfuller_report(samples, 'c')

def adf_report_with_trend(samples):
    return adfuller_report(samples, 'ct')

def adf_test(samples):
    return adfuller_test(samples, 'c')

def adfuller_test(samples, test_type):
    adf_result = sm.tsa.stattools.adfuller(samples, regression=test_type)
    return adf_result[0] < adf_result[4]["5%"]

def adfuller_report(samples, test_type):
    adf_result = sm.tsa.stattools.adfuller(samples, regression=test_type)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    isStationary = adf_result[0] < adf_result[4]["5%"]
    print(f"Is Stationary at 5%: {isStationary}")
    print("Critical Values")
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))
    return adf_result[0] < adf_result[4]["5%"]
