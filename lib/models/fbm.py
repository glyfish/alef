###############################################################################################
## Fractional Brownian Motion variance, covariance, simulators paramemeter esimation
## and statistical significance tests
import numpy

from lib.models import bm
from lib.models.dist import (TestHypothesis, Dist)
from lib.models.reports import VarianceRatioTestReport

###############################################################################################
## Variance, Covariance and Autocorrleation
def var(H, n):
    return n**(2.0*H)

def cov(H, s, n):
    return 0.5*(n**(2.0*H) + s**(2.0*H) - numpy.abs(n-s)**(2.0*H))

def acf(H, n):
    return 0.5*(abs(n-1.0)**(2.0*H) + (n+1.0)**(2.0*H) - 2.0*n**(2.0*H))

def acf_matrix(H, n):
    γ = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(n+1):
            if i != j :
                γ[i,j] = acf(H, numpy.abs(i-j))
            else:
                γ[i,j] = 1.0
    return γ

###############################################################################################
# Cholesky Method for FBM generation
def cholesky_decompose(H, n):
    l = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(i+1):
            if j == 0 and i == 0:
                l[i,j] = 1.0
            elif j == 0:
                l[i,j] = acf(H, i) / l[0,0]
            elif i == j:
                l[i,j] = numpy.sqrt(l[0,0] - numpy.sum(l[i,0:i]*l[i,0:i].T))
            else:
                l[i,j] = (acf(H, i - j) - numpy.sum(l[i,0:j]*l[j,0:j].T)) / l[j,j]
    return l

def cholesky_noise(H, n, Δt=1, dB=None, L=None):
    if dB is None:
        dB = bm.noise(n+1)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dB = numpy.matrix(dB)
    if L is None:
        R = acf_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(L) != n + 1:
        raise Exception(f"L should have length {n+1}")
    return numpy.squeeze(numpy.asarray(L*dB.T))

def generate_cholesky(H, n, Δt=1, dB=None, L=None):
    if dB is None:
        dB = bm.noise(n+1)
    if L is None:
        R = acf_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dZ = cholesky_noise(H, n, Δt, dB, L)
    Z = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

###############################################################################################
# FFT Method for FBM generation
def fft_noise(H, n, Δt=1, dB=None):
    if dB is None:
        dB = bm.noise(2*n)
    if len(dB) != 2*n:
        raise Exception(f"dB should have length {2*n}")

    # Compute first row of circulant matrix with embedded autocorrelation
    C = numpy.zeros(2*n)
    for i in range(2*n):
        if i == 0:
            C[i] = 1.0
        if i == n:
            C[i] = 0.0
        elif i < n:
            C[i] = acf(H, i)
        else:
            C[i] = acf(H, 2*n-i)

    # Compute circulant matrix eigen values
    Λ = numpy.fft.fft(C).real
    if numpy.any([l < 0 for l in Λ]):
        raise Exception(f"Eigenvalues are negative")

    # Compute product of Fourier Matrix and Brownian noise
    J = numpy.zeros(2*n, dtype=numpy.cdouble)
    J[0] = numpy.sqrt(Λ[0])*numpy.complex(dB[0], 0.0) / numpy.sqrt(2.0 * n)
    J[n] = numpy.sqrt(Λ[n])*numpy.complex(dB[n], 0.0) / numpy.sqrt(2.0 * n)

    for i in range(1, n):
        J[i] = numpy.sqrt(Λ[i])*numpy.complex(dB[i], dB[n+i]) / numpy.sqrt(4.0 * n)
        J[2*n-i] = numpy.sqrt(Λ[2*n-i])*numpy.complex(dB[i], -dB[n+i]) / numpy.sqrt(4.0 * n)

    Z = numpy.fft.fft(J)

    return Z[:n].real

# generate fractional brownian motion using the FFT method
def generate_fft(H, n, Δt=1, dB=None):
    if dB is None:
        dB = bm.noise(2*n)
    if len(dB) != 2*n:
        raise Exception(f"dB should have length {2*n}")
    dZ = fft_noise(H, n, Δt, dB=dB)
    Z = numpy.zeros(n)
    for i in range(1, n):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

###############################################################################################
## Variance Ratio Test
# The homoscedastic test statistic is used n the analysis.
def vr_test(samples, s_vals=[4, 6, 10, 16, 24], sig_level=0.1, hyp_type=TestHypothesis.TWO_TAIL):
    test_stats = [vr_stat_homo(samples, s) for s in s_vals]
    if hyp_type.value == TestHypothesis.TWO_TAIL.value:
        return _var_test_two_tail(test_stats, s_vals, sig_level, hyp_type)
    elif hyp_type.value == TestHypothesis.UPPER_TAIL.value:
        return _var_test_upper_tail(test_stats, s_vals, sig_level, hyp_type)
    elif hyp_type.value == TestHypothesis.LOWER_TAIL.value:
        return _var_test_lower_tail(test_stats, s_vals, sig_level, hyp_type)
    else:
        raise Exception(f"Hypothesis test type is invalid: {hyp_type}")

# perform two tail variance ratio test
def _var_test_two_tail(test_stats, s_vals, sig_level, hyp_type):
    sig_level = sig_level/2.0
    dist = Dist.NORMAL.create()
    lower_critical_value = dist.ppf(sig_level)
    upper_critical_value = dist.ppf(1.0 - sig_level)
    p_values = [2.0*(1.0 - dist.cdf(numpy.abs(stat))) for stat in test_stats]
    return VarianceRatioTestReport(2.0*sig_level, hyp_type, s_vals, test_stats,
                                   p_values, [lower_critical_value, upper_critical_value])

# perform upper tail variance ratio test
def _var_test_upper_tail(test_stats, s_vals, sig_level, hyp_type):
    dist = Dist.NORMAL.create()
    upper_critical_value = dist.ppf(1.0 - sig_level)
    p_values = [1.0 - dist.cdf(stat) for stat in test_stats]
    return VarianceRatioTestReport(sig_level, hyp_type, s_vals, test_stats, p_values, [None, upper_critical_value])

# perform lower tail variance ratio test
def _var_test_lower_tail(test_stats, s_vals, sig_level, hyp_type):
    dist = Dist.NORMAL.create()
    lower_critical_value = dist.ppf(sig_level)
    p_values = [dist.cdf(stat) for stat in test_stats]
    return VarianceRatioTestReport(sig_level, hyp_type, s_vals, test_stats, p_values, [lower_critical_value, None])

# lag variance
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

# variance ratio
def vr(samples, s):
    vars = lag_var(samples, s)
    var1 = lag_var(samples, 1)
    return vars/(s*var1)

def vr_scan(samples, s_vals):
    return [vr(samples, s) for s in s_vals]

# Homoscedastic variance Ratio
def vr_stat_homo(samples, s):
    if s == 1:
        return 0
    t = len(samples) - 1
    r = vr(samples, s)
    θ = 2.0*(2.0*s - 1.0)*(s - 1.0)/(3.0*s*t)
    return (r - 1.0)/numpy.sqrt(θ)

def vr_stat_homo_scan(samples, s_vals):
    return [vr_stat_homo(samples, s) for s in s_vals]

# Heteroscedastic variance Ratio
def vr_stat_hetero(samples, s):
    t = len(samples) - 1
    r = vr(samples, s)
    θ = theta_factor(samples, s)
    return (r - 1.0)/numpy.sqrt(θ)

def delta_factor(samples, j):
    t = len(samples) - 1
    μ = (samples[t] - samples[0]) / t
    factor = 0.0
    for i in range(j+1, t):
        f1 = (samples[i] - samples[i-1] - μ)**2
        f2 = (samples[i-j] - samples[i-j-1] - μ)**2
        factor += f1*f2
    return factor / lag_var(samples, 1)**2

def theta_factor(samples, s):
    t = len(samples) - 1
    μ = (samples[t] - samples[0]) / t
    factor = 0.0
    for j in range(1, s):
        delta = delta_factor(samples, j)
        factor += delta*(2.0*(s-j)/s)**2
    return factor/t**2

def vr_stat_hetero_scan(samples, s_vals):
    return [vr_stat_hetero(samples, s) for s in s_vals]
