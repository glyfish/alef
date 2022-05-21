###############################################################################################
## Fractional Brownian Motion variance, covariance, simulators paramemeter esimation
## and statistical significance tests
import numpy
from tabulate import tabulate

from lib.models import bm
from lib.dist import (HypothesisType, DistType, DistFuncType, distribution_function)

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
def vr_test(samples, s_vals=[4, 6, 10, 16, 24], sig_level=0.05, test_type=HypothesisType.TWO_TAIL, report=False, tablefmt="fancy_grid"):
    test_stats = [vr_stat_homo(samples, s) for s in s_vals]
    if test_type.value == HypothesisType.TWO_TAIL.value:
        return _var_test_two_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt)
    elif test_type.value == HypothesisType.UPPER_TAIL.value:
        return _var_test_upper_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt)
    elif test_type.value == HypothesisType.LOWER_TAIL.value:
        return _var_test_lower_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt)
    else:
        raise Exception(f"Hypothesis test type is invalid: {test_type}")

# perform two tail variance ratio test
def _var_test_two_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt):
    sig_level = sig_level/2.0
    dist_params = [1.0, 0.0]
    ppf = distribution_function(DistType.NORMAL, DistFuncType.PPF, dist_params)
    lower_critical_value = ppf(sig_level)
    upper_critical_value = ppf(1.0 - sig_level)

    nstats = len(test_stats)
    npass = 0

    for stat in test_stats:
        if stat >= lower_critical_value and stat <= upper_critical_value:
            npass += 1

    result = npass >= nstats/2.0

    cdf = distribution_function(DistType.NORMAL, DistFuncType.CDF, dist_params)
    p_values = [2.0*(1.0 - cdf(numpy.abs(stat))) for stat in test_stats]

    results = VarianceRatioTestReport(result,  2.0*sig_level, "Two Tail", s_vals, test_stats, p_values, [lower_critical_value, upper_critical_value])
    _var_test_report(results, report, tablefmt)
    return results

# perform upper tail variance ratio test
def _var_test_upper_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt):
    dist_params = [1.0, 0.0]
    ppf = distribution_function(DistType.NORMAL, DistFuncType.PPF, dist_params)
    upper_critical_value = ppf(1.0 - sig_level)

    nstats = len(test_stats)
    npass = 0

    for stat in test_stats:
        if stat <= upper_critical_value:
            npass += 1

    result = npass >= nstats/2.0

    cdf = distribution_function(DistType.NORMAL, DistFuncType.CDF, dist_params)
    p_values = [1.0 - cdf(stat) for stat in test_stats]

    results = VarianceRatioTestReport(result, sig_level, "Upper Tail", s_vals, test_stats, p_values, [None, upper_critical_value])
    _var_test_report(results, report, tablefmt)
    return results

# perform lower tail variance ratio test
def _var_test_lower_tail(test_stats, s_vals, sig_level, test_type, report, tablefmt):
    dist_params = [1.0, 0.0]
    ppf = distribution_function(DistType.NORMAL, DistFuncType.PPF, dist_params)
    lower_critical_value = ppf(sig_level)

    nstats = len(test_stats)
    npass = 0

    for stat in test_stats:
        if stat >= lower_critical_value:
            npass += 1

    result = npass >= nstats/2.0

    cdf = distribution_function(DistType.NORMAL, DistFuncType.CDF, dist_params)
    p_values = [cdf(stat) for stat in test_stats]

    results = VarianceRatioTestReport(result, sig_level, "Lower Tail", s_vals, test_stats, p_values, [lower_critical_value, None])
    _var_test_report(results, report, tablefmt)
    return results

# print test report
def _var_test_report(results, report, tablefmt):
    if not report:
        return
    table = results.table(tablefmt)
    print(table[0])
    print(table[1])

# lag variance
def lag_var(samples, s):
    t = len(samples) - 1
    μ = (samples[t] - samples[0]) / t
    m = (t - s + 1.0)*(1.0 - s/t)
    σ = 0.0
    for i in range(s, t+1):
        σ += (samples[i] - samples[i-s] - μ*s)**2
    return σ/m

# variance ratio
def vr(samples, s):
    vars = lag_var(samples, s)
    var1 = lag_var(samples, 1)
    return vars/(s*var1)

# Homoscedastic variance Ratio
def vr_stat_homo(samples, s):
    if s == 1:
        return 0
    t = len(samples) - 1
    r = vr(samples, s)
    θ = 2.0*(2.0*s - 1.0)*(s - 1.0)/(3.0*s*t)
    return (r - 1.0)/numpy.sqrt(θ)

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

# variance ratio test report
class VarianceRatioTestReport:
    def __init__(self, status, sig_level, test_type, s, statistics, p_values, critical_values):
        self.status = status
        self.sig_level = sig_level
        self.test_type = test_type
        self.s = s
        self.statistics = statistics
        self.p_values = p_values
        self.critical_values = critical_values

    def __repr__(self):
        return f"VarianceRatioTestReport(status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values})"

    def __str__(self):
        return f"status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values}"

    def _header(self, tablefmt):
        test_status = "Passed" if self.status else "Failed"
        header = [["Result", test_status], ["Test Type", self.test_type], ["Significance", f"{int(100.0*self.sig_level)}%"]]
        if self.critical_values[0] is not None:
            header.append(["Lower Critical Value", format(self.critical_values[0], '1.3f')])
        if self.critical_values[1] is not None:
            header.append(["Upper Critical Value", format(self.critical_values[1], '1.3f')])
        return tabulate(header, tablefmt=tablefmt)

    def _results(self, tablefmt):
        if self.critical_values[0] is None:
            z_result = [self.statistics[i] < self.critical_values[1] for i in range(len(self.statistics))]
        elif self.critical_values[1] is None:
            z_result = [self.statistics[i] > self.critical_values[0] for i in range(len(self.statistics))]
        else:
            z_result = [self.critical_values[1] > self.statistics[i] > self.critical_values[0] for i in range(len(self.statistics))]
        z_result = ["Passed" if zr else "Failed" for zr in z_result]
        s_result = [int(s_val) for s_val in self.s]
        stat_result = [format(stat, '1.3f') for stat in self.statistics]
        pval_result = [format(pval, '1.3f') for pval in self.p_values]
        results = [s_result]
        results.append(stat_result)
        results.append(pval_result)
        results.append(z_result)
        results = numpy.transpose(numpy.array(results))
        return tabulate(results, headers=["s", "Z(s)", "pvalue", "Result"], tablefmt=tablefmt)

    def table(self, tablefmt):
        header = self._header(tablefmt)
        result = self._results(tablefmt)
        return [header, result]
