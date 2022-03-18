import numpy
from lib import bm

## moments
def var(H, n):
    return n**(2.0*H)

def cov(H, s, n):
    return 0.5*(n**(2.0*H) + s**(2.0*H) - numpy.abs(n - s)**(2.0*H))

def autocorrelation(H, n):
    return 0.5*(abs(n-1.0)**(2.0*H) + (n+1.0)**(2.0*H) - 2.0*n**(2.0*H))

def autocorrelation_matrix(H, n):
    γ = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(n+1):
            if i != j :
                γ[i,j] = autocorrelation(H, numpy.abs(i-j))
            else:
                γ[i,j] = 1.0
    return γ

# Cholesky Method for FBM generation
def cholesky_decompose(H, n):
    l = numpy.matrix(numpy.zeros([n+1, n+1]))
    for i in range(n+1):
        for j in range(i+1):
            if j == 0 and i == 0:
                l[i,j] = 1.0
            elif j == 0:
                l[i,j] = autocorrelation(H, i) / l[0,0]
            elif i == j:
                l[i,j] = numpy.sqrt(l[0,0] - numpy.sum(l[i,0:i]*l[i,0:i].T))
            else:
                l[i,j] = (autocorrelation(H, i - j) - numpy.sum(l[i,0:j]*l[j,0:j].T)) / l[j,j]
    return l

def cholesky_noise(H, n, Δt=1, dB=None, L=None):
    if dB is None:
        dB = bm.noise(n+1)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dB = numpy.matrix(dB)
    if L is None:
        R = autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(L) != n + 1:
        raise Exception(f"L should have length {n+1}")
    return numpy.squeeze(numpy.asarray(L*dB.T))

def generate_cholesky(H, n, Δt=1, dB=None, L=None):
    if dB is None:
        dB = bm.noise(n+1)
    if L is None:
        R = autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dZ = cholesky_noise(H, n, Δt, dB, L)
    Z = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

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
            C[i] = autocorrelation(H, i)
        else:
            C[i] = autocorrelation(H, 2*n-i)

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

def to_geometric(s0, samples):
    return s0*numpy.exp(samples)

def from_geometric(s0, samples):
    return numpy.log(samples/s0)

def to_noise(samples):
    nsim, npts = samples.shape
    noise = numpy.zeros((nsim, npts-1))
    for i in range(nsim):
        for j in range(npts-1):
            noise[i,j] = samples[i,j+1] - samples[i,j]
    return noise

# Variance aggregation
def agg(samples, m):
    n = len(samples)
    d = int(n / m)
    agg = numpy.zeros(d)
    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k] / m
    return agg

def agg_var(samples, npts, m_min):
    n = len(samples)
    m_vals = numpy.logspace(numpy.log10(m_min), numpy.log10(n/10.0), npts)
    agg_var = numpy.zeros(npts)
    for i in range(npts):
        m = int(m_vals[i])
        agg = aggregated_process(samples, m)
        agg_mean = numpy.mean(agg)
        d = len(agg)
        for k in range(d):
            agg_var[i] += (agg[k] - agg_mean)**2/(d - 1)
    return agg_var

def m_logspace(nagg, m_min, npts):
    return numpy.logspace(numpy.log10(m_min), numpy.log10(npts/10.0), nagg)

def agg_var_H(agg_var, m_vals):
    x = numpy.log10(m_vals)
    y = numpy.log10(agg_var)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results

def agg_series(samples,m):
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
