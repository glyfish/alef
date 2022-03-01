import numpy

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

## Brownian Noise
def brownian_noise(n):
    return numpy.random.normal(0.0, 1.0, n)

def to_noise(samples):
    nsim, npts = samples.shape
    noise = numpy.zeros((nsim, npts-1))
    for i in range(nsim):
        for j in range(npts-1):
            noise[i,j] = samples[i,j+1] - samples[i,j]
    return noise

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

def cholesky_noise(H, Δt, n, dB=None, L=None):
    if dB is None:
        dB = brownian_noise(n+1)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dB = numpy.matrix(dB)
    if L is None:
        R = autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(L) != n + 1:
        raise Exception(f"L should have length {n+1}")
    return numpy.squeeze(numpy.asarray(L*dB.T))

def cholesky(H, Δt, n, dB=None, L=None):
    if dB is None:
        dB = brownian_noise(n+1)
    if L is None:
        R = autocorrelation_matrix(H, n)
        L = numpy.linalg.cholesky(R)
    if len(dB) != n + 1:
        raise Exception(f"dB should have length {n+1}")
    dZ = cholesky_noise(H, Δt, n, dB, L)
    Z = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

# FFT Method for FBM generation
def fft_noise(H, Δt, n, dB=None):

    if dB is None:
        dB = brownian_noise(2*n)
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

def fft(H, Δt, n, dB=None):
    if dB is None:
        dB = brownian_noise(2*n)
    if len(dB) != 2*n:
        raise Exception(f"dB should have length {2*n}")
    dZ = fft_noise(H, Δt, n, dB=dB)
    Z = numpy.zeros(n)
    for i in range(1, n):
        Z[i] = Z[i - 1] + dZ[i]
    return Z

# Brownian Motion Simulations
def brownian_motion_from_noise(dB):
    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i - 1] + dB[i]
    return B

def brownian_motion(Δt, n):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return samples

def brownian_motion_with_drift(μ, σ, Δt, n):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ * Δ * numpy.sqrt(Δt)) + (μ * Δt)
    return samples

def geometric_brownian_motion(μ, σ, s0, Δt, n):
    samples = brownian_motion_with_drift(μ, σ, Δt, n)
    return to_geometric(s0, samples)

def to_geometric(s0, samples):
    return s0*numpy.exp(samples)
