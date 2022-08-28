
import numpy
from statsmodels.tsa.api import VAR as pyvar

# Transformation to VAR(1) companion form

def phi_comp(φ):
    l, n, _ = φ.shape
    p = φ[0]
    for i in range(1,l):
        p = numpy.concatenate((p, φ[i]), axis=1)
    for i in range(1, n):
        if i == 1:
            r = numpy.eye(n)
        else:
            r = numpy.zeros((n, n))
        for j in range(1,l):
            if j == i - 1:
                r = numpy.concatenate((r, numpy.eye(n)), axis=1)
            else:
                r = numpy.concatenate((r, numpy.zeros((n, n))), axis=1)
        p = numpy.concatenate((p, r), axis=0)
    return numpy.matrix(p)

def mean_comp(μ):
    n = len(μ)
    p = numpy.zeros(n**2)
    p[:n] = μ
    return numpy.matrix([p]).T

def omega_comp(ω):
    n, _ = ω.shape
    p = numpy.zeros((n**2, n**2))
    p[:n, :n] = ω
    return numpy.matrix(p)

def vec(m):
    _, n = m.shape
    v = numpy.matrix(numpy.zeros(n**2)).T
    for i in range(n):
        d = i*n
        v[d:d+n] = m[:,i]
    return v

def unvec(v):
    n2, _ = v.shape
    n = int(numpy.sqrt(n2))
    m = numpy.matrix(numpy.zeros((n, n)))
    for i in range(n):
        d = i*n
        m[:,i] = v[d:d+n]
    return m

# First and second stationary order moments

def mean(φ, μ):
    Φ = phi_comp(φ)
    Μ = mean_comp(μ)
    n, _ = Φ.shape
    tmp = numpy.matrix(numpy.eye(n)) - Φ
    return numpy.linalg.inv(tmp)*Μ

def cov(φ, ω):
    Ω = omega_comp(ω)
    Φ = phi_comp(φ)
    n, _ = Φ.shape
    eye = numpy.matrix(numpy.eye(n**2))
    tmp = eye - numpy.kron(Φ, Φ)
    inv_tmp = numpy.linalg.inv(tmp)
    vec_var = inv_tmp * vec(Ω)
    return unvec(vec_var)

def sd(φ, ω):
    return numpy.sqrt(var(φ, ω))

def autocovariance(φ, ω, n):
    t = numpy.linspace(0, n-1, n)
    Φ = phi_comp(φ)
    Σ = cov(φ, ω)
    l, _ = Φ.shape
    γ = numpy.zeros((n, l, l))
    γ[0] = numpy.matrix(numpy.eye(l))
    for i in range(1,n):
        γ[i] = γ[i-1]*Φ
    for i in range(n):
        γ[i] = Σ*γ[i].T
    return γ

# Compute eigen values of parameter matrix (for stationarity all eigen values must satisfy |λ| < 1)
def eig(φ):
    Φ = phi_comp(φ)
    λ, _ = numpy.linalg.eig(Φ)
    return λ

def isStationary(φ):
    for λ in eig(φ):
        if abs(λ) >= 1:
            return False
    return True

# Simulators

def var(x0, μ, φ, Ω, n):
    x0 = numpy.array(x0.T)
    l, m = x0.shape
    xt = numpy.zeros((n, m))
    μ = numpy.squeeze(numpy.array(μ), axis=1)
    ε = numpy.random.multivariate_normal(μ, Ω, n)
    for i in range(l):
        xt[i] = x0[i]
    for i in range(l, n):
        xt[i] = ε[i]
        for j in range(l):
            t1 = φ[j]*numpy.matrix(xt[i-j-1]).T
            xt[i] += numpy.squeeze(numpy.array(t1), axis=1)
    return numpy.matrix(xt).T
