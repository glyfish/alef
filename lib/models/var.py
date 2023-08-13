
import numpy
from statsmodels.tsa.api import VAR as pyvar

def mean(φ: numpy.matrix[float], μ: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Compute the stationary mean matrix for a VAR(n) process with the given parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    μ: numpy.matrix[float]
        VAR(n) process offset matrix.

    Returns
    -------
    numpy.matrix[float]
        Stationary mean matrix.
    """

    Φ = phi_comp(φ)
    Μ = mean_comp(μ)
    n, _ = Φ.shape
    tmp = numpy.matrix(numpy.eye(n)) - Φ
    return numpy.linalg.inv(tmp)*Μ


def cov(φ: numpy.matrix[float], ω: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Compute the stationary covariance matrix for the given VAR(n) process
    parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    ω: numpy.matrix[float]
        VAR(n) process gaussian noise autocovariance matrix.

    Returns
    -------
    numpy.matrix[float]
        Stationary covariance matrix.
    """

    Ω = omega_comp(ω)
    Φ = phi_comp(φ)
    n, _ = Φ.shape
    eye = numpy.matrix(numpy.eye(n**2))
    tmp = eye - numpy.kron(Φ, Φ)
    inv_tmp = numpy.linalg.inv(tmp)
    vec_var = inv_tmp * vec(Ω)
    return unvec(vec_var)


def acf(φ: list[numpy.matrix[float]], ω: numpy.matrix[float], n: int) -> numpy.matrix[float]:
    """
    Compute the stationary auto covariance matrix for the given VAR(n)
    parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    ω: numpy.matrix[float]
        VAR(n) process gaussian noise autocovariance matrix.
    n: int
        Maximum lag.

    Returns
    -------
    numpy.matrix[float]
        Stationary mean matrix.
    """

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


def eig(φ: list[numpy.matrix[float]]) -> numpy.ndarray[float]:
    """
    Compute eigen values of VAR(n) parameter matrix transformed to VAR(1) companion form. 
    Stationarity requires that |λ| < 1.

    Parameters
    ----------
    φ: numpy.matrix[float]
       VAR(n) coefficient matrix in companion form.

    Returns
    -------
    numpy.ndarray[float]
        Array of eigen values.
    """

    Φ = phi_comp(φ)
    λ, _ = numpy.linalg.eig(Φ)
    return λ


def is_stationary(φ: list[numpy.matrix[float]]) -> bool:
    """
    Return True if the VAR(n) parameter matrix is stationary.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) covariance matrix.

    Returns
    -------
    bool
        True if VAR(n) process is stationary.
    """

    for λ in eig(φ):
        if abs(λ) >= 1:
            return False
    return True


def phi_comp(φ: list[numpy.matrix[float]]) -> numpy.matrix[float]:
    """
    Convert the VAR(n) coefficient matrix to the VAR(1) companion form used for calculations. 

    Parameters
    ----------
   φ: list[numpy.matrix[float]]
         VAR(n) coefficient matrix

    Returns
    -------
    numpy.matrix[float]
        Companion form of noise covariance matrix.
    """

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


def mean_comp(μ: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Convert the VAR(n) offset matrix the VAR(1) companion form used for calculations.

    Parameters
    ----------
    μ: numpy.matrix[float]
        VAR(n) offset matrix.

    Returns
    -------
    numpy.matrix[float]
        Companion form of VAR(n) offset matrix.
    """

    n = len(μ)
    p = numpy.zeros(n**2)
    p[:n] = μ
    return numpy.matrix([p]).T


def omega_comp(ω: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Convert VAR(n) gaussian noise covariance matrix to companion form.

    Parameters
    ----------
    ω: numpy.matrix[float]
        VAR(n) noise covariance matrix.

    Returns
    -------
    numpy.matrix[float]
        Companion form of noise covariance matrix.
    """

    n, _ = ω.shape
    p = numpy.zeros((n**2, n**2))
    p[:n, :n] = ω

    return numpy.matrix(p)


def vec(m: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Apply the vec operator to the given matrix. The vec operation 
    applied to the matrix,

    A = [[a11, a12],
         [a21, a22]]

    gives,

    vec(A) = [[a11],
              [a21],
              [a12],
              [a22]]

    Parameters
    ----------
    m: numpy.matrix[float]
        Matrix to be converted to vec form.

    Returns
    -------
    numpy.matrix[float]
        Input vector converted to vec form.
    """

    _, n = m.shape
    v = numpy.matrix(numpy.zeros(n**2)).T
    for i in range(n):
        d = i*n
        v[d:d+n] = m[:,i]
    return v


def unvec(v: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Apply the inverse of the vec operation to the given matrix. For the following
    matrix in vec form,

    A = [[a11],
         [a21],
         [a12],
         [a22]]

    apply unvec gives,

    unvec(A) = [[a11, a12],
                [a21, a22]]

    Parameters
    ----------
    m: numpy.matrix[float]
        Matrix to be converted to unvec form.

    Returns
    -------
    numpy.matrix[float]
        Input vector in unvec form.
    """

    n2, _ = v.shape
    n = int(numpy.sqrt(n2))
    m = numpy.matrix(numpy.zeros((n, n)))
    for i in range(n):
        d = i*n
        m[:,i] = v[d:d+n]
    return m


def var(x0: numpy.matrix[float], μ: numpy.matrix[float], φ: list[numpy.matrix[float]], Ω: numpy.matrix[float], n: int) -> numpy.matrix[float]:
    """
    Simulate a VAR(n) process using the provided parameters.
    
    Parameters
    ----------
    x0: numpy.matrix[float]
        VAR(n) process initial value matrix.
    μ: numpy.matrix[float]
        VAR(n) process offset matrix.
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    Ω: list[numpy.matrix[float]]
        VAR(n) process gaussian noise autocovariance function.

    Returns
    -------
    numpy.matrix[float]
        Simulation results.
    """

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

