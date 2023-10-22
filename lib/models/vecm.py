import numpy
from lib import stats

def vecm2(λ: numpy.ndarray[float, float], β: numpy.ndarray[float, float], a: numpy.ndarray[float, float], 
         Ω: numpy.ndarray[float, float], nsamp: int) -> numpy.ndarray[float, float]:
    """
    Simulate a second order Vector Error Correction Model (VECM) process with the specified parameters.

    Parameters
    ----------
    λ: numpy.ndarray[float, float]
        Damping matrix.
    β: numpy.ndarray[float, float]
        Transpose of cointegration vector.
    a: numpy.ndarray[float, float]
        Coefficient matrix.
    Ω: numpy.ndarray[float, float]
        Noise covariance matrix.
    nsamp: int
        Number of samples generated.

    Returns
    -------
    numpy.ndarray[float, float]
        Simulation results.
    """

    n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsamp)))
    εt = numpy.matrix(stats.multivariate_normal_samples(numpy.zeros(n), Ω, nsamp))
    for i in range(2, nsamp):
        Δxt1 = xt[:,i-1] - xt[:,i-2]
        Δxt = λ*β*xt[:,i-1] + a*Δxt1 + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return xt


