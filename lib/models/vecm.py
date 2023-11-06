import numpy
from statsmodels.tsa.vector_ar import vecm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR, LagOrderResults
from statsmodels.tsa.vector_ar.vecm import VECM

from lib import stats


def vecm1(λ: numpy.ndarray[float, float], β: numpy.ndarray[float, float], a: numpy.ndarray[float, float], 
          Ω: numpy.ndarray[float, float], nsamp: int) -> numpy.ndarray[float, float]:
    """
    Simulate a first order Vector Error Correction Model (VECM) process with the specified parameters.

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


def aic_order(samples: numpy.ndarray[float, float], maxlags: int=12) -> LagOrderResults:
    """
    Determine the order of a VAR process using the AIC criterion.

    Parameters
    ----------
    samples: numpy.ndarray[float, float]
        Samples analyzed.
    
    maxlags: int
        Maximum number of lags.

    Returns
    -------
    LagOrderResults
        Order results.
    """

    return VAR(samples).select_order(maxlags=maxlags).selected_orders['aic']



