import numpy
from statsmodels.tsa.vector_ar.var_model import LagOrderResults
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order, JohansenTestResult

from lib import stats

def order_estimate(samples: numpy.ndarray[float, float], maxlags: int=12, deterministic='n') -> LagOrderResults:
    """
    Estimate order of VECM samples.

    Parameters
    ----------
    samples: numpy.ndarray[float, float]
        Samples analyzed.    
    maxlags: int
        Maximum number of lags.
    deterministic: str
        Assumed trend (default 'n'). 
        Values 'n' -no deterministic terms, 'co' -constant outside the cointegration relation, 'ci' -constant within the cointegration relation, 
        'lo' -linear trend outside the cointegration relation, 'li' -linear trend within the cointegration relation.

    Returns
    -------
    LagOrderResults
        Order results.
    """

    return select_order(samples, maxlags=maxlags, deterministic=deterministic)


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


def johansen_coint(samples, max_lags, trend: int=0) -> JohansenTestResult:
    """
    Perform Johansen's cointegration test.

    Parameters
    ----------
    samples: numpy.ndarray[float, float]
        Samples analyzed.
    max_lags: int
        maximum number of lags.
    trend: int
        Trend to include in cointegration test.
            -1 - no trend
            0 - constant
            1 - linear trend,.

    Returns
    -------
    JohansenTestResult
        Johansen cointegration test result.
    """

    return coint_johansen(samples, trend, max_lags)


def __vecm_model(endog: numpy.ndarray[float, float]) -> VECM:
    """
    Estimate the parameters for and assumed VAR(n) model.

    Parameters
    ----------
    endog: DataFrame
        VAR(n) process endogenous variable samples.

    Returns
    -------
    VAR
        Analysis results.
    """

    return VECM(endog)


