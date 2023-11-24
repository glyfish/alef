import numpy
from typing import Tuple
from statsmodels.tsa.vector_ar.var_model import LagOrderResults
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult

from lib.models import vecm
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)
from lib.data.hyp_test import VAROrderTestReport, __var_order_test_report_from_result
from lib.data.reports import JohansenTestReport


def compute_order(samples: numpy.ndarray[float, float], **kwargs) -> Tuple[LagOrderResults, VAROrderTestReport]:
    """
    Determine the order of a VAR process using the AIC criterion.

    Parameters
    ----------
    samples: numpy.ndarray[float, float]
        Samples analyzed.    
    maxlags: int
        Maximum number of lags.
    trend: str
        Assumed trend (default 'c'). 
        Values 'n'=no trend, 'c'=constant offset, 'ct'=linear trend, 'ctt'=quadratic and linear trend.

    Returns
    -------
    LagOrderResults
        Order results.
    """

    maxlags = get_param_default_if_missing("maxlags", 12, **kwargs)
    trend = get_param_default_if_missing("trend", 'c', **kwargs)

    result = vecm.order_estimate(samples.T, maxlags, trend)
    return result, __var_order_test_report_from_result(result)


def compute_johansen_coint_test(samples: numpy.ndarray[float, float], max_lags: int, **kwargs) -> Tuple[JohansenTestResult]:
    """
    Compute the Johansen cointegration test.

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
             1 - linear trend.
        default is no trend.

    Returns
    -------
    numpy.ndarray[float, float]
        Eigenvalues.
    numpy.ndarray[float, float]
        Eigenvectors.
    numpy.ndarray[float, float]
        Trace statistic.
    """

    trend = get_param_default_if_missing("trend", 0, **kwargs)
    result = vecm.coint_johansen(samples.T, max_lags, trend)

    return JohansenTestReport(result), result


def create_vecm1_source(λ: numpy.ndarray[float, float], β: numpy.ndarray[float, float], a: numpy.ndarray[float, float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float, float]]:
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
        Noise covariance matrix. (default identity matrix)
    npts: int
        Number of samples generated (default 1000).

    Returns
    -------
    numpy.ndarray[float, float]
        Simulation results.
    """

    n, _ = a.shape
    Ω_default = numpy.matrix(numpy.eye(n))
    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    return create_space(npts=npts), numpy.array(vecm.vecm1(λ, β, a, Ω, npts))

