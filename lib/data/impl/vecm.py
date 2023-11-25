import numpy
from typing import Tuple
import uuid

from statsmodels.tsa.vector_ar.var_model import LagOrderResults
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult

from lib.models import vecm
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)
from lib.data.hyp_test import VAROrderTestReport, __var_order_test_report_from_result
from lib.data.reports import JohansenTestReport
from lib.data.hyp_test import JohansenCointTestReport, JohansenCointTestStatistic, JohansenCointTestRank, JohansenCointTestEigenVector


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


def compute_johansen_coint_test(samples: numpy.ndarray[float, float], max_lags: int, **kwargs) -> Tuple[JohansenTestReport, JohansenCointTestReport, JohansenTestResult]:
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

    return JohansenTestReport(result), __vecm_johansen_coint_test_report_from_result(result), result


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


def __vecm_johansen_coint_test_report_from_result(result: JohansenTestResult) -> JohansenCointTestReport:
    """
    Create a Johansen test report from a Johansen test result.

    Parameters
    ----------
    result: JohansenTestResult
        Johansen test result.

    Returns
    -------
    JohansenTestReport
        Johansen test report.
    """

    eigen_values = result.eig
    eigen_vectors = result.evec
    trace_critical_vals = result.cvt
    trace_statistic = result.lr1
    eigen_value_critical_values = result.cvm
    eigen_value_statistic = result.lr2

    def compute_rank():
        test_result = []
        n = len(trace_statistic)
        for i in range(n):
            test_result.append(trace_statistic[i] > trace_critical_vals[i])
        test_result = numpy.array(test_result)               
        return [len(test_result[:,i][test_result[:,i]]) for i in range(n)]

    ranks = compute_rank()
    n = len(eigen_values)
    test_id = str(uuid.uuid4())

    trace_statistic_report = [JohansenCointTestStatistic(test_id, i, trace_statistic[i], trace_critical_vals[i]) for i in range(n)]
    eigen_value_statistic_report = [JohansenCointTestStatistic(test_id, i, eigen_value_statistic[i], eigen_value_critical_values[i]) for i in range(n)]
    rank_report = JohansenCointTestRank(test_id, ranks)
    eigen_value_report = [JohansenCointTestEigenVector(test_id, eigen_values[i], eigen_vectors[i]) for i in range(n)]

    return JohansenCointTestReport(test_id, trace_statistic_report, eigen_value_statistic_report, rank_report, eigen_value_report)