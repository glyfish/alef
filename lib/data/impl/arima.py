"""
data.impl.arima.py

Interface to models.arima.py
"""
import numpy

from lib.models import arima

from lib.data.meta_data import (TestParam, TestData, TestReport,
                                ParamEst, ARMAEst)
from lib.models import (TestHypothesis)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)

def compute_pacf(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs):
    """
    Compute the partial autocorrelation function bu solving the Yule-Walker equations.

    Parameters
    ----------
    time: numpy.ndarray[float]
        Time
    data: numpy.ndarray[float]
        AR(p) processes samples
    nlags: int
        The assumed order of the AR(p) process.

    Returns
    -------
    numpy.ndarray[float]
        Estimate of AR(p) coefficients.

    """

    nlags = get_param_throw_if_missing("nlags", **kwargs)

    return time[1:nlags+1], arima.yw(data, nlags)

def compute_ar1_acf(**kwargs):
    """
    Compute the AR(1) Autocorrelation function.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    nlags: int
        number of lags.

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time lag and AR(1) autocorrelation function.

    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    
    lags = create_space(xmax=nlags - 1, npts=nlags)
    return lags, φ**lags

def compute_maq_acf(**kwargs):
    """
    Compute the AR(1) Autocorrelation function.

    Parameters
    ----------
    θ: list[float]
        MA(q) coefficients.
    nlags: int
        number of lags.
    σ : float
        Noise standard deviation.

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time lag and AR(1) autocorrelation function.
    """

    θ = get_param_throw_if_missing("θ", **kwargs)
    nlags = get_param_throw_if_missing("nlags", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)

    return create_space(xmax=nlags, npts=nlags + 1), arima.maq_acf(θ, σ, nlags)

def compute_arma_mean(**kwargs):
    """
    Compute the ARMA process mean value.

    Parameters
    ----------
    npts: int
        Number of points evaluate

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time and mean value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, 0.0)

def compute_ar1_sd(**kwargs):
    """
    Compute the AR(1) process standard deviation.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    σ : float
        Noise standard deviation.
    npts: int
        Number of points evaluate

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time and standard deviation value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    
    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, arima.ar1_sigma(φ, σ))

def compute_maq_sd(**kwargs):
    """
    Compute the MA(q) process standard deviation.

    Parameters
    ----------
    θ: list[float]
        MA(q) coefficients.
    σ : float
        Noise standard deviation.
    npts: int
        Number of points evaluate

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time and standard deviation value.
    """

    θ = get_param_throw_if_missing("θ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    verify_type(θ, list)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, arima.maq_sigma(θ, σ))

def compute_ar1_offset_mean(**kwargs):
    """
    Compute the AR(1) process with offset mean.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    μ : float
        Offset.
    npts: int
        Number of points evaluate

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, arima.ar1_offset_mean(φ, μ))

def compute_ar1_offset_sd(**kwargs):
    """
    Compute the AR(1) process with offset standard deviation.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    μ : float
        Offset.
    npts: int
        Number of points evaluate

    Returns
    -------
    numpy.ndarray[float], numpy.ndarray[float]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, arima.ar1_offset_sigma(φ, σ))

def create_ar_source(**kwargs):
    """
    Generate AR(p) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    npts: int
        number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    (numpy.ndarray[float], numpy.ndarray[float])
        time and Simulation results.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)

    return create_space(npts=npts), arima.arp(numpy.array(φ), npts, σ)

def create_ar_drift_source(**kwargs):
    """
    Generate AR(p) with drift source using specified parameters and the 
    statsmodels.tas simulator.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    u: float
        Offset.
    γ: float
        Drift parameter.
    npts: int
        number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    (numpy.ndarray[float], numpy.ndarray[float])
        time and Simulation results.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)

    return create_space(npts=npts), arima.arp_drift(numpy.array(φ), μ, γ, npts, σ)

def create_ar_offset_source(**kwargs):
    """
    Generate AR(p) with a constant offset using the specified parameters.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    u: float
        Offset.
    npts: int
        number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    numpy.ndarray[float]
        Simulation results.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(φ, list)

    return create_space(npts=npts), arima.arp_offset(numpy.array(φ), μ, npts, σ)

def create_ma_source(**kwargs):
    """
    Generate MA(q) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    θ: list[float]
        MA(q) parameters.
    npts: int
        number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    numpy.ndarray[float]
        Simulation results.
    """

    θ = get_param_throw_if_missing("θ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)

    return create_space(npts=npts), arima.maq(numpy.array(θ), npts, σ)

def create_arma_source(**kwargs):
    """
    Generate ARMA(p, q) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    θ: numpy.ndarray[float]
        MA(q) parameters.
    npts: int
        number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    numpy.ndarray[float]
        Simulation results.
    """

    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)

    return create_space(npts=npts), arima.arma(numpy.array(φ), numpy.array(θ), npts, σ)

def create_arima_source(**kwargs):
    """
    Generate ARIMA(p,d,q) using specified parameters and the statsmodels.tas simulator arma
    and integrate the result d times to obtain the ARIMA process.

    Parameters
    ----------
    φ: numpy.ndarray[float]
        AR(p) parameters.
    δ: numpy.ndarray[float]
        MA(q) parameters.
    d: int
        Number of integrations to perform (d = 1 or 2).
    npts: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    numpy.ndarray[float]
        Simulation results.

    Raises
    ______
    Exception
        d < 1 or d > 2
    """

    θ = get_param_throw_if_missing("θ", **kwargs)
    φ = get_param_throw_if_missing("φ", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    verify_type(θ, list)
    verify_type(φ, list)

    return create_space(npts=npts), arima.arima(numpy.array(φ), numpy.array(θ), d, npts, σ)

def create_arima_from_arma_source(**kwargs):
    """
    Generate ARIMA(p,d,q) using the samples from a ARMA(p,q) process
    by integrating d times,.

    Parameters
    ----------
    arma: numpy.ndarray[float]
        ARMA(p,q) processes samples
    d: int
        Number of integrations to perform (d = 1 or 2).

    Returns
    -------
    numpy.ndarray[float]
        Simulation results.

    Raises
    ______
    Exception
        d < 1 or d > 2
    """

    samples = get_param_throw_if_missing("arma", **kwargs)
    d = get_param_throw_if_missing("d", **kwargs)
    return create_space(npts=len(samples)), arima.arima_from_arma(samples, d)

# _TestImpl.ADF
def _adf_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

# _TestImpl.ADF_OFFSET
def _adf_offset_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test_offset(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

# _TestImpl.ADF_DRIFT
def _adf_drift_test(y, test_type, impl_type, **kwargs):
    result = arima.adf_test_drift(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

def __adf_report_from_result(result, test_type):
    sigs = [TestParam(label=result.sig_str[i], value=result.sig[i]) for i in range(3)]
    stat = TestParam(label=r"$t$", value=result.stat)
    pval = TestParam(label=r"$p-value$", value = result.pval)
    lower_vals = [TestParam(label=r"$t_L$", value=val) for val in result.critical_vals]
    test_data = []
    for i in range(3):
        data = TestData(status=result.status_vals[i],
                        stat=stat,
                        pval=pval,
                        params=[],
                        sig=sigs[i],
                        lower=lower_vals[i],
                        upper=None)
        test_data.append(data)
    return TestReport(status=test_type.status(result.status_vals),
                      hyp_type=TestHypothesis.LOWER_TAIL,
                      test_type=test_type,
                      impl_type=impl_type,
                      test_data=test_data,
                      dist=None)

# Est.AR
def __ar_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_fit(samples, order)
    return result, __arma_estimate_from_result(result)

# Est.AR_OFFSET
def __ar_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ar_offset_fit(samples, order)
    return result, __arma_estimate_from_result(result)

# Est.MA
def __ma_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_fit(samples, order)
    return result, __arma_estimate_from_result(result)

# Est.MA_OFFSET
def __ma_offset_estimate(samples, **kwargs):
    order = get_param_throw_if_missing("order", **kwargs)
    result = arima.ma_offset_fit(samples, order)
    return result, __arma_estimate_from_result(result)

def __arma_estimate_from_result(result, est_type):
    nparams = len(result.params)
    params = []
    for i in range(1, nparams-1):
        params.append(ParamEst.from_dict({"Estimate": result.params.iloc[i],
                                          "Error": result.bse.iloc[i]}))
    const = ParamEst.from_dict({"Estimate": result.params.iloc[0],
                                "Error": result.bse.iloc[0]})
    sigma2 = ParamEst.from_dict({"Estimate": result.params.iloc[nparams-1],
                                 "Error": result.bse.iloc[nparams-1]})
    return ARMAEst(est_type, const, sigma2, params)
