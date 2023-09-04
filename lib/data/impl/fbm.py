import numpy
from typing import Tuple
import statsmodels.api as sm
import uuid

from lib.models import fbm

from lib.data.param_est import (ParamEst, OLSSingleVarResult, OLSSinlgeVarTransform, OLSParamType)
from lib.data.impl.stats import OLS
from lib.data.hyp_test import (StatisticalTestParam, StatisticalTestData, StatisticalTestReport, HypothesisTestType, HypothesisType)
from lib.data.reports import VarianceRatioTestReport
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing, create_space, get_s_vals, verify_type)


def compute_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical FBM motion mean.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    μ: float
        Mean value. (default 0.0)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    npts = get_param_default_if_missing("npts", 11, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)

    return create_space(xmin=0, npts=npts, Δx=Δt), numpy.full(npts, μ)

def compute_sd(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical FBM standard deviation.

    Parameters
    ----------
    H: float
        Hurst parameter.
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and standard deviation.
    """
    
    t, var = compute_var(**kwargs)

    return t, numpy.sqrt(var)

def compute_var(t: numpy.ndarray[float]=None, **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical FBM motion variance.

    Parameters
    ----------
    t: numpy.ndarray[float]
        Time.
    H: float
        Hurst parameter.
    npts: int
        Number of points. (default None)
    tmax: int
        Maximum time. (default None)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and variance.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)

    if t is None:
        t = create_space(xmin=0, npts=npts, xmax=tmax, Δx=Δt)

    return t, fbm.var(H, t)

def compute_acf(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Fractional brownian motion autocorrelation function.

    Parameters
    ----------
    H: float
        Hurst parameter.
    nlags: int
        Number of lags. (default 11)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and Autocorrelation.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    nlags = get_param_default_if_missing("nlags", 11, **kwargs)

    t = create_space(xmin=0, npts=nlags, Δx=Δt)

    return t, fbm.acf(H, t)

def compute_cov(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical FBM covariance.

    Parameters
    ----------
    H: float
        Hurst parameter.
    s: float
        Time offset
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Covariance as a function of time.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    s = get_param_throw_if_missing("s", **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    tmin = get_param_default_if_missing("tmin", 0.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = create_space(xmin=tmin, xmax=tmax, npts=npts, Δx=Δt)

    return t, fbm.cov(H, s, t)

def compute_vr(t: numpy.ndarray[float]=None, **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute FBM variance ratio for zero lag. For brownian motion the variance ration is 1. If the 
    variance ration is less than one the samples are anticorrelated in time and if it 
    is greater thane 1 the samples are correlated in time.

    Parameters
    ----------
    t: numpy.ndarray[float]
        Time. (default None)
    H: float
        Hurst parameter.
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and variance ratio.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    if t is None:
        t = create_space(xmin=0, xmax=tmax, npts=npts, Δx=Δt)

    return t, t**(2*H - 1.0)

def compute_vr_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute FBM variance ratio for specified lags. The lag values, s, can be
    entered or generated. Use the svals keyword to specify values and linear, smin,
    smax and npts to generate values.
    
    Parameters
    ----------
    linear: bool
        If true s values are generated on a linear scale. If false they are 
        generated on a logarithmic scale. (default False)
    smin: int
        Minimum lag used in scan.
    smax: int
        Maximum lag used in scan.
    npts: int
        Number of points in scan
    svals: list[int]
        Specify lags used in scan.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Lags and variance ratio values.
    """

    s_vals = [int(s) for s in get_s_vals(**kwargs)]
    return s_vals, fbm.vr_scan(samples, s_vals)

def compute_homo_vr_stat_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute FBM homoscedastic variance ratio test statistic for specified lags. 
    The lag values, s, can be entered or generated. Use the svals keyword to specify 
    values and linear, smin, smax and npts to generate values.
    
    Parameters
    ----------
    linear: bool
        If true s values are generated on a linear scale. If false they are 
        generated on a logarithmic scale. (default False)
    smin: int
        Minimum lag used in scan.
    smax: int
        Maximum lag used in scan.
    npts: int
        Number of points in scan
    svals: list[int]
        Specify lags used in scan.

            Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Lags and variance ratio values.
    """

    s_vals = [int(s) for s in get_s_vals(**kwargs)]
    return s_vals, fbm.vr_stat_homo_scan(samples, s_vals)

def compute_hetero_vr_stat_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute FBM heteroscedastic variance ratio test statistic for specified lags. 
    The lag values, s, can be entered or generated. Use the svals keyword to specify 
    values and linear, smin, smax and npts to generate values.
    
    Parameters
    ----------
    linear: bool
        If true s values are generated on a linear scale. If false they are 
        generated on a logarithmic scale. (default False)
    smin: int
        Minimum lag used in scan.
    smax: int
        Maximum lag used in scan.
    npts: int
        Number of points in scan
    svals: list[int]
        Specify lags used in scan.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Lags and variance ratio values.
    """

    s_vals = [int(s) for s in get_s_vals(**kwargs)]
    return s_vals, fbm.vr_stat_hetero_scan(samples, s_vals)

def create_noise_cholesky_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate fractional brownian noise using the Cholesky method and the provided 
    parameters.

    Parameters
    ----------
    H: float
        Hurst parameter.
    npts: int
        Number of points.  (default 1024)
    Δt: float
        Width of time step. (default 1.0)
    dB: numpy.ndarray[float]
        Column vector of brownian noise.
    L: numpy.matrix[float]
        Lower diagonal Cholesky decomposition of FBM covariance matrix.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Fractional brownian noise as a function of time.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1024, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)

    return Δt * create_space(xmin=0, npts=npts), fbm.cholesky_noise(H, npts, dB, L)

def create_noise_fft_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate fractional brownian noise using the FFT method and the provided 
    parameters.

    Parameters
    ----------
    H: float
        Hurst parameter.
    npts: int
        Number of points.  (default 1024)
    Δt: float
        Width of time step. (default 1.0)
    dB: numpy.ndarray[float]
        Column vector of brownian noise. If value is none the brownian noise is generated.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Fractional brownian noise as a function of time.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1024, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)

    return Δt * create_space(xmin=0, npts=npts), fbm.fft_noise(H, npts, dB)

def create_cholesky_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate fractional brownian motion using the Cholesky method and the provided 
    parameters.

    Parameters
    ----------
    H: float
        Hurst parameter.
    npts: int
        Number of points.  (default 1024)
    dB: numpy.ndarray[float]
        Column vector of brownian noise. If value is none the brownian noise is generated.
    L: numpy.matrix[float]
        Lower diagonal Cholesky decomposition of FBM covariance matrix. If value is None
        The Cholesky method is used to compute L from the ACF matrix.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and Fractional brownian motion.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1024, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)

    return Δt * create_space(xmin=0, npts=npts), fbm.generate_cholesky(H, npts, dB, L)

def create_fft_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate fractional brownian motion using the FFT method with the provided 
    parameters.

    Parameters
    ----------
    H: float
        Hurst parameter.
    npts: int
        Number of points.  (default 1024)
    dB: numpy.ndarray[float]
        Column vector of brownian noise. If value is none the brownian noise is generated.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and Fractional brownian motion.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 1024, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)

    return Δt * create_space(xmin=0, npts=npts), fbm.generate_fft(H, npts, dB)
    
def compute_H_estimate_periodogram(freq: numpy.ndarray[float], pspec: numpy.ndarray[float]) -> Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]:
    """
    Estimate Hurst parameter using OLS on the periodogram assuming a power law.

    Parameters
    ----------
    freq: numpy.ndarray[float]
        Frequency.
    pspec: numpy.ndarray[float]
        Power spectrum.

    Returns
    -------
    Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]
        Ols report and result model.
    """

    report, result = OLS.LOG.single_variable_estimate(pspec, freq)
    __add_pergram_transform(result)
    return report, result

def __add_pergram_transform(result: OLSSingleVarResult):
    """
    Add transformation used for periodogram OLS analysis.

    Parameters
    ----------
    result: OLSSingleVarResult
        OLS analysis results.
    """

    model = r"$C\omega^{1 - 2H}$"

    param = ParamEst(est=(1.0 - result.param.est)/2.0,
                     err=result.param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$",
                     est_id=result.est_id,
                     param_type=OLSParamType.TRANS_PARAM.value)

    c = 10.0**result.const.est
    const = ParamEst(est=c,
                     err=c*result.const.err,
                     est_label=r"$\hat{C}$",
                     err_label=r"$\sigma_{\hat{C}}$",
                     est_id=result.est_id,
                     param_type=OLSParamType.TRANS_CONST.value)
    
    result.set_transform(OLSSinlgeVarTransform(model, const, param))

def compute_H_estimate_variance_aggregation(m_vals: numpy.ndarray[float], agg_var: numpy.ndarray[float]) -> Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]:
    """
    Estimate Hurst parameter using OLS on the aggregated variance.

    Parameters
    ----------
    m_vals: numpy.ndarray[float]
        Aggregation intervals.
    agg_var: numpy.ndarray[float]
        Aggregated variance.

    Returns
    -------
    Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]
        Ols report and result model.
    """

    report, result = OLS.LOG.single_variable_estimate(agg_var, m_vals)
    __add_agg_var_transform(result)
    return report, result

def __add_agg_var_transform(result: OLSSingleVarResult):
    """
    Add transformation used for variance aggregation OLS analysis.

    Parameters
    ----------
    result: OLSSingleVarResult
        OLS analysis results.
    """

    model = r"$\sigma^2 m^{2\left(H-1\right)}$"
    param = ParamEst(est=1.0 + result.param.est/2.0,
                     err=result.param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$",
                     est_id=result.est_id,
                     param_type=OLSParamType.TRANS_PARAM.value)

    c = 10.0**result.const.est
    const = ParamEst(est=c,
                     err= c*result.const.err,
                     est_label=r"$\hat{\sigma}^2$",
                     err_label=r"$\sigma^2_{\hat{\sigma}^2}$",
                     est_id=result.est_id,
                     param_type=OLSParamType.TRANS_CONST.value)
    
    result.set_transform(OLSSinlgeVarTransform(model, const, param))

def compute_vr_test(samples: numpy.ndarray[float], hyp_test_type: HypothesisTestType, **kwargs) -> Tuple[VarianceRatioTestReport, StatisticalTestReport]:
    """
    Use variance ratio test to test for brownian motion.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        samples to be tested.
    hyp_test_type: HypothesisTestType
        Hypothesis test performed,
    sig_level: float
        Significance level used in test (default 0.1).
    s: list[int]
        Lag values used in analysis (default [4, 6, 10, 16, 24]).

    Returns
    -------
    Tuple[VarianceRatioTestReport, StatisticalTestReport]
        Test report and result model.
    """

    if hyp_test_type.value == HypothesisTestType.BM.value:
        return __vr_homo_test(samples, HypothesisType.TWO_TAIL, **kwargs)
    elif hyp_test_type.value == HypothesisTestType.FBM_AUTO_CORR.value:
        return __vr_homo_test(samples, HypothesisType.UPPER_TAIL, **kwargs)
    elif hyp_test_type.value == HypothesisTestType.FBM_NEG_AUTO_CORR.value:
        return __vr_homo_test(samples, HypothesisType.LOWER_TAIL, **kwargs)
    else:
        raise Exception(f"Hypothesis test type is invalid: {hyp_test_type}")


def compute_hetero_vr_test(samples: numpy.ndarray[float], hyp_test_type: HypothesisTestType, **kwargs) -> Tuple[VarianceRatioTestReport, StatisticalTestReport]:
    """
    Use variance ratio test to test for brownian motion.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        samples to be tested.
    hyp_test_type: HypothesisTestType
        Hypothesis test performed,
    sig_level: float
        Significance level used in test (default 0.1).
    s: list[int]
        Lag values used in analysis (default [4, 6, 10, 16, 24]).

    Returns
    -------
    Tuple[VarianceRatioTestReport, StatisticalTestReport]
        Test report and result model.
    """

    if hyp_test_type.value == HypothesisTestType.BM.value:
        return __vr_hetero_test(samples, HypothesisType.TWO_TAIL, **kwargs)
    elif hyp_test_type.value == HypothesisTestType.FBM_AUTO_CORR.value:
        return __vr_hetero_test(samples, HypothesisType.UPPER_TAIL, **kwargs)
    elif hyp_test_type.value == HypothesisTestType.FBM_NEG_AUTO_CORR.value:
        return __vr_hetero_test(samples, HypothesisType.LOWER_TAIL, **kwargs)
    else:
        raise Exception(f"Hypothesis test type is invalid: {hyp_test_type}")

def __vr_homo_test(samples: numpy.ndarray[float], hyp_type: HypothesisType, **kwargs) -> Tuple[VarianceRatioTestReport, StatisticalTestReport]:
    sig_level = get_param_default_if_missing("sig_level", 0.1, **kwargs)
    s = get_param_default_if_missing("s", [4, 6, 10, 16, 24], **kwargs)
    verify_type(s, list)
    result = fbm.vr_homo_test(samples, s, sig_level, hyp_type)
    return result, __vr_report_from_result(result)

def __vr_hetero_test(samples: numpy.ndarray[float], hyp_type: HypothesisType, **kwargs) -> Tuple[VarianceRatioTestReport, StatisticalTestReport]:
    sig_level = get_param_default_if_missing("sig_level", 0.1, **kwargs)
    s = get_param_default_if_missing("s", [4, 6, 10, 16, 24], **kwargs)
    verify_type(s, list)
    result = fbm.vr_hetero_test(samples, s, sig_level, hyp_type)
    return result, __vr_report_from_result(result)

def __vr_report_from_result(result: VarianceRatioTestReport) -> StatisticalTestReport:
    sig = StatisticalTestParam(label=f"{int(100.0*result.sig_level)}%", value=result.sig_level)
    s_vals = [StatisticalTestParam(label=r"$s$", value=s) for s in result.s_vals]
    stats = [StatisticalTestParam(label=r"$Z(s)$", value=stat) for stat in result.stats]
    pvals = [StatisticalTestParam(label=r"$p-value$", value=pval) for pval in result.p_vals]
    lower = result.critical_values[0]
    if lower is not None:
        lower = StatisticalTestParam(label=r"$Z_L(s)$", value=lower)
    upper = result.critical_values[1]
    if upper is not None:
        upper = StatisticalTestParam(label=r"$Z_U(s)$", value=upper)
    test_data = []
    for i in range(len(s_vals)):
        data = StatisticalTestData(status=result.status_vals[i],
                                   stat=stats[i],
                                   pval=pvals[i],
                                   params=[s_vals[i]],
                                   sig=sig,
                                   lower=lower,
                                   upper=upper)
        test_data.append(data)
    return StatisticalTestReport(status=result.hyp_test_type.status(result.status_vals),
                                 hyp_type=result.hyp_type,
                                 hyp_test_type=result.hyp_test_type,
                                 test_data=test_data)
