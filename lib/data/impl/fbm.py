import numpy
from typing import Tuple

from lib.models import fbm

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space, create_logspace)


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

    return Δt * create_space(xmin=0, npts=npts), numpy.full(npts, μ)

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

def compute_var(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical FBM motion variance.

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
        Time and variance.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 11, **kwargs)

    t = Δt * create_space(xmin=0, npts=npts)

    return t, fbm.var(H, t)

def compute_acf(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Fractional brownian motion autocorrelation function.

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
        Time and Autocorrelation.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 11, **kwargs)

    t = Δt * create_space(xmin=0, npts=npts)

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
    npts = get_param_default_if_missing("npts", 11, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = Δt * create_space(xmin=0, npts=npts)
    s = Δt * s

    return t, fbm.cov(H, s, t)

def compute_variance_ratio(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute FBM variance ratio for zero lag. For brownian motion the variance ration is 1. If the 
    variance ration is less than one the samples are anticorrelated in time and if it 
    is greater thane 1 the samples are correlated in time.

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
        Time and variance ratio.
    """

    H = get_param_throw_if_missing("H", **kwargs)
    npts = get_param_default_if_missing("npts", 11, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = Δt * create_space(xmin=0, npts=npts)

    return t, t**(2*H - 1.0)

def compute_variance_ratio_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
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

    s_vals = [int(s) for s in __get_s_vals(**kwargs)]
    return s_vals, fbm.vr_scan(samples, s_vals)

def compute_variance_ratio_homo_stat_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
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

    s_vals = [int(s) for s in __get_s_vals(**kwargs)]
    return s_vals, fbm.vr_stat_homo_scan(samples, s_vals)

def compute_variance_ratio_hetero_stat_scan(samples: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
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

    s_vals = [int(s) for s in __get_s_vals(**kwargs)]
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

def __get_s_vals(**kwargs) -> list[int]:
    """
    Compute lags for variance ratio test using provided parameters.

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
    list[int]
        s values used in scan.
    """

    linear = get_param_default_if_missing("linear", False, **kwargs)
    smin = get_param_default_if_missing("smin", 1.0, **kwargs)
    smax = get_param_default_if_missing("smax", None, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    svals = get_param_default_if_missing("svals", None, **kwargs)
    if npts is not None and smax is not None:
        if linear:
            return create_space(npts=npts, xmax=smax, xmin=smin)
        else:
            return create_logspace(npts=npts, xmax=smax, xmin=smin)
    elif svals is not None:
        return svals
    else:
        raise Exception(f"smax and npts or svals is required")
    
# ##################################################################################################################
# # Perform test forspecified implementaion
# def _perform_test_for_impl(x, y, test_type, impl_type, **kwargs):
#     if impl_type.value == FBM._TestImpl.VR_TWO_TAILED.value:
#         return _vr_test(y, TestHypothesis.TWO_TAIL, test_type, impl_type, **kwargs)
#     elif impl_type.value == FBM._TestImpl.VR_LOWER_TAIL.value:
#         return _vr_test(y, TestHypothesis.LOWER_TAIL, test_type, impl_type, **kwargs)
#     elif impl_type.value == FBM._TestImpl.VR_UPPER_TAIL.value:
#         return _vr_test(y, TestHypothesis.UPPER_TAIL, test_type, impl_type, **kwargs)
#     else:
#         raise Exception(f"Test type is invalid: {self}")

# # _TestImpl.VR_TWO_TAILED, _TestImpl.VR_LOWER_TAIL, _TestImpl.VR_UPPER_TAIL.value
# def _vr_test(y, hypo_type, test_type, impl_type, **kwargs):
#     sig_level = get_param_default_if_missing("sig_level", 0.1, **kwargs)
#     s = get_param_default_if_missing("s", [4, 6, 10, 16, 24], **kwargs)
#     verify_type(s, list)
#     result = fbm.vr_test(y, s, sig_level, hypo_type)
#     return result, _vr_report_from_result(result, test_type, impl_type)

# ##################################################################################################################
# # Construct test report from result object
# def _vr_report_from_result(result, test_type, impl_type):
#     sig = TestParam(label=f"{int(100.0*result.sig_level)}%", value=result.sig_level)
#     s_vals = [TestParam(label=r"$s$", value=s) for s in result.s_vals]
#     stats = [TestParam(label=r"$Z(s)$", value=stat) for stat in result.stats]
#     pvals = [TestParam(label=r"$p-value$", value=pval) for pval in result.p_vals]
#     lower = result.critical_values[0]
#     if lower is not None:
#         lower = TestParam(label=r"$Z_L(s)$", value=lower)
#     upper = result.critical_values[1]
#     if upper is not None:
#         upper = TestParam(label=r"$Z_U(s)$", value=upper)
#     test_data = []
#     for i in range(len(s_vals)):
#         data = TestData(status=result.status_vals[i],
#                         stat=stats[i],
#                         pval=pvals[i],
#                         params=[s_vals[i]],
#                         sig=sig,
#                         lower=lower,
#                         upper=upper)
#         test_data.append(data)
#     return TestReport(status=test_type.status(result.status_vals),
#                       hyp_type=result.hyp_type,
#                       test_type=test_type,
#                       impl_type=impl_type,
#                       test_data=test_data,
#                       dist=Dist.NORMAL,
#                       loc=0.0,
#                       scale=1.0)

# ##################################################################################################################
# # OLS Variable Transforms
# # Est.AGG_VAR
# def _create_agg_var_trans(param, const):
#     formula = r"$\sigma^2 m^{2\left(H-1\right)}$"
#     param = ParamEst(est=1.0 + param.est/2.0,
#                      err=param.err/2.0,
#                      est_label=r"$\hat{Η}$",
#                      err_label=r"$\sigma_{\hat{Η}}$")
#     c = 10.0**const.est
#     const = ParamEst(est=c,
#                      err= c*const.err,
#                      est_label=r"$\hat{\sigma}^2$",
#                      err_label=r"$\sigma^2_{\hat{\sigma}^2}$")
#     return OLSSinlgeVarTrans(formula, const, param)

# # Est.PERGRAM
# def _create_pergram_trans(param, const):
#     formula = r"$C\omega^{1 - 2H}$"
#     param = ParamEst(est=(1.0 - param.est)/2.0,
#                      err=param.err/2.0,
#                      est_label=r"$\hat{Η}$",
#                      err_label=r"$\sigma_{\hat{Η}}$")
#     c = 10.0**const.est
#     const = ParamEst(est=c,
#                      err=c*const.err,
#                      est_label=r"$\hat{C}$",
#                      err_label=r"$\sigma_{\hat{C}}$")
#     return OLSSinlgeVarTrans(formula, const, param)
