"""
stats.py

Compute generic statistics functions.

"""

import numpy
from typing import Tuple

from lib import stats
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing, get_s_vals, 
                       create_logspace, create_space)

def compute_pspec(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Power spectrum computed using FFT methods.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Frequency and power spectrum.
    """

    return time[1:], stats.pspec(data)

def compute_acf(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Autocorrelation function of samples computed using sm.tsa.stattools.acf.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time lags and autocovariance of samples as a function of lag.
    """

    nlags = get_param_throw_if_missing("nlags", **kwargs)

    return time[:nlags + 1], stats.acf(data, nlags)

def compute_ndiff(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Take the specified number of differences of the samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    ndiff : int
        Number of differences taken.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and samples differenced n times.
    """

    ndiff = get_param_default_if_missing("ndiff", 1, **kwargs)
    return time[:-ndiff], stats.ndiff(data, ndiff)

def compute_cumu_mean(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Cumulative mean of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and cumulative mean of samples as a function of time.
    """

    return time, stats.cumu_mean(data)

def compute_cumu_sd(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Cumulative standard deviation of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and cumulative standard deviation of samples as a function of time.
    """
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.cumu_sd(data, Δt)

def compute_cumu_var(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Cumulative variance of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and cumulative variance of samples as a function of time.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.cumu_var(data, Δt)

def compute_agg_var(data: numpy.ndarray, **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the aggregated variance using the specified bin sizes.. 

    Parameters
    ----------
    data: numpy.ndarray[float]
        Sampled data.
    npts : int
        Number of aggregation steps
    m_max: int
        Maximum lags.
    m_min: int
        Minimum lag. (default 1)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Lags and lagged variance for each value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    m_max = get_param_throw_if_missing("m_max", **kwargs)
    m_min = get_param_default_if_missing("m_min", 1, **kwargs)

    m_vals = create_logspace(npts=npts, xmax=m_max, xmin=m_min)
    return m_vals, stats.agg_var(data, m_vals)

def compute_agg(time: numpy.ndarray, data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Aggregate sample averages of m elements into len(samples)/m bins. 

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: numpy.ndarray[float]
        Sampled data.
    m : int
        Number of aggregates

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Aggreated sample average.
    """

    m = get_param_throw_if_missing("m", **kwargs)
    return stats.agg_time(time, m), stats.agg(data, m)

def compute_lag_var(data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute lagged variance for a specified range of values.

    Parameters
    ----------
    data: numpy.ndarray[float]
        Unaggregated time values.
    s_max : int
        Maximum s-value.
    s_min : int
        Minimum s value.
    npts : int
        Number of s-values to create
    s_vals : list[int]
        List if s-values to use

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        lagged variance for specified lag values.
    """

    s_vals =  [int(s) for s in get_s_vals(**kwargs)]
    return s_vals, stats.lag_var_scan(data, s_vals)

def compute_ensemble_mean(time: numpy.ndarray, data: list[numpy.ndarray[float]]) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the time varying mean of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[numpy.ndarray[float]]
        Sampled data.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    return time, stats.ensemble_mean(data)

def compute_ensemble_sd(time: numpy.ndarray, data: list[numpy.ndarray[float]], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the time varying standard deviation of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[numpy.ndarray[float]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.ensemble_sd(data, Δt)

def compute_ensemble_var(time: numpy.ndarray[float], data: list[numpy.ndarray[float]], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the time varying variance of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[numpy.ndarray[float]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.ensemble_var(data, Δt)

def compute_ensemble_acf(time: numpy.ndarray, data: list[numpy.ndarray[float]], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ensemble averaged autocorrelation function of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[numpy.ndarray[float]]
        Sampled data.
    nlags: int
        Number of lags (default len(sample))

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    nlags = get_param_default_if_missing("nlags", None, **kwargs)

    return time[:nlags], stats.ensemble_acf(data, nlags)

def compute_pdf_hist(data: numpy.ndarray[float], **kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Create a PDF histogram for the provided data.

    Parameters
    ----------
    data : numpy.ndarray[float]
        Sampled data.
    xmin : float
        Minimum x value (required).
    xmax : float
        maximum x value (required).
    nbins : int
        Number of bins. (default 50)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        PDF histogram.

    Raises
    ------
    Exception
        xmin and xmax are missing.

    """

    xmin = get_param_throw_if_missing("xmin", **kwargs)
    xmax = get_param_throw_if_missing("xmax", **kwargs)
    nbins = get_param_default_if_missing("nbins", 50, **kwargs)
    pdf = stats.pdf_hist(data, [xmin, xmax], nbins)

    return pdf[1][:-1], pdf[0]

def compute_cdf_hist(x: numpy.ndarray[float], pdf: numpy.ndarray[float]) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Create a CDF histogram from the given PDF histogram.

    x : numpy.ndarray[float]
        Random variable values.
    pdf : numpy.ndarray[float]
        PDF.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        CDF histogram.
    """

    return x, stats.cdf_hist(x, pdf)

def compute_multivariate_normal_pdf(μ: numpy.ndarray[float], Ω: numpy.ndarray[float, float], n: int) -> numpy.ndarray[float]:
    """
    Return multivariate normal PDF with the specified parameters.

    Parameters
    ----------
    μ: numpy.ndarray[float]
        Distribution mean values contains m elements
    Ω: numpy.ndarray[float, float]
        Distribution correlation matrix contains mxm elements.
    n: int
        Number of points along an axis.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        coordinates and generated samples. Generated samples.

    Raises
    ------
        Exception invalid array dimensions.
    """

    σ = max(numpy.diag(Ω))
    δ = 6.0*σ/n
    nvars = len(μ)

    if nvars == 1 or nvars > 3:
        raise Exception("Number of variables must be between 2 or 3")

    x1 = -3.0*σ + μ[0]
    x2 = 3.0*σ + μ[0]
    y1 = -3.0*σ + μ[1]
    y2 = 3.0*σ + μ[1]

    if nvars == 2:
        vals = numpy.mgrid[x1:x2:δ, y1:y2:δ]
    else:
        z1 = -3.0*σ + μ[3]
        z2 = 3.0*σ + μ[3]
        vals = numpy.mgrid[x1:x2:δ, y1:y2:δ, z1:z2:δ]

    return vals, stats.multivariate_normal_pdf(vals, μ, Ω)

def create_multivariate_normal_samples_source(μ: numpy.ndarray[float], Ω: numpy.ndarray[float, float], n: int) -> numpy.ndarray[float]:
    """
    Return multivariate normal samples with the specified parameters.

    Parameters
    ----------
    μ: numpy.ndarray[float]
        Distribution mean values contains m elements
    Ω: numpy.ndarray[float, float]
        Distribution correlation matrix contains mxm elements.
    n: int
        Number of samples.

    Returns
    -------
    numpy.ndarray[float]
        Generated samples.
    """

    return stats.multivariate_normal_samples(μ, Ω, n)
