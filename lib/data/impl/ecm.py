"""
data.impl.ecm.py

Interface to models.ecm.py
"""

import numpy

from lib.models import ecm
import statsmodels.tsa as tsa
from typing import Tuple
import uuid

from lib.data.param_est import (ParamEst)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)

def compute_xt_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ARIMA process mean value.

    Parameters
    ----------
    npts: int
        Number of points to evaluate

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, 0.0)

def compute_yt_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ECM process mean value.

    Parameters
    ----------
    npts: int
        Number of points to evaluate

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), numpy.full(npts, 0.0)

def compute_xt_var(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ARIMA process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    σ: float
        Residual variance.
    npts: int
        Number of points to evaluate

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), ecm.xt_var(φ, σ, npts)

def compute_yt_var(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ECM process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    β: float
        ECM correlation parameter.
    σ: float
        Residual variance.
    npts: int
        Number of points to evaluate

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), ecm.yt_var(φ, σ, β, npts)

def compute_cov(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute the ECM process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    σ: float
        Residual variance.
    β: float
        ECM correlation parameter.
    npts: int
        Number of points to evaluate

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    return create_space(xmax=npts - 1, npts=npts), ecm.cov(φ, σ, β, npts)

def create_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate an ECM time series from an AR(1) process using the specified parameters.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    δ: float
        ECM term parameter. (default 0.0)
    α: float
        ECM term offset parameter. (default 0.0)
    β: float
        ECM correlation parameter.
    γ: float
        ECM X(t) scale parameter.
    λ: float
        ECM relaxation rate.
    σ: float
        Residual variance. (default 1.0)
    npts: int
        Number of samples generated. (default 1000)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Generated x(t) and y(t) ECM time series.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    λ = get_param_throw_if_missing("λ", **kwargs)
    δ = get_param_default_if_missing("δ", 0.0, **kwargs)
    α = get_param_default_if_missing("α", 0.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    xt, yt = ecm.ecm(φ, δ, α, β, γ, λ, npts, σ)

    return create_space(xmax=npts - 1, npts=npts), [xt, yt]
