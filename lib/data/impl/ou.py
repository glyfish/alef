"""
lib.data.impl.ou.py

Simulation and analysis of the Ornstein-Uhlenbeck process.
"""

from typing import Tuple
import uuid
import numpy

from lib.models import ou

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space, create_logspace)


def compute_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Mean value of Ornstein-Uhlenbeck process.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    x0: float
        Initial value.

    Returns
    -------
    numpy.ndarray[float]
        Mean as a function of time for given parameters.
    """

    npts = get_param_default_if_missing("npts", 11, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)

    t = create_space(xmin=0, npts=npts, Δx=Δt)

    return t, ou.mean(μ, λ, t, x0)

def compute_mean_limit(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Limit as t -> infinity of Ornstein-Uhlenbeck process mean value.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    μ: float
        Drift coefficient.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time anf mean
    """

    npts = get_param_default_if_missing("npts", 11, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    
    return create_space(xmin=0, npts=npts, Δx=Δt), numpy.full(npts, μ)


def compute_var(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Variance of Ornstein-Uhlenbeck process.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    λ: float
        Mean reversion rate.
    σ: float
        Standard deviation of random component.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and variance.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)

    t = create_space(xmin=0, npts=npts, Δx=Δt)

    return t, ou.var(λ, t, σ)

def compute_var_limit(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Limit as t -> infinity of Ornstein-Uhlenbeck process variance.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    λ: float
        Mean reversion rate.
    σ: float
        Standard deviation of random component.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and variance.
    """


    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)

    return create_space(xmin=0, npts=npts, Δx=Δt), numpy.full(npts, ou.var_limit(λ, σ))

def compute_cov(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Covariance of Ornstein-Uhlenbeck process.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    λ: float
        Mean reversion rate.
    s: float
        Time offset.
    σ: float
        Standard deviation of random component.

            Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and covariance.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    s = get_param_default_if_missing("s", 1.0, **kwargs)

    xmin = int(s/Δt)
    t = create_space(xmin=xmin, npts=npts, Δx=Δt)

    return t, ou.cov(λ, s, t, σ)

def compute_cov_limit(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Limit as t -> infinity of Ornstein-Uhlenbeck process variance.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δt: float
        Width of time step. (default 1.0)
    s: float
        Time offset.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and covariance limit.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    s = get_param_default_if_missing("s", 1.0, **kwargs)

    xmin = int(s/Δt)
    t = create_space(xmin=xmin, npts=npts, Δx=Δt)

    return t, numpy.full(npts, 0.0)

def compute_pdf(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Ornstein-Uhlenbeck process PDF for a specified time.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δx: float
        Width of variable increment. (default 1.0)
    xmin: float
        Minimum value of modeled variable. (default 0.0)
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    t: float
        Time
    σ: float
        Standard deviation of random component.
    x0: float
        Initial value.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Modeled variable and PDF.
    """

    t = get_param_throw_if_missing("t", **kwargs)
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)

    x = create_space(xmin=xmin, npts=npts, Δx=Δx)

    return x, ou.pdf(x, μ, λ, t, σ=σ, x0=x0)

def compute_cdf(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Ornstein-Uhlenbeck process CDF for a specified time.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δx: float
        Width of variable increment. (default 1.0)
    xmin: float
        Minimum value of modeled variable. (default 0.0)
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    t: float
        Time
    σ: float
        Standard deviation of random component.
    x0: float
        Initial value.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Modeled variable and CDF.
    """

    t = get_param_throw_if_missing("t", **kwargs)
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    
    x = create_space(xmin=xmin, npts=npts, Δx=Δx)

    return x, ou.cdf(x, μ, λ, t, σ=σ, x0=x0)

def compute_pdf_limit(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
     Limit as t -> infinity of Ornstein-Uhlenbeck process PDF.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δx: float
        Width of variable increment. (default 1.0)
    xmin: float
        Minimum value of modeled variable. (default 0.0)
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    σ: float
        Standard deviation of random component.
    x0: float
        Initial value.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]] 
        Modeled variable and PDF limit.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)

    x = create_space(xmin=xmin, npts=npts, Δx=Δx)

    return x, ou.pdf_limit(x, μ, λ, σ=σ, x0=x0)

def compute_cdf_limit(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Ornstein-Uhlenbeck process CDF for t -> infinity.

    Parameters
    ----------
    npts: int
        Number of points. (default 11)
    Δx: float
        Width of variable increment. (default 1.0)
    xmin: float
        Minimum value of modeled variable. (default 0.0)
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    σ: float
        Standard deviation of random component.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]] 
        Modeled variable and PDF limit.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)

    x = create_space(xmin=xmin, npts=npts, Δx=Δx)

    return x, ou.cdf_limit(x, μ, λ, σ=σ, x0=x0)

def compute_mean_half_life(**kwargs) -> float:
    """
    Ornstein-Uhlenbeck half life to limiting mean.

    Parameters
    ----------
    λ: float
        Mean reversion rate.

    Returns
    -------
    float
        Mean half life
    """

    λ = get_param_default_if_missing("λ", 1.0, **kwargs)

    return ou.mean_halflife(λ)

def create_xt_source(**kwargs) -> numpy.ndarray[float]:
    """
    Simulation of modeled variable at a specified time with the specified parameters.

    Parameters
    ----------
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    t: float
        Time
    σ: float
        Standard deviation of random component.
    x0: float
        Initial value.
    n: int
        Number of values simulated.

    Returns
    -------
    numpy.ndarray[float]
        Simulation of modeled variable at specified time using given parameters.
    """

    t = get_param_throw_if_missing("t", **kwargs)
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    
    return ou.xt(μ, λ, t, σ, x0, npts)

def create_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Simulation of Ornstein-Uhlenbeck process using provide parameters.

    Parameters
    ----------
    μ: float
        Drift coefficient.
    λ: float
        Mean reversion rate.
    Δt: float
        Time increment.
    n: int
        Number of values simulated.
    σ: float
        Standard deviation of random component.
    x0: float
        Initial value.

    Returns
    -------
    numpy.ndarray[float]
        Simulation of Ornstein-Uhlenbeck process using provide parameters.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)

    t = create_space(xmin=0, npts=npts, Δx=Δt)

    return t, ou.ou(μ, λ, Δt, npts, σ, x0)
