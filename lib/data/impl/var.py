from enum import Enum
import numpy

from lib.models import var

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_type, verify_condition, create_space, create_logspace)

def compute_mean(φ: list[numpy.matrix[float]], **kwargs):
    μ = get_param_throw_if_missing("μ", **kwargs)

def compute_acf(φ: list[numpy.matrix[float]], **kwargs) -> numpy.matrix[float]:
    """
    Compute the stationary auto covariance matrix for the given VAR(n)
    parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    ω: numpy.matrix[float]
        VAR(n) process gaussian noise autocovariance matrix.
    nlag: int
        Maximum lag.

    Returns
    -------
    numpy.matrix[float]
        Stationary mean matrix.
    """

    m, _ = φ[0].shape
    ω_default = numpy.matrix(numpy.eye(m))

    ω = get_param_default_if_missing("ω", ω_default, **kwargs)
    verify_type(ω, numpy.matrix[float])
    nlag = get_param_default_if_missing("nlag", 25, **kwargs)

    return  create_space(npts=nlag), var.acf(φ, ω, nlag)



def create_source(Φ: list[numpy.matrix[float]], **kwargs) -> numpy.matrix[float]:
    """
    Simulate a VAR(n) process using the provided parameters.
    
    Parameters
    ----------
    Φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    x0: numpy.matrix[float]
        VAR(n) process initial value matrix. (default zero column matrix)
    μ: numpy.matrix[float]
        VAR(n) process offset matrix.(default zero column matrix)
    Ω: list[numpy.matrix[float]]
        VAR(n) process gaussian noise autocovariance function. (identity matrix)
    npts: int
        Number of steps simulated. (default 1000)

    Returns
    -------
    numpy.matrix[float]
        Simulation results.
    """

    verify_condition(Φ, len(Φ) > 0, "len(φ) > 0")
    verify_type(Φ[0], numpy.matrix[float])
    n = len(Φ)
    m, _ = Φ[0].shape

    Ω_default = numpy.matrix(numpy.eye(m))
    μ_default = numpy.matrix(numpy.zeros(m)).T
    x0_default = numpy.matrix(numpy.zeros((m, n)))

    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    μ = get_param_default_if_missing("μ", μ_default, **kwargs)
    x0 = get_param_default_if_missing("x0", x0_default, **kwargs)
    verify_type(x0, numpy.matrix[float])
    verify_type(Ω, numpy.matrix[float])
    verify_type(μ, numpy.matrix[float])

    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    return create_space(npts=npts), var.var(x0, μ, Φ, Ω, npts)

    
