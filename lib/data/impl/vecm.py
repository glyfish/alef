import numpy
from typing import Tuple
from statsmodels.tsa.vector_ar.var_model import LagOrderResults

from lib.models import vecm
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)


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

