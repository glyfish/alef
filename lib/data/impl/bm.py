"""
data.impl.bm.oy

Interface to data.models.bm.py
"""

from enum import Enum
import uuid
import numpy

from lib.models import bm, stats
from typing import Tuple

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, create_space)

def compute_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical brownian motion mean.

    Parameters
    ----------
    npts: int
        Number of points.  (default 10)
    μ: float
        Mean value. (default 0.0)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)

    return Δt * create_space(xmin=1, npts=npts), numpy.full(npts, μ)

def compute_bm_drift_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical brownian motion with drift mean.

    Parameters
    ----------
    npts: int
        Number of points.  (default 10)
    μ: float
        Mean value. (default 0.0)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and mean value.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)

    t = Δt * create_space(xmin=1, npts=npts)

    return t, μ*t

def compute_bm_drift_sd(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical brownian motion with drift standard deviation.

    Parameters
    ----------
    npts: int
        Number of points.  (default 10)
    μ: float
        Mean value. (default 0.0)
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and standard deviation.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = Δt * create_space(xmin=1, npts=npts)

    return t, σ*numpy.sqrt(t)

def compute_gbm_mean(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Compute theoretical geometrical brownian motion mean.

    Parameters
    ----------
    npts: int
        Number of points.  (default 10)
    μ: float
        Mean value. (default 0.0)
    S0: float
        Initial value (default 1.0).
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and standard deviation.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = Δt * create_space(xmin=1, npts=npts)

    return t, S0*numpy.exp(μ*t)


def compute_gbm_sd(func_type, **kwargs):
    """
    Compute theoretical geometrical brownian motion standard deviation.

    Parameters
    ----------
    npts: int
        Number of points.  (default 10)
    μ: float
        Mean value. (default 0.0)
        σ: float
    Standard deviation factor of brownian motion term. The actual standard 
        deviation is given by σ * sqrt(Δt). (default 1)
    S0: float
        Initial value (default 1.0).
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and standard deviation.
    """

    npts = get_param_default_if_missing("npts", 10, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    t = Δt * create_space(xmin=1, npts=npts)

    return t, numpy.sqrt(S0**2*numpy.exp(2*μ*t)*(numpy.exp(t*σ**2)-1))


def compute_bm_from_noise(**kwargs):
    """
    Compute brownian motion from brownian noise. 

    Parameters
    ----------
    dB: numpy.ndarray[float]
        Brownian noise.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        Time and brownian motion time series.
    """

    dB = get_param_throw_if_missing("dB", **kwargs)
    verify_type(dB, numpy.ndarray[float])

    npts = len(dB)

    return create_space(xmax=npts - 1, npts=npts), stats.from_noise(dB)

def _create_bm_noise_source(source_type, x, **kwargs):
    f = lambda x : bm.noise(len(x))
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={},
                      ylabel=r"$\Delta S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Noise",
                      f=f,
                      x=x)

# Source.MOTION
def _create_bm_source(source_type, x, **kwargs):
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm(len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      params={"Δx": Δx},
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion",
                      f=f,
                      x=x)

# Source.DRIFT_MOTION
def _create_bm_drift_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_with_drift(μ, σ, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Brownian Motion With Drift",
                      f=f,
                      x=x)

# Source.GEO_MOTION
def _create_bm_geo_source(source_type, x, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    S0 = get_param_default_if_missing("S0", 1.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    f = lambda x : bm.bm_geometric(μ, σ, S0, len(x), Δx)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Geometric-BM-Simulation-{str(uuid.uuid4())}",
                      params={"σ": σ, "μ": μ, "Δt": Δx, "S0": S0},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Geometric Brownian Motion",
                      f=f,
                      x=x)
