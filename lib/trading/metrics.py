"""
metrics.py

Metrics in used in financial analysis.

"""

import numpy
from pandas import DataFrame

import statsmodels.api as sm
from scipy.stats import multivariate_normal
from typing import Tuple
from statsmodels.tsa.stattools import grangercausalitytests

def zscore(samples: numpy.ndarray[float]) -> float:
    """
    Calculate the z-score using samples to compute the mean and standard deviation
    and use the last value in samples as the test value.

    Parameters
    ----------
    samples : numpy.ndarray
        The time series.

    Returns
    -------
    float
        The z-score.
    """

    mean = numpy.mean(samples)
    std = numpy.std(samples)
    val = samples[-1]

    return (val - mean) / std if std > 0 else 0.0


def zscore_series(series: numpy.ndarray[float], window: int) -> numpy.ndarray[float]:
    """
    for a time series using a rolling window.

    Parameters
    ----------
    series : numpy.ndarray[float]
        The time series.
    window : int
        The lookback window.

    Returns
    -------
    numpy.ndarray[float]
        The z-score series.
    """

    return 