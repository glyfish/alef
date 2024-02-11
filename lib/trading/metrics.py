"""
metrics.py

Metrics in used in financial analysis.

"""

import numpy
from pandas import DataFrame

from typing import Tuple

from lib.utils import verify_condition

def zscore(samples: numpy.ndarray[float]) -> float:
    """
    Calculate the z-score using samples to compute the mean and standard deviation
    and use the first value in samples as the test value. It is assumed that the
    data is backtrader line order which hs the most recent value at the beginning of the array.

    Parameters
    ----------
    samples : numpy.ndarray
        The time series.

    Returns
    -------
    float
        The z-score.
    """

    verify_condition(samples, len(samples) > 0, "No samples to compute z-score")

    mean = numpy.mean(samples)
    std = numpy.std(samples)
    val = samples[0]

    return (val - mean) / std if std > 0 else 0.0


def compute_zscore(time: numpy.ndarray[float], data: numpy.ndarray[float], window: int) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Calculate the z-score for a time series using a rolling window. The order of the
    time series is assumed oldest data to most recent data.

    Parameters
    ----------
    data : numpy.ndarray[float]
        The time series.
    time : numpy.ndarray[float]
        The time series time.
    window : int
        The lookback window.

    Returns
    -------
    numpy.ndarray[float]
        The z-score series.
    """

    npts = len(data) - window + 1
    zscores = [zscore(numpy.flip(data[i:i + window])) for i in range(npts)]
    return time[window - 1:], numpy.array(zscores)