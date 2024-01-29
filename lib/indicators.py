from __future__ import (absolute_import, division, print_function, unicode_literals)

from lib.stats import moving_std, zscore

# Import the backtrader platform
import backtrader as bt


class ZScore(bt.Indicator):
    """
    Implementation of the z-score indicator described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan
    
    described in Example 2.8, 'Backtesting a Linear Mean-Reverting Strategy on a Portfolio'.  

    Properties
    ----------
    zscore : numpy.ndarray[float]
        The z-score line
    window : int
        The lookback window
    """

    lines = ('zscore',)

    params = (
        ('window', 15),
    )

    def next(self):
        self.lines.zscore[0] = zscore(self.data.get(size=self.p.window), self.p.window)


class MovingStandardDeviation(bt.Strategy):
    """
    Implementation of moving standard deviation.

    Properties
    ----------
    mstd : numpy.ndarray[float]
        The z-Moving standard deviation line
    window : int
        The lookback window
    """

    lines = ('mstd',)

    params = (
        ('window', 15),
    )

    def next(self):
        self.lines.mstd[0] = moving_std(self.data.get(size=self.p.window), self.p.window)