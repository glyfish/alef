from __future__ import (absolute_import, division, print_function, unicode_literals)

from lib.stats import moving_std, zscore
from lib.utils import get_param_default_if_missing

# Import the backtrader platform
import backtrader as bt


class ZScore(bt.ind.PeriodN):
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
    alias = ('ZS', 'ZScore',)

    params = (
        ('period', 15),
    )

    def __init__(self, **kwargs):
        period = get_param_default_if_missing("period", self.p.period, **kwargs)
        setattr(self.params, "period", period)
        super(ZScore, self).__init__()

        
    def next(self):
        data = self.data.get(size=self.p.period)
        self.lines.zscore[0] = zscore(data, self.p.period)

    
    def once(self, start, end):
        src = self.data.array
        dst = self.line.array
        period = self.p.period

        for i in range(start, end):
            pass

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