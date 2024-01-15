from __future__ import (absolute_import, division, print_function, unicode_literals)

from lib.data import stats

# Import the backtrader platform
import backtrader as bt


class ZScore(bt.Indicator):
    """
    Implementation of the z-score indicator described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan
    
    described in Example 2.8, 'Backtesting a Linear Mean-Reverting Strategy on a Portfolio'.  
    """
    
    lines = ('zscore',)

    params = (
        ('window', 15),
    )

    def next(self):
        self.lines.zscore[0] = max(0.0, self.params.value)

    def once(self, start, end):
       zscore_array = self.lines.zscore.array

       for i in xrange(start, end):
           dummy_array[i] = max(0.0, self.params.value)