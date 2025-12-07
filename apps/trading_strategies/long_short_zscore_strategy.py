from datetime import datetime
import os.path
import sys
import random

import backtrader as bt
import shortuuid

from lib.trading.indicators import ZScore
from lib.db.backtest_db import BacktestDb
from lib.trading.strategy import GlyfishStrategy

ensemble_id = shortuuid.ShortUUID().random(length=12)

class LongShortZScore(GlyfishStrategy):
    """
    Implementation of the mean reverting time series strategy described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan

    The strategy uses the time series z-score to scale the position size. In this implementation
    a short position is taken when the z-score is greater than zero and the position size is a multiple
    of the z-score value. A long position is taken if the z-score is less than zero.  
    """

    params = (
        # Half-life of mean reversion estimate
        ('half_life', 124),
        # Multiple applied to zscore to determine stake size
        ('stake_multiple', 100)
    )

    def __init__(self):
        super().__init__(ensemble_id)
        
        self.zscore = ZScore(self.datas[0], period=self.params.half_life)
        self.zscore.csv = True


    def next(self):
        """
        Called on each new bar.
        """

        super().next()

        self.db.insert_zscore_indicator(self.run_id, self.current_date(), self.datas[0]._name, 
                                        self.zscore[0], self.params.half_life, self.params.stake_multiple, ensemble_id)

        if self.order:
            return

        # Calculate the desired stake size and trade identifier
        size = abs(int(self.params.stake_multiple * self.zscore[0]))
        
        self.log(f"Z-Score {self.zscore[0]:.3f}, Size {size}, Position {self.position.size}")

        # Check if a position is held
        if not self.position:
            # If zscore > 0.0 short sell a multiple of the negative z-score value. For this case price is below average
            # and nothing is owned.
            if self.zscore[0] > 0.0:
                self.log(f"SHORT SELL CREATE, {self.dataclose[0]:.3f}, Z-Score {self.zscore[0]:.3f}, Size {size}")
                self.order = self.sell(size=size, tradeid=self.get_tradeid())
            elif self.zscore[0] < 0.0:
                self.log(f"LONG BUY CREATE, {self.dataclose[0]:.3f}, Z-Score {self.zscore[0]:.3f}, Size {size}")
                self.order = self.buy(size=size, tradeid=self.get_tradeid())
        else:
            self.db.insert_position(self.run_id, self.current_date(), self.datas[0]._name, self.position, ensemble_id)
            # If zscore > 0.0 short sell or cover what is needed to obtain a multiple of the negative z-score value.
            if self.zscore[0] > 0.0:
                if self.position.size > 0:
                    # Exit Long Position
                    self.log(f"EXITING LONG POSITION SELL CREATE, {self.dataclose[0]:.2f}, Z-Score, {self.zscore[0]:.3f}, Position {self.position.size}")
                    self.order = self.sell(size=self.position.size, tradeid=self.get_tradeid())
                elif self.position.size < 0:
                    # Update short position
                    delta = size + self.position.size
                    self.log(f"ADJUSTING SHORT POSITION, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, " \
                                f"Position {self.position.size}, Size {size}, Delta {delta}")
                    if delta < 0:
                        # Must sell delta to maintain position.
                        self.log(f"COVER BUY CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {-delta}")
                        self.order = self.buy(size=-delta, tradeid=self.get_tradeid())
                    elif delta > 0:
                        # Must buy delta to maintain position.
                        self.log(f"SHORT SELL CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {delta}")
                        self.order = self.sell(size=delta, tradeid=self.get_tradeid())
            elif self.zscore[0] < 0.0:
                # If z-score is < 0.0 Cover position.
                if self.position.size < 0:
                    # Exit short position
                    self.log(f"EXITING SHORT POSITION COVER BUY CREATE, {self.dataclose[0]:.2f}, Z-Score, {self.zscore[0]:.3f}, Position {self.position.size}")
                    self.order = self.buy(size=self.position.size, tradeid=self.get_tradeid())
                elif self.position.size > 0:
                    # Update long position
                    delta = size - self.position.size
                    self.log(f"ADJUSTING LONG POSITION, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, " \
                            f"Position {self.position.size}, Size {size}, Delta {delta}")
                    if delta < 0:
                        # Must sell delta to maintain position.
                        self.log(f"LONG SELL CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {-delta}")
                        self.order = self.sell(size=-delta, tradeid=self.get_tradeid())
                    elif delta > 0:
                        # Must buy delta to maintain position.
                        self.log(f"LONG BUY CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {delta}")
                        self.order = self.buy(size=delta, tradeid=self.get_tradeid())



if __name__ == '__main__':
    data = GlyfishStrategy.load_yahoo_finance_data('data/algorithmic_trading/CAD=X.csv', 
                                                   datetime(2007, 7, 23), 
                                                   datetime(2012, 3, 28))

    cerebro = GlyfishStrategy.backtest(data, LongShortZScore, ensemble_id=ensemble_id)
    cerebro.plot()
