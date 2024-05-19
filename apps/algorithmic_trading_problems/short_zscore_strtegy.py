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

class ShortZScore(GlyfishStrategy):
    """
    Implementation of the mean reverting time series strategy described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan

    The strategy uses the time series z-score to scale the position size. In this implementation
    a short position is taken when the z-score is greater than zero and the position size is a multiple
    of the z-score value. The position is exited when the z-score is greater less than zero.  
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


    def notify_order(self, order: bt.Order):
        """
        Called when an order has a state change.

        Parameters
        ----------
        order : bt.Order
            The order that has changed state.
        """
        
        super().notify_order(order)

        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY COVER EXECUTED, Price {order.executed.price:.2f}, Cost {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f"SHORT SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            # save bar when order was executed
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None


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
        else:
            self.db.insert_position(self.run_id, self.current_date(), self.datas[0]._name, self.position, ensemble_id)
            # If zscore > 0.0 short sell or cover what is needed to obtain a multiple of the negative z-score value.
            if self.zscore[0] > 0.0:
                delta = size + self.position.size
                self.log(f"ADJUSTING POSITION, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, " \
                         f"Position {self.position.size}, Size {size}, Delta {delta}")
                # Must sell delta to maintain position.
                if delta < 0:
                    self.log(f"COVER BUY CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {-delta}")
                    self.order = self.buy(size=-delta, tradeid=self.get_tradeid())
                # Must buy delta to maintain position.
                elif delta > 0:
                    self.log(f"SHORT SELL CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {delta}")
                    self.order = self.sell(size=delta, tradeid=self.get_tradeid())
            # If z-score is < 0.0 Cover position.
            elif self.zscore[0] < 0.0:
                self.log(f"EXITING POSITION COVER BUY CREATE, {self.dataclose[0]:.2f}, Z-Score, {self.zscore[0]:.3f}, Position {self.position.size}")
                self.order = self.buy(size=self.position.size, tradeid=self.get_tradeid())


if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()

    dataname = os.path.abspath('data/algorithmic_trading/CAD=X.csv')
    data = bt.feeds.YahooFinanceCSVData(
        dataname=dataname,
        fromdate = datetime(2007, 7, 23),
        todate = datetime(2012, 3, 28),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Add a strategy
    cerebro.addstrategy(ShortZScore)

    # Set cash start
    cerebro.broker.setcash(1000.0)

    # Decrease cash and add value when short asset is sold.
    cerebro.broker.set_shortcash(False)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run over everything
    strats = cerebro.run()

    # Print out the final result
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}, Run ID: {strats[0].run_id}, Ensemble ID: {ensemble_id}")

    # Plot the result
    cerebro.plot()
