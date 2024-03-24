from __future__ import (absolute_import, division, print_function, unicode_literals)

from datetime import datetime, date
import os.path
import sys

import backtrader as bt
import shortuuid

from lib.trading.indicators import ZScore
from lib.db.backtest_db import BacktestDb


class MeanRevertingTimeSeries(bt.Strategy):
    """
    Implementation of the mean reverting time series strategy described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan

    described in Example 2.8, 'Backtesting a Linear Mean-Reverting Strategy on a Portfolio'.  
    """

    params = (
        # Half-life of mean reversion estimate
        ('half_life', 124),
        # Multiple applied to zscore to determine stake size
        ('stake_multiple', 100)
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission, current  bar_executed
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a ZScore indicator
        self.zscore = ZScore(self.datas[0], period=self.params.half_life)
        self.zscore.csv = True

        # Add database interface
        self.db = BacktestDb()

        # Create run identifier
        self.run_id = shortuuid.ShortUUID().random(length=12)
        self.time_stamp = datetime.utcnow()


    def log(self, txt: str, dt: datetime=None):
        """
        Logging function for strategy.

        Parameters
        ----------
        txt : str
            Text to be logged.
        dt : datetime, optional
            Date and time to be logged. The default is None.
        """

        dt = dt or self.current_date()
        print(f"{dt.isoformat()}, {txt}")


    def current_date(self):
        """
        Get the current date.

        Returns
        -------
        date
            The current date.
        """

        return self.datas[0].datetime.date(0)
    
    
    def notify_order(self, order: bt.Order):
        """
        Called when an order has a state change.

        Parameters
        ----------
        order : bt.Order
            The order that has changed state.
        """
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price {order.executed.price:.2f}, Cost {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")

            # save bar when order was executed
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None


    def notify_trade(self, trade: bt.Trade):
        """
        Called when a trade has a state change.

        Parameters
        ----------
        trade : bt.Trade
            The trade that has changed state.
        """

        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


    def next(self):
        """
        Called on each new bar.
        """

        #  Log the closing price
        self.log(f"Close {self.dataclose[0]:.2f}")

        self.db.insert_backtest(self.run_id, self.current_date(), self.__class__.__name__, self.time_stamp, self.broker)
        self.db.insert_yahoo_asset_price(self.run_id, self.datas[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Calculate the desired stake size
        size = abs(int(self.params.stake_multiple * self.zscore[0]))
        self.log(f"Z-Score {self.zscore[0]:.3f}, Size {size}, Position {self.position.size}")

        # Check if a position is held
        if not self.position:
            # If zscore < 0.0 buy a multiple of the negative z-score value. For this case price is below average
            # and nothing is owned.
            if self.zscore[0] < 0.0:
                self.log(f"BUY CREATE, {self.dataclose[0]:.3f}, Z-Score {self.zscore[0]:.3f}, Size {size}")
                self.order = self.buy(size=size)
        else:
            # If zscore < 0.0 buy or sell what is needed to obtain a multiple of the negative z-score value.
            if self.zscore[0] < 0.0:
                delta = size - self.position.size
                self.log(f"ADJUSTING POSITION, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, " \
                         f"Position {self.position.size}, Size {size}, Delta {delta}")
                # Must sell delta to maintain position.
                if delta < 0:
                    self.log(f"SELL CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {-delta}")
                    self.order = self.sell(size=-delta)
                # Must buy delta to maintain position.
                elif delta > 0:
                    self.log(f"BUY CREATE, {self.dataclose[0]:.2f}, Z-Score {self.zscore[0]:.3f}, Size {delta}")
                    self.order = self.buy(size=delta)
            # If z-score is > 0.0 sell everything.
            elif self.zscore[0] > 0.0:
                self.log(f"EXITING POSITION SELL CREATE, {self.dataclose[0]:.2f}, Z-Score, {self.zscore[0]:.3f}, Position {self.position.size}")
                self.order = self.sell(size=self.position.size)


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
    cerebro.addstrategy(MeanRevertingTimeSeries)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='strat_sharpe_ration')

    # Set cash start
    cerebro.broker.setcash(1000.0)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.0)

    # Write output to file
    cerebro.addwriter(bt.WriterFile, csv=True, out='apps/output/mean-reversion-timeseries-CAD=X.csv')

    # Print out the starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run over everything
    strats = cerebro.run()

    # Print out the final result
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    print('Sharpe Ratio:', strats[0].analyzers.strat_sharpe_ration.get_analysis())

    # Plot the result
    cerebro.plot()
