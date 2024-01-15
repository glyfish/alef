from __future__ import (absolute_import, division, print_function, unicode_literals)

from datetime import datetime, date # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


class MeanRevertingTimeSeries(bt.Strategy):
    """
    Implementation of the mean reverting time series strategy described in,

        'Algorithmic Trading: Winning Strategies and Their Rationale' - Ernest Chan

    described in Example 2.8, 'Backtesting a Linear Mean-Reverting Strategy on a Portfolio'.  
    """

    params = (
        ('window', 15),
    )


    def __init__(self):
        self.dataclose = self.datas[0].close


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

        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))


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


    def notify_trade(self, trade):
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

        self.log('Close, %.2f' % self.dataclose[0])


if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MeanRevertingTimeSeries)

    dataname = os.path.abspath('data/algorithmic_trading/CAD=X.csv')
    data = bt.feeds.YahooFinanceCSVData(
        dataname=dataname,
        fromdate = datetime(2007, 7, 22),
        todate = datetime(2007, 12, 31),
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
