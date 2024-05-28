from __future__ import (absolute_import, division, print_function, unicode_literals)

from datetime import datetime, date # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


# Create a Strategy
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 150),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.maperiod)
        self.sma.csv = True

    def notify_order(self, order):
        # If order is submitted/accepted, do nothing
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # If price is above moving average buy
            if self.dataclose[0] > self.sma[0]:
                # previous close less than the previous close
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.2f, Moving Average, %.2f' % (self.dataclose[0], self.sma[0]))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            # Already in the market .. sell if below the moving average
            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f, Moving Average, %.2f' % (self.dataclose[0], self.sma[0]))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

if __name__ == '__main__':
    # Create a cerebro instance
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    dataname = os.path.abspath('data/algorithmic_trading/CAD=X.csv')
    data = bt.feeds.YahooFinanceCSVData(
        dataname=dataname,
        fromdate = datetime(2007, 7, 22),
        todate = datetime(2012, 3, 28),
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

    # Write output to file
    cerebro.addwriter(bt.WriterFile, csv=True, out='apps/output/getting_started.csv')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = "sharpe")

    # Run over everything
    back = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    back[0].analyzers.sharpe.get_analysis()
    # Plot the result
    cerebro.plot()