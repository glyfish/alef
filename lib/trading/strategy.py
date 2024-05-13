import numpy
from datetime import datetime
import random

from lib.trading.metrics import std, zscore
from lib.utils import get_param_default_if_missing

import backtrader as bt
import shortuuid

from lib.db.backtest_db import BacktestDb


class GlyfishStrategy(bt.Strategy):
    """
    The GlyfishStrategy is a container for reusable elements in Strategies
    """

    def __init__(self, ensemble_id: str):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission, current  bar_executed
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add database interface
        self.db = BacktestDb()

        # Create run identifier
        self.run_id = shortuuid.ShortUUID().random(length=12)
        self.ensemble_id = ensemble_id
        self.time_stamp = datetime.utcnow()

        # Maintain trade ID
        self.tradeid = None
        self.db.insert_backtest(self.run_id, self.__class__.__name__, self.time_stamp, ensemble_id)


    def get_tradeid(self):
        """
        Create a new trade ID if one does not exist and return it
        or the current value.
        """

        if self.tradeid is None:
            self.tradeid = random.getrandbits(32)
        return self.tradeid
    

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


    def notify_cashvalue(self, cash, value):
        self.log(f"Cash={cash:.2f}, Value={value:.2f}")


    def notify_trade(self, trade: bt.Trade):
        """
        Called when a trade has a state change.

        Parameters
        ----------
        trade : bt.Trade
            The trade that has changed state.
        """
        
        self.db.insert_trade(self.run_id, self.current_date(), self.datas[0]._name, trade, self.ensemble_id)
        
        if not trade.isclosed:
            return
        self.tradeid = None

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


    def notify_order(self, order: bt.Order):
        """
        Called when an order has a state change.

        Parameters
        ----------
        order : bt.Order
            The order that has changed state.
        """
        
        self.db.insert_order(self.run_id, self.current_date(), self.datas[0]._name, order, self.ensemble_id)    


    def next(self):
        """
        Called on each new bar.
        """

        #  Log the closing price
        self.log(f"Close {self.dataclose[0]:.2f}")

        # Insert broker and asset price data into database
        self.db.insert_broker(self.run_id, self.current_date(), self.broker, self.ensemble_id)
        self.db.insert_yahoo_asset_price(self.run_id, self.datas[0], self.ensemble_id)

