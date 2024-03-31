from typing import List, Tuple, Optional
from enum import Enum

from datetime import datetime
import json
import pandas
import os
import numpy

from sqlalchemy import create_engine, String, Float, Date, Integer, ForeignKey, JSON, Boolean, DateTime
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column, relationship

import backtrader as bt

from lib.utils import read_yahoo_data

class MappedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class OrderExecutionType(str, MappedEnum):
    """
    Order execution type.
    """

    Market = 'Market'
    Close = 'Close'
    Limit = 'Limit'
    Stop = 'Stop'
    StopLimit = 'StopLimit'
    StopTrail = 'StopTrail'
    StopTrailLimit = 'StopTrailLimit'
    Historical = 'Historical'


class OrderStatusType(str, MappedEnum):
    """
    Order status type.
    """

    Created = 'Created'
    Submitted = 'Submitted'
    Accepted = 'Accepted'
    Partial = 'Partial'
    Completed = 'Completed'
    Canceled = 'Canceled'
    Expired = 'Expired'
    Margin = 'Margin'
    Rejected = 'Rejected'


class OrderType(str, MappedEnum):
    """
    Order type.
    """

    Buy = 'Buy'
    Sell = 'Sell'


class TradeStatus(str, MappedEnum):
    """
    Trade status.
    """

    Created ='Created'
    Open = 'Open'
    Closed = 'Closed'


class Base(DeclarativeBase):
    pass


class BackTest(Base):
    __tablename__ = "backtests"

    run_id: Mapped[str]          = mapped_column(String(256), primary_key=True)
    time_stamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    date: Mapped[datetime.date]  = mapped_column(Date, primary_key=True)
    strategy: Mapped[str]        = mapped_column(String(256), primary_key=True)
    cash: Mapped[float]          = mapped_column(Float, nullable=False)
    value: Mapped[float]         = mapped_column(Float, nullable=False)


class Position(Base):
    __tablename__ = "positions"

    run_id: Mapped[str]          = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date]  = mapped_column(Date, primary_key=True)
    ticker: Mapped[str]          = mapped_column(String(256), nullable=False)
    adjbase: Mapped[float]       = mapped_column(Float, nullable=False)
    price: Mapped[float]         = mapped_column(Float, nullable=False)
    price_orig: Mapped[float]    = mapped_column(Float, nullable=False)
    size: Mapped[int]            = mapped_column(Integer, nullable=False)
    upclosed: Mapped[float]      = mapped_column(Float, nullable=False)
    upopened: Mapped[float]      = mapped_column(Float, nullable=False)
    updt: Mapped[datetime.date]  = mapped_column(Date, nullable=False)


class Trade(Base):
    __tablename__ = "trades"

    run_id: Mapped[str]             = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date]     = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]             = mapped_column(String(256), nullable=False)
    status: Mapped[str]             = mapped_column(String(256), nullable=False)
    trade_id: Mapped[int]           = mapped_column(Integer, nullable=False)
    size: Mapped[float]             = mapped_column(Integer, nullable=False)
    price: Mapped[float]            = mapped_column(Float, nullable=False)
    value: Mapped[float]            = mapped_column(Float, nullable=False)
    commission: Mapped[float]       = mapped_column(Float, nullable=False)
    pnl: Mapped[float]              = mapped_column(Float, nullable=False)
    pnlcomm: Mapped[float]          = mapped_column(Float, nullable=False)
    dtclose: Mapped[datetime.date]  = mapped_column(Date, nullable=False)
    dtopen: Mapped[datetime.date]   = mapped_column(Date, nullable=False)


class Order(Base):
    __tablename__ = "orders"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]         = mapped_column(String(256), nullable=False)
    order_status: Mapped[str]   = mapped_column(String(256), nullable=False)
    order_type: Mapped[str]     = mapped_column(String(256), nullable=False)
    price: Mapped[float]        = mapped_column(Float, nullable=False)
    value: Mapped[float]        = mapped_column(Float, nullable=False)
    size: Mapped[int]           = mapped_column(Integer, nullable=False)
    commission: Mapped[float]   = mapped_column(Float, nullable=False)
    pnl: Mapped[float]          = mapped_column(Float, nullable=False)
    exec_type: Mapped[str]      = mapped_column(String(256), nullable=False)


class Analyzer(Base):
    __tablename__ = "analyzers"

    run_id: Mapped[str]                 = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date]         = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]                 = mapped_column(String(256))
    analyzer: Mapped[str]               = mapped_column(String(256), nullable=False)
    value: Mapped[dict]                 = mapped_column(JSON, nullable=False)
    parameters: Mapped[Optional[dict]]  = mapped_column(JSON, nullable=True)


class Indicator(Base):
    __tablename__ = "indicators"

    run_id: Mapped[str]                = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date]        = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]                = mapped_column(String(256))
    indicator: Mapped[str]             = mapped_column(String(256), nullable=False)
    value: Mapped[dict]                = mapped_column(JSON, nullable=False)
    params: Mapped[Optional[dict]]     = mapped_column(JSON, nullable=True)


class AssetPrice(Base):
    __tablename__ = "asset_prices"

    run_id: Mapped[str]          = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date]  = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]          = mapped_column(String(256))
    open_price: Mapped[float]    = mapped_column(Float, nullable=False)
    high_price: Mapped[float]    = mapped_column(Float, nullable=False)
    low_price: Mapped[float]     = mapped_column(Float, nullable=False)
    close_price: Mapped[float]   = mapped_column(Float, nullable=False)


class PriceSeries(Base):
    __tablename__ = "price_series"

    ticker: Mapped[str]              = mapped_column(String(256), primary_key=True)
    date: Mapped[datetime.date]      = mapped_column(Date, nullable=False, primary_key=True)
    open_price: Mapped[float]        = mapped_column(Float, nullable=False)
    high_price: Mapped[float]        = mapped_column(Float, nullable=False)
    low_price: Mapped[float]         = mapped_column(Float, nullable=False)
    close_price: Mapped[float]       = mapped_column(Float, nullable=False)
    adj_close_price: Mapped[float]   = mapped_column(Float, nullable=False)
    volume: Mapped[float]            = mapped_column(Float, nullable=False)
    open_interest: Mapped[float]     = mapped_column(Float, nullable=False)


class BacktestDb:
    """
    Interface to backtrader database.

    Properties
    ----------
    engine : sqlalchemy.engine.base.Engine
        Database engine.
    """


    def __init__(self):
        self.__db_url = "postgresql://backtrader@localhost/backtest"
        self.engine = create_engine(self.__db_url, isolation_level="AUTOCOMMIT")


    def insert_backtest(self, run_id: str, date: datetime.date, strategy: str, time_stamp: datetime, 
                        broker: bt.BrokerBase):
        """
        Insert current backtest financials into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        strategy: str
            Strategy used in backtest.
        broker : bt.BrokerBase
            backtrader broker
        time_stamp: datetime
            Time stamp of the backtest.
        """

        with self.engine.connect() as connection:
            connection.execute(BackTest.__table__.insert().values(
                run_id=run_id, 
                time_stamp=time_stamp,
                date=date, 
                strategy=strategy,
                cash=broker.getcash(), 
                value=broker.getvalue()
            ))


    def insert_position(self, run_id: str, date: datetime.date, ticker: str, position: bt.Position):
        """
        Insert current position into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        ticker : str
            Ticker symbol.
        position: bt.Position
            Backtrader Position object.
        """

        updt =  None

        with self.engine.connect() as connection:
            connection.execute(Position.__table__.insert().values(
                run_id=run_id, 
                date=date,
                ticker=ticker,
                adjbase=position.adjbase,
                price=position.price,
                price_orig=position.price_orig,
                size=position.size,
                upclosed=position.upclosed,
                upopened=position.upopened,
                updt=position.updt
            ))


    def insert_trade(self, run_id: str, date: datetime.date, ticker: str, trade: bt.Trade):
        """
        Insert current trade into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        ticker : str
            Ticker symbol.
        trade: bt.Trade
            Backtrader Trade object.
        """

        dtclose = trade.close_datetime() if trade.dtclose > 0.0 else None
        dtopen = trade.open_datetime() if trade.dtopen > 0.0 else None

        with self.engine.connect() as connection:
            connection.execute(Trade.__table__.insert().values(
                run_id=run_id, 
                date=date,
                ticker=ticker,
                status=trade.status,
                trade_id=trade.tradeid,
                size=trade.size,
                price=trade.price,
                value=trade.value,
                commission=trade.commission,
                pnl=trade.pnl,
                pnlcomm=trade.pnlcomm,
                dtclose=dtclose,
                dtopen=dtopen
            ))
        

    def insert_order(self, run_id: str, date: datetime.date, ticker: str, order: bt.Order):
        """
        Insert current order into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        ticker : str
            Ticker symbol.
        order: bt.Order
            Backtrader Order object.
        """

        order_type = order.ordtypename()
        order_exec_type = OrderExecutionType.list()[order.exectype]
        order_status = OrderStatusType.list()[order.status]

        order_data = order.executed if order_status == OrderStatusType.Completed.value else order.created
        price = order_data.price
        value = order_data.value
        size = order_data.size
        comm = order_data.comm
        pnl = order_data.pnl

        with self.engine.connect() as connection:
            connection.execute(Order.__table__.insert().values(
                run_id=run_id, 
                date=date,
                ticker=ticker,
                order_status=order_status,
                order_type=order_type,
                price=price,
                value=value,
                size=size,
                commission=comm,
                pnl=pnl,
                exec_type=order_exec_type
            ))


    def insert_yahoo_asset_price(self, run_id: str, datas):
        """
        Insert current position into the database from a yahoo CSV input feed.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        datas : 
            List of data feeds.
        """

        self.__insert_asset_price(run_id, datas.datetime.date(0), datas._name, datas.open[0], 
                                 datas.high[0], datas.low[0], datas.close[0])


    def insert_price_series(self, ticker: str, date: datetime.date, open_price: float, high_price: float, low_price: float, 
                           close_price: float, adj_close_price: float, volume: float, open_interest: float):
        """
        Insert current price series into the database.

        Parameters
        ----------
        ticker: str
            Ticker symbol.
        date: datetime.date
            Date of the indicator.
        open_price: float
            Opening price.
        high_price: float
            High price for day.
        low_price: float 
            Low price for day.
        close_price: float
            Closing price.
        adj_close_price: float
            Adjusted closing price.
        volume: float
            Trade volume.
        open_interest: float
            Open interest.
        """        

        with self.engine.connect() as connection:
            connection.execute(PriceSeries.__table__.insert().values(
                ticker=ticker, 
                date=date, 
                open_price=open_price, 
                high_price=high_price, 
                low_price=low_price, 
                close_price=close_price, 
                adj_close_price=adj_close_price, 
                volume=volume, 
                open_interest=open_interest
            ))

    
    def insert_zscore_indicator(self, run_id: str, date: datetime.date, ticker: str, zscore: float, period: int):
        params = json.dumps({'period': period})
        value = json.dumps({'zscore': zscore})
        self.__insert_indicator(run_id, date, ticker, 'zscore', value, params)


    def insert_yahoo_price_series(self, file_root: str, ticker: str):
        """
        Insert current price series into the database from a yahoo CSV input feed.        

        Parameters
        ----------
        file_path: str
            File path.            
        """

        file_path = os.path.abspath(f"{file_root}/{ticker}.csv")
        data = read_yahoo_data(file_path)
        data.rename(columns={"Open": "open_price", "High": "high_price", "Low": "low_price", "Close": "close_price", 
                             "Adj Close": "adj_close_price", "Volume": "volume"}, inplace=True)
        data.index.names = ['date']
        data['ticker'] = numpy.full(len(data), ticker)
        data.to_sql("price_series", self.engine, if_exists="append")


    def fetch_price_series(self, ticker: str, start_date: str=None, end_date: str=None) -> pandas.DataFrame:
        """
        Fetch price series from the backtest database.

        Parameters
        ----------
        ticker: str
            Ticker symbol.
        start_date: str
            Start date.
        end_date: str
            End date.

        Returns
        -------
        pandas.DataFrame
            Price series.
        """

        query = f"SELECT * FROM price_series WHERE ticker='{ticker}'"
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"

        return pandas.read_sql(query, self.engine)

    def __insert_asset_price(self, run_id: str, date: datetime.date, ticker: str, open_price: float, 
                             high_price: float, low_price: float, close_price: float):
        """
        Insert current position into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        ticker : str
            Ticker symbol.
        open_price : float
            Opening price.
        high_price : float
            High price.
        low_price : float
            Low price.
        close_price : float
            Closing price.            
        """

        with self.engine.connect() as connection:
            connection.execute(AssetPrice.__table__.insert().values(
                run_id=run_id, 
                date=date, 
                ticker=ticker, 
                open_price=open_price, 
                high_price=high_price, 
                low_price=low_price, 
                close_price=close_price
            ))


    def __insert_analyzer(self, run_id: str, date: datetime.date, analyzer: str, value: float):
        """
        Insert current analyzer value into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        analyzer : str
            Name of the analyzer.
        value : float
            Value of the analyzer.
        """

        with self.engine.connect() as connection:
            connection.execute(Analyzer.__table__.insert().values(
                run_id=run_id, 
                date=date, 
                analyzer=analyzer, 
                value=value
            ))


    def __insert_indicator(self, run_id: str, date: datetime.date, ticker: str, indicator: str, value: str, 
                          params: str = None):
        """
        Insert current indicator value into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        indicator : str
            Name of the indicator.
        value : str
            Value of the indicator.
        params : str
            Indicator parameters.
        """

        with self.engine.connect() as connection:
            connection.execute(Indicator.__table__.insert().values(
                run_id=run_id, 
                date=date,
                ticker=ticker,
                indicator=indicator, 
                value=value,
                params=params
            ))


