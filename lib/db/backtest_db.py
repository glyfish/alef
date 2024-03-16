from typing import List
from typing import Optional

from datetime import datetime

from sqlalchemy import create_engine, String, Float, Date, Integer, ForeignKey
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class BackTest(Base):
    __tablename__ = "backtests"

    run_id: Mapped[str]         = mapped_column(String(256), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    cash: Mapped[float]         = mapped_column(Float, nullable=False)
    value: Mapped[float]        = mapped_column(Float, nullable=False)


class Position(Base):
    __tablename__ = "positions"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)


class Trade(Base):
    __tablename__ = "trades"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)


class Order(Base):
    __tablename__ = "orders"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)


class Analyzer(Base):
    __tablename__ = "analyzers"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    analyzer: Mapped[str]       = mapped_column(String(256), nullable=False)
    value: Mapped[float]        = mapped_column(Float, nullable=False)


class Indicator(Base):
    __tablename__ = "indicators"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    indicator: Mapped[str]      = mapped_column(String(256), nullable=False)
    value: Mapped[float]        = mapped_column(Float, nullable=False)


class AssetPrice(Base):
    __tablename__ = "asset_prices"

    run_id: Mapped[str]         = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]         = mapped_column(String(256))
    open_price: Mapped[float]   = mapped_column(Float, nullable=False)
    high_price: Mapped[float]   = mapped_column(Float, nullable=False)
    low_price: Mapped[float]    = mapped_column(Float, nullable=False)
    close_price: Mapped[float]  = mapped_column(Float, nullable=False)


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


    def insert_backtest(self, run_id: str, date: datetime.date, strategy: str, cash: float, value: float):
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
        cash : float
            Cash balance.
        value : float
            Portfolio value.
        """

        with self.engine.connect() as connection:
            connection.execute(BackTest.__table__.insert().values(
                run_id=run_id, 
                date=date, 
                cash=cash, 
                value=value
            ))


    def insert_position(self, run_id: str, date: datetime.date):
        """
        Insert current position into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Position.__table__.insert().values(
                run_id=run_id, 
                date=date
            ))


    def insert_trade(self, run_id: str, date: datetime.date):
        """
        Insert current trade into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Trade.__table__.insert().values(
                run_id=run_id, 
                date=date
            ))
        

    def insert_order(self, run_id: str, date: datetime.date):
        """
        Insert current order into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : datetime.date 
            Date of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Order.__table__.insert().values(
                run_id=run_id, 
                date=date
            ))


    def insert_indicator(self, run_id: str, date: datetime.date, indicator: str, value: float):
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
        value : float
            Value of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Indicator.__table__.insert().values(
                run_id=run_id, 
                date=date, 
                indicator=indicator, 
                value=value
            ))


    def insert_analyzer(self, run_id: str, date: datetime.date, analyzer: str, value: float):
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

        self.insert_asset_price(run_id, datas.datetime.date(0), datas._name, datas.open[0], 
                                datas.high[0], datas.low[0], datas.close[0])

    def insert_asset_price(self, run_id: str, date: datetime.date, ticker: str, open_price: float, 
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
            High price.
        low_price: float 
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