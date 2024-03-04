from typing import List
from typing import Optional

from sqlalchemy import create_engine, String, Float, Date, Integer, ForeignKey
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class BackTest(Base):
    __tablename__ = "backtests"

    run_id: Mapped[str]     = mapped_column(String(256), primary_key=True)
    date: Mapped[str]       = mapped_column(Date, primary_key=True)
    cash: Mapped[float]     = mapped_column(Float, nullable=False)
    value: Mapped[float]    = mapped_column(Float, nullable=False)


class Position(Base):
    __tablename__ = "positions"

    run_id: Mapped[str]      = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]        = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)


class Trade(Base):
    __tablename__ = "trades"

    run_id: Mapped[str]      = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]        = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)


class Order(Base):
    __tablename__ = "orders"

    run_id: Mapped[str]      = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]        = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)


class Analyzer(Base):
    __tablename__ = "analyzers"

    run_id: Mapped[str]   = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]     = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    analyzer: Mapped[str] = mapped_column(String(256), nullable=False)
    value: Mapped[float]  = mapped_column(Float, nullable=False)


class Indicator(Base):
    __tablename__ = "indicators"

    run_id: Mapped[str]     = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]       = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    indicator: Mapped[str]  = mapped_column(String(256), nullable=False)
    value: Mapped[float]    = mapped_column(Float, nullable=False)


class AssetPrice(Base):
    __tablename__ = "asset_prices"

    run_id: Mapped[str]        = mapped_column(String(256), ForeignKey("backtests.id"), primary_key=True)
    date: Mapped[str]          = mapped_column(Date, ForeignKey("backtests.date"), primary_key=True)
    ticker: Mapped[str]        = mapped_column(String(256))
    open_price: Mapped[float]  = mapped_column(Float, nullable=False)
    high_price: Mapped[float]  = mapped_column(Float, nullable=False)
    low_price: Mapped[float]   = mapped_column(Float, nullable=False)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)


class PriceSeries(Base):
    __tablename__ = "price_series"

    ticker: Mapped[str]              = mapped_column(String(256), primary_key=True)
    date: Mapped[str]                = mapped_column(Date, nullable=False, primary_key=True)
    open_price: Mapped[float]        = mapped_column(Float, nullable=False)
    high_price: Mapped[float]        = mapped_column(Float, nullable=False)
    low_price: Mapped[float]         = mapped_column(Float, nullable=False)
    close_price: Mapped[float]       = mapped_column(Float, nullable=False)
    adj_close_price: Mapped[float]   = mapped_column(Float, nullable=False)
    volume: Mapped[float]            = mapped_column(Float, nullable=False)
    open_interest: Mapped[float]     = mapped_column(Float, nullable=False)


class BacktraderDb:
    """
    Interface to backtrader database.

    Properties
    ----------
    engine : sqlalchemy.engine.base.Engine
        Database engine.
    """


    def __init__(self):
        self.engine = create_engine("postgresql://backtrader@localhost/backtrader", isolation_level="AUTOCOMMIT")


    def insert_indicator(self, run_id: str, date: str, indicator: str, value: float):
        """
        Insert current indicator value into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : str 
            Date of the indicator.
        indicator : str
            Name of the indicator.
        value : float
            Value of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Indicator.__table__.insert().values(run_id=run_id, 
                                                                   date=date, 
                                                                   indicator=indicator, 
                                                                   value=value))

    def insert_position(self, run_id: str, date: str):
        """
        Insert current position into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : str 
            Date of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(Position.__table__.insert().values())


    def insert_backtest(self, run_id: str, date: str):
        """
        Insert current backtest financials into the database.

        Parameters
        ----------
        run_id : str
            Unique identifier for the backtest.
        date : str 
            Date of the indicator.
        """

        with self.engine.connect() as connection:
            connection.execute(BackTest.__table__.insert().values())
