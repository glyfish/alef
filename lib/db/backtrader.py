from typing import List
from typing import Optional

from sqlalchemy import create_engine, String, Float, Date, Integer
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column, relationship

engine = create_engine("postgresql://backtrader@localhost/backtrader",
                       isolation_level="AUTOCOMMIT")

class Base(DeclarativeBase):
    pass


class BackTest(Base):
    __tablename__ = "backtests"

    run_id: Mapped[str]         = mapped_column(String(256), nullable=False)
    date: Mapped[str]           = mapped_column(Date, nullable=False, unique=True)
    strategy: Mapped[str]       = mapped_column(String(255), nullable=False)
    cash: Mapped[float]         = mapped_column(Float, nullable=False)
    value: Mapped[float]        = mapped_column(Float, nullable=False)
    buy: Mapped[float]          = mapped_column(Float, nullable=False)
    buy_size: Mapped[float]     = mapped_column(Float, nullable=False)
    sell: Mapped[float]         = mapped_column(Float, nullable=False)
    trade_size: Mapped[float]   = mapped_column(Float, nullable=False)
    pnlplus: Mapped[float]      = mapped_column(Float, nullable=False)
    pnlminus: Mapped[float]     = mapped_column(Float, nullable=False)



class Indicator(Base):
    __tablename__ = "indicators"

    run_id: Mapped[str]      = mapped_column(String(256), nullable=False)
    date: Mapped[str]        = mapped_column(Date, nullable=False, unique=True)
    indicator: Mapped[str]   = mapped_column(String(255), nullable=False)
    value: Mapped[float]     = mapped_column(Float, nullable=False)


class Analyzer(Base):
    __tablename__ = "analyzers"

    run_id: Mapped[str]   = mapped_column(String(256), nullable=False)
    analyzer: Mapped[str] = mapped_column(String(256), nullable=False)
    value: Mapped[float]  = mapped_column(Float, nullable=False)


class AssetPrice(Base):
    __tablename__ = "asset_prices"

    run_id: Mapped[str]        = mapped_column(String(256), nullable=False)
    ticker: Mapped[str]        = mapped_column(String(256))
    date: Mapped[str]          = mapped_column(Date, nullable=False, unique=True)
    open_price: Mapped[float]  = mapped_column(Float, nullable=False)
    high_price: Mapped[float]  = mapped_column(Float, nullable=False)
    low_price: Mapped[float]   = mapped_column(Float, nullable=False)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)

class PriceSeries(Base):
    __tablename__ = "price_series"

    ticker: Mapped[str]              = mapped_column(String(256))
    date: Mapped[str]                = mapped_column(Date, nullable=False, unique=True)
    open_price: Mapped[float]        = mapped_column(Float, nullable=False)
    high_price: Mapped[float]        = mapped_column(Float, nullable=False)
    low_price: Mapped[float]         = mapped_column(Float, nullable=False)
    close_price: Mapped[float]       = mapped_column(Float, nullable=False)
    adj_close_price: Mapped[float]   = mapped_column(Float, nullable=False)
    volume: Mapped[float]            = mapped_column(Float, nullable=False)
    open_interest: Mapped[float]     = mapped_column(Float, nullable=False)

