"""create back test tables

Revision ID: 659095aec2e8
Revises: 
Create Date: 2024-02-25 15:03:48.690416

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '659095aec2e8'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "backtests",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("time_stamp", sa.DateTime, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("strategy", sa.String(256), nullable=False),
        sa.PrimaryKeyConstraint("run_id")
    )

    op.create_table(
        "broker",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date")
    )

    op.create_table(
        "positions",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("ticker", sa.String(256), nullable=False),
        sa.Column("adjbase", sa.Float, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("price_orig", sa.Float, nullable=False),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("upclosed", sa.Float, nullable=False),
        sa.Column("upopened", sa.Float, nullable=False),
        sa.Column("updt", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("run_id", "date")
    )


    op.create_table(
        "trades",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("ticker", sa.String(256), nullable=False),
        sa.Column("status", sa.String(256), nullable=False),
        sa.Column("trade_id", sa.BigInteger, nullable=False),
        sa.Column("size", sa.Integer, nullable=False),        
        sa.Column("price", sa.Float, nullable=False),        
        sa.Column("value", sa.Float, nullable=False),        
        sa.Column("commission", sa.Float, nullable=False),        
        sa.Column("pnl", sa.Float, nullable=False),        
        sa.Column("pnlcomm", sa.Float, nullable=False),        
        sa.Column("dtclose", sa.Date, nullable=True),        
        sa.Column("dtopen", sa.Date, nullable=True),
        sa.Column("baropen", sa.Integer, nullable=True),        
        sa.Column("barclose", sa.Integer, nullable=True),
        sa.Column("barlen", sa.Integer, nullable=True),
        sa.Column("is_long", sa.Boolean, nullable=True),
        sa.PrimaryKeyConstraint("run_id", "date")
    )


    op.create_table(
        "orders",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("ticker", sa.String(256), nullable=False),
        sa.Column("order_status", sa.String(256), nullable=False),
        sa.Column("order_type", sa.String(256), nullable=False),
        sa.Column("trade_id", sa.BigInteger, nullable=False),
        sa.Column("price", sa.Float, nullable=False),        
        sa.Column("value", sa.Float, nullable=False),        
        sa.Column("size", sa.Integer, nullable=False),        
        sa.Column("commission", sa.Float, nullable=False),        
        sa.Column("pnl", sa.Float, nullable=False),        
        sa.Column("exec_type", sa.String, nullable=False),        
        sa.PrimaryKeyConstraint("run_id", "date", "order_status")
    )


    op.create_table(
        "indicators",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("indicator", sa.String(255), nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("ticker", sa.String(256), nullable=False),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("params", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("run_id", "date", )
    )


    op.create_table(
        "analyzers",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("analyzer", sa.String(256), nullable=False),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("params", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("run_id", "date")
    )


    op.create_table(
        "asset_prices",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("ticker", sa.String(256)),
        sa.Column("ensemble_id", sa.String(256), nullable=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open_price", sa.Float, nullable=False),
        sa.Column("high_price", sa.Float, nullable=False),
        sa.Column("low_price", sa.Float, nullable=False),
        sa.Column("close_price", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date")
    )


    op.create_table(
        "price_series",
        sa.Column("ticker", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open_price", sa.Float, nullable=False),
        sa.Column("high_price", sa.Float, nullable=False),
        sa.Column("low_price", sa.Float, nullable=False),
        sa.Column("close_price", sa.Float, nullable=False),
        sa.Column("adj_close_price", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("ticker", "date")
    )


def downgrade() -> None:
    op.drop_table("positions")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("indicators")
    op.drop_table("analyzers")
    op.drop_table("asset_prices")
    op.drop_table("price_series")
    op.drop_table("backtests")
    op.drop_table("broker")
