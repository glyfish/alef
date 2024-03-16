"""create back test tables

Revision ID: 659095aec2e8
Revises: 
Create Date: 2024-02-25 15:03:48.690416

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '659095aec2e8'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "backtests",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("strategy", sa.String(256), nullable=False),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date")
    )


    op.create_table(
        "positions",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
    )


    op.create_table(
        "trades",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
    )


    op.create_table(
        "orders",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
    )


    op.create_table(
        "indicators",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("indicator", sa.String(255), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
    )


    op.create_table(
        "analyzers",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("analyzer", sa.String(256), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
    )


    op.create_table(
        "asset_prices",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("ticker", sa.String(256)),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open_price", sa.Float, nullable=False),
        sa.Column("high_price", sa.Float, nullable=False),
        sa.Column("low_price", sa.Float, nullable=False),
        sa.Column("close_price", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("run_id", "date"),
        sa.ForeignKeyConstraint(["run_id", "date"], ["backtests.run_id", "backtests.date"], name="fk_run_id_date")
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
        sa.Column("open_interest", sa.Float, nullable=False)
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
