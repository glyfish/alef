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
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("strategy", sa.String(255), nullable=False),
        sa.Column("cash", sa.Float, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("buy", sa.Float, nullable=False),
        sa.Column("buy_size", sa.Float, nullable=False),
        sa.Column("sell", sa.Float, nullable=False),
        sa.Column("trade_size", sa.Float, nullable=False),
        sa.Column("pnlplus", sa.Float, nullable=False),
        sa.Column("pnlminus", sa.Float, nullable=False),
    )

    op.create_table(
        "indicators",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("indicator", sa.String(255), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
    )

    op.create_table(
        "analyzers",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("analyzer", sa.String(256), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
    )

    op.create_table(
        "asset_prices",
        sa.Column("run_id", sa.String(256), nullable=False),
        sa.Column("ticker", sa.String(256)),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
    )

    op.create_table(
        "price_series",
        sa.Column("ticker", sa.String(256)),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("adj_close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=False),
        sa.Column("open_interest", sa.Float, nullable=False),
    )

    op.create_index("ix_backtests_run_id", "backtests", ["run_id"], unique=False)
    op.create_index("ix_indicators_run_id", "indicators", ["run_id"], unique=False)
    op.create_index("ix_analyzers_run_id", "analyzers", ["run_id"], unique=False)
    op.create_index("ix_asset_prices_run_id", "asset_prices", ["run_id"], unique=False)
    op.create_index("ix_price_series_ticker", "price_series", ["ticker"], unique=False)

def downgrade() -> None:
    op.drop_table("backtests")
    op.drop_table("indicators")
    op.drop_table("analyzers")
    op.drop_table("asset_prices")
    op.drop_table("price_series")
