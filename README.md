# alef

Research code for quantitative finance and stochastic processes: a collection of
Jupyter notebooks exploring time series models and random processes, alongside
runnable [backtrader](https://www.backtrader.com/) trading strategies and the
supporting data layer used to record backtest results.

The numerical and modeling library lives in a companion package,
[`navi`](https://github.com/glyfish/navi), which is installed as an editable
dependency.

## Contents

### `notebooks/`

Exploratory analysis and simulations, organized by topic.

- **`random_processes/`** — ARIMA (AR/MA/ARMA estimation, ADF test), Brownian
  and fractional Brownian motion (ensembles, expectations, Hurst estimation,
  variance-ratio tests), the error-correction model (ECM), Ornstein–Uhlenbeck
  processes (simulation, mean half-life estimation), vector autoregression (VAR),
  and the vector error-correction model (VECM, including cointegration analysis
  and prediction).
- **`algorithmic_trading/`** — mean-reversion analysis of cointegrated time
  series and backtrader z-score strategy verification.

### `apps/`

Runnable scripts.

- **`trading_strategies/`** — long, short, and long/short z-score mean-reversion
  strategies, following the approach in Ernest Chan's *Algorithmic Trading:
  Winning Strategies and Their Rationale*. The z-score scales position size; runs
  are tagged with an ensemble id and persisted to the backtest database.
- **`back_trader_examples/`** — getting-started backtrader examples.
- **`output/`** — CSV output produced by the scripts.

### `data/`

Sample price series (`EWA`, `EWC`, `IGE`, `CAD=X`) used by the mean-reversion
notebooks and strategies.

### `alembic/`

Database migrations for the backtest result tables (run metadata, daily
portfolio value, and per-trade records).

## Requirements

- Python 3.11.2 (see `.python-version`; managed with [pyenv](https://github.com/pyenv/pyenv))
- PostgreSQL (for persisting backtest results)
- The companion [`navi`](https://github.com/glyfish/navi) package, checked out
  alongside this repo at `../navi`

## Setup

```sh
pyenv install 3.11.2
pyenv virtualenv 3.11.2 alef-3.11.2
pyenv activate alef-3.11.2

pip install -r requirements.txt
```

`requirements.in` is the source of truth for dependencies; `requirements.txt`
is the pinned, compiled output (e.g. via `pip-compile`).

### Environment

The repo expects the project root on `PYTHONPATH` (see `.env`):

```sh
PYTHONPATH=$HOME/Develop/gly.fish/alef:$PYTHONPATH
```

### Database

Configure the connection in `alembic.ini` (`sqlalchemy.url`, default
`postgresql://backtrader@localhost/backtest`) and apply the migrations:

```sh
alembic upgrade head
```

## Running

Launch the notebooks:

```sh
jupyter lab
```

Run a strategy:

```sh
python apps/trading_strategies/long_short_zscore_strategy.py
```

Backtest output is written to `apps/output/` and recorded in the backtest
database.

## Plot style

`gly.fish.mplstyle` is a Matplotlib style sheet used for figures throughout the
notebooks.
