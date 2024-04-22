from matplotlib import pyplot
import numpy
from pandas import DataFrame

from lib.utils import get_param_default_if_missing
from lib.plots import comp
from typing import Callable

def price_series(data: DataFrame, **kwargs):
    """
    Plot asset price series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(figsize=figsize)


    date = data.date.to_numpy()
    close = data.close_price.to_numpy()
    ticker = data.ticker[0]

    mean = numpy.full(len(data), numpy.mean(close))

    title = f"{ticker} Price Series"

    comp.comparison(axis, [close, mean], date, title=title, xlabel="Date", ylabel="Price", labels=[ticker, 'Mean'], lw=1)


def asset_price(data: DataFrame, **kwargs):
    """
    Plot asset price series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(figsize=figsize)

    date = data.date.to_numpy()
    close = data.close_price.to_numpy()
    ticker = data.ticker[0]
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]


    title = f"{ticker} Asset Price Series\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.curve(axis, close, date, title=title, xlabel="Date", ylabel="Price", label=ticker, lw=1)


def zscore_indicator(data: DataFrame, mean_reversion_half_life: int, **kwargs):
    """
    Plot zscore indicator time series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    mean_reversion_half_life : int
        Mean reversion half life for price series.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(figsize=figsize)

    date = data.date.to_numpy()
    zscore = data.zscore.to_numpy()
    ticker = data.ticker[0]
    mean_reversion_half_life = int(mean_reversion_half_life)
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    zero = numpy.full(len(zscore), 0.0)

    title = f"{ticker}, Z-Score Indicator Time Series, $t_{{1/2}}$={mean_reversion_half_life}\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.comparison(axis, [zscore[mean_reversion_half_life:], zero], date[mean_reversion_half_life:], title=title, xlabel="Date", 
                    ylabel="Z-Score", lw=1)


def cash_value(data: DataFrame, **kwargs):
    """
    Plot cash and value time series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(figsize=figsize)

    date = data.date.to_numpy()
    cash = data.cash.to_numpy()
    value = data.value.to_numpy()
    spend = value - cash
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    title = f"Account Balance\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.comparison(axis, [cash, value, spend], date, title=title, xlabel="Date", ylabel="Dollars", lw=1, 
                    labels=["Cash", "Value", "Spend"])


def position(data: DataFrame, **kwargs):
    """
    Plot position size and value time series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(2, figsize=figsize, sharex=True, sharey=False)

    position = data['size'].to_numpy()
    price = data.price.to_numpy()
    date = data.date.to_numpy()
    value = price * position

    date = data.date.to_numpy()
    ticker = data.ticker.iloc[0]
    run_id = data.run_id.iloc[0]
    ensemble_id = data.ensemble_id.iloc[0]
    
    title = f"{ticker} Position Size and Value\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.bar(axis[0], position, date, xlabel=None, title=title, ylabel="Size", alpha=1.0)
    comp.bar(axis[1], value, date, xlabel="Date", ylabel="Dollars", alpha=1.0)


def orders(order_data: DataFrame, asset_price_data: DataFrame, **kwargs):
    """
    Plot order size and value time series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    symbol_offset_factor : float
        Symbol offset factor.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,6), **kwargs)
    _, axis = pyplot.subplots(figsize=figsize, sharex=True, sharey=False)

    price = asset_price_data.close_price.to_numpy()
    price_date = asset_price_data.date.to_numpy()
    offset = (numpy.max(price) - numpy.min(price)) * 0.05

    buy_orders = order_data.query('order_type == "Buy" and order_status == "Completed"')
    sell_orders = order_data.query('order_type == "Sell" and order_status == "Completed"')

    buy_price = buy_orders.price.to_numpy() - offset
    buy_date = buy_orders.date.to_numpy()
    sell_price = sell_orders.price.to_numpy() + offset
    sell_date = sell_orders.date.to_numpy()

    ticker = sell_orders.ticker.iloc[0]
    run_id = sell_orders.run_id.iloc[0]
    ensemble_id = sell_orders.ensemble_id.iloc[0]
    
    title = f"{ticker} Order Size and Value\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.fcurve_scatter_comparison(axis, [buy_price, sell_price], price, [buy_date, sell_date], price_date, title=title, xlabel='Date', 
                                    ylabel='Price', lw=1, labels=[ticker, 'Buy', 'Sell'], markers=['^', 'v'], marker_colors=['#007735', '#BB0000'],
                                    marker_size=6.0) 


def pnl(data: DataFrame, **kwargs):
    """
    Plot profit and loss time series.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,8), **kwargs)
    _, axis = pyplot.subplots(2, figsize=figsize, sharex=True, sharey=False)

    pnl = data.pnl.to_numpy()
    pnl_date = data.date.to_numpy()
    ticker = data.ticker.iloc[0]
    run_id = data.run_id.iloc[0]
    ensemble_id = data.ensemble_id.iloc[0]

    cumulative_pnl = data.pnl.cumsum().to_numpy()

    title = f"{ticker} Profit and Loss\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    comp.positive_negative_bar(axis[0], pnl, pnl_date, title=title, ylabel='PnL (Dollars)', alpha=1.0, xlabel=None)
    comp.curve(axis[1], cumulative_pnl, pnl_date, ylabel='Value (Dollars)', xlabel='Date')