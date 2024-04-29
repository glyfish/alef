from matplotlib import pyplot
import matplotlib.gridspec as gridspec
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

    ticker = data.ticker[0]
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    title = f"{ticker} Asset Price Series\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    __asset_price(axis, data, title=title)


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

    ticker = data.ticker[0]
    mean_reversion_half_life = int(mean_reversion_half_life)
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    title = f"{ticker}, Z-Score Indicator Time Series, $t_{{1/2}}$={mean_reversion_half_life}\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    __zscore_indicator(axis, data, mean_reversion_half_life, title=title)  


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

    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    title = f"Account Balance\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    __cash_value(axis, data, title=title)


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

    ticker = order_data.ticker.iloc[0]
    run_id = order_data.run_id.iloc[0]
    ensemble_id = order_data.ensemble_id.iloc[0]

    title = f"{ticker} Order Size and Value\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    __orders(axis, order_data, asset_price_data, title=title)


def order_pnl(data: DataFrame, **kwargs):
    """
    Plot profit and loss time series computed from orders.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    pnl(data.query('order_status == "Completed"'))


def trade_pnl(data: DataFrame, **kwargs):
    """
    Plot profit and loss time series computed from trades.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    pnl(data.query('status == "Closed"'))


def pnl(data: DataFrame, **kwargs):
    """
    Plot profit and loss time series. Inputs data must be a closed trade or order,

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,8), **kwargs)
    _, axis = pyplot.subplots(2, figsize=figsize, sharex=True, sharey=False)

    ticker = data.ticker.iloc[0]
    run_id = data.run_id.iloc[0]
    ensemble_id = data.ensemble_id.iloc[0]

    title = f"{ticker} Profit and Loss\nRun ID: {run_id}, Ensemble ID: {ensemble_id}"

    __pnl(axis, data, title=title)


def zscore_backtest(broker: DataFrame, zscore_indicator: DataFrame, position: DataFrame, asset: DataFrame, 
                    orders: DataFrame, mean_reversion_half_life: int, **kwargs):
    """
    Plot backtest results.

    Parameters
    ----------
    broker : DataFrame
        Broker data.
    indicator : DataFrame
        Indicator data.
    position : DataFrame
        Position data.
    asset : DataFrame
        Asset data.
    orders : DataFrame
        Order data.
    mean_reversion_half_life : int
        Mean reversion half life for price series.
    figsize : Tuple[int, int]
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (10,10), **kwargs)

    fig = pyplot.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)

    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1:, 0])

    __cash_value(ax1, broker)
    __orders(ax2, orders, asset)


"""
Reusable plot components.

__zscore_indicator
    Plot zscore indicator time series.
__asset_price
    Plot asset price series.
__pnl
    Plot profit and loss time series.
__orders
    Plot when orders type occurs compared with asset price.
__cash_value
"""
def __zscore_indicator(axis: pyplot.axis, data: DataFrame, mean_reversion_half_life: int, **kwargs):
    """
    Plot zscore indicator time series.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : DataFrame
        Data to plot.
    mean_reversion_half_life : int
        Mean reversion half life for price series.
    """

    title = get_param_default_if_missing("title", None, **kwargs)


    date = data.date.to_numpy()
    zscore = data.zscore.to_numpy()
    mean_reversion_half_life = int(mean_reversion_half_life)

    zero = numpy.full(len(zscore), 0.0)

    comp.comparison(axis, [zscore[mean_reversion_half_life:], zero], date[mean_reversion_half_life:], title=title, xlabel="Date", 
                    ylabel="Z-Score", lw=1)


def __asset_price(axis: pyplot.axis, data: DataFrame, **kwargs):
    """
    Plot asset price series.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : DataFrame
        Data to plot.
    title : str
        Figure title.
    """

    title = get_param_default_if_missing("title", None, **kwargs)

    date = data.date.to_numpy()
    close = data.close_price.to_numpy()
    ticker = data.ticker.iloc[0]
    run_id = data.run_id.iloc[0]
    ensemble_id = data.ensemble_id.iloc[0]

    comp.curve(axis, close, date, title=title, xlabel="Date", ylabel="Price", label=ticker, lw=1)


def __pnl(axis: pyplot.axis, data: DataFrame, **kwargs):
    """
    Plot profit and loss time series.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : DataFrame
        Data to plot.
    title : str
        Figure title.
    """

    title = get_param_default_if_missing("title", None, **kwargs)

    pnl = data.pnl.to_numpy()
    pnl_date = data.date.to_numpy()

    cumulative_pnl = data.pnl.cumsum().to_numpy()

    comp.positive_negative_bar(axis[0], pnl, pnl_date, title=title, ylabel='PnL (Dollars)', alpha=1.0, xlabel=None)
    comp.curve(axis[1], cumulative_pnl, pnl_date, ylabel='Value (Dollars)', xlabel='Date')


def __orders(axis: pyplot.axis, order_data: DataFrame, asset_price_data: DataFrame, **kwargs):
    """
    Plot order size and value time series.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : DataFrame
        Data to plot.
    symbol_offset_factor : float
        Symbol offset factor.
    title : str
        Figure title.
    """

    title = get_param_default_if_missing("title", None, **kwargs)

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
    
    comp.fcurve_scatter_comparison(axis, [buy_price, sell_price], price, [buy_date, sell_date], price_date, title=title, xlabel='Date', 
                                   ylabel='Price', lw=1, labels=[ticker, 'Buy', 'Sell'], markers=['^', 'v'], marker_colors=['#007735', '#BB0000'],
                                   marker_size=6.0) 



def __cash_value(axis: pyplot.axis, data: DataFrame, **kwargs):
    """
    Plot cash and value time series.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        Axis used to draw plot.
    data : DataFrame
        Data to plot.
    title : str
        Figure title.
    """

    title = get_param_default_if_missing("title", None, **kwargs)

    date = data.date.to_numpy()
    cash = data.cash.to_numpy()
    value = data.value.to_numpy()
    spend = value - cash
    run_id = data.run_id[0]
    ensemble_id = data.ensemble_id[0]

    comp.comparison(axis, [cash, value, spend], date, title=title, xlabel="Date", ylabel="Dollars", lw=1, 
                    labels=["Cash", "Value", "Spend"])


