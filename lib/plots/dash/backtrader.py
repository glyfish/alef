from matplotlib import pyplot
import numpy
from pandas import DataFrame

from lib.utils import get_param_default_if_missing
from lib.plots import comp, bar
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
    ticker = data.ticker[0]
    
    title = f"{ticker} Position Size and Value"

    comp.bar(axis[0], position, date, xlabel=None, title=title, ylabel="Size", alpha=1.0)
    comp.bar(axis[1], position, date, xlabel="Date", ylabel="Dollars", alpha=1.0)