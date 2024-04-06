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

    title = f"{ticker} Asset Price Series"

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

    title = f"{ticker} ZScore Indicator Time Series, $t_{{1/2}}$={mean_reversion_half_life}"

    comp.curve(axis, zscore[mean_reversion_half_life:], date[mean_reversion_half_life:], title=title, xlabel="Date", 
               ylabel="Z-Score", label=ticker, lw=1)
