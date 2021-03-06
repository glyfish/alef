import numpy
from enum import Enum
from matplotlib import pyplot

from lib import stats

from lib.models import (TestHypothesis, Dist)
from lib.data.meta_data import (MetaData)
from lib.plots.axis import (PlotType, logStyle, logXStyle, logYStyle)

###############################################################################################
## Specify DistPlotConfig for distributions plot
class HypTestPlotType(Enum):
    VR_TEST = "VR_TEST"         # Variance ration test used to detect brownian motion

###############################################################################################
## HypTestPlotConfig for distributions plot type
class HypTestPlotConfig:
    def __init__(self, xlabel, ylabel, plot_type=PlotType.LINEAR, legend_labels=None, dist_type=None, **dist_params):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_type = plot_type
        self.legend_labels = legend_labels
        self.dist_type = dist_type
        self.dist_params = dist_params

###############################################################################################
# Create distribution plot type
def create_dist_plot_type(plot_type):
    if plot_type.value == HypTestPlotType.VR_TEST.value:
        return HypTestPlotConfig(xlabel=r"$Z(s)$",
                                 ylabel=r"Normal(CDF)",
                                 plot_type=PlotType.LINEAR,
                                 dist_type=Dist.NORMAL,
                                 loc=0.0,
                                 scale=1.0)
    else:
        raise Exception(f"Distribution plot type is invalid: {plot_type}")

###############################################################################################
# Hypothesis test plot (Uses DistPlotType config)
def hyp_test(df, plot_type, **kwargs):
    title        = kwargs["title"]        if "title"        in kwargs else None
    test_type    = kwargs["test_type"]    if "test_type"    in kwargs else HypothesisType.TWO_TAIL
    npts         = kwargs["npts"]         if "npts"         in kwargs else 100
    sig_level    = kwargs["sig_level"]    if "sig_level"    in kwargs else 0.05
    labels       = kwargs["labels"]       if "labels"       in kwargs else None
    title_offset = kwargs["title_offset"] if "title_offset" in kwargs else 1.0

    plot_config = create_dist_plot_type(plot_type)
    if plot_config.dist_params is not None:
        dist_params = plot_config.dist_params

    _, test_stats = MetaData.get_schema_data(df)

    dist = Dist.NORMAL.create(**kwargs)
    x_vals = dist.range(npts)
    min_stats = min(test_stats)
    max_stats = max(test_stats)
    min_stats = x_vals[0] if x_vals[0] < min_stats else min_stats
    max_stats = x_vals[-1] if x_vals[-1] > max_stats else max_stats
    x_vals = numpy.linspace(min_stats, max_stats, npts)
    y_vals = dist.cdf(x_vals)

    lower_critical_value = None
    lower_label = None
    upper_critical_value = None
    upper_label = None

    if test_type == TestHypothesis.TWO_TAIL:
        sig_level_2 = sig_level/2.0
        lower_critical_value = dist.ppf(sig_level_2)
        upper_critical_value = dist.ppf(1.0 - sig_level_2)
        lower_label = f"Lower Tail={format(sig_level_2, '1.3f')}"
        upper_label = f"Upper Tail={format(1.0 - sig_level_2, '1.3f')}"
    elif test_type == TestHypothesis.LOWER_TAIL:
        lower_critical_value = dist.ppf(sig_level)
        lower_label = f"Lower Tail={format(sig_level, '1.3f')}"
    elif test_type == TestHypothesis.UPPER_TAIL:
        upper_critical_value = dist.ppf(1.0 - sig_level)
        upper_label = f"Upper Tail={format(1.0 - sig_level, '1.3f')}"

    figure, axis = pyplot.subplots(figsize=(12, 8))

    text = axis.text(x_vals[0], 0.05*y_vals[-1], f"Signicance={format(100.0*sig_level, '2.0f')}%", fontsize=18)
    text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))

    if title is not None:
        axis.set_title(title, y=title_offset)

    axis.set_ylabel(plot_config.ylabel)
    axis.set_ylim([-0.05, 1.05])
    axis.set_xlabel(plot_config.xlabel)

    axis.plot(x_vals, y_vals)
    if lower_critical_value is not None:
        axis.plot([lower_critical_value, lower_critical_value], [0.0, 1.0], color='red', label=lower_label, lw=4)
    if upper_critical_value is not None:
        axis.plot([upper_critical_value, upper_critical_value], [0.0, 1.0], color='black', label=upper_label, lw=4)

    # if z_vals is None:
    for i in range(len(test_stats)):
        if labels is None:
            axis.plot([test_stats[i], test_stats[i]], [0.0, 1.0])
        else:
            axis.plot([test_stats[i], test_stats[i]], [0.0, 1.0], label=labels[i])

    axis.legend(loc='best', bbox_to_anchor=(0.1, 0.1, 0.8, 0.8))
