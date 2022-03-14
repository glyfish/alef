import numpy
from matplotlib import pyplot
from lib import config

def time_series(samples, time, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$X_t$")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)

def time_series_comparison(samples, time, labels, lengend_location, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$X_t$")
    axis.set_title(title)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1, label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)

def time_series_stack(series, labels, ylim, title):
    nplot = len(series)
    nsample = len(series[0])
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(15, 12))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        axis[i].set_ylabel(r"$X_t$")
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, nsample])
        text = axis[i].text(time[int(0.9*nsample)], 0.65*ylim[-1], labels[i], fontsize=18)
        text.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
        axis[i].plot(time, series[i], lw=1.0)

def cumulative(accum, target, title, label):
    range = max(accum) - min(accum)
    legend_pos=[0.85, 0.95]
    nsample = len(accum)
    time = numpy.linspace(1.0, nsample, nsample)
    figure, axis = pyplot.subplots(figsize=(15, 12))
    axis.set_ylim([min(accum)-0.25*range, max(accum)+0.25*range])
    axis.set_xlabel("Time")
    axis.set_ylabel(label)
    axis.set_title(title)
    axis.set_xlim([1.0, nsample])
    axis.semilogx(time, accum, label=f"Cumulative "+label)
    axis.semilogx(time, numpy.full((len(time)), target), label="Target "+label)
    axis.legend(bbox_to_anchor=legend_pos)

def auto_regression(series, result, legend_anchor, title, plot_name, lim=None):
    nsample = len(series)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_{t}$")
    axis.set_xlabel(r"$x_{t-1}$")
    if lim is not None:
        axis.set_xlim(lim)
        axis.set_ylim(lim)
        x = numpy.linspace(lim[0], lim[1], 100)
    else:
        x = numpy.linspace(numpy.min(series), numpy.max(series), 100)
    y_hat = x * φ_hat
    axis.set_title(title)
    axis.plot(series[1:], series[0:-1], marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Simulation")
    axis.plot(x, y_hat, lw=3.0, color="#000000", zorder=6, label=r"$x_{t}=\hat{\phi}x_{t-1}$")
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    axis.text(x[80], x[0],
              r"$\hat{\phi}=$" + f"{format(φ_hat, '2.3f')}\n" +
              r"$\sigma_{\hat{\phi}}=$" + f"{format(numpy.sqrt(φ_hat_var), '2.3f')}\n"
              r"$R^2=$"+f"{format(φ_r_squared, '2.3f')}\n",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.legend(bbox_to_anchor=legend_anchor).set_zorder(7)
    config.save_post_asset(figure, "regression", plot_name)
