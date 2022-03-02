import numpy
from matplotlib import pyplot
from lib import config

def timeseries(samples, time, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)

def timeseries_comparison(samples, time, labels, lengend_location, title):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1, label=labels[i])
    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
