import numpy
from enum import Enum

###############################################################################################
# Supported plot types supported
class PlotType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

###############################################################################################
## Add axes for log plots for 1 to 3 decades
def logStyle(axis, x, y):
    minx = min(x) if min(x) > 0.0 else 1.0
    if numpy.log10(max(x)/minx) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.spines['left'].set_color("#b0b0b0")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logXStyle(axis, x, y):
    minx = min(x) if min(x) > 0.0 else 1.0
    if numpy.log10(max(x)/minx) < 4:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['bottom'].set_color("#b0b0b0")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logYStyle(axis, x, y):
    if numpy.log10(max(y)/min(y)) < 4:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        axis.spines['left'].set_color("#b0b0b0")
