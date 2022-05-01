import numpy

def get_param_throw_if_missing(param, **kwargs):
    if param in kwargs:
        return kwargs[param]
    else:
        raise Exception(f"{param} parameter is required")

def get_param_default_if_missing(param, default, **kwargs):
    return kwargs[param] if param in kwargs else default

def calculate_ticks(ax, ticks, round_to=0.1, center=False):
    upperbound = numpy.ceil(ax.get_ybound()[1]/round_to)
    lowerbound = numpy.floor(ax.get_ybound()[0]/round_to)
    dy = upperbound - lowerbound
    fit = numpy.floor(dy/(ticks - 1)) + 1
    dy_new = (ticks - 1)*fit
    if center:
        offset = numpy.floor((dy_new - dy)/2)
        lowerbound = lowerbound - offset
    values = numpy.linspace(lowerbound, lowerbound + dy_new, ticks)
    return values*round_to
