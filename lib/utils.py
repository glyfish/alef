import numpy

def get_param_throw_if_missing(param, **kwargs):
    if param in kwargs:
        return kwargs[param]
    else:
        raise Exception(f"{param} parameter is required")

def get_param_default_if_missing(param, default, **kwargs):
    return kwargs[param] if param in kwargs else default

def verify_type(param, expected_type):
    if not isinstance(param, expected_type):
        raise Exception(f"{param} is type {type(param)}. Expected {expected_type}")

def verify_types(param, expected_types):
    if not isinstance(param, expected_types):
        raise Exception(f"{param} is type {type(param)}. Expected {expected_types}")

def create_space(**kwargs):
    npts = get_param_default_if_missing("npts", None, **kwargs)
    xmax = get_param_default_if_missing("xmax", None, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    if xmax is None and npts is None:
        raise Exception(f"xmax or npts is required")
    if xmax is None:
        xmax = (npts - 1)*Δx + xmin
    elif npts is None:
        npts = int((xmax-xmin)/Δx) + 1
    return numpy.linspace(xmin, xmax, npts)

def create_logspace(**kwargs):
    npts = get_param_throw_if_missing("npts", **kwargs)
    xmax = get_param_throw_if_missing("xmax", **kwargs)
    xmin = get_param_default_if_missing("xmin", 1.0, **kwargs)
    return numpy.logspace(numpy.log10(xmin), numpy.log10(xmax/xmin), npts)
