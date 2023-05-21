import numpy

def get_param_throw_if_missing(param: str, **kwargs):
    """
    Raise exception if parameter is missing from kwargs.

    Parameters
    ----------
    param: str
        Parameter to type check.
    kwargs
        key word arguments

    Raises
    ------
        Exception(param does not match expected type)

    Returns
    -------
        Specified kwargs parameter.
    """
    if param in kwargs:
        return kwargs[param]
    else:
        raise Exception(f"{param} parameter is required")

def get_param_default_if_missing(param, default, **kwargs):
    """
    Get parameter from kwargs and return specified default value if it is missing.

    Parameters
    ----------
    param: str
        Parameter to type check.
    default
        value returned if specified parameter is not in kwargs.
    kwargs
        key word arguments

    Returns
    -------
        Specified kwargs parameter.
    """
    return kwargs[param] if param in kwargs else default

def verify_condition(param, condition: bool, condition_string: str):
    """
    Raise exception if parameter does not satisfy specified condition.

    Parameters
    ----------
    param: str
        Parameter to type check.
    default
        value returned if specified parameter is not in kwargs.
    kwargs
        key word arguments

    Raises
    ------
        Exception(param does satisfy condition)
    """
    if not condition:
        raise Exception(f"{param} should satisfy {condition_string}")

def verify_type(param, expected_type):
    """
    Raise exception if parameter is not specified type.

    Parameters
    ----------
    param
        Parameter to type check.
    expected_type
        Expected tpe

    Raises
    ------
        Exception(param does not match expected type)
    """
    if not isinstance(param, expected_type):
        raise Exception(f"{param} is type {type(param)}. Expected {expected_type}")

def create_space(**kwargs):
    """
    Create linear space with specified parameters.

    Parameters
    ----------
    npts: float
        number of steps in simulation.
    xmax: int
        Space maximum value.
    xmin: float
        Space minimum value (default 0.0).
    Δx : float
        Space grid spacing (default 1).

    Raises
    ------
        Exception(xmax or npts is required)

    Returns
    -------
    numpy.ndarray[float]
        Linear space.
    """
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
    kwargs["npts"] = npts
    kwargs["xmax"] = xmax
    kwargs["xmin"] = xmin
    kwargs["Δx"] = Δx
    return numpy.linspace(xmin, xmax, npts)

def create_logspace(**kwargs):
    """
    Create log space with specified parameters.

    Parameters
    ----------
    npts: float
        number of steps in simulation.
    xmax: int
        Space maximum value.
    xmin: float
        Space minimum value (default 0.0).
    Returns
    -------
    numpy.ndarray[float]
        Linear space.
    """
    npts = get_param_throw_if_missing("npts", **kwargs)
    xmax = get_param_throw_if_missing("xmax", **kwargs)
    xmin = get_param_default_if_missing("xmin", 1.0, **kwargs)
    kwargs["npts"] = npts
    kwargs["xmax"] = xmax
    kwargs["xmin"] = xmin
    return numpy.logspace(numpy.log10(xmin), numpy.log10(xmax/xmin), npts)
