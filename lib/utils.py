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
    return numpy.logspace(numpy.log10(xmin), numpy.log10(xmax/xmin), npts)

def create_parameter_scan(source, *args):
    """
    Generate a parameter scan for the specified data source using the 
    specified parameters

    Parameters
    ----------
    source: lambda(**kwargs) -> (numpy.ndarray, numpy.ndarray)
        lambda calling source create.
    args : *args
        Array of parameter scan kwargs

    Returns
    -------
    (numpy.ndarray[float], list[numpy.ndarray[float]])
        time and ensemble simulation results.
    """

    scan = []
    for kwargs in args:
        _, samples = source(**kwargs)
        scan.append(samples)
    return create_space(npts=len(scan[0])), scan

def create_ensemble(source, nsim: int, **kwargs):
    """
    Generate a parameter scan for the specified data source using the 
    specified parameters

    Parameters
    ----------
    source: lambda(**kwargs) -> (numpy.ndarray, numpy.ndarray)
        lambda calling source create.
    nsim : int
        Number of simulations in ensemble
    kwargs : **kwargs
        Simulation parameters.

    Returns
    -------
    (numpy.ndarray[float], list[numpy.ndarray[float]])
        time and ensemble simulation results.
    """

    ensemble = []
    for _ in range(nsim):
        _, samples = source(**kwargs)
        ensemble.append(samples)
    return create_space(npts=len(ensemble[0])), ensemble

def apply_to_list(func, data_list, **kwargs):
    """
    Apply specified function to list of data arrays.
    
    Parameters
    ----------
    func: lambda(**kwargs) -> result
        lambda calling source create.
    data : list[numpy.ndarray]
        list of data arrays.
    kwargs : **kwargs
       Functions parameters.

    Returns
    -------
    list[results]
        List of function results.
    """

    return [func(data, **kwargs) for data in data_list]

def apply_parameter_scan(func, data, *args):
    """
    Apply specified list of parameters to data samples.
    
    Parameters
    ----------
    func: lambda(**kwargs) -> result
        lambda calling source create.
    data : numpy.ndarray
        list of data arrays.
    args : *args
        Array of parameter scan kwargs

    Returns
    -------
    list[results]
        List of function results.
    """

    return [func(data, **kwargs) for kwargs in args]

def get_s_vals(**kwargs):
    """
    Create s values used in Lo and Mackinlay lagged variance analysis.
    
    Parameters
    ----------
    linear: bool
        If True create s-values using linear spacing. If False use logarithmic spacing.
        (default is logarithmic)
    s_max : int
        Maximum s-value.
    s_min : int
        Minimum s value.
    npts : int
        Number of s-values to create
    s_vals : list[int]
        List if s-values to use

    Returns
    -------
    list[results]
        List of function results.
    """

    linear = get_param_default_if_missing("linear", False, **kwargs)
    s_min = get_param_default_if_missing("s_min", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    s_max = get_param_default_if_missing("s_max", None, **kwargs)
    s_vals = get_param_default_if_missing("s_vals", None, **kwargs)
    if npts is not None and s_max is not None:
        if linear:
            return create_space(npts=npts, xmax=s_max, xmin=s_min)
        else:
            return create_logspace(npts=npts, xmax=s_max, xmin=s_min)
    elif s_vals is not None:
        return s_vals
    else:
        raise Exception(f"s_max and npts or s_vals is required")
