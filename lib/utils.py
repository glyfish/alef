import numpy

def get_param_throw_if_missing(param, **kwargs):
    if param in kwargs:
        return kwargs[param]
    else:
        Exception(f"{param} parameter is required")

def get_param_default_if_missing(param, default, **kwargs):
    return kwargs[param] if param in kwargs else default
