import numpy

def throw_if_missing(param, **kwargs):
    if param in kwargs:
        return kwargs[param]
    else:
        Exception(f"{param} parameter is required")
