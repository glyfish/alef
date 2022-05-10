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
