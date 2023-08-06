from enum import Enum
import uuid
import numpy


from lib.models import var

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_type, verify_condition, create_space, create_logspace)

def _create_mean(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)


def _create_var_func(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    ω = get_param_throw_if_missing("ω", **kwargs)

def _create_acf(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    ω = get_param_throw_if_missing("ω", **kwargs)

###################################################################################################
# Source.VAR
def _create_var_source(source_type, x, **kwargs):
    Φ = get_param_throw_if_missing("Φ", **kwargs)
    verify_type(Φ, list)
    verify_condition(Φ, len(Φ) > 0, "len(φ) > 0")
    verify_type(Φ[0], numpy.matrix)
    n = len(Φ)
    m, _ = Φ[0].shape

    Ω_default = numpy.matrix(numpy.eye(m))
    μ_default = numpy.matrix(numpy.zeros(m)).T
    x0_default = numpy.matrix(numpy.zeros((m, n)))

    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    μ = get_param_default_if_missing("μ", μ_default, **kwargs)
    x0 = get_param_default_if_missing("x0", x0_default, **kwargs)
    verify_type(x0, numpy.matrix)
    verify_type(Ω, numpy.matrix)
    verify_type(μ, numpy.matrix)

    f = lambda x : var.var(x0, μ, Φ, Ω, len(x))
