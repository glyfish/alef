from enum import Enum
import uuid
import numpy

from lib.models import var

from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.source import (DataSource, SourceBase)
from lib.data.schema import (DataType)
from lib.data.meta_data import (EstBase, TestBase, TestImplBase,
                                TestParam, TestData, TestReport,
                                ParamEst)
from lib.models import (TestHypothesis)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Define VAR
class VAR:
    # Func
    class Func(FuncBase):
        MEAN = "MEAN"                     # Stationary Mean vector
        COV = "COV"                       # Stationary covariance matrix
        ACF = "ACF"                       # Stationary Autocovarinace matrix

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Sources
    class Source(SourceBase):
        VAR = "VAR"                      # Create VAR simulation with specified parameters

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

###################################################################################################
## Create function definition for data type
###################################################################################################
def _create_func(func_type, **kwargs):
    if func_type.value == VAR.Func.MEAN.value:
        return _create_mean(func_type, **kwargs)
    elif func_type.value == VAR.Func.VAR.value:
        return _create_var_func(func_type, **kwargs)
    elif func_type.value == VAR.Func.ACF.value:
        return _create_acf(func_type, **kwargs)
    else:
        raise Exception(f"func_type is invalid: {func_type}")

###################################################################################################
# Func.MEAN
def _create_mean(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    μ = get_param_throw_if_missing("μ", **kwargs)


###################################################################################################
# Func.VAR
def _create_var_func(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    ω = get_param_throw_if_missing("ω", **kwargs)

###################################################################################################
# Func.ACF
def _create_acf(func_type, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    ω = get_param_throw_if_missing("ω", **kwargs)

###################################################################################################
## Create data source for specified type
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == ARIMA.Source.AR.value:
        return _create_var_source(source_type, x, **kwargs)
    else:
        raise Exception(f"source_type is invalid: {source_type}")

###################################################################################################
# Source.VAR
def _create_var_source(source_type, x, **kwargs):
    φ = get_param_throw_if_missing("φ", **kwargs)
    verify_type(φ, list)
    verify_condition(φ, len(φ) > 0, "len(φ) > 0")
    verify_type(φ[0], numpy.matrix)
    n = len(φ)
    m, _ = φ[0].shape

    ω_default = numpy.matrix(numpy.eye(m))
    μ_default = numpy.matrix(numpy.zeros(m)).T
    x0_default = numpy.matrix(numpy.zeros((m, n)))

    ω = get_param_default_if_missing("ω", ω_default, **kwargs)
    μ = get_param_default_if_missing("μ", μ_default, **kwargs)
    x0 = get_param_default_if_missing("x0", x0_default, **kwargs)
    verify_type(x0, numpy.matrix)
    verify_type(ω, numpy.matrix)
    verify_type(μ, numpy.matrix)

    f = lambda x : var.var(x0, μ, φ, Ω, len(x))
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"VAR({len(φ)})-Simulation-{str(uuid.uuid4())}",
                      params={"φ": φ, "σ": σ},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"AR({len(φ)})",
                      f=f,
                      x=x)
