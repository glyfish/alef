from scipy import stats
from enum import Enum
import numpy

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

# Supported Distributions
class DistType(Enum):
    NORMAL = 1  # Normal distribution with parameters σ and μ

    def create(self, **kwargs):
        return _create_distribution(**kwargs)

# Supported distribution functions
class DistFuncType(Enum):
    PDF = 1     # Probability density funcxtion
    CDF = 2     # Cummulative distribution function
    PPF = 3     # Percent point function (Inverse of CDF)
    TAIL = 4    # Tail distribution function (1 - CDF)
    SAMP = 5    # Return specied number of distribution samples
    RANGE = 6   # Distribution range for plotting

# Specify hypothesis test type
class HypothesisType(Enum):
    TWO_TAIL = 1
    LOWER_TAIL = 2
    UPPER_TAIL = 3

# Create specified distribution function with specifoed parameters
def _create_distribution(**kwargs):
    dist_type = get_param_throw_if_missing("dist_type", **kwargs)
    func_type = get_param_throw_if_missing("func_type", **kwargs)
    if dist_type.value == DistType.NORMAL.value:
        return _normal(func_type, **kwargs)
    else:
        raise Exception(f"Distribution type is invalid: {type}")

# Normal distributions with scale σ and loc μ
def _normal(func_type, **kwargs):
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    if func_type.value == DistFuncType.PDF.value:
        return lambda x : stats.norm.pdf(x, loc=μ, scale=σ)
    elif func_type.value == DistFuncType.CDF.value:
        return lambda x : stats.norm.cdf(x, loc=μ, scale=σ)
    elif func_type.value == DistFuncType.PPF.value:
        return lambda x : stats.norm.ppf(x, loc=μ, scale=σ)
    elif func_type.value == DistFuncType.TAIL.value:
        return lambda x : 1.0 - stats.norm.cdf(x, loc=μ, scale=σ)
    elif func_type.value == DistFuncType.SAMP.value:
        return lambda x=1 : stats.norm.rvs(loc=μ, scale=σ, size=x)
    elif func_type.value == DistFuncType.RANGE.value:
        return lambda x : numpy.linspace(-5.0*σ, 5.0*σ, x)
    else:
        raise Exception(f"Distribution function type is invalid: {func_type}")
