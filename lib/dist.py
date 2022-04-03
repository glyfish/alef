from scipy import stats
from enum import Enum
import numpy

# Supported Distributions
class DistributionType(Enum):
    NORMAL = 1  # Normal distribution with parameters σ and μ

# Supported distribution functions
class DistributionFuncType(Enum):
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
def distribution_function(type, func_type, params):
    if type.value == DistributionType.NORMAL.value:
        return _normal(func_type, params)
    else:
        raise Exception(f"Distribution type is invalid: {type}")

# Normal distributions with scale σ and loc μ
def _normal(func_type, params):
    if len(params) > 0:
        σ = params[0]
    else:
        σ = 1.0
    if len(params) > 1:
        μ = params[1]
    else:
        μ = 0.0

    if func_type.value == DistributionFuncType.PDF.value:
        return lambda x : stats.norm.pdf(x, loc=μ, scale=σ)
    elif func_type.value == DistributionFuncType.CDF.value:
        return lambda x : stats.norm.cdf(x, loc=μ, scale=σ)
    elif func_type.value == DistributionFuncType.PPF.value:
        return lambda x : stats.norm.ppf(x, loc=μ, scale=σ)
    elif func_type.value == DistributionFuncType.TAIL.value:
        return lambda x : 1.0 - stats.norm.cdf(x, loc=μ, scale=σ)
    elif func_type.value == DistributionFuncType.SAMP.value:
        return lambda x=1 : stats.norm.rvs(loc=μ, scale=σ, size=x)
    elif func_type.value == DistributionFuncType.RANGE.value:
        return lambda x : numpy.linspace(-5.0*σ, 5.0*σ, x)
    else:
        raise Exception(f"Distribution function type is invalid: {func_type}")
