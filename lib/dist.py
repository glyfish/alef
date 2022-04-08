from scipy import stats
from enum import Enum
import numpy
from tabulate import tabulate

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

class VarianceRatioTestResults:
    def __init__(self, status, sig_level, test_type, s, statistics, p_values, critical_values):
        self.status = status
        self.sig_level = sig_level
        self.test_type = test_type
        self.s = s
        self.statistics = statistics
        self.p_values = p_values
        self.critical_values = critical_values

    def __repr__(self):
        f"VarianceRatioTestResults(status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values})"

    def __str__(self):
        return f"status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values}"

    def table(self, tablefmt="grid"):
        test_status = "Passed" if self.status else "Failed"
        header = [["Status", test_status], ["Significance", f"{int(100.0*self.sig_level)}%"], ["Test Type", self.test_type]]
        if self.critical_values[0] is not None:
            header.append(["Lower Critical Value", format(self.critical_values[0], '1.3f')])
        if self.critical_values[1] is not None:
            header.append(["Upper Critical Value", format(self.critical_values[1], '1.3f')])
        return [tabulate(header, tablefmt=tablefmt)]


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
