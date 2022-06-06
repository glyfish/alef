from scipy import stats
from enum import Enum
import numpy

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

# Supported Distributions
class Dist(Enum):
    NORMAL = 1          # Normal distribution

    def create(self, **kwargs):
        return _create_distribution(self, **kwargs)

# Specify hypothesis test type
class TestHypothesis(Enum):
    TWO_TAIL = "TWO_TAIL"
    LOWER_TAIL = "LOWER_TAIL"
    UPPER_TAIL = "UPPER_TAIL"

# Create specified distribution function with specifoed parameters
def _create_distribution(dist_type, **kwargs):
    loc = get_param_default_if_missing("loc", 0.0, **kwargs)
    scale = get_param_default_if_missing("scale", 1.0, **kwargs)
    if dist_type.value == DistType.NORMAL.value:
        return DistSingleVar(dist=stats.norm, loc=loc, scale=scale)
    else:
        raise Exception(f"Distribution type is invalid: {type}")

# Normal distributions with scale σ and loc μ
class DistSingleVar:
    def _init_(self, dist, loc, scale):
        self.dist = dist
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return self.dist.pdf(x, loc=self.loc, scale=self.scale)

    def cdf(self, x):
        return self.dist.cdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, x):
        return self.dist.ppf(x, loc=self.loc, scale=self.scale)
