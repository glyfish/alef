from enum import (Enum, EnumMeta)
import numpy

from lib import stats
from lib.data.func import (DataFunc, FuncBase)
from lib.data.schema import (DataType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create Ornstien-Uhlenbeck Process Functions
class OU:
    class Func(FuncBase):
        MEAN = "OU_MEAN"                    # Ornstein-Uhelenbeck process mean
        VAR = "OU_VAR"                      # Ornstein-Uhelenbeck process variance
        COV = "OU_COV"                      # Ornstein-Uhelenbeck process covariance
        PDF = "OU_PDF"                      # Ornstein-Uhelenbeck process PDF
        CDF = "OU_CDF"                      # Ornstein-Uhelenbeck process CDF
        PDF_LIMIT = "OU_PDF"                # Ornstein-Uhelenbeck process PDF
        CDF_LIMIT = "OU_CDF"                # Ornstein-Uhelenbeck process CDF

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

###################################################################################################
## create function definition for data type
def _create_func(func_type, **kwargs):
    Exception(f"Func is invalid: {func_type}")

###################################################################################################
# Create DataFunc objects for specified Func
# Source.OU_MEAN
def _create_ou_mean(source_type, x, **kwargs):
    μ = get_param_default_if_missing("μ", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", 1.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    x0 = get_param_default_if_missing("x0", 0.0, **kwargs)
    f = lambda x : ou.ou(μ, λ, Δt, len(x), σ, x0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=source_type,
                    params={"nlags": nlags},
                    ylabel=r"$\rho_\tau$",
                    xlabel=r"$\tau$",
                    desc="Ensemble ACF",
                    fy=fy,
                    fx=fx)
