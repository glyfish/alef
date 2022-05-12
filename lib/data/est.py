import numpy
from enum import Enum
from pandas import (DataFrame)

##################################################################################################################
# Parameter Estimates
class EstType(Enum):
    AR = "AR"                     # Autoregressive model parameters
    AR_OFFSET = "AR_OFFSET"       # Autoregressive model with constant offset parameters
    MA = "MA"                     # Moving average model parameters
    MA_OFFSET = "MA_OFFSET"       # Moving average model  with constant offset parameters
    PERGRAM = "PERGRAM"           # Periodogram esimate of FBM Hurst parameter using OLS
    VAR_AGG = "VAR_AGG"           # variance Aggregation esimate of FBM Hurst parameter using OLS

##################################################################################################################
# Estimated parameter
class ParamEst:
    def __init__(self, est, err):
            self.est = est
            self.err = err
            self.data = [est, err]

    def __repr__(self):
        return f"ParamEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est=({self.est}), err=({self.err}), data=({self.data})"

    @staticmethod
    def from_array(meta_data):
        return ParamEst(meta_data[0], meta_data[1])

##################################################################################################################
# ARMA estimated parameters
class ARMAEst:
    def __init__(self, type, const, sigma2, params):
        self.type = type
        self.const = const
        self.order = len(params)
        self.params = params
        self.data = {"Type": type,
                     "Const": const.data,
                     "Parameters": [p.data for p in params],
                     "Sigma2": sigma2.data}

    def __repr__(self):
        return f"ARMAEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), const=({self.const}), params=({self.params}), data=({self.data})"

    def key(self):
        return f"{self.type.value}({self.order})"


    @staticmethod
    def from_dict(meta_data):
        return ARMAEst(
            type=meta_data["Type"],
            const=ParamEst.from_array(meta_data["Const"]),
            sigma2=ParamEst.from_array(meta_data["Sigma2"]),
            params=[ParamEst.from_array(est) for est in  meta_data["Parameters"]]
        )

##################################################################################################################
# OLS estimated parameters
class OLSEst:
    def __init__(self, type, const, params):
        self.type = type
        self.const = const
        self.params = params
        self.data = {"Const": const.data, "Parameters": params.data}

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"type=({self.type}), const=({self.const}), params=({self.params}), data=({self.data})"

    def key(self):
        return self.type.key

    @staticmethod
    def from_dict(meta_data):
        return ARMAEst(
            const=ParamEst.from_array(meta_data["Const"]),
            params=[ParamEst.from_array(est) for est in  meta_data["Parameters"]]
        )

##################################################################################################################
# Create estimates
def create_estimates_from_dict(dict):
    result = {}
    for key in dict.keys():
        result[key] = create_estimate_from_dict(dict[key])
    return result

def create_estimate_from_dict(dict):
    est_type = dict["Type"]
    if est_type.value == EstType.AR.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.AR_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.MA.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.MA_OFFSET.value:
        return ARMAEst.from_dict(dict)
    elif est_type.value == EstType.PERGRAM.value:
        return OLSEst.from_dict(dict)
    elif est_type.value == EstType.VAR_AGG.value:
        return OLSEst.from_dict(dict)
    else:
        raise Exception(f"Esitmate type is invalid: {est_type}")

def create_dict_from_estimates(ests):
    result = {}
    for key in ests.keys():
        result[key] = ests[key].data
    return result
