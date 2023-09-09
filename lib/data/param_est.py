import numpy
from enum import Enum
from json import dumps

class EstModel(str, Enum):
    """
    Estimate model.

    Values
    ------
    ARMA
        Assume an ARMA(p,q) model when performing estimate.
    OLS_SING_VAR
        Assume a single variable OLS model when performing regression.
    VAR
        ASSUME a VAR(p) model when performing estimate.
    """

    ARMA = "ARMA"
    OLS_SING_VAR = "OLS_SING_VAR"
    VAR = "VAR"


class ParamEst:
    """
    Model used to store a parameter estimate result.

    Properties
    ----------
    est: float
        Estimate value.
    err: float
        Estimate error.
    est_label: str
        Estimate label used when display results.
    err_label: str
        Estimate error label used when display results.
    order: int
        Parameter order index.
    row: int
        Parameter row index.
    column: int
        Parameter column index. 
    est_id: str
        Estimate ID..
    param_type: str
        Parameter type.
    """

    def __init__(self, est: float, err: float=None, est_label: str=None, err_label: str=None, 
                 order: int=1, row: int=0, column: int=0, est_id: str=None, param_type: str=None):
            self.est = est
            self.err = err
            self.est_label = est_label
            self.err_label = err_label
            self.row = row
            self.column = column
            self.order = order
            self.est_id = est_id
            self.param_type = param_type

    def set_labels(self, est_label, err_label):
        self.est_label = est_label
        self.err_label = err_label

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return dumps(self, indent=indent, default=lambda o: o.__dict__)

    def __repr__(self):
        return f"ParamEst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est=({self.est}), " \
               f"err=({self.err}, " \
               f"est_label=({self.est_label}), " \
               f"err_label=({self.err_label}), " \
               f"order=({self.order}), " \
               f"row=({self.row}), " \
               f"column=({self.column}), " \
               f"est_id=({self.est_id}), " \
               f"param_type=({self.param_type})"

    @staticmethod
    def from_dict(data):
        est = data["est"] if "est" in data else None
        est_label = data["est_label"] if "est_label" in data else None
        err = data["err"] if "err" in data else None
        err_label = data["err_label"] if "err_label" in data else None
        order = data["order"] if "order" in data else 1
        row = data["row"] if "row" in data else 0
        column = data["column"] if "column" in data else 0
        err_label = data["err_label"] if "err_label" in data else None
        est_id = data["est_id"] if "est_id" in data else None
        param_type = data["param_type"] if "param_type" in data else None
        return ParamEst(est, err, est_label, err_label, order, row, column, est_id, param_type)
    

class OLSSinlgeVarTransform:
    """
    OLS result transformation.

    Properties
    ----------
    model: str
        Transformation model.
    const: ParamEst
        Constant estimate.
    param: ParamEst
        Parameter estimate.
    """

    def __init__(self, model: str, const: ParamEst, param: ParamEst):
        self.model = model
        self.param = param
        self.const = const

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return dumps(self, indent=indent, default=lambda o: o.__dict__)

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"model=({self.model}), " \
               f"param=({self.param}), " \
               f"const=({self.const})"


class OLSParamType(str, Enum):
    """
    OLS single variable estimate parameter type.

    Values
    ------
    OLS_CONST
        Estimate of constant parameter.
    OLS_R2
        Estimate of R2 parameter.
    OLS_PARAM
        Estimate of slope parameter.
    TRANS_CONST
        Estimate of slope parameter.
    TRANS_PARAM
        Estimate of slope parameter.
    """

    OLS_CONST = "OLS_CONST"
    OLS_R2 = "OLS_R2"
    OLS_PARAM = "OLS_PARAM"
    TRANS_CONST = "TRANS_CONST"
    TRANS_PARAM = "TRANS_PARAM"


class OLSSingleVarResult:
    """
    Single variable OLS estimate result.

    Properties
    ----------
    est_model: EstModel
        Estimation model.
    const: ParamEst
        Constant estimate.
    param: ParamEst
        Parameter estimate.
    r2: ParamEst
        Estimate r^2.
    transform: OLSSinlgeVarTransform
        Estimated parameter transformation.
    """

    def __init__(self, const: ParamEst, param: ParamEst, r2: float, est_id: str=None):
        self.est_model = EstModel.OLS_SING_VAR
        self.const = const
        self.param = param
        self.r2 = r2
        self.transform = None
        self.est_id = est_id

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()
    
    def _props(self):
        return f"est_id={self.est_id}, " \
               f"est_model=({self.est_model}), " \
               f"const=({self.const}), " \
               f"params=({self.param}, "\
               f"r2=({self.r2}), " \
               f"transform=({self.transform})"
    
    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=lambda o: o.__dict__)
        else:
            return dumps(self, default=lambda o: o.__dict__)
    
    def set_transform(self, transform: OLSSinlgeVarTransform):
        self.transform = transform


class ARMAEstType(str, Enum):
    """
    ARMA model type.

    Values
    ------
    AR
        AR(p) model.
    AR_OFFSET
        AR(p) model with constant offset.
    MA
        MA(q) model.
    MA_OFFSET
        MA(q) model with constant offset.
    """

    AR = "AR"
    AR_OFFSET = "AR_OFFSET"
    MA = "MA"
    MA_OFFSET = "MA_OFFSET"

    def formula(self):
        if self.value == ARMAEstType.AR.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.value == ARMAEstType.AR_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        elif self.value == ARMAEstType.MA.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.value == ARMAEstType.MA_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        else:
            raise Exception(f"Estimate type is invalid: {self}")

    def set_param_labels(self, param, i):
        if self.value == ARMAEstType.AR.value or self.value == ARMAEstType.AR_OFFSET.value:
            param.set_labels(est_label=f"$\hat{{\phi_{{{i}}}}}$",
                             err_label=f"$\sigma_{{$\hat{{\phi_{{{i}}}}}}}$")
        elif self.value == ARMAEstType.MA.value or self.value == ARMAEstType.MA_OFFSET.value:
            param.set_labels(est_label=f"$\hat{{\\theta_{{{i}}}}}$",
                             err_label=f"$\sigma_{{$\hat{{\theta_{{{i}}}}}}}$")
        else:
            raise Exception(f"Estimate type is invalid: {self}")

class ARMAParamType(str, Enum):
    """
    ARAM estimate parameter type.

    Values
    ------
    ARMA_CONST
        Estimate of constant parameter.
    ARMA_PARAM
        Estimate of R2 parameter.
    ARMA_SIG2
        Estimate of slope parameter.
    """

    ARMA_CONST = "ARMA_CONST"
    ARMA_PARAM = "ARMA_PARAM"
    ARMA_SIG2 = "ARMA_SIG2"


class ARMAEst:
    """
    ARMA parameter estimate result.

    Parameters
    ----------
    const: ParamEst
        Estimate of model constant parameter.
    params: list[ParamEst]
        Estimate of model Parameters.
    sigma2: ParamEst
        Estimate of variance of model random component.
    arma_est_type: ARMAEstType
        ARMA model estimate type.
    """

    def __init__(self, const: ParamEst, params: list[ParamEst], sigma2: ParamEst, arma_est_type: ARMAEstType=ARMAEstType.AR, est_id: str=None):
        self.est_model = EstModel.ARMA
        self.arma_est_type = arma_est_type
        self.const = const
        self.order = len(params)
        self.params = params
        self.sigma2 = sigma2
        self.est_id = est_id
        self.__set_const_labels()
        self.__set_params_labels()
        self.__set_sigma2_labels()

    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=lambda o: o.__dict__)
        else:
            return dumps(self, default=lambda o: o.__dict__)

    def __repr__(self):
        return f"ARMAEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est_model=({self.__est_model}), " \
               f"arma_est_type=({self.__arma_est_type}), " \
               f"est_id={self.__est_id}, " \
               f"const=({self.__const}), " \
               f"order=({self.__order}), " \
               f"params=({self.__params}), " \
               f"sigma2=({self.__sigma2})"

    def __set_const_labels(self):
        self.const.set_labels(est_label=r"$\hat{\mu^*}$",
                              err_label=r"$\sigma_{\hat{\mu^*}}$")

    def __set_params_labels(self):
        for i in range(len(self.params)):
            self.arma_est_type.set_param_labels(self.params[i], i)

    def __set_sigma2_labels(self):
        self.sigma2.set_labels(est_label=r"$\hat{\sigma^2}$",
                               err_label=r"$\sigma_{\hat{\sigma^2}}$")


class VARParamType(str, Enum):
    """
    VAR estimate parameter type.

    Values
    ------
    VAR_CONST
        Estimate of constant parameter.
    VAR_PARAM
        Estimate of R2 parameter.
    VAR_OMEGA
        Estimate of slope parameter.
    """

    VAR_CONST = "VAR_CONST"
    VAR_PARAM = "VAR_PARAM"
    VAR_OMEGA = "VAR_OMEGA"


class VAREst:
    """
    VAR parameter estimate result.

    Parameters
    ----------
    const: list[ParamEst]
        Estimate of model constant parameter.
    params: list[ParamEst]
        Estimate of model Parameters.
    omega: ParamEst
        Estimate of variance of model random component.
    arma_est_type: ARMAEstType
        ARMA model estimate type.
    """

    def __init__(self, const: list[ParamEst], params: list[list[list[ParamEst]]], omega: list[list[ParamEst]]):
        self.__est_model = EstModel.VAR
        self.__const = const
        self.__order = len(params)
        self.__params = params
        self.__omega = omega

    def __repr__(self):
        return f"VAREst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est_model=({self.__est_model}), " \
               f"arma_est_type=({self.__arma_est_type}), " \
               f"const=({self.__const}), " \
               f"order=({self.__order}), " \
               f"params=({self.__params}), " \
               f"omega=({self.__omega})"

    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=lambda o: o.__dict__)
        else:
            return dumps(self, default=lambda o: o.__dict__)

