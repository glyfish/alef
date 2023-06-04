import numpy
from enum import Enum
from lib import stats

class EstModel(str, Enum):
    """
    Estimate model.

    Values
    ------
    ARMA
        Assume an ARMA(p,q) model when performing estimate.
    OLS_SING_VAR
        Assume a single variable OLS model when performing regression/

    """

    ARMA = "ARMA"
    OLS_SING_VAR = "OLS_SING_VAR"

class ParamEst:
    """
    Model used to store a parameter estimate result.

    Properties
    ----------
    est : float
        Estimate value.
    err : float
        Estimate error.
    est_label : str
        Estimate label used when display results.
    err_label : str
        Estimate error label used when display results.
    """

    def __init__(self, est: float, err: float, est_label: str=None, err_label: str=None):
            self.est = est
            self.err = err
            self.est_label = est_label
            self.err_label = err_label
            self.__set_dict()

    def set_labels(self, est_label, err_label):
        self.est_label = est_label
        self.err_label = err_label
        self.__set_dict()

    def __repr__(self):
        return f"ParamEst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est=({self.est}), " \
               f"err=({self.err}, " \
               f"est_label=({self.est_label}), "\
               f"err_label=({self.err_label})"

    def __set_dict(self):
        self.dict = {"Estimate": self.est,
                     "Error": self.err,
                     "Estimate Label": self.est_label,
                     "Error Label": self.err_label}

    @staticmethod
    def from_dict(meta_data):
        if "Estimate Label" in meta_data:
            est_label = meta_data["Estimate Label"]
        else:
            est_label = None
        if "Error Label" in meta_data:
            err_label = meta_data["Error Label"]
        else:
            err_label = None

        return ParamEst(meta_data["Estimate"],
                        meta_data["Error"],
                        est_label,
                        err_label)

class OLSEstModel(Enum):
    """
    Model used in OLS estimate.

    Values
    ------
    LINEAR
        Assume a linear model of the form y = a + b * x.
    POWER
        Assume a power law model of the form y = a * x**b
    EXPONENTIAL
        Assume an exponential model of the form y = 10**(a + b * x)
    LOG
        Assume a logarithmic model of the form y = a + b * log(x)
    """

    LINEAR = "LINEAR"
    POWER = "POWER"
    EXPONENTIAL = "EXPONENTIAL"
    LOG = "LOG"

    def formula(self) -> str:
        if self.value == OLSEstModel.LINEAR:
            return r"$\alpha + \beta x$"
        elif self.value == OLSEstModel.POWER:
            return r"$10^\alpha x^\beta$"
        elif self.value == OLSEstModel.EXPONENTIAL:
            return r"100^{$\alpha + \beta x}$"
        elif self.value == OLSEstModel.LOG:
            return r"$\alpha + \beta \log_{10}x$"
        else:
            raise Exception(f"OLS Regression model type is invalid: {self.reg_type}")

class OLSSingleVarResult:
    """
    Single variable OLS estimate result.

    Properties
    ----------
    const: ParamEst
        Constant estimate.
    param: ParamEst
        Parameter estimate.
    ols_est_model: OLSEstMode
        OLS model used in estimate. (default OLSEstModel.LINEAR)
    """

    def __init__(self, const: ParamEst, param: ParamEst, r2: float, ols_est_model: OLSEstModel=OLSEstModel.LINEAR):
        self.__ols_est_model = ols_est_model
        self.__est_model = EstModel.OLS_SING_VAR
        self.__const = const
        self.__param = param
        self.__r2 = r2
        self.__const.set_labels(est_label=r"$\hat{\alpha}$", err_label=r"$\sigma_{\hat{\alpha}}$")
        self.__param.set_labels(est_label=r"$\hat{\beta}$", err_label=r"$\sigma_{\hat{beta}}$")

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"ols_est_model=({self.__ols_est_model}), " \
               f"est_model=({self.__est_model}), " \
               f"const=({self.__const}), " \
               f"params=({self.__param}, "\
               f"r2=({self.__r2})"

    def formula(self) -> str:
        return self.__ols_est_model.formula()

    def trans_param(self):
        return self.__ols_est_model.param

    def trans_const(self):
        return self.__ols_est_model.const

    def fit(self):
        if self.__ols_est_model.value == OLSEstModel.LINEAR.value:
            return self.__linear_fit()
        elif self.__ols_est_model.value == OLSEstModel.POWER.value:
            return self.__power_fit()
        elif self.__ols_est_model.value == OLSEstModel.EXPONENTIAL.value:
            raise self.__exp_fit()
        elif self.__ols_est_model.value == OLSEstModel.LOG.value:
            raise self.__log_fit()
        else:
            raise Exception(f"OLS Regression model type is invalid: {self.reg_type}")

    def __power_fit(self):
        return lambda x : 10**self.const.est * x**self.param.est

    def __linear_fit(self):
        return lambda x : self.const.est + x*self.param.est

    def __exp_fit(self):
        return lambda x : 10**(self.const.est * x*self.param.est)

    def __log_fit(self):
        return lambda x :self.const.est + self.param.est * numpy.log10(x)
    

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
    def __init__(self, const: ParamEst, params: list[ParamEst], sigma2: ParamEst, arma_est_type: ARMAEstType=ARMAEstType.AR):
        self.__est_model = EstModel.ARMA
        self.__arma_est_type = arma_est_type
        self.__const = const
        self.__order = len(params)
        self.__params = params
        self.__sigma2 = sigma2
        self.__set_const_labels()
        self.__set_params_labels()
        self.__set_sigma2_labels()

    def __repr__(self):
        return f"ARMAEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"est_model=({self.__est_model}), " \
               f"arma_est_type=({self.__arma_est_type}), " \
               f"const=({self.__const}), " \
               f"order=({self.__order}), " \
               f"params=({self.__params}), " \
               f"sigma2=({self.__sigma2})"

    def __set_const_labels(self):
        self.__const.set_labels(est_label=r"$\hat{\mu^*}$",
                                err_label=r"$\sigma_{\hat{\mu^*}}$")

    def __set_params_labels(self):
        for i in range(len(self.__params)):
            self.__arma_est_type.set_param_labels(self.__params[i], i)

    def __set_sigma2_labels(self):
        self.__sigma2.set_labels(est_label=r"$\hat{\sigma^2}$",
                                 err_label=r"$\sigma_{\hat{\sigma^2}}$")
