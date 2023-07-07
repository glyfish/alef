import numpy
from enum import Enum
from typing import Tuple
import statsmodels.api as sm

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
        self.__set_dict()

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"model=({self.model}), " \
               f"param=({self.param}), " \
               f"const=({self.const})"

    def __set_dict(self):
        self.dict = {"model": self.model,
                     "param": self.param.dict,
                     "const": self.const.dict}

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

    def __init__(self, const: ParamEst, param: ParamEst, r2: float):
        self.__est_model = EstModel.OLS_SING_VAR
        self.const = const
        self.param = param
        self.r2 = r2
        self.transform = None
        self.__set_dict()

    def __repr__(self):
        return f"OLSEst({self._props()})"

    def __str__(self):
        return self._props()
    def _props(self):
        return f"est_model=({self.__est_model}), " \
               f"const=({self.__const}), " \
               f"params=({self.__param}, "\
               f"r2=({self.__r2}), " \
               f"transform=({self.__transform})"
    
    def __set_dict(self):
        self.dict = {"est_model": self.__est_model.value,
                     "param": self.param.dict,
                     "const": self.const.dict,
                     "r2": self.r2,
                     "transform": self.transform.dict if self.transform is not None else "{}"}

    def set_transform(self, transform: OLSSinlgeVarTransform):
        self.transform = transform
        self.__set_dict()

class OLSSingleVariable(Enum):
    """
    Specify regression model to use for linear ols repression models.

    Values
    ------
    LINEAR : str
        Assume a linear relation between regression variables.
            y = a*x + b
        where a and b be are regression constants.
    LOG : str
        Assume power law relation between the regression variables.
            y = b*x**a
        where a and b be are regression constants.
    XLOG : str
        Assume an exponential relationship between the regression variables.
            y = b*exp(a*x)
        where a and b be are regression constants.
    YLOG : str
        Assume a logarithmic relation between the regression variables.
            y = b*ln(a*x)
        where a and b be are regression constants.
    """

    LINEAR = "LINEAR"
    LOG = "LOG"
    XLOG = "XLOG"
    YLOG = "YLOG"

    def estimate(self, y: numpy.ndarray[float], x: numpy.ndarray[float]) -> Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]:
        """
        Perform single variable OLS regression on the provided data.

        Parameters
        ----------
        y: numpy.ndarray[float]
            Dependent variable
        x: numpy.ndarray[float]
            Variable
 
        Return
        ------
        Tuple[sm.regression.linear_model.RegressionResults, OLSSingleVarResult]
            OLS report and result model.
        """

        report = self.___OLS_fit(y, x)
        return report, self.__result_from_report(report)

    def __result_from_report(self, report: sm.regression.linear_model.RegressionResults) -> OLSSingleVarResult:
        """
        Create an OLS result model from the returned report.

        Parameters
        ----------
        report: sm.regression.linear_model.RegressionResults
            OLS results report.
 
        Return
        ------
        OLSSingleVarResult
            OLS result model.
        """
        
        const = ParamEst.from_dict({"Estimate": report.params[0],
                                    "Error": report.bse[0]})
        param = ParamEst.from_dict({"Estimate": report.params[1],
                                    "Error": report.bse[1]})
        r2 = report.rsquared
        return OLSSingleVarResult(const, param, r2)
    
    def ___OLS_fit(self, y: numpy.ndarray[float], x: numpy.ndarray[float]) -> sm.OLS:
        """ 
        Create statsmodels OLS object using specified samples assuming a single dependent variable.

        Parameters
        ----------
        y: numpy.ndarray[float]
            Dependent variable
        x: numpy.ndarray[float]
            Variable

        Returns
        -------
        sm.OLS
            OLS object
        """

        if self.value == OLSSingleVariable.LOG.value:
            x = numpy.log10(x)
            y = numpy.log10(y)

        x = sm.add_constant(x)
        return sm.OLS(y, x, missing='drop').fit()

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
        self.__set_dict()

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

    def __set_dict(self):
        self.dict = {"est_model": self.__est_model.value,
                     "arma_est_type": self.__arma_est_type.value,
                     "order": self.__order,
                     "const": self.__const.dict,
                     "sigma2": self.__sigma2.dict,
                     "params": [param.dict for param in self.__params]}
