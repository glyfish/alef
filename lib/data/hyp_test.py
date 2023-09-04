from enum import Enum
import json
import numpy

class HypothesisTestStatus(str, Enum):
    """
    Hypothesis test status.

    Values
    ------

    PASSED
        The test passed.
    FAILED
        The test failed
    """

    PASSED = "PASSED"
    FAILED = "FAILED"

    def to_bool(self) -> bool:
        return self.value == "PASSED"

    @staticmethod
    def from_bool(status: bool):
        return HypothesisTestStatus.PASSED if status else HypothesisTestStatus.FAILED

class HypothesisTestType(str, Enum):
    """
    Supported hypothesis tests.

    Values
    ------
    STATIONARITY
        ADF test for stationarity of an AR(p) process. To pass a failure of the ADF 
        test at a significance level of 10% is required.
    STATIONARITY_OFFSET
        ADF test for a stationarity of an AR(p) process with a constant offset. To pass a failure of the ADF 
        test at a significance level of 10% is required.
    STATIONARITY_DRIFT
        ADF test for a stationarity of an AR(p) process with drift.To pass a failure of the ADF 
        test at a significance level of 10% is required.
    BM
        Test for brownian motion using the Lo and Mackinlay variance ratio test. The process is 
        brownian motion if the test passes at the specified significance level.
    FBM_AUTO_CORR
        Test for positive correlation in a fractional brownian motion process using the Lo and Mackinlay 
        variance ratio test. The FBM process has positive autocorrelation if the upper tail test fails at the specified
        significance level.
    FBM_NEG_AUTO_CORR
        Test for negative correlation in a fractional brownian motion process using the Lo and Mackinlay variance ratio 
        test. The FBM process has negative autocorrelation if the lower tail test fails at the specified
        significance level.
    """

    STATIONARITY = "STATIONARITY"
    STATIONARITY_OFFSET = "STATIONARITY_OFFSET"
    STATIONARITY_DRIFT = "STATIONARITY_DRIFT"
    BM = "BM"
    FBM_AUTO_CORR = "AUTO_CORR"
    FBM_NEG_AUTO_CORR = "NEG_AUTO_CORR"

    def status(self, status) -> HypothesisTestStatus:
        if self.value == HypothesisTestType.STATIONARITY.value:
            return HypothesisTestStatus.from_bool(not status[2])
        elif self.value == HypothesisTestType.STATIONARITY_OFFSET.value:
            return HypothesisTestStatus.from_bool(not status[2])
        elif self.value == HypothesisTestType.STATIONARITY_DRIFT.value:
            return HypothesisTestStatus.from_bool(not status[2])
        elif self.value == HypothesisTestType.BM.value:
            npass = 0
            for stat in status:
                if stat:
                    npass += 1
            return HypothesisTestStatus.from_bool(npass >= 1)
        elif self.value == HypothesisTestType.FBM_AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return HypothesisTestStatus.PASSED
            return HypothesisTestStatus.FAILED
        elif self.value == HypothesisTestType.FBM_NEG_AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return HypothesisTestStatus.PASSED
            return HypothesisTestStatus.FAILED
        else:
            raise Exception(f"Test type is invalid: {self}")

    def desc(self):
        if self.value == HypothesisTestType.STATIONARITY.value:
            return "Stationarity Test"
        elif self.value == HypothesisTestType.STATIONARITY_OFFSET.value:
            return "Stationarity Test with Constant Offset."
        elif self.value == HypothesisTestType.STATIONARITY_DRIFT.value:
            return "Stationarity Test with Drift."
        elif self.value == HypothesisTestType.BM.value:
            return "Brownian Motion Test"
        elif self.value == HypothesisTestType.FBM_AUTO_CORR.value:
            return "Autocorrelation Test"
        elif self.value == HypothesisTestType.FBM_NEG_AUTO_CORR.value:
            return "Negative Autocorrelation Test"
        else:
            raise Exception(f"Test type is invalid: {self}")


class HypothesisType(str, Enum):
    """
    Hypotheses type.

    Values
    ------
    TWO_TAIL
        Two tail test type.
    LOWER_TAIL
        Lower tail test type.
    UPPER_TAIL
        Upper tail test type.
    """

    TWO_TAIL = "TWO_TAIL"
    LOWER_TAIL = "LOWER_TAIL"
    UPPER_TAIL = "UPPER_TAIL"

class StatisticalTestParam:
    """
    Statistical test parameter value.

    Properties
    ----------
    label: str
        Test parameter label.
    value: float
        Test parameter value.
    hyp_test_id: str
        Hypothesis test identifier.   
    """

    def __init__(self, label: str, value: float, hyp_test_id: str):
        self.label = label
        self.value = value
        self.hyp_test_id = hyp_test_id

    def __repr__(self):
        return f"TestParam({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"label=({self.label}), " \
               f"value=({self.value}), " \
               f"hyp_test_id=({self.hyp_test_id})"

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return json.dumps(self, indent=indent, default=lambda o: o.__dict__)

    @staticmethod
    def from_dict(data):
        return StatisticalTestParam(label=data["label"],
                                    value=data["value"],
                                    hyp_test_id=data["hyp_test_id"])

class StatisticalTestData:
    """
    Statistical test data.

    Properties
    ----------
    status: HypothesisTestStatus
        Test status.
    stat: StatisticalTestParam
        Value of test statistic.
    pval: StatisticalTestParam
        Probability of occurrence of test statistic value.
    params: list[StatisticalTestParam]
        Any parameters used to configure test.
    sig: StatisticalTestParam
        Statistical test significance.
    lower: StatisticalTestParam
        Value of test statistic used for lower tail test.
    upper: StatisticalTestParam
        Value of test statistic used for upper tail test.
    hyp_test_id: str
        Hypothesis test identifier.   
    """

    def __init__(self, 
                 status: HypothesisTestStatus, 
                 stat: StatisticalTestParam, 
                 pval: StatisticalTestParam, 
                 params: list[StatisticalTestParam], 
                 sig: StatisticalTestParam, 
                 lower: StatisticalTestParam, 
                 upper: StatisticalTestParam,
                 hyp_test_id: str):
        self.status = status
        self.stat = stat
        self.pval = pval
        self.params = params
        self.sig = sig
        self.lower = lower
        self.upper = upper
        self.hyp_test_id = hyp_test_id

    def __repr__(self):
        return f"StatisticalTestData({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"status=({self.status}), " \
               f"stat=({self.stat}), " \
               f"pval=({self.pval}, " \
               f"params=({self.params}), " \
               f"sig=({self.sig}), " \
               f"lower=({self.lower}), " \
               f"upper=({self.upper}), " \
               f"hyp_test_id=({self.hyp_test_id})"
               

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return json.dumps(self, indent=indent, default=lambda o: o.__dict__)

    @staticmethod
    def from_dict(data):
        status = data["status"] if "status" in data else HypothesisTestStatus.FAILED
        stat = StatisticalTestParam.from_dict(data["stat"]) if "stat" in data else None
        pval = StatisticalTestParam.from_dict(data["pval"]) if "pval" in data else None
        params = [StatisticalTestParam.from_dict(param) for param in dict["params"]]
        sig = StatisticalTestParam.from_dict(data["sig"]) if "sig" in data else None
        lower = StatisticalTestParam.from_dict(data["lower"]) if "lower" in data else None
        upper = StatisticalTestParam.from_dict(data["upper"]) if "upper" in data else None
        hyp_test_id = data["hyp_test_id"] if "hyp_test_id" in data else None

        return StatisticalTestData(status=status, stat=stat, pval=pval, params=params, sig=sig,
                                   lower=lower,upper=upper, hyp_test_id=hyp_test_id)

class StatisticalTestReport:
    """
    Data used to construct the statistical test report.

    Parameters
    ----------
    status: HypothesisTestStatus
        Test status. This status may be the negation from the status of the performed test if the 
        desired result is the alternative hypothesis not the null hypothesis.
    hyp_type: HypothesistType
        Hypothesis test type performed (two tailed, upper tail or lower tail).
    hyp_test_type: HypothesisTestType
        Type of hypothesis test performed.
    test_data: list[StatisticalTestData]
        Results from test.
    hyp_test_id: str
        Hypothesis test identifier.   
    """

    def __init__(self, 
                 status: HypothesisTestStatus, 
                 hyp_type: HypothesisType, 
                 hyp_test_type: HypothesisTestType, 
                 test_data: list[StatisticalTestData],
                 hyp_test_id: str=None):
        self.status = status
        self.hyp_type = hyp_type
        self.hyp_test_type = hyp_test_type
        self.test_data = test_data
        self.desc = hyp_test_type.desc()
        self.hyp_test_id = hyp_test_id

    def __repr__(self):
        return f"TestReport({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"status=({self.status}), " \
               f"hyp_type=({self.hyp_type}), " \
               f"hyp_test_type=({self.hyp_test_type}), " \
               f"desc=({self.desc}, " \
               f"test_data=({self.test_data}), " \
               f"hyp_test_id=({self.hyp_test_id})"

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return json.dumps(self, indent=indent, default=lambda o: o.__dict__)

    @staticmethod
    def from_dict(data):
        status = data["status"] if "status" in data else HypothesisTestStatus.FAILED
        hyp_type = data["hyp_type"] if "hyp_type" in data else None
        hyp_test_type = data["hyp_test_type"] if "hyp_test_type" in data else None
        test_data = [StatisticalTestData.from_dict(test_data) for test_data in data["test_data"]]
        hyp_test_id = data["hyp_test_id"] if "hyp_test_id" in data else None

        return StatisticalTestReport(status=status, hyp_type=hyp_type, hyp_test_type=hyp_test_type, test_data=test_data,
                                     hyp_test_id=hyp_test_id)
