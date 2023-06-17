from enum import Enum

class HypothesisTestType(Enum):
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

    def status(self, status):
        if self.value == HypothesisTestType.STATIONARITY.value:
            return not status[2]
        elif self.value == HypothesisTestType.STATIONARITY_OFFSET.value:
            return not status[2]
        elif self.value == HypothesisTestType.STATIONARITY_DRIFT.value:
            return not status[2]
        elif self.value == HypothesisTestType.BM.value:
            npass = 0
            for stat in status:
                if stat:
                    npass += 1
            return npass >= len(status)/2
        elif self.value == HypothesisTestType.FBM_AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return True
            return False
        elif self.value == HypothesisTestType.FBM_NEG_AUTO_CORR.value:
            for stat in status:
                if not stat:
                    return True
            return False
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


class HypothesisType(Enum):
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
    """

    def __init__(self, label: str, value: float):
        self.label = label
        self.value = value
        self.dict = {"Value": value,
                     "Label": label}

    def __repr__(self):
        return f"TestParam({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"label=({self.label}), " \
               f"value=({self.value})"

    @staticmethod
    def from_dict(meta_data):
        return StatisticalTestParam(label=meta_data["Label"],
                                    value=meta_data["Value"])

class StatisticalTestData:
    """
    Statistical test data.

    Properties
    ----------
    status: bool
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
    """

    def __init__(self, 
                 status: bool, 
                 stat: StatisticalTestParam, 
                 pval: StatisticalTestParam, 
                 params: list[StatisticalTestParam], 
                 sig: StatisticalTestParam, 
                 lower: StatisticalTestParam, 
                 upper: StatisticalTestParam):
        self.status = status
        self.stat = stat
        self.pval = pval
        self.params = params
        self.sig = sig
        self.lower = lower
        self.upper = upper
        self.dict = {"Status": status,
                     "Statistic": stat.dict,
                     "PValue": pval.dict,
                     "Parameters": [param.dict for param in params],
                     "Significance": sig.dict,
                     "Lower Critical Value": lower.dict if lower is not None else None,
                     "Upper Critical Value": upper.dict if upper is not None else None}

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
               f"upper=({self.upper})"

    @staticmethod
    def from_dict(dict):
        lower = dict["Lower Critical Value"]
        upper = dict["Upper Critical Value"]
        return StatisticalTestData(status=dict["Status"],
                                   stat=StatisticalTestParam.from_dict(dict["Statistic"]),
                                   pval=StatisticalTestParam.from_dict(dict["PValue"]),
                                   params=[StatisticalTestParam.from_dict(param) for param in dict["Parameters"]],
                                   sig=StatisticalTestParam.from_dict(dict["Significance"]),
                                   lower=StatisticalTestParam.from_dict(lower) if lower is not None else None,
                                   upper=StatisticalTestParam.from_dict(upper) if upper is not None else None)

class StatisticalTestReport:
    """
    Data used to construct the statistical test report.

    Parameters
    ----------
    status: bool
        Test status. This status may be the negation from the status of the performed test if the 
        desired result is the alternative hypothesis not the null hypothesis.
    hyp_type: HypothesistType
        Hypothesis test type performed (two tailed, upper tail or lower tail).
    test_type: HypothesisTestType
        Type of hypothesis test performed.
    test_data: StatisticalTestData
        Results from test.
    """

    def __init__(self, 
                 status: bool, 
                 hyp_type: HypothesisType, 
                 test_type: HypothesisTestType, 
                 test_data: StatisticalTestData):
        self.status = status
        self.hyp_type = hyp_type
        self.test_type = test_type
        self.test_data = test_data
        self.desc = test_type.desc()
        self.dict = {"Status": status,
                     "HypothesisType": hyp_type,
                     "TestType": test_type,
                     "Description": self.desc,
                     "TestData": [data.dict for data in test_data]}

    def __repr__(self):
        return f"TestReport({self.__props()})"

    def __str__(self):
        return self._props()

    def __props(self):
        return f"status=({self.status}), " \
               f"hyp_type=({self.hyp_type}), " \
               f"test_type=({self.test_type}), " \
               f"desc=({self.desc}, " \
               f"test_data=({self.test_data})"

    def key(self):
        return self.test_type.value

    @classmethod
    def from_dict(cls, dict):
        return StatisticalTestReport(status=dict["Status"],
                          hyp_type=dict["HypothesisType"],
                          test_type=dict["TestType"],
                          impl_type=dict["ImplType"],
                          desc=dict["Description"],
                          test_data=[StatisticalTestData.from_dict(data) for data in dict["TestData"]])
