from enum import Enum

class TestHypothesis(Enum):
    """
    Hypotheses test type.

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
        return f"StatisticalTestData({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
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
    """
    def __init__(self, status, hyp_type, test_type, impl_type, test_data, dist, **dist_params):
        self.status = status
        self.hyp_type = hyp_type
        self.test_type = test_type
        self.impl_type = impl_type
        self.test_data = test_data
        self.desc = test_type._desc()
        self.dist = dist
        self.dict = {"Status": status,
                     "TestHypothesis": hyp_type,
                     "TestType": test_type,
                     "ImplType": impl_type,
                     "Description": self.desc,
                     "Distribution": dist,
                     "Distribution Params": dist_params,
                     "TestData": [data.dict for data in test_data]}

    def __repr__(self):
        return f"TestReport({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status=({self.status}), " \
               f"hyp_type=({self.hyp_type}), " \
               f"test_type=({self.test_type}), " \
               f"impl_type=({self.impl_type}, " \
               f"desc=({self.desc}, " \
               f"test_data=({self.test_data})"

    def key(self):
        return self.test_type.value

    @classmethod
    def from_dict(cls, dict):
        return StatisticalTestReport(status=dict["Status"],
                          hyp_type=dict["TestHypothesis"],
                          test_type=dict["TestType"],
                          impl_type=dict["ImplType"],
                          desc=dict["Description"],
                          dist=dict["Distribution"],
                          dist_params=dict["Distribution Params"],
                          test_data=[StatisticalTestData.from_dict(data) for data in dict["TestData"]])
