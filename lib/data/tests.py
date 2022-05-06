import numpy
from enum import Enum
from pandas import (DataFrame)

##################################################################################################################
# Tests
class TestType(Enum):
    ADF = "ADF"                    # Augmented Dickey Fuller test
    ADF_OFF_SET = "ADF_OFF_SET"    # Augmented Dickey Fuller with off set test
    ADF_DRIFT = "ADF_DRIFT"        # Augmented Dickey Fuller with drift test
    VR = "VR"                      # Variance Ratio test

def create_dict_from_tests(tests):
    result = {}
    for key in tests.keys():
        result[key] = tests[key].data
    return result

def create_tests_from_dict(dict):
    result = {}
    for key in dict.keys():
        result[key] = create_esimate_from_dict(dict[key])
    return result

def create_test_from_dict(dict):
    test_type = dict["Type"]
    return dict
    # if est_type.value == TestType.ADF.value:
    # else:
    #     raise Exception(f"Esitmate type is invalid: {est_type}")
