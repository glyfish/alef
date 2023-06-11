
from typing import Tuple
import numpy

from lib.models import adf
from lib.data.hyp_test import (StatisticalTestParam, StatisticalTestData, StatisticalTestReport, 
                               HypothesisTestType, HypothesisType)
from lib.data.reports import ADFTestReport

from lib.utils import get_param_default_if_missing, create_space

def create_df_source(**kwargs) -> Tuple[numpy.ndarray[float], numpy.ndarray[float]]:
    """
    Generate the Dickey-Fuller distribution by simulating an ensemble of solutions
    to the stochastic integral that defines it.

    Parameters
    ----------
    nstep: int
        Number od steps in integral simulations.
    nsim: int
        Number of simulations in ensemble.
    """
    
    nstep = get_param_default_if_missing("nstep", 100, **kwargs)
    nsim = get_param_default_if_missing("nsim", 1000, **kwargs)

    return create_space(xmax=nsim, npts=nsim + 1), adf.dist_ensemble(nstep, nsim)

def compute_adf_test(samples: numpy.ndarray[float]) -> Tuple[ADFTestReport, StatisticalTestReport]:
    """
    Perform ADF test on provided samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Samples to test.

    Returns
    -------
    Tuple[ADFTestReport, StatisticalTestReport]
        ADF result report and test result model.
    """

    result = adf.adf_test(samples)
    return result, __adf_report_from_result(result, HypothesisTestType.STATIONARITY)

def compute_adf_offset_test(samples: numpy.ndarray[float]) -> Tuple[ADFTestReport, StatisticalTestReport]:
    """
    Perform ADF test assuming a constant offset on provided samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Samples to test.

    Returns
    -------
    Tuple[ADFTestReport, StatisticalTestReport]
        ADF result report and test result model.
    """
    
    result = adf.adf_test_offset(samples)
    return result, __adf_report_from_result(result, HypothesisTestType.STATIONARITY_OFFSET)

def compute_adf_drift_test(samples: numpy.ndarray[float]) -> Tuple[ADFTestReport, StatisticalTestReport]:
    """
    Perform ADF test assuming offset and drift terms on provided samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Samples to test.

    Returns
    -------
    Tuple[ADFTestReport, StatisticalTestReport]
        ADF result report and test result model.
    """
    
    result = adf.adf_test_drift(samples)
    return result, __adf_report_from_result(result, HypothesisTestType.STATIONARITY_DRIFT)

def __adf_report_from_result(result: ADFTestReport, test_type: HypothesisTestType) -> StatisticalTestReport:
    """
    Perform ADF test on provided samples.

    Parameters
    ----------
    samples: numpy.ndarray[float]
        Samples to test.

    Returns
    -------
    Tuple[ADFTestReport, StatisticalTestReport]
        ADF result report and test result model.
    """
    
    sigs = [StatisticalTestParam(label=result.sig_str[i], value=result.sig[i]) for i in range(3)]
    stat = StatisticalTestParam(label=r"$t$", value=result.stat)
    pval = StatisticalTestParam(label=r"$p-value$", value = result.pval)
    lower_vals = [StatisticalTestParam(label=r"$t_L$", value=val) for val in result.critical_vals]
    test_data = []
    for i in range(3):
        data = StatisticalTestData(status=result.status_vals[i],
                                   stat=stat,
                                   pval=pval,
                                   params=[],
                                   sig=sigs[i],
                                   lower=lower_vals[i],
                                   upper=None)
        test_data.append(data)
    return StatisticalTestReport(status=test_type.status(result.status_vals),
                      hyp_type=HypothesisType.LOWER_TAIL,
                      test_type=test_type,
                      test_data=test_data,
                      dist=None)
