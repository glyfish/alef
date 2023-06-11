
from typing import Tuple
import numpy

from lib.models import adf
from lib.data.hyp_test import StatisticalTestParam, StatisticalTestData, StatisticalTestReport

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

def compute_adf_test(y, test_type, impl_type, **kwargs):
    result = adf.adf_test(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

def compute_adf_offset_test(y, test_type, impl_type, **kwargs):
    result = adf.adf_test_offset(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

def compute_adf_drift_test(y, test_type, impl_type, **kwargs):
    result = adf.adf_test_drift(y)
    return result, __adf_report_from_result(result, test_type, impl_type)

def __adf_report_from_result(result, test_type):
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
                      hyp_type=TestHypothesis.LOWER_TAIL,
                      test_type=test_type,
                      impl_type=impl_type,
                      test_data=test_data,
                      dist=None)
