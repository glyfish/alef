from enum import Enum
import uuid
import numpy

from lib.models import fbm
from lib import stats

from lib.data.func import (DataFunc, FuncBase, _get_s_vals)
from lib.data.source import (DataSource, SourceBase)
from lib.data.schema import (DataType, DataSchema)
from lib.data.meta_data import (EstBase, TestBase, TestImplBase,
                                TestParam, TestData, TestReport,
                                OLSSinlgeVarTrans, ParamEst)
from lib.models import (TestHypothesis, Dist)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Fractional Brownian Motion Fuctions
class FBM:
    # Funcs
    class Func(FuncBase):
        MEAN = "FBM_MEAN"                      # Fractional Brownian Motion mean
        VAR = "FBM_VAR"                        # Fractional Brownian Motion variance
        SD = "FBM_SD"                          # Fractional Brownian Motion standard deviation
        ACF = "FBM_ACF"                        # Fractional Brownian Motion autocorrelation function
        COV = "FBM_COV"                        # Fractional Brownian Motion covariance function
        VR = "FBM_VR"                          # Fractional Brownian Motion Variance Ratio
        VR_STAT = "FBM_VR_STAT"                # Variance ratio test statistic
        VR_HOMO_STAT = "FBM_VR_HOMO_STAT"      # Homoscedastic variance ratio test statistic
        VR_HETERO_STAT = "FBM_VR_HETERO_STAT"  # Heteroscedastic variance ratio test statistic

        def _create_func(self, **kwargs):
            return _create_func(self, **kwargs)

    # Source
    class Source(SourceBase):
        NOISE_CHOL = "FBM_NOISE_CHOL"           # FBM Noise simulation implemented using the Cholesky method
        NOISE_FFT = "FBM_NOISE_FFT"             # FBM Noise simulation implemented using the FFT method
        MOTION_CHOL = "FBM_MOTION_CHOL"         # FBM simulation implemented using the Cholesky method
        MOTION_FFT = "FBM_MOTION_FFT"           # FBM Noise simulation implemented using the FFT method

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

    # Test
    class Test(TestBase):
        BM = "BM"                               # Test for brownian motion
        AUTO_CORR = "AUTO_CORR"                 # Test for autocorrelated fractional brownian motion
        NEG_AUTO_CORR = "NEG_AUTO_CORR"         # Test for negative autocorrelated fractional brownian motion

        def status(self, status):
            if self.value == FBM.Test.BM.value:
                npass = 0
                for stat in status:
                    if stat:
                        npass += 1
                return npass >= len(status)/2
            elif self.value == FBM.Test.AUTO_CORR.value:
                for stat in status:
                    if not stat:
                        return True
                return False
            elif self.value == FBM.Test.NEG_AUTO_CORR.value:
                for stat in status:
                    if not stat:
                        return True
                return False
            else:
                raise Exception(f"Test type is invalid: {self}")

        def _desc(self):
            if self.value == FBM.Test.BM.value:
                return "Brownian Motion Test"
            elif self.value == FBM.Test.AUTO_CORR.value:
                return "Autocorrelation Test"
            elif self.value == FBM.Test.NEG_AUTO_CORR.value:
                return "Negative Autocorrelation Test"
            else:
                raise Exception(f"Test type is invalid: {self}")

        def _impl(self):
            if self.value == FBM.Test.BM.value:
                return FBM._TestImpl.VR_TWO_TAILED
            elif self.value == FBM.Test.AUTO_CORR.value:
                return FBM._TestImpl.VR_UPPER_TAIL
            elif self.value == FBM.Test.NEG_AUTO_CORR.value:
                return FBM._TestImpl.VR_LOWER_TAIL
            else:
                raise Exception(f"Test type is invalid: {self}")

    # _TestImpl
    class _TestImpl(TestImplBase):
        VR_TWO_TAILED = "VR_TWO_TAILED"      # Variance ratio test for brownian motion
        VR_LOWER_TAIL = "VR_LOWER_TAIL"      # Variance ratio test for anti-autocorrelated fractional brownian motion
        VR_UPPER_TAIL = "VR_UPPER_TAIL"      # Variance ratio test for autocorrelated fractional brownian motion

        def _perform_test_for_impl(self, x, y, test_type, **kwargs):
            return _perform_test_for_impl(x, y, test_type, self, **kwargs)

    # Est
    class Est(EstBase):
        PERGRAM = "PERGRAM"           # Periodogram esimate of FBM Hurst parameter using OLS
        AGG_VAR = "AGG_VAR"           # Variance Aggregation esimate of FBM Hurst parameter using OLS

        def _perform_est_for_type(self, x, y, **kwargs):
            if self.value == FBM.Est.PERGRAM.value:
                return self._ols_estimate(x, y, stats.RegType.LOG, **kwargs)
            elif self.value == FBM.Est.AGG_VAR.value:
                return self._ols_estimate(x, y, stats.RegType.LOG, **kwargs)
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

        def _create_ols_single_var_trans(self, param, const):
            if self.value == FBM.Est.AGG_VAR.value:
                return _create_agg_var_trans(param, const)
            elif self.value == FBM.Est.PERGRAM.value:
                return _create_pergram_trans(param, const)
            else:
                raise Exception(f"Esitmate type is invalid: {self}")

###################################################################################################
## Create DataFunc object for func type
###################################################################################################
def _create_func(func_type, **kwargs):
    if func_type.value == FBM.Func.MEAN.value:
        return _create_fbm_mean(func_type, **kwargs)
    elif func_type.value == FBM.Func.VAR.value:
        return _create_fbm_var(func_type, **kwargs)
    elif func_type.value == FBM.Func.SD.value:
        return _create_fbm_sd(func_type, **kwargs)
    elif func_type.value == FBM.Func.ACF.value:
        return _create_fbm_acf(func_type, **kwargs)
    elif func_type.value == FBM.Func.COV.value:
        return _create_fbm_cov(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR.value:
        return _create_fbm_vr(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_STAT.value:
        return _create_vr_stat(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_HOMO_STAT.value:
        return _create_vr_homo_stat(func_type, **kwargs)
    elif func_type.value == FBM.Func.VR_HETERO_STAT.value:
        return _create_vr_hetero_stat(func_type, **kwargs)
    else:
        raise Exception(f"func_type is invalid: {func_type}")

###################################################################################################
# Func.MEAN
def _create_fbm_mean(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : numpy.full(len(x), 0.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={},
                    ylabel=r"$\mu_t$",
                    xlabel=r"$t$",
                    formula=r"$0$",
                    desc="FBM Mean",
                    fy=fy,
                    fx=fx)

# Func.SD
def _create_fbm_sd(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1., **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : Δt*numpy.sqrt(fbm.var(H, x))
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H, "Δt": Δt},
                    ylabel=r"$\sigma^H_t$",
                    xlabel=r"$t$",
                    desc="FBM SD",
                    formula=r"$(Δt) t^H$",
                    fy=fy,
                    fx=fx)

# Func.VAR
def _create_fbm_var(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1., **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : Δt**2*fbm.var(H, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H, "Δt": Δt},
                    ylabel=r"$\sigma^2$",
                    xlabel=r"$t$",
                    desc="FBM VAR",
                    formula=r"$Δt^2 t^{2H}$",
                    fy=fy,
                    fx=fx)

# Func.ACF
def _create_fbm_acf(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : fbm.acf(H, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"H": H},
                    ylabel=r"$\rho^H_n$",
                    xlabel=r"$n$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[(n-1)^{2H} + (n+1)^{2H} - 2n^{2H}]$",
                    fy=fy,
                    fx=fx)

# Func.VR
def _create_fbm_vr(func_type, **kwargs):
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    H = get_param_throw_if_missing("H", **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y :  x**(2*H - 1.0)
    return DataFunc(func_type=func_type,
                    data_type=DataType.TIME_SERIES,
                    source_type=DataType.TIME_SERIES,
                    params={"H": H},
                    ylabel=r"$VR(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    formula=r"$s^{2H-1}$",
                    fy=fy,
                    fx=fx)

# DataType.COV
def _create_fbm_cov(func_type, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    s = get_param_throw_if_missing("s", **kwargs)
    nplot = get_param_default_if_missing("nplot", 10, **kwargs)
    fx = lambda x : x[::int(len(x)/(nplot - 1))]
    fy = lambda x, y : fbm.cov(H, s, x)
    return DataFunc(func_type=func_type,
                    data_type=DataType.ACF,
                    source_type=DataType.ACF,
                    params={"H": H, "s": s},
                    ylabel=r"$R^H(t,s)$",
                    xlabel=r"$t$",
                    desc="FBM ACF",
                    formula=r"$\frac{1}{2}[t^{2H}+s^{2H}-(t-s)^{2H}]$",
                    fy=fy,
                    fx=fx)

# Func.VR_STAT
def _create_vr_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$VR(s)$",
                    xlabel=r"$s$",
                    desc="Variance Ratio",
                    formula=r"$\frac{\sigma^2(s)}{\sigma_B^2(s)}$",
                    fy=fy,
                    fx=fx)

# Func.VR_HOMO_STAT
def _create_vr_homo_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_stat_homo_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Homoscedastic Variance Ratio Statistic",
                    formula=r"$\frac{VR(s) - 1}{\sqrt{\theta^\ast (s)}}$",
                    fy=fy,
                    fx=fx)

# Func.VR_HETERO_STAT
def _create_vr_hetero_stat(func_type, **kwargs):
    source_type = get_param_default_if_missing("source_type", DataType.TIME_SERIES, **kwargs)
    s_vals = _get_s_vals(**kwargs)
    fx = lambda x : [int(s) for s in s_vals]
    fy = lambda x, y : fbm.vr_stat_hetero_scan(y, x)
    return DataFunc(func_type=func_type,
                    data_type=source_type,
                    source_type=source_type,
                    params={},
                    ylabel=r"$Z^*(s)$",
                    xlabel=r"$s$",
                    desc="Heteroscedastic Variance Ratio Statistic",
                    formula=r"$\frac{VR(s) - 1}{\sqrt{\theta^\ast (s)}}$",
                    fy=fy,
                    fx=fx)

###################################################################################################
# Create DataSource objects for specified DataType
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == FBM.Source.NOISE_CHOL.value:
        return _create_fbm_noise_chol_source(source_type, x, **kwargs)
    elif source_type.value == FBM.Source.NOISE_FFT.value:
        return _create_fbm_noise_fft_source(source_type, x, **kwargs)
    elif source_type.value == FBM.Source.MOTION_CHOL.value:
        return _create_fbm_motion_chol_source(source_type, x, **kwargs)
    elif source_type.value == FBM.Source.MOTION_FFT.value:
        return _create_fbm_motion_fft_source(source_type, x, **kwargs)
    else:
        raise Exception(f"source_type is invalid: {source_type}")

###################################################################################################
# Source.NOISE_CHOL
def _create_fbm_noise_chol_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_schema_data(dB)
    else:
        ΔB = None
    f = lambda x : fbm.cholesky_noise(H, len(x[:-1]), Δx, ΔB, L)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Cholesky-FBM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Cholesky FBM Noise",
                      f=f,
                      x=x)

# Source.NOISE_FFT
def _create_fbm_noise_fft_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_schema_data(dB)
    else:
        ΔB = None
    f = lambda x : fbm.fft_noise(H, len(x), Δx, ΔB)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"FFT-FBM-Noise-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"FFT FBM Noise",
                      f=f,
                      x=x)

# Source.CHOL
def _create_fbm_motion_chol_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    L = get_param_default_if_missing("L", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB)
    else:
        ΔB = None
    f = lambda x : fbm.generate_cholesky(H, len(x[:-1]), Δx, ΔB, L)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Cholesky-FBM-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Cholesky FBM",
                      f=f,
                      x=x)

# Source.FFT
def _create_fbm_motion_fft_source(source_type, x, **kwargs):
    H = get_param_throw_if_missing("H", **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)
    dB = get_param_default_if_missing("dB", None, **kwargs)
    if dB is not None:
        _, ΔB = DataSchema.get_data_type(dB, DataType.TIME_SERIES)
    else:
        ΔB = None
    f = lambda x : fbm.generate_fft(H, len(x), Δx, ΔB)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"FFT-FBM-Simulation-{str(uuid.uuid4())}",
                      params={"H": H, "Δt": Δx},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"FFT FBM",
                      f=f,
                      x=x)

##################################################################################################################
# Perform test forspecified implementaion
def _perform_test_for_impl(x, y, test_type, impl_type, **kwargs):
    if impl_type.value == FBM._TestImpl.VR_TWO_TAILED.value:
        return _vr_test(y, TestHypothesis.TWO_TAIL, test_type, impl_type, **kwargs)
    elif impl_type.value == FBM._TestImpl.VR_LOWER_TAIL.value:
        return _vr_test(y, TestHypothesis.LOWER_TAIL, test_type, impl_type, **kwargs)
    elif impl_type.value == FBM._TestImpl.VR_UPPER_TAIL.value:
        return _vr_test(y, TestHypothesis.UPPER_TAIL, test_type, impl_type, **kwargs)
    else:
        raise Exception(f"Test type is invalid: {self}")

# _TestImpl.VR_TWO_TAILED, _TestImpl.VR_LOWER_TAIL, _TestImpl.VR_UPPER_TAIL.value
def _vr_test(y, hypo_type, test_type, impl_type, **kwargs):
    sig_level = get_param_default_if_missing("sig_level", 0.1, **kwargs)
    s = get_param_default_if_missing("s", [4, 6, 10, 16, 24], **kwargs)
    verify_type(s, list)
    result = fbm.vr_test(y, s, sig_level, hypo_type)
    return result, _vr_report_from_result(result, test_type, impl_type)

##################################################################################################################
# Construct test report from result object
def _vr_report_from_result(result, test_type, impl_type):
    sig = TestParam(label=f"{int(100.0*result.sig_level)}%", value=result.sig_level)
    s_vals = [TestParam(label=r"$s$", value=s) for s in result.s_vals]
    stats = [TestParam(label=r"$Z(s)$", value=stat) for stat in result.stats]
    pvals = [TestParam(label=r"$p-value$", value=pval) for pval in result.p_vals]
    lower = result.critical_values[0]
    if lower is not None:
        lower = TestParam(label=r"$Z_L(s)$", value=lower)
    upper = result.critical_values[1]
    if upper is not None:
        upper = TestParam(label=r"$Z_U(s)$", value=upper)
    test_data = []
    for i in range(len(s_vals)):
        data = TestData(status=result.status_vals[i],
                        stat=stats[i],
                        pval=pvals[i],
                        params=[s_vals[i]],
                        sig=sig,
                        lower=lower,
                        upper=upper)
        test_data.append(data)
    return TestReport(status=test_type.status(result.status_vals),
                      hyp_type=result.hyp_type,
                      test_type=test_type,
                      impl_type=impl_type,
                      test_data=test_data,
                      dist=Dist.NORMAL,
                      loc=0.0,
                      scale=1.0)

##################################################################################################################
# OLS Variable Transforms
# Est.AGG_VAR
def _create_agg_var_trans(param, const):
    formula = r"$\sigma^2 m^{2\left(H-1\right)}$"
    param = ParamEst(est=1.0 + param.est/2.0,
                     err=param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$")
    c = 10.0**const.est
    const = ParamEst(est=c,
                     err= c*const.err,
                     est_label=r"$\hat{\sigma}^2$",
                     err_label=r"$\sigma^2_{\hat{\sigma}^2}$")
    return OLSSinlgeVarTrans(formula, const, param)

# Est.PERGRAM
def _create_pergram_trans(param, const):
    formula = r"$C\omega^{1 - 2H}$"
    param = ParamEst(est=(1.0 - param.est)/2.0,
                     err=param.err/2.0,
                     est_label=r"$\hat{Η}$",
                     err_label=r"$\sigma_{\hat{Η}}$")
    c = 10.0**const.est
    const = ParamEst(est=c,
                     err=c*const.err,
                     est_label=r"$\hat{C}$",
                     err_label=r"$\sigma_{\hat{C}}$")
    return OLSSinlgeVarTrans(formula, const, param)
