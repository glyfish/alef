"""
Tests for ``lib.data.reports`` — the four formatted text report containers that
navi's model layer returns to the notebooks.

  VarianceRatioTestReport  -- built by lib.models.fbm's Lo/MacKinlay variance
                              ratio tests; owns the tail logic that turns test
                              statistics plus critical values into pass/fail.
  ADFTestReport            -- wraps a statsmodels ``adfuller`` result tuple.
  OUEstReport              -- turns an AR(1) fit of an Ornstein-Uhlenbeck path
                              into (mu, lambda, sigma^2) estimates.
  JohansenTestReport       -- wraps a statsmodels ``JohansenTestResult`` and
                              derives the cointegration rank.

The classes are pure presentation/derivation code, so most tests feed in
hand-picked inputs and check hand-computed outputs. The rest close the loop with
real estimator output: ``adfuller`` on a random walk and on a stationary AR(1),
an OLS AR(1) fit and navi's own ``arima.ar_offset_fit`` of a simulated OU
process, and ``coint_johansen`` on a simulated cointegrated pair - the two
variable shape whose rank table ``compute_rank`` truncates.

Every simulation draws from numpy's global RNG, which tests/conftest.py reseeds
before each test.
"""
import warnings

import numpy
import pandas
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from lib.data.hyp_test import HypothesisTestType, HypothesisType
from lib.data.reports import (ADFTestReport, JohansenTestReport, OUEstReport,
                              VarianceRatioTestReport)
from lib.models import arima
from typing import cast
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult

# statsmodels' Johansen trace/max-eigenvalue critical values for a system of
# three variables with no deterministic term (det_order=0), one row per null
# hypothesis r <= i, columns 90%, 95%, 99%.
CVT_3VAR = numpy.array([[27.0669, 29.7961, 35.4628],
                        [13.4294, 15.4943, 19.9349],
                        [2.7055, 3.8415, 6.6349]])
CVT_2VAR = CVT_3VAR[1:]


# ===========================================================================
# Helpers
# ===========================================================================

def vr_report(stats, critical_values, sig_level=0.1,
              hyp_type=HypothesisType.TWO_TAIL,
              hyp_test_type=HypothesisTestType.BM,
              s_vals=None, p_vals=None):
    """Build a VarianceRatioTestReport from hand-picked statistics."""
    stats = numpy.asarray(stats, dtype=float)
    s_vals = list(range(2, 2 + len(stats))) if s_vals is None else s_vals
    p_vals = [0.5]*len(stats) if p_vals is None else p_vals
    return VarianceRatioTestReport(sig_level, hyp_type, hyp_test_type,
                                   s_vals, stats, p_vals, critical_values)


def adf_result(stat, pval=0.5, lags=3, nobs=997, criticals=(-3.4, -2.86, -2.57)):
    """Mimic the tuple returned by statsmodels.tsa.stattools.adfuller."""
    crit = {"5%": criticals[1], "1%": criticals[0], "10%": criticals[2]}  # unordered on purpose
    return (stat, pval, lags, nobs, crit, 1234.5)


class ARFitStub:
    """
    Minimal stand-in for the AR(1) fit OUEstReport consumes: ``params`` and
    ``bse`` addressed positionally with ``.iloc`` as [intercept, AR
    coefficient, innovation variance].
    """

    NAMES = ["const", "ar.L1", "sigma2"]

    def __init__(self, params, bse):
        self.params = pandas.Series(list(params), index=self.NAMES, dtype=float)
        self.bse = pandas.Series(list(bse), index=self.NAMES, dtype=float)


class JohansenResultStub:
    """Stand-in for statsmodels JohansenTestResult: the six fields the report reads."""

    def __init__(self, lr1, cvt, lr2=None, cvm=None, eig=None, evec=None):
        n = len(lr1)
        self.lr1 = numpy.asarray(lr1, dtype=float)
        self.cvt = numpy.asarray(cvt, dtype=float)
        self.lr2 = numpy.asarray(self.lr1 if lr2 is None else lr2, dtype=float)
        self.cvm = numpy.asarray(self.cvt if cvm is None else cvm, dtype=float)
        self.eig = numpy.asarray(numpy.linspace(0.4, 0.01, n) if eig is None else eig, dtype=float)
        self.evec = numpy.asarray(numpy.eye(n) if evec is None else evec, dtype=float)


def ou_path(mu, lam, sigma, dt, n, x0=None):
    """
    Exact discretisation of an OU process: x[i+1] = mu(1-phi) + phi x[i] + eps,
    phi = exp(-lambda dt), Var[eps] = sigma^2 (1 - phi^2)/(2 lambda). Sampling
    the exact transition (rather than an Euler step) keeps the AR(1) fit an
    unbiased inverse of the continuous parameters.
    """
    phi = numpy.exp(-lam*dt)
    sd = numpy.sqrt(sigma**2*(1.0 - phi**2)/(2.0*lam))
    eps = numpy.random.normal(0.0, sd, n)
    x = numpy.zeros(n)
    x[0] = mu if x0 is None else x0
    for i in range(1, n):
        x[i] = mu*(1.0 - phi) + phi*x[i-1] + eps[i]
    return x


def fit_ar1(x):
    """OLS regression of x[1:] on a constant and x[:-1], packed as an ARFitStub."""
    ols = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit()
    intercept, coeff = ols.params
    intercept_err, coeff_err = ols.bse
    var = ols.mse_resid
    # standard error of a variance estimate from m residuals is var*sqrt(2/m)
    var_err = var*numpy.sqrt(2.0/(len(x) - 3))
    return ARFitStub([intercept, coeff, var], [intercept_err, coeff_err, var_err])


def cointegrated_pair(n=1000, beta=2.0, phi=0.5):
    """
    y1 is a random walk, y2 = beta*y1 + z with z a stationary AR(1). The single
    cointegrating vector is proportional to (beta, -1).
    """
    y1 = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))
    eps = numpy.random.normal(0.0, 1.0, n)
    z = numpy.zeros(n)
    for i in range(1, n):
        z[i] = phi*z[i-1] + eps[i]
    return numpy.column_stack([y1, beta*y1 + z])


def real_pair_report(n=1000, beta=2.0):
    """
    JohansenTestReport over a real ``coint_johansen`` result for a two variable
    system - the shape the notebooks actually use, and the one whose rank table
    the (nvars vs. three significance levels) slice in compute_rank truncates.

    coint_johansen itself emits ComplexWarning while building lr1/lr2, which is
    unrelated to the report; suppress it so tests can pin the report's own
    warnings.
    """
    data = cointegrated_pair(n=n, beta=beta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", numpy.exceptions.ComplexWarning)
        return JohansenTestReport(coint_johansen(data, 0, 1))


# ===========================================================================
# VarianceRatioTestReport
# ===========================================================================

class TestVarianceRatioTestReport:

    # Two tail test: the null is accepted while lower < Z(s) < upper.
    @pytest.mark.parametrize("stats,expected", [
        ([-2.5, 0.0, 1.0, 3.0], [False, True, True, False]),
        ([-1.96, 1.96], [False, False]),          # boundary: comparison is strict
        ([-1.9599, 1.9599], [True, True]),
    ])
    def test_two_tail_status(self, stats, expected):
        report = vr_report(stats, [-1.96, 1.96])
        assert [bool(s) for s in report.status_vals] == expected

    # Upper tail test: only the upper critical value is supplied and the null
    # is accepted while Z(s) sits below it.
    @pytest.mark.parametrize("stats,expected", [
        ([-4.0, 0.0, 1.6448], [True, True, True]),
        ([1.6449, 5.0], [False, False]),
    ])
    def test_upper_tail_status(self, stats, expected):
        report = vr_report(stats, [None, 1.6449],
                           hyp_type=HypothesisType.UPPER_TAIL,
                           hyp_test_type=HypothesisTestType.FBM_AUTO_CORR)
        assert [bool(s) for s in report.status_vals] == expected

    # Lower tail test: mirror image of the upper tail branch.
    @pytest.mark.parametrize("stats,expected", [
        ([-5.0, -1.6449], [False, False]),
        ([-1.6448, 0.0, 4.0], [True, True, True]),
    ])
    def test_lower_tail_status(self, stats, expected):
        report = vr_report(stats, [-1.6449, None],
                           hyp_type=HypothesisType.LOWER_TAIL,
                           hyp_test_type=HypothesisTestType.FBM_NEG_AUTO_CORR)
        assert [bool(s) for s in report.status_vals] == expected

    def test_fields_stored_verbatim(self):
        stats = numpy.array([0.5, -0.5])
        report = VarianceRatioTestReport(0.05, HypothesisType.TWO_TAIL, HypothesisTestType.BM,
                                         [4, 16], stats, [0.61, 0.62], [-1.96, 1.96])
        assert report.sig_level == 0.05
        assert report.hyp_type is HypothesisType.TWO_TAIL
        assert report.hyp_test_type is HypothesisTestType.BM
        assert report.s_vals == [4, 16]
        assert report.stats is stats
        assert report.p_vals == [0.61, 0.62]
        assert report.critical_values == [-1.96, 1.96]
        assert len(report.status_vals) == 2

    def test_repr_and_str(self):
        report = vr_report([-2.5, 1.0], [-1.96, 1.96], sig_level=0.1,
                           s_vals=[4, 10], p_vals=[0.012, 0.317])
        text = repr(report)
        assert text.startswith("VarianceRatioTestReport(status_vals=")
        assert text.endswith(")")
        # enum members are rendered through .value, not their python repr
        assert "hyp_type=TWO_TAIL" in text
        assert "hyp_test_type=BM" in text
        assert "sig_level=0.1" in text
        assert "s_vals=[4, 10]" in text
        assert "p_vals=[0.012, 0.317]" in text
        assert "critical_values=[-1.96, 1.96]" in text
        assert str(report) == text[len("VarianceRatioTestReport("):-1]

    def test_header_includes_both_critical_values(self):
        header, _ = vr_report([1.0], [-1.959964, 1.959964], sig_level=0.05)._table("plain")
        assert "Hypothesis Type       HypothesisType.TWO_TAIL" in header
        assert "Hypothesis Test Type  HypothesisTestType.BM" in header
        assert "Significance          5%" in header
        assert "Lower Critical Value  -1.960" in header      # formatted 1.3f
        assert "Upper Critical Value  1.960" in header

    @pytest.mark.parametrize("critical_values,present,absent", [
        ([None, 1.6449], "Upper Critical Value  1.645", "Lower Critical Value"),
        ([-1.6449, None], "Lower Critical Value  -1.645", "Upper Critical Value"),
    ])
    def test_header_omits_missing_critical_value(self, critical_values, present, absent):
        header, _ = vr_report([1.0], critical_values)._table("plain")
        assert present in header
        assert absent not in header

    def test_results_table(self):
        # float lags are coerced to int, statistics and p values to 3 decimals,
        # and the boolean status becomes Passed/Failed
        report = vr_report([-2.5, 0.4321], [-1.96, 1.96],
                           s_vals=numpy.array([4.0, 10.7]), p_vals=[0.0124, 0.6656])
        _, results = report._table("plain")
        rows = [line.split() for line in results.strip().split("\n")]
        assert rows[0] == ["s", "Z(s)", "pvalue", "Result"]
        assert rows[1] == ["4", "-2.5", "0.012", "Failed"]
        assert rows[2] == ["10", "0.432", "0.666", "Passed"]

    def test_summary_prints_header_then_results(self, capsys):
        report = vr_report([-2.5, 1.0], [-1.96, 1.96], s_vals=[4, 10], p_vals=[0.012, 0.317])
        report.summary()
        out = capsys.readouterr().out
        # header table first, then the per lag results table
        assert out.index("Hypothesis Type") < out.index("Significance") < out.index("Z(s)")
        assert "10%" in out and "-1.960" in out and "1.960" in out
        assert out.count("Failed") == 1 and out.count("Passed") == 1
        assert "╒" in out                      # fancy_grid is the default format

    def test_summary_tablefmt_passthrough(self, capsys):
        vr_report([1.0], [-1.96, 1.96]).summary(tablefmt="plain")
        out = capsys.readouterr().out
        assert "╒" not in out and "|" not in out
        assert "Significance          10%" in out

    def test_hyp_type_does_not_select_the_tail(self):
        # __init__ never reads hyp_type: the tail is chosen by which critical
        # value is None. Passing UPPER_TAIL together with two critical values
        # therefore yields two tail decisions - a genuine upper tail test would
        # accept -2.5, this one rejects it - while the header still prints
        # UPPER_TAIL. The two arguments are free to disagree.
        report = vr_report([-2.5, 0.0, 2.5], [-1.96, 1.96],
                           hyp_type=HypothesisType.UPPER_TAIL,
                           hyp_test_type=HypothesisTestType.FBM_AUTO_CORR)
        assert [bool(s) for s in report.status_vals] == [False, True, False]
        header, _ = report._table("plain")
        assert "Hypothesis Type       HypothesisType.UPPER_TAIL" in header
        assert "Lower Critical Value  -1.960" in header
        assert "Upper Critical Value  1.960" in header

    def test_two_missing_critical_values_are_unguarded(self):
        # lib.models.fbm only ever emits [lo, hi], [None, hi] or [lo, None], so
        # the [None, None] combination is unreachable in production; it falls
        # into the first branch and compares a float against None.
        with pytest.raises(TypeError):
            vr_report([1.0], [None, None])


# ===========================================================================
# ADFTestReport
# ===========================================================================

class TestADFTestReport:

    def test_fields_from_result_tuple(self):
        report = ADFTestReport(adf_result(-2.1, pval=0.24, lags=5, nobs=994,
                                          criticals=(-3.43, -2.86, -2.57)))
        assert report.stat == -2.1
        assert report.pval == 0.24
        assert report.lags == 5
        assert report.nobs == 994
        assert report.sig_str == ["1%", "5%", "10%"]
        assert report.sig == [0.01, 0.05, 0.1]
        # critical values are pulled by significance label, not by dict order
        assert report.critical_vals == [-3.43, -2.86, -2.57]

    # The ADF null is a unit root: the test "passes" (fails to reject) when the
    # statistic sits above the critical value at that significance level.
    @pytest.mark.parametrize("stat,expected", [
        (-0.5, [True, True, True]),            # above every critical value
        (-2.7, [True, True, False]),           # rejected only at 10%
        (-3.0, [True, False, False]),
        (-5.0, [False, False, False]),         # rejected everywhere
        (-2.86, [True, True, False]),          # boundary: comparison is >=
    ])
    def test_status_vals(self, stat, expected):
        report = ADFTestReport(adf_result(stat, criticals=(-3.43, -2.86, -2.57)))
        assert [bool(s) for s in report.status_vals] == expected
        assert report.status_str == ["Passed" if s else "Failed" for s in expected]

    def test_summary(self, capsys):
        report = ADFTestReport(adf_result(-2.7, pval=0.073, lags=2, nobs=997,
                                          criticals=(-3.43, -2.86, -2.57)))
        report.summary(tablefmt="plain")
        rows = [line.split() for line in capsys.readouterr().out.strip().split("\n")]
        assert rows[0] == ["Test", "Statistic", "-2.7"]
        assert rows[1] == ["pvalue", "0.073"]
        assert rows[2] == ["Lags", "2"]
        assert rows[3] == ["Number", "Obs", "997"]
        assert rows[4] == ["Significance", "Critical", "Value", "Result"]
        assert rows[5] == ["1%", "-3.43", "Passed"]
        assert rows[6] == ["5%", "-2.86", "Passed"]
        assert rows[7] == ["10%", "-2.57", "Failed"]

    def test_summary_default_tablefmt_is_fancy_grid(self, capsys):
        ADFTestReport(adf_result(-2.7, pval=0.073, lags=2, nobs=997)).summary()
        out = capsys.readouterr().out
        # both tables are boxed: header table then the per significance results
        assert out.count("╒") == 2 and out.count("╘") == 2
        assert "│" in out
        assert "Significance" in out and "Critical Value" in out and "Result" in out
        assert out.index("Test Statistic") < out.index("Significance")

    def test_random_walk_fails_to_reject_unit_root(self):
        # A random walk has a unit root, so the ADF null holds and the test
        # "passes" at every level except on the size of the test: at 10% that
        # is ~10% of paths, so this is asserted over an ensemble rather than a
        # single path (>= 12 of 20 clears p = 0.9 by more than 5 sigma).
        # Within a path the three statuses must be ordered, since the critical
        # values grow from 1% to 10%.
        npass = 0
        for _ in range(20):
            x = numpy.cumsum(numpy.random.normal(0.0, 1.0, 400))
            res = sm.tsa.stattools.adfuller(x, regression="c", maxlag=6, result_object=False)
            report = ADFTestReport(res)
            assert report.nobs + report.lags + 1 == 400
            # the critical values are pulled out of statsmodels' unordered dict
            # by label and re-ordered 1%, 5%, 10% (increasing)
            assert report.critical_vals == [res[4]["1%"], res[4]["5%"], res[4]["10%"]]  # pyright: ignore[reportIndexIssue, reportGeneralTypeIssues, reportOptionalSubscript]
            assert report.critical_vals[0] < report.critical_vals[1] < report.critical_vals[2]
            if all(report.status_vals):
                assert report.status_str == ["Passed"]*3
                npass += 1
        assert npass >= 12

        # the status of each level flips exactly at its own critical value:
        # a statistic a hair below the 5% value of the last fitted report
        # passes at 1% only
        criticals = tuple(report.critical_vals)  # pyright: ignore[reportPossiblyUnboundVariable]
        nudged = ADFTestReport(adf_result(criticals[1] - 1e-9, criticals=criticals))
        assert [bool(s) for s in nudged.status_vals] == [True, False, False]
        nudged = ADFTestReport(adf_result(criticals[1], criticals=criticals))
        assert [bool(s) for s in nudged.status_vals] == [True, True, False]

    def test_stationary_ar1_rejects_unit_root(self):
        # AR(1) with phi = 0.5 over 1000 points: the ADF statistic is around
        # -22 (roughly -sqrt(n)*(1-phi)/sqrt(1-phi^2)), decades below the 1%
        # critical value, so rejection is certain for any seed.
        eps = numpy.random.normal(0.0, 1.0, 1000)
        x = numpy.zeros(1000)
        for i in range(1, 1000):
            x[i] = 0.5*x[i-1] + eps[i]
        report = ADFTestReport(sm.tsa.stattools.adfuller(x, regression="c", maxlag=12, result_object=False))
        assert not any(report.status_vals)
        assert report.status_str == ["Failed"]*3
        assert report.pval < 0.01
        assert report.stat < min(report.critical_vals)


# ===========================================================================
# OUEstReport
# ===========================================================================

# AR(1) fit of an OU process: intercept a = mu(1 - phi), coefficient phi, and
# innovation variance. mu = 0.4/0.2 = 2, lambda = -ln(0.8)/0.1.
STUB_PARAMS = [0.4, 0.8, 0.25]
STUB_ERRORS = [0.01, 0.02, 0.005]
STUB_DT = 0.1


def ou_stub_report():
    return OUEstReport(ARFitStub(STUB_PARAMS, STUB_ERRORS), STUB_DT, 1.5)


class TestOUEstReport:

    def test_stores_fit_values(self):
        fit = ARFitStub(STUB_PARAMS, STUB_ERRORS)
        report = OUEstReport(fit, STUB_DT, 1.5)
        assert report.ar_result is fit
        assert report.delta_t == STUB_DT
        assert report.x0 == 1.5
        assert report.offset_est == 0.4
        assert report.offset_error == 0.01
        assert report.coeff_est == 0.8
        assert report.coeff_error == 0.02

    def test_mu_est_hand_computed(self):
        # mu = a/(1 - phi) = 0.4/0.2
        assert ou_stub_report().mu_est() == pytest.approx(2.0)

    def test_lambda_est_hand_computed(self):
        # lambda = -ln(phi)/dt = -ln(0.8)/0.1
        assert ou_stub_report().lambda_est() == pytest.approx(2.2314355131420974)

    @pytest.mark.parametrize("lam,dt", [(0.5, 0.01), (1.0, 0.1), (3.25, 0.25)])
    def test_lambda_est_inverts_the_discretisation(self, lam, dt):
        # an AR(1) coefficient of exp(-lambda dt) must map back to lambda
        report = OUEstReport(ARFitStub([0.0, numpy.exp(-lam*dt), 1.0], [0.0, 0.0, 0.0]), dt, 0.0)
        assert report.lambda_est() == pytest.approx(lam)

    def test_mu_error_propagation(self):
        # mu = a/(1 - phi) => delta_mu = delta_a/(1 - phi) + a*delta_phi/(1 - phi)**2
        #                              = (0.01 + 0.4*0.02/0.2)/0.2 = 0.25
        assert ou_stub_report().mu_error() == pytest.approx(0.25)

    def test_lambda_error_is_a_magnitude(self):
        # |d(lambda)/d(phi)|*delta_phi = 0.02/(0.8*0.1) = 0.25
        assert ou_stub_report().lambda_error() == pytest.approx(0.25)

    def test_sigma2_est(self):
        # sigma^2 = 2*lambda*var(eps)/(1 - phi^2) with lambda = -ln(0.8)/0.1:
        # 2*2.2314355131420974*0.25/(1 - 0.64)
        assert ou_stub_report().sigma2_est() == pytest.approx(3.0992159904751354)

    def test_sigma2_est_body_is_correct(self):
        # only the binding is broken, not the arithmetic: called off the class
        # (so the instance attribute cannot shadow it) the method body reads the
        # stored AR innovation variance and rescales it correctly
        assert OUEstReport.sigma2_est(ou_stub_report()) == pytest.approx(3.0992159904751354)

    def test_sigma2_error(self):
        # sigma^2 = 2 lambda s2/(1 - phi^2), so the propagated uncertainty is
        #   4 lambda phi s2 delta_phi/(1 - phi^2)^2  = 0.275486
        # + 2 s2 |delta_lambda|/(1 - phi^2)          = 0.347222
        # + 2 lambda delta_s2/(1 - phi^2)            = 0.061984
        assert ou_stub_report().sigma2_error() == pytest.approx(0.6846924078517371)

    def test_sigma2_error_sign_cancellation(self):
        assert OUEstReport.sigma2_error(ou_stub_report()) == pytest.approx(0.6846924078517371)

    def test_raw_fit_values_are_kept_apart_from_the_derived_ones(self):
        # the AR innovation variance and its standard error live on their own
        # names, leaving sigma2_est/sigma2_error free to be the derived methods
        report = ou_stub_report()
        assert report.ar_var_est == 0.25
        assert report.ar_var_error == 0.005
        assert callable(report.sigma2_est)
        assert callable(report.sigma2_error)

    def test_summary(self, capsys):
        ou_stub_report().summary(tablefmt="plain")
        out = capsys.readouterr().out
        assert "Parameter" in out and "Estimate" in out and "Error" in out
        assert "μ" in out and "λ" in out and "σ2" in out

    # An AR(1) fit of a slowly mean reverting series can land outside (0, 1),
    # and nothing in OUEstReport guards against it: the estimates degenerate
    # silently (inf/nan) rather than raising.
    def test_unit_root_coefficient_blows_up_mu(self):
        report = OUEstReport(ARFitStub([0.4, 1.0, 0.25], STUB_ERRORS), STUB_DT, 0.0)
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            mu = report.mu_est()
        assert numpy.isinf(mu)                    # a/(1 - phi) with phi = 1
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            assert numpy.isinf(report.mu_error())
        assert report.lambda_est() == 0.0         # -ln(1)/dt: no mean reversion
        assert report.lambda_error() == pytest.approx(0.2)

    def test_explosive_coefficient_gives_negative_lambda(self):
        # phi > 1 is a mean averting fit: lambda = -ln(phi)/dt < 0 and mu falls
        # on the wrong side of the data, both without any warning
        report = OUEstReport(ARFitStub([0.4, 1.05, 0.25], STUB_ERRORS), STUB_DT, 0.0)
        assert report.lambda_est() == pytest.approx(-numpy.log(1.05)/STUB_DT)
        assert report.lambda_est() < 0.0
        assert report.mu_est() == pytest.approx(-8.0)

    def test_negative_coefficient_makes_lambda_nan(self):
        # phi <= 0 (an oscillating fit) puts numpy.log outside its domain
        report = OUEstReport(ARFitStub([0.4, -0.5, 0.25], STUB_ERRORS), STUB_DT, 0.0)
        with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
            assert numpy.isnan(report.lambda_est())
        # and the error divides by a negative phi, so the sign of lambda_error
        # flips relative to the phi > 0 case tested above
        assert report.lambda_error() == pytest.approx(0.4)

    def test_zero_coefficient_makes_lambda_infinite(self):
        report = OUEstReport(ARFitStub([0.4, 0.0, 0.25], STUB_ERRORS), STUB_DT, 0.0)
        with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
            assert numpy.isinf(report.lambda_est())
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            assert numpy.isposinf(report.lambda_error())

    def test_round_trip_recovers_ou_parameters(self):
        # Simulate an exactly discretised OU path, fit AR(1) by OLS and read the
        # continuous parameters back. Tolerances are ~5 standard errors of the
        # respective estimators for n = 4000, dt = 0.1, lambda = 1:
        #   se(mu) = sqrt(sigma^2/(2 lambda) * (1+phi)/(n(1-phi))) ~ 0.05
        #   se(lambda) = se(phi)/(phi dt), se(phi) = sqrt((1-phi^2)/n) ~ 0.075
        mu, lam, sigma, dt, n = 2.0, 1.0, 1.0, 0.1, 4000
        x = ou_path(mu, lam, sigma, dt, n)
        report = OUEstReport(fit_ar1(x), dt, x[0])

        phi = numpy.exp(-lam*dt)
        assert report.coeff_est == pytest.approx(phi, abs=0.04)
        assert report.offset_est == pytest.approx(mu*(1.0 - phi), abs=0.08)
        assert report.mu_est() == pytest.approx(mu, abs=0.25)
        assert report.lambda_est() == pytest.approx(lam, abs=0.4)
        # the propagated error is the OLS se scale, ~0.15 here: bracket it well
        # inside an order of magnitude either side rather than just > 0
        assert 0.05 < report.mu_error() < 0.5

        # sigma2_est() is shadowed, so apply its formula to the innovation
        # variance read back through the fit object (not through the shadowing
        # attribute, which the fix to that bug will remove):
        # sigma^2 = 2 lambda var(eps)/(1 - phi^2) recovers the continuous
        # diffusion coefficient. Tolerance is 5 sd of the recovered value
        # (sd ~ 0.024 over seeds), matching the other tolerances here.
        ar_var = report.ar_result.params.iloc[2]
        sigma2 = 2.0*report.lambda_est()*ar_var/(1.0 - report.coeff_est**2)
        assert sigma2 == pytest.approx(sigma**2, abs=0.12)
        # the AR innovation variance itself is a factor (1 - phi^2)/(2 lambda) smaller
        assert ar_var == pytest.approx(sigma**2*(1.0 - phi**2)/(2.0*lam), rel=0.1)

    def test_ar_offset_fit_supplies_the_positional_layout(self):
        # OUEstReport addresses params/bse as [offset, AR coefficient,
        # innovation variance] via .iloc, an assumption no navi caller
        # validates (the class has no producer). navi's own AR(1) with offset
        # estimator is the only fit of that shape: check the layout against it.
        # Tolerances are ~3x the largest deviation seen over 60 seeds for
        # n = 4000 (max |phi err| 0.017, |lambda err| 0.19, |const err| 0.14,
        # |sigma2 rel err| 0.07), the maximum likelihood fit being noisier than
        # the OLS one used above.
        mu, lam, sigma, dt, n = 2.0, 1.0, 1.0, 0.1, 4000
        x = ou_path(mu, lam, sigma, dt, n)
        fit = arima.ar_offset_fit(pandas.Series(x), 1)
        assert list(fit.params.index) == ARFitStub.NAMES
        assert list(fit.bse.index) == ARFitStub.NAMES  # pyright: ignore[reportAttributeAccessIssue]

        report = OUEstReport(fit, dt, x[0])
        phi = numpy.exp(-lam*dt)
        assert report.coeff_est == pytest.approx(phi, abs=0.05)
        assert report.lambda_est() == pytest.approx(lam, abs=0.5)
        # statsmodels' ARIMA trend='c' reports the process MEAN as 'const',
        # not the OLS intercept mu(1 - phi) that OUEstReport expects
        assert report.offset_est == pytest.approx(mu, abs=0.4)
        assert report.ar_var_est == pytest.approx(sigma**2*(1.0 - phi**2)/(2.0*lam), rel=0.2)

    def test_mu_est_from_navi_ar_offset_fit(self):
        mu, lam, sigma, dt, n = 2.0, 1.0, 1.0, 0.1, 4000
        x = ou_path(mu, lam, sigma, dt, n)
        report = OUEstReport(arima.ar_offset_fit(pandas.Series(x), 1), dt, x[0])
        assert report.mu_est() == pytest.approx(mu, abs=0.4)

    def test_fit_must_be_pandas_backed(self):
        # ar_offset_fit is typed for NDArray input, and for an ndarray sample
        # statsmodels returns params/bse as plain arrays - which .iloc cannot
        # address. Callers must hand OUEstReport a pandas backed fit.
        x = ou_path(2.0, 1.0, 1.0, 0.1, 500)
        fit = arima.ar_offset_fit(x, 1)
        assert isinstance(fit.params, numpy.ndarray)
        with pytest.raises(AttributeError, match="iloc"):
            OUEstReport(fit, 0.1, x[0])


# ===========================================================================
# JohansenTestReport
# ===========================================================================

class TestJohansenTestReport:

    def test_fields_from_result(self):
        evec = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = JohansenResultStub([50.0, 14.0, 1.0], CVT_3VAR, lr2=[36.0, 13.0, 1.0],
                                    cvm=CVT_3VAR - 1.0, eig=[0.3, 0.1, 0.01], evec=evec)
        report = JohansenTestReport(cast(JohansenTestResult, result))
        assert_allclose(report.eigen_values, [0.3, 0.1, 0.01])
        assert_allclose(report.eigen_vectors, evec)
        assert_allclose(report.trace_statistic, [50.0, 14.0, 1.0])
        assert_allclose(report.trace_critical_vals, CVT_3VAR)
        assert_allclose(report.eigen_value_statistic, [36.0, 13.0, 1.0])
        assert_allclose(report.eigen_value_critical_values, CVT_3VAR - 1.0)
        assert isinstance(report.eigen_vectors, numpy.ndarray)
        assert isinstance(report.trace_critical_vals, numpy.ndarray)

    def test_compute_rank_monotone_rejections(self):
        # r <= 0 rejected at every level; r <= 1 only at 90% (14 > 13.4294 but
        # 14 < 15.4943); r <= 2 nowhere. Rank per level: [2, 1, 1].
        report = JohansenTestReport(cast(JohansenTestResult, JohansenResultStub([50.0, 14.0, 1.0], CVT_3VAR)))
        assert [int(r) for r in report.compute_rank()] == [2, 1, 1]

    @pytest.mark.parametrize("trace_stats,expected", [
        ([100.0, 50.0, 20.0], [3, 3, 3]),      # every null rejected -> full rank
        ([1.0, 1.0, 1.0], [0, 0, 0]),          # nothing rejected -> no cointegration
        ([30.0, 1.0, 1.0], [1, 1, 0]),         # r <= 0 rejected at 90/95 only
    ])
    def test_compute_rank(self, trace_stats, expected):
        report = JohansenTestReport(cast(JohansenTestResult, JohansenResultStub(trace_stats, CVT_3VAR)))
        assert [int(r) for r in report.compute_rank()] == expected

    def test_compute_rank_non_monotone_rejections(self):
        # r <= 0 rejected everywhere, r <= 1 accepted everywhere (5 < 13.4294),
        # r <= 2 rejected everywhere (8 > 6.6349)
        report = JohansenTestReport(cast(JohansenTestResult, JohansenResultStub([50.0, 5.0, 8.0], CVT_3VAR)))
        assert [int(r) for r in report.compute_rank()] == [1, 1, 1]

    def test_compute_rank_two_variables(self):
        # r <= 0 rejected at every level, r <= 1 only at 90%: one rank per
        # significance level is expected whatever the system size
        report = JohansenTestReport(cast(JohansenTestResult, JohansenResultStub([236.9, 3.3], CVT_2VAR)))
        ranks = report.compute_rank()
        assert len(ranks) == 3
        assert int(ranks[2]) == 1

    def test_compute_rank_four_variables(self):
        cvt = numpy.vstack([CVT_3VAR[0] + 20.0, CVT_3VAR])
        report = JohansenTestReport(cast(JohansenTestResult, JohansenResultStub([90.0, 50.0, 14.0, 1.0], cvt)))
        assert len(report.compute_rank()) == 3

    def test_summary(self, capsys):
        evec = numpy.array([[1.0, 0.1, 0.01], [-2.0, 0.2, 0.02], [0.5, 0.3, 0.03]])
        result = JohansenResultStub([50.0, 14.0, 1.0], CVT_3VAR, lr2=[36.0, 13.0, 1.0],
                                    cvm=CVT_3VAR - 1.0, eig=[0.25, 0.1, 0.01], evec=evec)
        JohansenTestReport(cast(JohansenTestResult, result)).summary(tablefmt="plain")
        out = capsys.readouterr().out

        assert "Trace Statistic" in out and "Rank" in out
        assert "Eigenvalue Statistic" in out and "Eigenvalues and Eigenvectors" in out
        for i in range(3):
            assert f"r <= {i}" in out
        assert "50.000" in out and "27.067" in out and "35.463" in out   # trace row, floatfmt .3f
        assert "36.000" in out and "26.067" in out                       # eigenvalue statistic row
        assert "2.50e-01" in out and "1.00e-02" in out                   # eigenvalues, floatfmt .2e
        # rank row: [2, 1, 1] under the three significance headers
        rank_header = out.split("Rank\n")[1].split("\n")[1]
        assert rank_header.split() == ["2", "1", "1"]
        # eigenvectors are printed as columns of evec
        assert "[ 1.  -2.   0.5]" in out

    def test_summary_default_tablefmt_is_fancy_grid(self, capsys):
        result = JohansenResultStub([50.0, 14.0, 1.0], CVT_3VAR)
        JohansenTestReport(cast(JohansenTestResult, result)).summary()
        out = capsys.readouterr().out
        # four boxed tables: trace, rank, eigenvalue statistic, eigenvectors
        assert out.count("╒") == 4 and out.count("╘") == 4
        assert "│" in out
        for section in ["Trace Statistic", "Rank", "Eigenvalue Statistic",
                        "Eigenvalues and Eigenvectors"]:
            assert section in out

    def test_cointegrated_pair_round_trip(self):
        # y2 = 2*y1 + stationary noise, so exactly one cointegrating relation
        # exists and its vector is proportional to (2, -1).
        report = real_pair_report(n=1000, beta=2.0)

        # The r <= 0 trace and max-eigenvalue statistics diverge with n for a
        # genuinely cointegrated pair (both land near 230 for n = 1000 with sd
        # ~18, against a 99% critical value of ~20). Three times the critical
        # value (~60) sits more than 9 sd below the mean, so rejection holds
        # for any seed; ten times (~200) is only 1.7 sd below it and flakes.
        assert report.trace_statistic[0] > 3.0*report.trace_critical_vals[0][2]
        assert report.eigen_value_statistic[0] > 3.0*report.eigen_value_critical_values[0][2]
        # lr1 is a decreasing partial sum by construction, so trace[1] < trace[0]
        # holds for any data at all: what separates a cointegrated pair from an
        # unrelated one is the size of the gap (the ratio stays under 0.09).
        assert report.trace_statistic[1] < 0.2*report.trace_statistic[0]

        # the leading eigenvector is the first *column* of evec, normalised
        # here by its first entry; (2, -1) ~ (1, -0.5)
        lead = numpy.real(report.eigen_vectors[:, 0])
        assert_allclose(lead/lead[0], [1.0, -0.5], atol=0.05)
        assert numpy.real(report.eigen_values[0]) > numpy.real(report.eigen_values[1])

    def test_real_pair_compute_rank(self):
        # compute_rank on the shape the notebooks use: one rank per
        # significance level. The r <= 0 null is rejected by a factor of ten
        # at every level, so every column carries at least rank 1 - the
        # r <= 1 null is size controlled and strays above its 90/95% critical
        # values often enough that the individual values are not pinned.
        report = real_pair_report()
        ranks = report.compute_rank()
        assert len(ranks) == 3
        assert all(int(rank) >= 1 for rank in ranks)

    def test_real_pair_compute_rank_covers_every_significance_level(self):
        assert len(real_pair_report().compute_rank()) == 3

    def test_real_pair_summary_rank_table_covers_every_significance_level(self, capsys):
        real_pair_report().summary(tablefmt="plain")
        out = capsys.readouterr().out
        rank_table = out.split("Rank\n")[1].split("Eigenvalue Statistic")[0]
        assert "Critical Value 99%" in rank_table

    def test_real_pair_summary_rank_table_all_columns(self, capsys):
        # one value under each of the three significance headers
        real_pair_report().summary(tablefmt="plain")
        out = capsys.readouterr().out
        rank_table = out.split("Rank\n")[1].split("Eigenvalue Statistic")[0]
        header, values = [line for line in rank_table.split("\n") if line.strip()][:2]
        assert header.split() == ["Critical", "Value", "90%", "Critical", "Value", "95%",
                                  "Critical", "Value", "99%"]
        assert len(values.split()) == 3

    def test_real_result_stores_real_eigen_values(self):
        # numpy >= 2 linalg.eig returns complex128 even when every imaginary
        # part is zero. JohansenTestReport takes numpy.real of eig/evec, the
        # same thing hyp_test.JohansenCointTestEigenVector does, so the two
        # report layers agree and tabulate has nothing to downcast.
        report = real_pair_report()
        assert not numpy.iscomplexobj(report.eigen_values)
        assert not numpy.iscomplexobj(report.eigen_vectors)
        assert report.eigen_values.dtype == numpy.float64
        assert report.eigen_vectors.dtype == numpy.float64

    def test_summary_of_real_result_does_not_discard_imaginary_parts(self):
        report = real_pair_report()
        with warnings.catch_warnings():
            warnings.simplefilter("error", numpy.exceptions.ComplexWarning)
            report.summary(tablefmt="plain")

    def test_summary_of_real_result_prints_leading_eigenvector(self, capsys):
        report = real_pair_report(n=1000, beta=2.0)
        report.summary(tablefmt="plain")
        out = capsys.readouterr().out
        assert "r <= 0" in out and "r <= 1" in out

        # The eigenvector cell of the leading row holds evec[:, 0] printed
        # whole: parse it back out and check it is the cointegrating vector
        # (2, -1) rather than a row of evec.
        tail = out.split("Eigenvalues and Eigenvectors")[1]
        lead_row = [line for line in tail.split("\n") if "[" in line][0]
        cell = lead_row[lead_row.index("["):lead_row.index("]") + 1]
        vec = [float(v) for v in cell.replace("+0.j", "").strip("[]").split()]
        assert len(vec) == 2
        assert vec[0]/vec[1] == pytest.approx(-2.0, abs=0.2)
