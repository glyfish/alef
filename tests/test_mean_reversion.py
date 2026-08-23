"""Tests for ``lib.data.mean_reversion``.

The module exposes a single façade used by the mean-reversion trading
notebooks::

    t½, report, result = compute_mean_reversion_halflife(data)

Internally it subtracts the sample mean, regresses ``ΔX_t`` on ``X_{t-1}``
(``lib.data.impl.ou.compute_mean_half_life_estimate`` with the default
``dt = 1``) and attaches two transformed parameters to the ``OLSResult``:

    λ̂ = slope / dt          t½ = -ln 2 / λ̂          σ_{t½} = (ln 2 / λ̂²) σ_λ̂

Note the sign convention: the estimation model printed on the result is
``ΔX_t = λ X_{t-1} Δt + μ Δt + √Δt ε_t``, so the reported ``λ̂`` is the *slope*
and is negative for a mean-reverting series — the negation of the ``λ`` that
``lib.models.ou.ou`` simulates with.  ``t½`` carries the compensating minus
sign and comes out positive.  Because ``dt`` is hard-coded to 1 the half-life
is measured in *sample steps*, which is what the notebooks want when they size
a moving-average window.

**The reported half-life is a first-order approximation.**  ``t½ = -ln2/λ̂``
inverts ``φ = 1 + λ̂Δt``, not ``φ = e^{λ̂Δt}``, so for a series whose deviation
from the mean really decays like ``φ^t`` the façade returns ``ln2/(1 - φ)``
while the series actually halves in ``ln2/(-ln φ)`` steps — a factor of 2.01
apart at ``φ = 0.2``.  ``lib.models.ou.ou`` is itself the matching first-order
Euler scheme (``φ = 1 - λΔt``), so a round trip through it cancels the error on
both sides and is silent about it.  Two tests below make the approximation
visible instead: ``test_reported_half_life_is_a_first_order_approximation``
(deterministic, noiseless) and ``TestRoundTrip.test_recovers_an_exactly_
simulated_ou`` / ``test_exact_ou_exposes_the_discretisation_bias``, which
simulate the *exact* AR(1) OU (``φ = e^{-λΔt}``) and pin the resulting bias.

The kinds of check below are:

* closed form — a noiseless geometric decay ``X_t = μ + A φ^t`` satisfies
  ``ΔX_t = (φ - 1)(X_{t-1} - μ)`` exactly, so the fit must be perfect and
  return ``t½ = ln 2 / (1 - φ)``; plus one four-point series whose OLS slope is
  worked out by hand as an exact fraction;
* round trip — simulate OU paths with a known ``λ`` and recover the half-life
  inside a tolerance built from the AR(1) slope standard error
  ``√((1 - φ²)/n)``, both for the Euler simulator and for an exact OU;
* contract — tuple shape and types, the transform bookkeeping, translation
  invariance of the demeaning step, ``nobs = n - 1`` from the differencing,
  serialization of the returned result, accepted input containers/dtypes, and
  the degenerate inputs (no variation, too short, NaN/inf, 2-D).

Every stochastic tolerance was checked over 40+ independent seeds (see the
comment on each) so nothing here is tuned to the seed the autouse conftest
fixture installs.
"""

import json
import math
import warnings

import numpy
import numpy.testing as npt
import pandas
import pytest
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib.data.impl import ou as ou_impl
from lib.data.mean_reversion import compute_mean_reversion_halflife
from lib.data.param_est import EstModel, OLSParamType, OLSResult, ParamEst
from lib.models import ou

LN2 = math.log(2.0)


def _ou_path(λ: float, n: int, μ: float = 5.0, σ: float = 1.0, Δt: float = 1.0):
    """OU path started at the mean, through the façade the notebooks call.

    ``lib.models.ou.ou`` is the Euler scheme
    ``x[i+1] = x[i] + λ(μ - x[i])Δt + σΔt ε[i]``, i.e. an exact AR(1) in
    ``x - μ`` with ``φ = 1 - λΔt`` — the *same* first-order discretisation the
    estimator inverts.  Use ``_exact_ou_path`` when the point of the test is
    that discretisation.
    """
    _, x = ou_impl.create_source(μ=μ, λ=λ, Δt=Δt, σ=σ, x0=μ, npts=n)
    return x


def _exact_ou_path(λ: float, n: int, μ: float = 5.0, σ: float = 1.0, Δt: float = 1.0):
    """Exactly simulated OU: the AR(1) with ``φ = e^{-λΔt}``.

    ``x[t+1] = μ + φ(x[t] - μ) + σ√((1 - φ²)/(2λ)) ε[t]`` is the exact
    transition law of ``dX = λ(μ - X)dt + σ dW`` sampled every ``Δt``, so its
    deviation from the mean genuinely halves in ``ln2/λ`` time units.  Nothing
    here shares an approximation with the estimator.
    """
    φ = math.exp(-λ * Δt)
    scale = σ * math.sqrt((1.0 - φ**2) / (2.0 * λ))
    ε = numpy.random.normal(0.0, 1.0, n)
    x = numpy.empty(n)
    x[0] = μ
    for i in range(n - 1):
        x[i + 1] = μ + φ * (x[i] - μ) + scale * ε[i]
    return x


def _slope_se(λΔt: float, n: int) -> float:
    """Large-sample SE of the OLS slope of ΔX on X_{t-1} for AR(1) φ = 1 - λΔt.

    SE² = Var[ε] / (n Var[X]) and the stationary Var[X] = Var[ε]/(1 - φ²), so
    the innovation scale cancels: SE = √((1 - φ²)/n).
    """
    φ = 1.0 - λΔt
    return math.sqrt((1.0 - φ**2) / n)


def _hand_slope(x) -> float:
    """OLS slope of ΔX on X_{t-1} from the covariance formula.

    Independent of statsmodels: b = Σ(x̃ - x̄)(Δx - Δx̄) / Σ(x̃ - x̄)², where
    x̃ = x[:-1].  Used to cross-check the façade rather than re-run it.
    """
    x = numpy.asarray(x, dtype=float)
    lag = x[:-1]
    dx = numpy.diff(x)
    lag_c = lag - lag.mean()
    return float(numpy.dot(lag_c, dx - dx.mean()) / numpy.dot(lag_c, lag_c))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    n = 2000

    @pytest.fixture
    def fit(self):
        x = _ou_path(0.5, self.n)
        return x, compute_mean_reversion_halflife(x)

    def test_returns_half_life_report_and_result(self, fit):
        _, (half_life, report, result) = fit
        assert isinstance(half_life, float)  # numpy.float64 subclasses float
        assert isinstance(report, RegressionResultsWrapper)
        assert isinstance(result, OLSResult)
        assert result.est_model == EstModel.OLS
        assert numpy.isfinite(half_life)
        assert half_life > 0.0

    def test_single_variable_regression_shape(self, fit):
        x, (_, report, result) = fit
        # One observation is consumed by the difference, and the design is
        # [constant, X_{t-1}] so exactly one slope parameter comes back.
        assert report.nobs == len(x) - 1
        assert numpy.asarray(report.params).shape == (2,)
        assert len(result.params) == 1
        assert result.params[0].param_type == OLSParamType.OLS_PARAM.value
        assert result.const.param_type == OLSParamType.OLS_CONST.value
        assert result.r2 == pytest.approx(report.rsquared, rel=1e-12)

    def test_half_life_derives_from_the_raw_report_slope(self, fit):
        # Independent of the OLSResult bookkeeping: read the slope straight out
        # of the statsmodels parameter vector and apply t½ = -ln2/λ̂ by hand.
        _, (half_life, report, result) = fit
        params = numpy.asarray(report.params)
        bse = numpy.asarray(report.bse)
        assert half_life == pytest.approx(-LN2 / params[1], rel=1e-12)
        assert result.params[0].est == pytest.approx(params[1], rel=1e-12)
        assert result.const.est == pytest.approx(params[0], rel=1e-12)
        # The standard errors are carried over untransformed, positionally:
        # design column 0 is the constant and column 1 the lagged level. Pin
        # them against the report directly so a mis-scaled σ_λ̂ is localised
        # here rather than only showing up through the delta-method transform.
        assert result.params[0].err == pytest.approx(bse[1], rel=1e-12)
        assert result.const.err == pytest.approx(bse[0], rel=1e-12)
        # delta method on t½(λ) = -ln2/λ: σ_{t½} = ln2 σ_λ / λ²
        assert result.param_transforms[0].param.err == pytest.approx(
            LN2 * bse[1] / params[1] ** 2, rel=1e-12
        )

    def test_slope_agrees_with_the_covariance_formula(self, fit):
        # statsmodels solves by pinv; the closed-form covariance ratio is a
        # different route to the same number.
        x, (half_life, _, result) = fit
        assert result.params[0].est == pytest.approx(_hand_slope(x), rel=1e-9)
        assert half_life == pytest.approx(-LN2 / _hand_slope(x), rel=1e-9)

    def test_transform_bookkeeping(self, fit):
        _, (half_life, report, result) = fit
        assert result.param_transforms is not None
        assert len(result.param_transforms) == 2
        t_half, λ_est = (t.param for t in result.param_transforms)
        assert half_life == t_half.est
        assert t_half.est_label == r"$t_{1/2}$"
        assert t_half.err_label == r"$\sigma_{t_H}$"
        assert λ_est.est_label == r"$\lambda$"
        for p in (t_half, λ_est):
            assert p.param_type == OLSParamType.TRANS_PARAM.value
            assert p.est_id == result.est_id
        # dt defaults to 1 in the façade, so the λ̂ transform must be the
        # untouched OLS slope. Anchor it on the statsmodels report rather than
        # on result.params (which the transform is derived from), otherwise the
        # assertion is an identity that cannot fail.
        assert λ_est.est == pytest.approx(numpy.asarray(report.params)[1], rel=1e-12)
        assert λ_est.err == pytest.approx(numpy.asarray(report.bse)[1], rel=1e-12)
        assert λ_est.est < 0.0 < half_life  # sign convention of the printed model

        assert result.const_transform is not None
        const = result.const_transform.param
        assert const.param_type == OLSParamType.TRANS_CONST.value
        assert const.est == pytest.approx(result.const.est, rel=1e-12)
        assert const.est_label == r"$\mu$"

        assert isinstance(result.model, str)
        assert r"\Delta X_t" in result.model
        assert r"\lambda X_{t-1}" in result.model

    def test_result_serializes_and_round_trips(self, fit):
        _, (half_life, _, result) = fit
        payload = json.loads(result.to_json())
        assert payload["est_model"] == EstModel.OLS.value
        assert payload["est_id"] == result.est_id
        assert payload["param_transforms"][0]["param"]["est"] == half_life
        restored = ParamEst.from_dict(payload["param_transforms"][0]["param"])
        assert restored.est == half_life
        assert restored.est_label == r"$t_{1/2}$"
        assert restored.param_type == OLSParamType.TRANS_PARAM.value
        assert "$t_{1/2}$" in repr(restored)
        assert repr(result).startswith("OLSEst(")
        assert result.est_id in repr(result)

    def test_report_summary_is_usable(self, fit):
        # The notebooks print the returned statsmodels report directly.
        _, (_, report, _) = fit
        text = str(report.summary())
        assert "OLS" in text
        assert "R-squared" in text

    def test_input_is_not_modified(self):
        # Build the path here rather than reusing the `fit` fixture: that
        # fixture has already called the façade once, so a baseline taken after
        # it would itself carry any idempotent in-place mutation (a clip, a
        # fillna, a sort) and the comparison would pass regardless.
        x = _ou_path(0.5, 500)
        before = x.copy()
        compute_mean_reversion_halflife(x)
        npt.assert_array_equal(x, before)

    def test_accepts_a_python_list(self):
        # Pin a value, not just list/array agreement: the hand-computed series
        # from test_hand_computed_four_point_regression has slope -33/35 and
        # t½ = 35 ln2 / 33, so a regression affecting *both* containers still
        # fails here.
        x = [0.0, 2.0, 1.0, 4.0, 3.0]
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert half_life == pytest.approx(35.0 * LN2 / 33.0, rel=1e-9)
        assert result.params[0].est == pytest.approx(-33.0 / 35.0, rel=1e-9)
        assert half_life == pytest.approx(
            compute_mean_reversion_halflife(numpy.array(x))[0], rel=1e-12
        )

    def test_accepts_a_pandas_series(self):
        # The yfinance-fed notebooks pass a pandas Series of closes straight in.
        # numpy.mean/len/subtraction all work on a Series, and statsmodels takes
        # it from there.
        s = pandas.Series([0.0, 2.0, 1.0, 4.0, 3.0])
        half_life, report, result = compute_mean_reversion_halflife(s)
        assert half_life == pytest.approx(35.0 * LN2 / 33.0, rel=1e-9)
        assert result.params[0].est == pytest.approx(-33.0 / 35.0, rel=1e-9)
        assert report.nobs == 4
        # …and an indexed Series matches the bare array it wraps.
        x = _ou_path(0.5, 400)
        indexed = pandas.Series(x, index=pandas.RangeIndex(100, 100 + len(x)))
        assert compute_mean_reversion_halflife(indexed)[0] == pytest.approx(
            compute_mean_reversion_halflife(x)[0], rel=1e-12
        )

    def test_accepts_integer_dtype(self):
        # No dtype guard: an int64 series is promoted by the demeaning and fitted
        # like any other.
        x = numpy.array([0, 2, 1, 4, 3])
        assert x.dtype == numpy.dtype(numpy.int64)
        assert compute_mean_reversion_halflife(x)[0] == pytest.approx(
            35.0 * LN2 / 33.0, rel=1e-9
        )
        # An int ramp is accepted too and returns round-off garbage rather than
        # raising — same degenerate path as the float ramp below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half_life, _, result = compute_mean_reversion_halflife(numpy.arange(50))
        assert result.params[0].est == pytest.approx(0.0, abs=1e-12)
        assert abs(half_life) > 1e12

    def test_two_dimensional_input_raises(self):
        # numpy.full(len(data), numpy.mean(data)) builds a (n,) mean for an
        # (n, k) input, so the demeaning on mean_reversion.py:22 cannot
        # broadcast. Pinned so adding column support is a deliberate change.
        with pytest.raises(ValueError, match="could not be broadcast together"):
            compute_mean_reversion_halflife(numpy.random.normal(0.0, 1.0, (50, 2)))

    def test_dataframe_input_raises(self):
        # A one-column DataFrame is not a Series: numpy.mean returns a length-1
        # Series, and numpy.full cannot fill an array with it.
        df = pandas.DataFrame({"close": numpy.random.normal(0.0, 1.0, 50)})
        with pytest.raises(ValueError):
            compute_mean_reversion_halflife(df)


# ---------------------------------------------------------------------------
# Degenerate and malformed input
# ---------------------------------------------------------------------------


class TestDegenerateInput:
    """Contract of the short/non-finite inputs a notebook can hand over.

    None of these are guarded in the façade; the failures come out of numpy and
    statsmodels with opaque messages. They are pinned so that adding a guard
    clause (a clear ValueError, say) is a deliberate, visible change rather
    than a silent one.
    """

    @pytest.mark.parametrize("x", [[], [3.0]])
    def test_fewer_than_two_points_raises_value_error(self, x):
        # numpy.diff leaves an empty regressand, and statsmodels' add_constant
        # reduces over an empty axis.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError, match="zero-size array to reduction"):
                compute_mean_reversion_halflife(numpy.array(x, dtype=float))

    def test_exactly_two_points_raises_index_error(self):
        # One observation survives the difference, so statsmodels returns a
        # single parameter and the result model's params[0] lookup runs off the
        # end of the list.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(IndexError):
                compute_mean_reversion_halflife(numpy.array([1.0, 2.0]))

    def test_three_points_returns_a_saturated_fit(self):
        # Pinning the garbage: three points leave 2 observations for 2
        # parameters, so the fit interpolates exactly (R² = 1, zero residual
        # degrees of freedom) and every standard error is non-finite — yet a
        # perfectly ordinary-looking half-life comes back.
        # x = [1, 2, 1.5] → X_{t-1} = [-0.5, 0.5] about the mean 1.5 and
        # ΔX = [1, -0.5], so the slope is exactly (-0.5 - 1)/(0.5 + 0.5) = -1.5
        # and t½ = ln2/1.5.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half_life, report, result = compute_mean_reversion_halflife(
                numpy.array([1.0, 2.0, 1.5])
            )
        assert report.nobs == 2.0
        assert report.df_resid == 0.0
        assert result.params[0].est == pytest.approx(-1.5, rel=1e-12)
        assert half_life == pytest.approx(LN2 / 1.5, rel=1e-12)
        assert not numpy.isfinite(result.params[0].err)
        assert not numpy.isfinite(result.param_transforms[0].param.err)

    @pytest.mark.parametrize("bad", [numpy.nan, numpy.inf, -numpy.inf])
    def test_non_finite_values_raise_value_error(self, bad):
        # The mean of a series containing NaN/inf is itself non-finite, so the
        # demeaning poisons *every* observation and statsmodels' missing='drop'
        # discards the lot — leaving the same empty-reduction ValueError as an
        # empty input. Utterly opaque for a price series with one gap.
        x = numpy.r_[numpy.random.normal(0.0, 1.0, 50), bad]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError, match="zero-size array to reduction"):
                compute_mean_reversion_halflife(x)


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------


class TestClosedForm:
    @pytest.mark.parametrize("φ, n", [(0.25, 15), (0.5, 20), (0.9, 60), (0.99, 200)])
    def test_geometric_decay_is_fitted_exactly(self, φ, n):
        # X_t = μ + A φ^t ⇒ ΔX_t = (φ - 1)(X_{t-1} - μ) is an exact straight
        # line in X_{t-1} whatever constant is removed first, so the fit is
        # perfect: slope φ - 1, R² = 1, zero standard error, and
        # t½ = -ln2/(φ - 1) = ln2/(1 - φ).
        μ, A = 3.0, 2.0
        x = μ + A * φ ** numpy.arange(n)
        half_life, report, result = compute_mean_reversion_halflife(x)
        assert half_life == pytest.approx(LN2 / (1.0 - φ), rel=1e-9)
        assert result.params[0].est == pytest.approx(φ - 1.0, rel=1e-9)
        assert result.r2 == pytest.approx(1.0, abs=1e-9)
        assert result.param_transforms[0].param.err == pytest.approx(0.0, abs=1e-8)
        assert result.param_transforms[1].param.err == pytest.approx(0.0, abs=1e-8)
        assert report.nobs == n - 1

    @pytest.mark.parametrize(
        "φ, factor", [(0.2, 2.0118), (0.5, 1.3863), (0.9, 1.0536)]
    )
    def test_reported_half_life_is_a_first_order_approximation(self, φ, factor):
        # Deterministic companion to the round trips: the same noiseless decay
        # X_t = μ + A φ^t, whose deviation from μ provably halves after
        # ln2/(-ln φ) steps (asserted directly below from the series itself).
        # The façade inverts φ = 1 + λ̂ instead of φ = e^{λ̂} and reports
        # ln2/(1 - φ), which is larger by -ln φ/(1 - φ) — a factor of 2.01 at
        # φ = 0.2. A notebook sizing a moving-average window off it uses twice
        # the correct window. Fixing the estimator to -ln2/ln(1 + slope) is
        # what would flip this test, which is the point of pinning it.
        μ, A, n = 3.0, 2.0, 80
        x = μ + A * φ ** numpy.arange(n)
        half_life, _, _ = compute_mean_reversion_halflife(x)

        true_halving = LN2 / (-math.log(φ))
        # verify `true_halving` against the series rather than trusting algebra:
        # the deviation really is halved after that many steps.
        assert A * φ**true_halving == pytest.approx(A / 2.0, rel=1e-12)

        assert half_life == pytest.approx(LN2 / (1.0 - φ), rel=1e-9)
        assert half_life / true_halving == pytest.approx(factor, rel=1e-4)

    def test_half_life_of_a_hand_computed_decay(self):
        # X = 4·(1/2)^t about a zero mean: ΔX = -X/2 exactly, so λ̂ = -1/2 and
        # t½ = 2 ln 2 ≈ 1.386 steps.
        x = 4.0 * 0.5 ** numpy.arange(12)
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(-0.5, rel=1e-9)
        assert half_life == pytest.approx(2.0 * LN2, rel=1e-9)

    def test_hand_computed_four_point_regression(self):
        # x = [0, 2, 1, 4, 3] → X_{t-1} = [0, 2, 1, 4], ΔX = [2, -1, 3, -1].
        # Means 7/4 and 3/4; Σ(x̃-x̄)(Δx-Δx̄) = -33/4 and Σ(x̃-x̄)² = 35/4, so
        # the slope is exactly -33/35 and t½ = 35 ln2 / 33.
        x = numpy.array([0.0, 2.0, 1.0, 4.0, 3.0])
        half_life, report, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(-33.0 / 35.0, rel=1e-12)
        assert half_life == pytest.approx(35.0 * LN2 / 33.0, rel=1e-12)
        assert report.nobs == 4

    def test_divergence_reports_a_negative_half_life(self):
        # φ > 1 runs away from the mean: the slope is φ - 1 > 0 so t½ < 0,
        # the sign the notebooks use to reject a series as non-mean-reverting.
        x = 1.0 + 1.1 ** numpy.arange(30)
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(0.1, rel=1e-9)
        assert half_life == pytest.approx(-LN2 / 0.1, rel=1e-9)

    def test_no_variation_yields_an_infinite_half_life_and_nan_errors(self):
        # Degenerate contract: with X_{t-1} constant the design is singular,
        # pinv returns slope 0 and t½ = -ln2/0 = -∞ (with numpy warnings)
        # rather than raising. The transformed *error* is 0/0 = NaN, so a
        # caller that only checks isfinite(half_life) still gets a NaN in the
        # error column — pin the whole degenerate tuple, not just t½.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half_life, _, result = compute_mean_reversion_halflife(numpy.full(50, 3.0))
        assert result.params[0].est == 0.0
        assert numpy.isneginf(half_life)
        assert numpy.isnan(result.param_transforms[0].param.err)
        assert (
            numpy.isnan(result.param_transforms[1].param.err)
            or result.param_transforms[1].param.err == 0.0
        )

    @pytest.mark.parametrize("n", [20, 50, 101, 200])
    def test_constant_drift_yields_no_reversion(self, n):
        # A pure ramp has ΔX identically 1: no dependence on X_{t-1}, so the
        # slope is a round-off zero and the half-life diverges. Only the
        # *magnitude* is a property of the code — the sign is decided by the
        # last bit of the pinv solve and flips with n (measured: -∞ at n = 20,
        # +6.9e16 at n = 50, +2.1e17 at n = 101, -2.0e18 at n = 200) and would
        # flip again on a different BLAS. Assert what is actually invariant.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half_life, _, result = compute_mean_reversion_halflife(
                numpy.arange(n, dtype=float)
            )
        assert result.params[0].est == pytest.approx(0.0, abs=1e-12)
        assert abs(half_life) > 1e12


# ---------------------------------------------------------------------------
# The demeaning step
# ---------------------------------------------------------------------------


class TestDemeaning:
    def test_estimates_are_translation_invariant(self):
        # The façade removes the sample mean, and an OLS slope with an
        # intercept is invariant to shifting the regressor anyway, so every
        # reported quantity must survive an arbitrary offset.
        x = _ou_path(0.5, 1500, μ=0.0)
        base_hl, _, base = compute_mean_reversion_halflife(x)
        for shift in (-250.0, 3.0, 1.0e4):
            hl, _, shifted = compute_mean_reversion_halflife(x + shift)
            assert hl == pytest.approx(base_hl, rel=1e-8)
            assert shifted.params[0].est == pytest.approx(base.params[0].est, rel=1e-8)
            assert shifted.params[0].err == pytest.approx(base.params[0].err, rel=1e-8)
            assert shifted.const.est == pytest.approx(base.const.est, abs=1e-8)
            assert shifted.r2 == pytest.approx(base.r2, rel=1e-8)

    def test_demeaning_removes_the_intercept_the_raw_fit_keeps(self):
        # Fitted on the raw series the intercept is -slope·μ = λμ ≈ 25; the
        # façade's demeaning collapses it to O(σ/n) while leaving the slope —
        # and therefore the half-life — untouched.
        μ = 50.0
        x = _ou_path(0.5, 1500, μ=μ)
        half_life, _, demeaned = compute_mean_reversion_halflife(x)
        _, raw = ou_impl.compute_mean_half_life_estimate(x)
        assert raw.params[0].est == pytest.approx(demeaned.params[0].est, rel=1e-8)
        assert raw.param_transforms[0].param.est == pytest.approx(half_life, rel=1e-8)
        # The informative version of the intercept check: OLS-with-intercept
        # satisfies const = mean(ΔX) - b·mean(X_{t-1}) identically, so assert
        # that at machine precision (an independent recomputation from the
        # data, not from the fit)…
        identity = numpy.mean(numpy.diff(x)) - raw.params[0].est * numpy.mean(x[:-1])
        assert raw.const.est == pytest.approx(identity, rel=1e-10)
        # …and separately that it sits near the population value -b·μ. The gap
        # is b·(x̄ - μ), about 0.1% of the 25.0 target; over 40 seeds the worst
        # relative deviation was 0.23%, so 0.5% is a real constraint rather
        # than the near-vacuous 5% it replaces.
        assert raw.const.est == pytest.approx(-raw.params[0].est * μ, rel=0.005)
        assert abs(demeaned.const.est) < 1.0e-3 * abs(raw.const.est)
        assert abs(demeaned.const.est) < 3.0 * demeaned.const.err


# ---------------------------------------------------------------------------
# Round trip: simulate with known λ, estimate it back
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("λ, n", [(0.8, 4000), (0.3, 4000), (0.1, 8000)])
    def test_recovers_the_euler_ou_half_life(self, λ, n):
        # SE(λ̂) = √((1 - φ²)/n) with φ = 1 - λ is 1.9%, 3.8% and 4.9% of λ for
        # the three cases. 4.5 SE on λ̂ and 5 SE (relative) on the half-life,
        # which absorbs the convexity of 1/λ̂; the OLS (Hurwicz) bias
        # (1 + 3φ)/n stays below 0.4 SE. Over 120 seeds the worst |z| seen was
        # 2.7, so these bounds are not seed specific.
        #
        # NOTE the target: lib.models.ou.ou is the Euler scheme, an AR(1) with
        # φ = 1 - λΔt, and the estimator inverts that same first-order relation
        # — so ou.mean_halflife(λ) = ln2/λ and the AR(1) target ln2/(1 - φ) are
        # the *same number* and this test cannot see the discretisation error.
        # test_recovers_an_exactly_simulated_ou below is the one that can.
        φ = 1.0 - λ
        ar1_target = LN2 / (1.0 - φ)
        assert ar1_target == pytest.approx(ou.mean_halflife(λ), rel=1e-12)

        x = _ou_path(λ, n)
        half_life, _, result = compute_mean_reversion_halflife(x)
        se = _slope_se(λ, n)
        λ_hat = -result.param_transforms[1].param.est
        assert abs(λ_hat - λ) < 4.5 * se
        assert half_life == pytest.approx(ar1_target, rel=5.0 * se / λ)

    @pytest.mark.parametrize(
        "λ, n, rtol", [(0.8, 4000, 0.128), (0.3, 4000, 0.205), (0.1, 8000, 0.250)]
    )
    def test_recovers_an_exactly_simulated_ou(self, λ, n, rtol):
        # The exact OU transition (φ = e^{-λΔt}) shares no approximation with
        # the estimator. What comes back is the first-order inversion of the
        # true φ, i.e. ln2/(1 - e^{-λ}) — NOT ln2/λ.
        # rtol = 5·SE(λ̂)/|φ - 1| with SE = √((1 - φ²)/n): 12.8%, 20.5%, 25.0%.
        # Over 60 seeds the worst deviation from the target was 6.8%, 10.3% and
        # 9.8% (and 7.3%, 9.3%, 8.9% on the slope), so each bound carries
        # roughly a factor-of-two margin.
        φ = math.exp(-λ)
        target = LN2 / (1.0 - φ)
        x = _exact_ou_path(λ, n)
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(φ - 1.0, rel=rtol)
        assert half_life == pytest.approx(target, rel=rtol)
        # …and the same statement as an explicit bias against continuous-time
        # truth: the reported half-life is high by λ/(1 - e^{-λ}), which is
        # +45%, +16% and +5% for these three λ.
        assert half_life / ou.mean_halflife(λ) == pytest.approx(
            λ / (1.0 - φ), rel=rtol
        )

    def test_exact_ou_exposes_the_discretisation_bias(self):
        # The λ = 0.8 case sharply: on a path that genuinely halves in
        # ln2/0.8 = 0.866 steps the façade reports ≈ 1.26, and the error is far
        # larger than the sampling noise (rel SE ≈ 2.6% at n = 4000; over 60
        # seeds the ratio below stayed in [1.354, 1.536] against a bound of
        # 1.25). This is the failure the Euler round trip cancels out.
        λ, n = 0.8, 4000
        x = _exact_ou_path(λ, n)
        half_life, _, _ = compute_mean_reversion_halflife(x)
        truth = ou.mean_halflife(λ)
        assert half_life / truth > 1.25
        assert half_life == pytest.approx(LN2 / (1.0 - math.exp(-λ)), rel=0.128)

    def test_half_life_is_measured_in_sample_steps(self):
        # dt is hard-coded to 1, so a path sampled at Δt = 0.25 reports the
        # per-step rate λΔt = 0.2 and t½ = ln2/(λΔt) steps — the time half-life
        # divided by Δt. Supplying dt to the underlying estimator converts back
        # to time units exactly.
        λ, Δt, n = 0.8, 0.25, 4000
        x = _ou_path(λ, n, Δt=Δt)
        half_life, _, result = compute_mean_reversion_halflife(x)
        se = _slope_se(λ * Δt, n)
        assert abs(-result.params[0].est - λ * Δt) < 4.5 * se
        assert half_life == pytest.approx(
            ou.mean_halflife(λ) / Δt, rel=5.0 * se / (λ * Δt)
        )
        _, in_time = ou_impl.compute_mean_half_life_estimate(x - x.mean(), dt=Δt)
        assert in_time.param_transforms[0].param.est == pytest.approx(
            half_life * Δt, rel=1e-12
        )
        assert in_time.param_transforms[1].param.est == pytest.approx(
            result.params[0].est / Δt, rel=1e-12
        )

    @pytest.mark.parametrize("λ, n", [(0.8, 4000), (0.3, 4000), (0.1, 8000)])
    def test_reported_errors_match_ar1_theory(self, λ, n):
        # The regression's own standard error must reproduce √((1 - φ²)/n);
        # over 60 seeds the ratio stayed inside [0.95, 1.06]. The transformed
        # half-life error follows by the delta method, evaluated at the true λ
        # (ratio inside [0.82, 1.18] over the same seeds).
        x = _ou_path(λ, n)
        _, _, result = compute_mean_reversion_halflife(x)
        se = _slope_se(λ, n)
        assert result.param_transforms[1].param.err == pytest.approx(se, rel=0.15)
        assert result.param_transforms[0].param.err == pytest.approx(
            LN2 * se / λ**2, rel=0.3
        )

    def test_reported_error_is_calibrated_over_an_ensemble(self):
        # 80 independent paths at λ = 0.5, n = 1000 (σ_λ̂ ≈ 0.027, 5.5%): the
        # z-scores (t½ - ln2/λ)/σ_{t½} should look standard normal. The sd of a
        # sample sd of 80 draws is 1/√158 = 0.080, so ±3 SE is [0.76, 1.24];
        # over 30 independent ensembles the observed sd stayed in [0.81, 1.16],
        # the mean in [-0.38, 0.09] (its own SE is 1/√80 = 0.11) and the ±2σ
        # coverage never fell below 0.90. The window matters: at nsim = 40 the
        # old [0.6, 1.55] bound was ±4–5 SE and a reported error 1.5× too small
        # (z.sd ≈ 1.5, coverage ≈ 0.82) sailed through both it and a 0.8
        # coverage floor. Both are now inside the rejection region.
        λ, n, nsim = 0.5, 1000, 80
        truth = ou.mean_halflife(λ)
        est = numpy.empty(nsim)
        err = numpy.empty(nsim)
        for i in range(nsim):
            half_life, _, result = compute_mean_reversion_halflife(_ou_path(λ, n))
            est[i] = half_life
            err[i] = result.param_transforms[0].param.err
        z = (est - truth) / err
        assert abs(z.mean()) < 0.75
        assert 0.76 < z.std(ddof=1) < 1.24
        assert numpy.mean(numpy.abs(z) < 2.0) >= 0.85
        # the ensemble mean is 9× more precise than a single estimate
        assert est.mean() == pytest.approx(truth, abs=4.0 * err.mean() / math.sqrt(nsim))


# ---------------------------------------------------------------------------
# Discrimination between reverting and non-reverting series
# ---------------------------------------------------------------------------


class TestDiscrimination:
    def test_random_walk_has_no_short_half_life(self):
        # Under a unit root n·slope follows the Dickey-Fuller distribution
        # (1% quantile ≈ -21), so |n·slope| < 70 and |t½| > ln2·n/70 ≈ 20 for
        # n = 2000 on essentially any seed — over 200 seeds the largest
        # |n·slope| was 30 and the smallest |t½| was 46. A strongly reverting
        # path is under 2 steps (range [0.81, 0.95] over the same 200 seeds).
        n = 2000
        walk = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))
        walk_hl, _, walk_result = compute_mean_reversion_halflife(walk)
        assert abs(walk_result.params[0].est) < 70.0 / n
        assert abs(walk_hl) > 20.0
        reverting_hl, _, _ = compute_mean_reversion_halflife(_ou_path(0.8, n))
        assert 0.0 < reverting_hl < 2.0

    def test_slower_reversion_gives_a_longer_half_life(self):
        # The λ are 2× apart while each estimate lands within ~15% of truth at
        # n = 4000, so the ordering never inverted over 60 seeds.
        n = 4000
        λs = (0.8, 0.4, 0.2)
        hl = [compute_mean_reversion_halflife(_ou_path(λ, n))[0] for λ in λs]
        assert hl[0] < hl[1] < hl[2]
        npt.assert_allclose(hl, [ou.mean_halflife(λ) for λ in λs], rtol=0.25)
