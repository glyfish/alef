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
(deterministic, noiseless) and
``TestRoundTrip.test_exact_ou_exposes_the_discretisation_bias``, which simulates
the *exact* AR(1) OU (``φ = e^{-λΔt}``) and pins the resulting bias one-sidedly
at the two ``λ`` where it is larger than the sampling noise.

**The façade removes a constant, never a trend** (``mean_reversion.py:21-22``),
so a deterministic drift — the cointegration spread the notebooks actually feed
it — inflates the reported half-life without any other symptom.
``TestTrendSensitivity`` pins how badly.

The kinds of check below are:

* closed form — a noiseless geometric decay ``X_t = μ + A φ^t`` satisfies
  ``ΔX_t = (φ - 1)(X_{t-1} - μ)`` exactly, so the fit must be perfect and
  return ``t½ = ln 2 / (1 - φ)``; this holds for ``φ < 0`` (oscillating) and
  ``|φ| > 1`` (divergent) too, which is where a ``0 < t½ < small`` screen goes
  wrong; plus one four-point series whose OLS slope is worked out by hand as an
  exact fraction;
* round trip — simulate OU paths with a known ``λ`` and recover the half-life
  inside a tolerance built from the AR(1) slope standard error
  ``√((1 - φ²)/n)``, both for the Euler simulator and for an exact OU;
* contract — tuple shape and types, the transform bookkeeping, translation
  invariance of the demeaning step, ``nobs = n - 1`` from the differencing,
  serialization of the returned result, accepted input containers/dtypes, and
  the degenerate inputs (no variation, too short, NaN/inf, 2-D).  Note that
  non-finite input has *two* contracts, not one: an ndarray raises, while a
  ``pandas.Series`` carrying a NaN silently fits the rows that survive
  ``missing='drop'`` (``TestNonFiniteSeries``).

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
from helpers import present

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
        assert result.r2.est == pytest.approx(report.rsquared, rel=1e-12)

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
        # An ordering pin, not a numeric check: the façade returns
        # param_transforms[0] verbatim (mean_reversion.py:25), so this is the
        # same object — what it asserts is that slot 0 is t½ and slot 1 is λ̂.
        # The numbers are anchored on the statsmodels report below.
        assert half_life is result.param_transforms[0].param.est
        assert λ_est is result.param_transforms[1].param
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
        assert repr(result).startswith("OLSResult(")
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
            half_life, _, result = compute_mean_reversion_halflife(numpy.arange(50, dtype=float))
        assert result.params[0].est == pytest.approx(0.0, abs=1e-12)
        assert abs(half_life) > 1e12

    def test_two_dimensional_input_raises(self):
        # numpy.full(len(data), numpy.mean(data)) builds a (n,) mean for an
        # (n, k) input, so the demeaning on mean_reversion.py:22 cannot
        # broadcast. Pinned so adding column support is a deliberate change.
        with pytest.raises(ValueError, match="could not be broadcast together"):
            compute_mean_reversion_halflife(numpy.random.normal(0.0, 1.0, (50, 2)))

    def test_dataframe_input_raises(self):
        # A one-column DataFrame is not a Series, but the failure is *not* in
        # the mean: numpy.mean(df) on one column returns a numpy.float64 scalar
        # and numpy.full(len(df), scalar) builds a fine (50,) array. It is the
        # subtraction on mean_reversion.py:22 that fails — pandas will not
        # broadcast a length-50 ndarray against a DataFrame's single column and
        # raises "Unable to coerce to Series, length must be 1: given 50".
        # Match the message so a different ValueError from elsewhere (a real
        # guard clause, say) shows up as a failure rather than passing silently.
        df = pandas.DataFrame({"close": numpy.random.normal(0.0, 1.0, 50)})
        assert isinstance(numpy.mean(df), numpy.floating)
        with pytest.raises(ValueError, match="Unable to coerce to Series"):
            compute_mean_reversion_halflife(df)  # pyright: ignore[reportArgumentType]


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
        assert not numpy.isfinite(present(result.param_transforms)[0].param.err)

    @pytest.mark.parametrize("bad", [numpy.nan, numpy.inf, -numpy.inf])
    def test_non_finite_values_in_an_ndarray_raise_value_error(self, bad):
        # numpy.mean of an ndarray containing NaN/inf is itself non-finite, so
        # the demeaning poisons *every* observation and statsmodels'
        # missing='drop' discards the lot — leaving the same empty-reduction
        # ValueError as an empty input. Utterly opaque for a price series with
        # one gap. The container decides: see TestNonFiniteSeries for the
        # pandas path, which does not raise at all.
        x = numpy.r_[numpy.random.normal(0.0, 1.0, 50), bad]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError, match="zero-size array to reduction"):
                compute_mean_reversion_halflife(x)


class TestNonFiniteSeries:
    """A NaN in a ``pandas.Series`` does **not** behave like a NaN in an array.

    ``numpy.mean(series)`` dispatches to ``Series.mean()``, which *skips* NaN,
    so the demeaning is not poisoned; only the one or two regression rows that
    actually touch the gap are non-finite and statsmodels' ``missing='drop'``
    quietly discards exactly those.  The call therefore succeeds, with a
    silently reduced ``nobs`` and no warning.  This is the branch that matters:
    the notebooks feed Series straight from the price loaders, and a single
    missing bar changes the answer without changing the return contract.

    ``±inf`` is different again — a Series mean of ``inf`` *is* propagated, so
    the array behaviour (ValueError) comes back.
    """

    def test_a_nan_in_a_series_silently_drops_the_affected_rows(self):
        # x = [0, 2, 1, NaN, 4, 3]: the mean over the finite entries is 2, and
        # of the five differencing rows the two that touch the gap (lag = 1 and
        # lag = NaN) are dropped, leaving demeaned lags [-2, 0, 2] against
        # ΔX = [2, -1, -1].  Hand OLS: Σ(l-l̄)(Δ-Δ̄) = -6, Σ(l-l̄)² = 8, so the
        # slope is exactly -3/4, t½ = ln2/0.75, the intercept is exactly 0 and
        # R² = 1 - 1.5/6 = 0.75.  Nothing flags the missing bar.
        x = [0.0, 2.0, 1.0, numpy.nan, 4.0, 3.0]
        half_life, report, result = compute_mean_reversion_halflife(pandas.Series(x))
        assert report.nobs == 3.0
        assert result.params[0].est == pytest.approx(-0.75, rel=1e-12)
        assert half_life == pytest.approx(LN2 / 0.75, rel=1e-12)
        assert result.r2.est == pytest.approx(0.75, rel=1e-12)
        assert abs(result.const.est) < 1e-12
        assert numpy.isfinite(present(result.param_transforms)[0].param.err)
        # …and the *identical* data as a bare ndarray raises instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError, match="zero-size array to reduction"):
                compute_mean_reversion_halflife(numpy.array(x))

    def test_one_interior_nan_costs_exactly_two_observations(self):
        # On a real-sized path: n points give n - 1 rows, and one interior NaN
        # kills the row that uses it as the lag and the row whose difference
        # spans it, so nobs = n - 3. The surviving fit is recomputed here from
        # the finite (lag, ΔX) pairs by the covariance formula — an independent
        # route to the same slope, so this pins *which* rows were dropped, not
        # merely how many.
        n, gap = 400, 137
        clean = _ou_path(0.5, n)
        x = clean.copy()
        x[gap] = numpy.nan
        half_life, report, result = compute_mean_reversion_halflife(pandas.Series(x))
        assert report.nobs == n - 3

        centred = x - numpy.nanmean(x)
        lag, dx = centred[:-1], numpy.diff(centred)
        keep = numpy.isfinite(lag) & numpy.isfinite(dx)
        assert keep.sum() == n - 3
        lag_c = lag[keep] - lag[keep].mean()
        slope = numpy.dot(lag_c, dx[keep] - dx[keep].mean()) / numpy.dot(lag_c, lag_c)
        assert result.params[0].est == pytest.approx(slope, rel=1e-9)
        assert half_life == pytest.approx(-LN2 / slope, rel=1e-9)
        # The gap moves the answer on the *same* path, but only by ~one
        # observation's worth — the drop is local, not a wholesale corruption.
        clean_hl, _, _ = compute_mean_reversion_halflife(clean)
        assert half_life == pytest.approx(clean_hl, rel=0.05)

    @pytest.mark.parametrize("bad", [numpy.inf, -numpy.inf])
    def test_an_infinite_value_in_a_series_still_raises(self, bad):
        # Series.mean() skips NaN but not ±inf, so the mean is infinite, the
        # demeaning poisons the whole column and every row is dropped.
        x = pandas.Series(numpy.r_[numpy.random.normal(0.0, 1.0, 50), bad])
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
        assert result.r2.est == pytest.approx(1.0, abs=1e-9)
        assert present(result.param_transforms)[0].param.err == pytest.approx(0.0, abs=1e-8)
        assert present(result.param_transforms)[1].param.err == pytest.approx(0.0, abs=1e-8)
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
        # Verify `true_halving` against the *generated series* rather than
        # against the φ it was derived from: read the decay ratio back out of x
        # (the early terms only — 2·0.2^t underflows the 3.0 offset by t ≈ 24),
        # confirm the constructed series really is that geometric decay, and
        # halve it with the ratio the data supplies. Writing it as
        # A·φ**true_halving instead would be pure algebra: exp(-ln2) = ½ for
        # every φ, touching neither x nor the estimator.
        dev = x - μ
        ratios = dev[1:6] / dev[0:5]
        npt.assert_allclose(ratios, φ, rtol=1e-12)
        ratio = float(ratios[0])
        assert dev[0] * ratio**true_halving == pytest.approx(dev[0] / 2.0, rel=1e-12)

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

    @pytest.mark.parametrize("φ", [-0.5, -0.9, -1.5])
    def test_oscillating_regimes_are_fitted_exactly_too(self, φ):
        # ΔX_t = (φ - 1)(X_{t-1} - μ) is exact for *any* φ, negative included,
        # so the same closed form holds when the deviation alternates sign:
        # slope φ - 1 ∈ (-3, -1), R² = 1 and t½ = ln2/(1 - φ) < ln2. Nothing in
        # the façade restricts the fitted AR(1) coefficient to (0, 1), so a
        # rapidly alternating series reports a *shorter* half-life than any
        # genuinely reverting one — see the divergence trap below.
        μ, A, n = 3.0, 2.0, 30
        x = μ + A * φ ** numpy.arange(n)
        half_life, report, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(φ - 1.0, rel=1e-9)
        assert half_life == pytest.approx(LN2 / (1.0 - φ), rel=1e-9)
        assert result.r2.est == pytest.approx(1.0, abs=1e-9)
        assert report.nobs == n - 1

    def test_an_explosive_oscillation_reports_a_short_positive_half_life(self):
        # The trap the sign convention leaves open: φ = -1.5 diverges — |x - μ|
        # grows without bound — yet slope = -2.5 < -1 gives t½ = ln2/2.5 ≈ 0.277,
        # smaller (and just as positive) as the strongly reverting φ = 0.5 case.
        # A notebook screening on "0 < t½ < small" accepts this series. Only
        # slope > -2 (i.e. |φ| < 1) is actually stationary, and the façade
        # reports nothing about that.
        μ, A, n = 3.0, 2.0, 30
        x = μ + A * (-1.5) ** numpy.arange(n)
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert abs(x[-1] - μ) > 1.0e4 * abs(x[0] - μ)  # it really does explode
        assert result.params[0].est == pytest.approx(-2.5, rel=1e-9)
        assert half_life == pytest.approx(LN2 / 2.5, rel=1e-9)
        assert 0.0 < half_life
        # …strictly shorter than a genuinely (and fast) mean-reverting series.
        reverting, _, _ = compute_mean_reversion_halflife(
            μ + A * 0.5 ** numpy.arange(n)
        )
        assert half_life < reverting

    def test_no_variation_yields_an_infinite_half_life_and_nan_errors(self):
        # Degenerate contract: with X_{t-1} constant the design is singular,
        # pinv returns slope 0 and t½ = -ln2/0 = -∞ (with numpy warnings)
        # rather than raising. The transformed *error* is 0/0 = NaN, so a
        # caller that only checks isfinite(half_life) still gets a NaN in the
        # error column — pin the whole degenerate tuple, not just t½.
        #
        # The RuntimeWarnings are *asserted*, not suppressed: the divide-by-zero
        # out of ou.py's __half_life_transform (-ln2/λ with λ = 0) is the only
        # signal a caller gets that the fit was singular, so a change that stops
        # emitting it — or moves it — must fail here. pytest.warns swallows the
        # rest of the batch (two more RuntimeWarnings from the 1/λ² error
        # transform) the way catch_warnings did.
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            half_life, _, result = compute_mean_reversion_halflife(numpy.full(50, 3.0))
        assert result.params[0].est == 0.0
        assert numpy.isneginf(half_life)
        # Errors: the OLS slope error is an exact zero (the design column has no
        # variation at all), and only the t½ transform is NaN, via 0/0² · 0.
        assert result.params[0].err == 0.0
        assert result.const.err == 0.0
        assert numpy.isnan(present(result.param_transforms)[0].param.err)
        assert present(result.param_transforms)[1].param.err == 0.0
        # R² of the singular fit is NaN, not 0 or 1 — a caller screening a fit
        # on r2 gets neither a pass nor a fail out of any ordinary comparison.
        assert numpy.isnan(result.r2.est)

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
            assert shifted.r2.est == pytest.approx(base.r2.est, rel=1e-8)

    def test_demeaning_removes_the_intercept_the_raw_fit_keeps(self):
        # Fitted on the raw series the intercept is -slope·μ = λμ ≈ 25; the
        # façade's demeaning collapses it to O(σ/n) while leaving the slope —
        # and therefore the half-life — untouched.
        μ = 50.0
        x = _ou_path(0.5, 1500, μ=μ)
        half_life, _, demeaned = compute_mean_reversion_halflife(x)
        _, raw = ou_impl.compute_mean_half_life_estimate(x)
        assert raw.params[0].est == pytest.approx(demeaned.params[0].est, rel=1e-8)
        assert present(raw.param_transforms)[0].param.est == pytest.approx(half_life, rel=1e-8)
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
# Trend sensitivity: a constant is removed, a trend is not
# ---------------------------------------------------------------------------


class TestTrendSensitivity:
    """``mean_reversion.py:21-22`` subtracts the sample *mean* and nothing else.

    A deterministic drift therefore survives into the regressor, where it looks
    like persistence: ΔX picks up a constant the intercept absorbs while
    X_{t-1} carries a ramp uncorrelated with the reverting part, so the fitted
    slope shrinks toward zero and t½ blows up. This is exactly the case the
    mean-reversion notebooks live on — a cointegration spread whose hedge ratio
    is slightly off leaves a small drift in the residual — and it degrades
    silently: R² stays sane, the errors stay finite, only the number is wrong.
    """

    λ, n = 0.5, 4000

    @pytest.fixture
    def path(self):
        return _ou_path(self.λ, self.n), numpy.arange(self.n, dtype=float)

    def test_a_small_drift_inflates_the_half_life(self, path):
        # λ = 0.5 ⇒ truth 1.386 steps, per-step innovation σΔt = 1.0. A drift of
        # 0.002 per step — 0.2% of that innovation, invisible on a plot —
        # returns ≈ 7 (5× too long); 0.01 per step returns ≈ 147 (>100×).
        # Bounds measured over 41 seeds: driftless [1.31, 1.48], 0.002 →
        # [6.45, 7.32], 0.010 → [132, 148], so nothing here is seed specific.
        x, t = path
        truth = ou.mean_halflife(self.λ)
        hl = [compute_mean_reversion_halflife(x + drift * t)[0]
              for drift in (0.0, 0.002, 0.01)]

        assert hl[0] == pytest.approx(truth, rel=0.15)
        assert 4.0 < hl[1] < 10.0
        assert hl[1] > 4.0 * truth
        assert 60.0 < hl[2] < 250.0
        assert hl[0] < hl[1] < hl[2]

    def test_the_drift_shrinks_the_slope_without_spoiling_the_fit(self, path):
        # Why it is silent: the reported slope collapses from ≈ -0.5 toward 0
        # (t½ = -ln2/slope is what explodes), yet every quality signal a caller
        # might screen on stays perfectly healthy — R² in the same ballpark as
        # the clean fit, a finite and *smaller* standard error, and a positive,
        # finite, ordinary-looking half-life.
        x, t = path
        _, _, clean = compute_mean_reversion_halflife(x)
        half_life, report, drifted = compute_mean_reversion_halflife(x + 0.01 * t)

        assert clean.params[0].est == pytest.approx(-self.λ, rel=0.15)
        assert -0.02 < drifted.params[0].est < 0.0
        assert abs(drifted.params[0].est) < 0.1 * abs(clean.params[0].est)
        assert numpy.isfinite(half_life) and half_life > 0.0
        assert numpy.isfinite(drifted.params[0].err)
        assert 0.0 < drifted.r2.est < 1.0
        assert report.nobs == self.n - 1

    def test_removing_the_trend_restores_the_estimate(self, path):
        # Confirms the drift is the whole story: subtracting an OLS line in t
        # (the demeaning the façade does *not* do) recovers the driftless
        # answer to well under a percent. Over 5 seeds the worst gap was 0.2%.
        x, t = path
        baseline, _, _ = compute_mean_reversion_halflife(x)
        y = x + 0.01 * t
        design = numpy.column_stack([numpy.ones_like(t), t])
        coef, *_ = numpy.linalg.lstsq(design, y, rcond=None)
        detrended, _, _ = compute_mean_reversion_halflife(y - design @ coef)
        assert detrended == pytest.approx(baseline, rel=0.02)
        assert detrended == pytest.approx(ou.mean_halflife(self.λ), rel=0.15)


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
        λ_hat = -present(result.param_transforms)[1].param.est
        assert abs(λ_hat - λ) < 4.5 * se
        assert half_life == pytest.approx(ar1_target, rel=5.0 * se / λ)

    @pytest.mark.parametrize(
        "λ, n, rtol", [(0.8, 4000, 0.128), (0.3, 4000, 0.205), (0.1, 8000, 0.250)]
    )
    def test_recovers_an_exactly_simulated_ou(self, λ, n, rtol):
        # Plain round trip against the AR(1) target. The exact OU transition
        # (φ = e^{-λΔt}) shares no approximation with the estimator, so what
        # must come back is the first-order inversion of the *true* φ, i.e.
        # slope = φ - 1 and t½ = ln2/(1 - e^{-λ}).
        # rtol = 5·SE(λ̂)/|φ - 1| with SE = √((1 - φ²)/n): 12.8%, 20.5%, 25.0%.
        # Over 41 seeds the worst deviation from the target was 6.7%, 13.9% and
        # 22.3% (slope: 6.3%, 12.2%, 18.3%), so every bound holds with margin.
        #
        # Deliberately NOT a claim about ln2/λ: at λ = 0.3 and λ = 0.1 the two
        # candidate targets differ by only 15.7% and 5.1%, well inside these
        # tolerances — and at λ = 0.1 inside the sampling noise itself (the
        # measured ratio to ln2/λ ranged over [0.977, 1.286] across those
        # seeds, i.e. it lands below 1 as readily as above). Discriminating the
        # two is test_exact_ou_exposes_the_discretisation_bias' job, at the λ
        # and n where it can actually be done.
        φ = math.exp(-λ)
        target = LN2 / (1.0 - φ)
        x = _exact_ou_path(λ, n)
        half_life, _, result = compute_mean_reversion_halflife(x)
        assert result.params[0].est == pytest.approx(φ - 1.0, rel=rtol)
        assert half_life == pytest.approx(target, rel=rtol)

    @pytest.mark.parametrize(
        "λ, n, floor, rtol", [(0.8, 4000, 1.25, 0.128), (0.3, 34000, 1.09, 0.071)]
    )
    def test_exact_ou_exposes_the_discretisation_bias(self, λ, n, floor, rtol):
        # The discriminating cases, stated one-sidedly against continuous-time
        # truth: on a path that genuinely halves in ln2/λ the façade reports
        # ln2/(1 - e^{-λ}), high by the factor λ/(1 - e^{-λ}) = 1.453 at
        # λ = 0.8 and 1.157 at λ = 0.3. An estimator "fixed" to -ln2/ln(1 +
        # slope) returns the truth and fails these floors, which is the point.
        #
        # n is chosen so the bias clears the sampling noise. Over 41 seeds the
        # ratio below stayed in [1.376, 1.550] at λ = 0.8, n = 4000 and in
        # [1.128, 1.191] at λ = 0.3, n = 34000 — the floors sit ~10% and ~3.5%
        # under those minima. λ = 0.3 needs the long path: at n = 4000 the same
        # ratio ranged over [1.049, 1.318] and a 1.09 floor would be a coin
        # toss. λ = 0.1 is left out entirely (bias 5% vs a spread of ±25%);
        # test_recovers_an_exactly_simulated_ou covers it as a round trip only.
        x = _exact_ou_path(λ, n)
        half_life, _, _ = compute_mean_reversion_halflife(x)
        assert half_life / ou.mean_halflife(λ) > floor
        assert half_life == pytest.approx(LN2 / (1.0 - math.exp(-λ)), rel=rtol)

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
        assert present(in_time.param_transforms)[0].param.est == pytest.approx(
            half_life * Δt, rel=1e-12
        )
        assert present(in_time.param_transforms)[1].param.est == pytest.approx(
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
        assert present(result.param_transforms)[1].param.err == pytest.approx(se, rel=0.15)
        assert present(result.param_transforms)[0].param.err == pytest.approx(
            LN2 * se / λ**2, rel=0.3
        )

    def test_reported_error_is_calibrated_over_an_ensemble(self):
        # 150 independent paths at λ = 0.5, n = 1000 (σ_λ̂ ≈ 0.027, 5.5%): the
        # z-scores (t½ - ln2/λ)/σ_{t½} should look standard normal. The sd of a
        # sample sd of 150 draws is 1/√298 = 0.058, so ±3 SE is [0.83, 1.17];
        # over 21 independent ensembles the observed sd stayed in [0.94, 1.09],
        # the mean in [-0.29, 0.03] (its own SE is 1/√150 = 0.082), the ±2σ
        # coverage never fell below 0.93 and the last assertion's margin never
        # exceeded 0.61 of its bound. nsim was raised from 80 precisely to
        # afford this window: at 80 the same ±3 SE window is [0.76, 1.24], and
        # a sd bound that wide self-rejects ≈0.3% of runs while catching less.
        # The constraint still bites — a reported error 1.5× too small
        # (z.sd ≈ 1.5, coverage ≈ 0.82) fails both the sd window and the 0.85
        # coverage floor.
        λ, n, nsim = 0.5, 1000, 150
        truth = ou.mean_halflife(λ)
        est = numpy.empty(nsim)
        err = numpy.empty(nsim)
        for i in range(nsim):
            half_life, _, result = compute_mean_reversion_halflife(_ou_path(λ, n))
            est[i] = half_life
            err[i] = present(result.param_transforms)[0].param.err
        z = (est - truth) / err
        assert abs(z.mean()) < 0.75
        assert 0.83 < z.std(ddof=1) < 1.17
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
