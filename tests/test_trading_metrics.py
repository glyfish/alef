"""
Tests for ``lib.trading.metrics`` — the rolling z-score and rolling standard
deviation that drive the mean-reversion trading strategies:

    Z_i = (y_{i+w-1} - mean(y_i..y_{i+w-1})) / std_pop(y_i..y_{i+w-1})

Three kinds of test, in priority order:

1. CLOSED FORM on ramps, spikes, sinusoids and textbook samples, where the
   population mean/std of the window is known exactly.
2. ROUND TRIP on simulated data with known parameters: white noise with known
   σ, a volatility regime change, and a stationary AR(1) whose stationary
   standard deviation σ/sqrt(1-φ²) must come back out of ``compute_std`` (and
   whose mean reversion must show up as a negative correlation between the
   z-score and the next increment).
3. CONTRACT: the ``(t, values)`` tuple shape and time alignment, the
   ``std == 0`` branch, the empty-input guard, return types, aliasing of the
   returned time array, package re-exports, and agreement with the independent
   ``pandas`` and ``lib.stats`` (cumulative-sum) rolling implementations.

Known library defects are pinned with ``@pytest.mark.xfail(strict=True)``
asserting the CORRECT behaviour, so the marker turns into an XPASS failure the
day the defect is fixed. They are, with the fix that would flip each one:

* ``zscore`` swallows non-finite windows: ``std`` is NaN, ``std > 0`` is False,
  so the degenerate branch returns ``0.0`` — "the price sits exactly on its
  rolling mean", a tradeable flat signal — instead of propagating NaN. (Fix at
  ``metrics.py:88``: return NaN when ``std`` is not finite, keeping ``0.0`` for
  the genuinely constant window where ``std == 0``.)
* ``zscore`` returns ``numpy.float64`` from the dividing branch though it is
  annotated and documented ``-> float``; the sibling ``std`` already wraps its
  return in ``float()``.
* ``zscore``/``compute_zscore`` cannot take a ``pandas.Series`` — ``samples[-1]``
  is label lookup on a ``RangeIndex`` and raises ``KeyError: -1`` — although
  quotes reach the strategies from the CSV/DB loaders as Series, and
  ``compute_std`` accepts them happily.
* neither ``compute_`` facade validates ``len(time) == len(data)`` or
  ``window > 0``: ``values`` is sized from ``len(data)`` while ``time`` is
  sliced from the ``time`` argument, so a mismatched ``(t, values)`` pair is
  returned silently.

Every simulation draws from numpy's global RNG, which ``conftest.py`` reseeds
before each test.
"""

import math
import warnings

import numpy
import pandas
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import lib.trading
from lib.data.impl import stats as dstats
from lib.trading import metrics
from lib.trading.metrics import compute_std, compute_zscore, std, zscore


# ---------------------------------------------------------------------------
# Independent reference implementations / helpers
# ---------------------------------------------------------------------------

def _pandas_rolling_std(data: numpy.ndarray, window: int) -> numpy.ndarray:
    """
    Population (ddof=0) rolling std via pandas — independent of lib.

    The warm-up is sliced off by position rather than with ``.dropna()``: pandas
    emits NaN for a zero-std window (0/0) where the library returns 0.0, and
    dropping NaNs would silently shorten the reference array, turning a value
    disagreement into a confusing shape error inside ``assert_allclose``.
    """
    return pandas.Series(data).rolling(window).std(ddof=0).to_numpy()[window - 1:]


def _pandas_rolling_mean_deviation(data: numpy.ndarray, window: int) -> numpy.ndarray:
    """x_{i+w-1} - mean(window_i) via pandas — an outside reference for z·s."""
    s = pandas.Series(data)
    return (s - s.rolling(window).mean()).to_numpy()[window - 1:]


def _pandas_rolling_zscore(data: numpy.ndarray, window: int) -> numpy.ndarray:
    """Rolling z-score of the window's last value via pandas — independent of lib."""
    s = pandas.Series(data)
    mean = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return ((s - mean) / sd).to_numpy()[window - 1:]


def _ramp_zscore(window: int) -> float:
    """
    z-score of the last point of an arithmetic progression of length w:
    deviation (w-1)/2 over population std sqrt((w²-1)/12) = sqrt(3(w-1)/(w+1)).
    """
    return math.sqrt(3.0 * (window - 1) / (window + 1))


def _ramp_std(window: int) -> float:
    """Population std of an arithmetic progression of length w with unit step."""
    return math.sqrt((window**2 - 1) / 12.0)


def _ar1(phi: float, sigma: float, n: int) -> numpy.ndarray:
    """
    Stationary AR(1): x_t = φ x_{t-1} + σ ε_t, started from its stationary
    distribution N(0, σ²/(1-φ²)) so no burn-in is needed.
    """
    eps = numpy.random.normal(0.0, sigma, n)
    x = numpy.zeros(n)
    x[0] = numpy.random.normal(0.0, sigma / math.sqrt(1.0 - phi**2))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------

class TestZscore:

    def test_ramp_closed_form(self):
        # mean 3, population std sqrt(2), last value 5 -> 2/sqrt(2) = sqrt(2)
        assert zscore(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(math.sqrt(2.0))

    def test_uses_population_std_not_sample_std(self):
        # [0, 2]: mean 1, population std 1 -> z = 1. Sample std (ddof=1) would
        # give sqrt(2) and z = 1/sqrt(2), so this pins the ddof convention.
        assert zscore(numpy.array([0.0, 2.0])) == pytest.approx(1.0)

    def test_uses_last_sample_as_test_value(self):
        # Same multiset, reversed order: only the last value changes. This pins
        # the documented "oldest first, newest last" ordering convention.
        assert zscore(numpy.array([5.0, 4.0, 3.0, 2.0, 1.0])) == pytest.approx(-math.sqrt(2.0))

    @pytest.mark.parametrize("window", [2, 5, 10, 50])
    def test_trailing_spike_closed_form(self, window):
        # zeros with a trailing 1: mean 1/w, var (w-1)/w², so z = sqrt(w-1),
        # which is also the maximum attainable |z| for w points.
        samples = numpy.zeros(window)
        samples[-1] = 1.0
        assert zscore(samples) == pytest.approx(math.sqrt(window - 1))

    @pytest.mark.parametrize("value", [0.0, 3.5, -2.0])
    @pytest.mark.parametrize("n", [1, 2, 7])
    def test_constant_samples_return_zero(self, value, n):
        # std == 0 branch: must not divide by zero.
        result = zscore(numpy.full(n, value))
        assert result == 0.0
        # This branch returns the Python literal 0.0 and would keep returning a
        # Python float if the dividing branch were wrapped in float(), so the
        # strict type check here is safe in both worlds.
        assert type(result) is float

    def test_empty_samples_raise(self):
        # guarded by lib.utils.verify_condition
        with pytest.raises(Exception, match="No samples to compute z-score"):
            zscore(numpy.array([]))

    @pytest.mark.parametrize("bad", [numpy.inf, -numpy.inf])
    def test_std_propagates_an_infinite_sample(self, bad):
        # The half of the divergence that is already correct: std of a window
        # holding ±inf is NaN (inf - inf) and the facade hands that straight back.
        samples = numpy.array([1.0, 2.0, bad])
        with numpy.errstate(invalid="ignore"):
            result = std(samples)
        assert math.isnan(result)
        assert type(result) is float

    @pytest.mark.parametrize("bad", [numpy.inf, -numpy.inf])
    @pytest.mark.xfail(strict=True,
                       reason="zscore swallows non-finite samples: mean is ±inf, std is NaN "
                              "(inf - inf), and `std > 0` is False for NaN, so metrics.py:88 takes "
                              "the degenerate branch and returns 0.0 — 'the price sits exactly on "
                              "its rolling mean', a tradeable flat signal — instead of propagating "
                              "the non-finite input the way std() does. Fix: return NaN when std "
                              "is not finite, reserving the 0.0 branch for std == 0")
    def test_infinite_sample_should_propagate_not_be_silently_zero(self, bad):
        samples = numpy.array([1.0, 2.0, bad])
        with numpy.errstate(invalid="ignore"):
            result = zscore(samples)
        assert math.isnan(result)

    def test_affine_invariance_and_sign_flip(self):
        # z is invariant under x -> a*x + b for a > 0 and odd under a < 0.
        x = numpy.random.normal(0.0, 1.0, 40)
        z = zscore(x)
        assert zscore(3.0 * x + 7.0) == pytest.approx(z)
        assert zscore(-x) == pytest.approx(-z)

    def test_return_is_a_float_in_both_branches(self):
        # The contract from the signature and docstring (metrics.py:66 `-> float`)
        # — true today (numpy.float64 subclasses float) and after the library is
        # made to return a Python float.
        assert isinstance(zscore(numpy.array([1.0, 2.0, 4.0])), float)
        assert isinstance(zscore(numpy.array([1.0, 1.0])), float)

    @pytest.mark.xfail(strict=True,
                       reason="zscore is annotated and documented `-> float` (metrics.py:66) but "
                              "the dividing branch returns numpy.float64; only the std == 0 guard "
                              "returns a Python float. The sibling std() already wraps its return "
                              "in float() (metrics.py:106). Fix: `return float((val - mean) / std)`")
    def test_return_type_should_be_python_float_in_both_branches(self):
        assert type(zscore(numpy.array([1.0, 2.0, 4.0]))) is float
        assert type(zscore(numpy.array([1.0, 1.0]))) is float

    @pytest.mark.xfail(strict=True,
                       reason="zscore cannot take a pandas.Series: samples[-1] (metrics.py:86) is "
                              "label-based lookup on a default RangeIndex and raises KeyError: -1. "
                              "Quotes reach the strategies from the CSV/DB loaders as Series, and "
                              "std() handles them fine. Fix: use samples.iloc[-1] / "
                              "numpy.asarray(samples)[-1]")
    def test_pandas_series_input(self):
        # mean 3, population std sqrt(2), last value 5 -> sqrt(2), the same
        # closed form as test_ramp_closed_form on the numpy array.
        assert zscore(pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(math.sqrt(2.0))

    @pytest.mark.parametrize("n", [2, 5, 30])
    def test_bounded_by_population_maximum(self, n):
        # For any n points, |x_last - mean| / std_pop <= sqrt(n-1) (attained by
        # the trailing-spike sample above). The bound alone is an algebraic
        # identity — it also holds trivially whenever the std == 0 guard fires —
        # so the complementary check below pins that heavy-tailed draws actually
        # push |z| up near the bound. A uniformly shrunken z-score fails there.
        bound = math.sqrt(n - 1)
        largest = 0.0
        for _ in range(200):
            x = numpy.random.standard_cauchy(n)
            z = abs(zscore(x))
            assert z <= bound + 1e-12
            largest = max(largest, z)
        # A single Cauchy outlier dominates its window and drives |z| to
        # sqrt(n-1) — but only when the outlier is the newest sample, which
        # happens for roughly 1 draw in n, hence 200 draws rather than 50. Over
        # 1000 seeds the smallest max|z|/bound seen is 0.59 (n = 30), so 0.5 is
        # a safe floor that a uniformly halved z-score would still break.
        assert largest > 0.5 * bound

    @pytest.mark.parametrize("w", [2, 10, 50])
    def test_iid_normal_windows_are_approximately_standard_normal(self, w):
        # Independent windows of w iid N(μ, σ²) draws. The tested point is part
        # of its own window's mean/std, and the w z-scores of a window satisfy
        # Σ_i z_i² = Σ_i (x_i-m)² / [(1/w) Σ_j (x_j-m)²] = w identically. The
        # draws are exchangeable, so E[z²] = 1 EXACTLY for every w (and E[z] = 0
        # by symmetry) — independent of the window size, which is why w is
        # parametrized here. With 5000 windows SE(mean) ≈ 1/sqrt(5000) ≈ 0.014
        # and SE(var) ≈ sqrt(2/5000) ≈ 0.020, so abs=0.15 is ~7 SE — comfortably
        # seed independent.
        nwin = 5000
        draws = numpy.random.normal(10.0, 3.0, (nwin, w))
        z = numpy.array([zscore(row) for row in draws])
        assert z.mean() == pytest.approx(0.0, abs=0.1)
        assert z.var() == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# std
# ---------------------------------------------------------------------------

class TestStd:

    def test_textbook_closed_form(self):
        # Classic example: mean 5, squared deviations sum 32, population var 4.
        assert std(numpy.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])) == pytest.approx(2.0)

    def test_population_not_sample(self):
        # [0, 2]: population std 1.0; sample std (ddof=1) would be sqrt(2).
        assert std(numpy.array([0.0, 2.0])) == pytest.approx(1.0)

    @pytest.mark.parametrize("window", [2, 5, 12])
    def test_ramp_closed_form(self, window):
        assert std(numpy.arange(window, dtype=float)) == pytest.approx(_ramp_std(window))

    @pytest.mark.parametrize("samples", [numpy.array([4, 4, 4]), numpy.array([2.5]), numpy.zeros(10)])
    def test_constant_or_single_sample_is_zero(self, samples):
        assert std(samples) == 0.0

    def test_returns_python_float_even_for_integer_input(self):
        result = std(numpy.array([1, 2, 3, 4]))
        assert type(result) is float
        assert result == pytest.approx(math.sqrt(1.25))

    def test_empty_samples_return_nan_while_zscore_raises(self):
        # The two scalar facades guard their empty input asymmetrically: zscore
        # has an explicit verify_condition, std has none and degrades to
        # numpy's nan — a silent nan into the strategies rather than a raise.
        # Both halves are pinned here. Whether numpy also emits a RuntimeWarning
        # for the empty slice is numpy's business, not part of this module's
        # contract, so the warning is silenced rather than asserted.
        with numpy.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = std(numpy.array([]))
        assert math.isnan(result)
        assert type(result) is float
        with pytest.raises(Exception, match="No samples to compute z-score"):
            zscore(numpy.array([]))

    def test_pandas_series_input(self):
        # std takes a Series (numpy.std handles it) where zscore raises
        # KeyError: -1 — see TestZscore.test_pandas_series_input. Quote data
        # arrives from the loaders as a Series, so this asymmetry is load bearing.
        assert std(pandas.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])) == pytest.approx(2.0)

    def test_scale_and_shift(self):
        x = numpy.random.normal(0.0, 1.0, 100)
        s = std(x)
        assert std(-2.5 * x + 100.0) == pytest.approx(2.5 * s)
        assert std(x + 1e3) == pytest.approx(s, rel=1e-9)

    def test_white_noise_round_trip(self):
        # n = 4000 draws: SE of the sample std ≈ σ/sqrt(2n) ≈ 0.028 for σ = 2.5,
        # so a 5 % relative tolerance (0.125) is ~4.5 SE — seed independent.
        sigma = 2.5
        x = numpy.random.normal(1.0, sigma, 4000)
        assert std(x) == pytest.approx(sigma, rel=0.05)


# ---------------------------------------------------------------------------
# compute_zscore
# ---------------------------------------------------------------------------

class TestComputeZscore:

    def test_output_shapes_and_time_alignment(self):
        n, w = 50, 7
        time = numpy.linspace(0.0, 49.0, n)
        data = numpy.random.normal(0.0, 1.0, n)
        result = compute_zscore(time, data, w)
        assert isinstance(result, tuple) and len(result) == 2
        t, z = result
        assert isinstance(t, numpy.ndarray) and isinstance(z, numpy.ndarray)
        assert t.shape == z.shape == (n - w + 1,)
        assert z.dtype == numpy.float64
        # each z-score is stamped with the time of the *last* point of its window
        assert_array_equal(t, numpy.arange(w - 1, n, dtype=float))
        assert t[-1] == time[-1]

    def test_datetime_time_passes_through(self):
        n, w = 12, 4
        time = numpy.arange("2024-01-01", "2024-01-13", dtype="datetime64[D]")
        data = numpy.random.normal(0.0, 1.0, n)
        t, z = compute_zscore(time, data, w)
        assert t.dtype == time.dtype
        assert t[0] == numpy.datetime64("2024-01-04")
        assert t[-1] == numpy.datetime64("2024-01-12")
        assert len(z) == n - w + 1
        # the datetime stamps must not disturb the values themselves
        assert_allclose(z, _pandas_rolling_zscore(data, w), rtol=1e-10)

    def test_returned_time_aliases_the_input_time_array(self):
        # `time[window - 1:]` is a numpy view, not a copy, so the returned stamps
        # track later mutations of the caller's array while the values array is
        # freshly built and does not. Pinned as the current (plain numpy slice)
        # semantics: if the library ever starts copying defensively, this is the
        # test that says so.
        n, w = 12, 3
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        t, z = compute_zscore(time, data, w)
        assert numpy.shares_memory(t, time)
        assert not numpy.shares_memory(z, data)
        time[w - 1] = -999.0
        assert t[0] == -999.0

    @pytest.mark.xfail(strict=True,
                       reason="compute_zscore cannot take a pandas.Series: the per-window slice is "
                              "still a Series and zscore's samples[-1] (metrics.py:86) is label "
                              "lookup on a RangeIndex, so it raises KeyError: -1. compute_std "
                              "accepts the same input. Fix: samples.iloc[-1] / numpy.asarray()")
    def test_pandas_series_data(self):
        n, w = 40, 6
        time = numpy.arange(n, dtype=float)
        values = numpy.random.normal(0.0, 1.0, n)
        _, z = compute_zscore(time, pandas.Series(values), w)
        assert_allclose(numpy.asarray(z), _pandas_rolling_zscore(values, w), rtol=1e-10)

    @pytest.mark.parametrize("window", [2, 5, 10, 25])
    def test_linear_ramp_closed_form(self, window):
        n = 60
        time = numpy.arange(n, dtype=float)
        up = 0.3 * time + 5.0
        _, z_up = compute_zscore(time, up, window)
        assert_allclose(z_up, numpy.full(n - window + 1, _ramp_zscore(window)), rtol=1e-10)
        _, z_down = compute_zscore(time, -up, window)
        assert_allclose(z_down, -z_up, rtol=1e-10)

    def test_single_spike_hand_computed_vector(self):
        # zeros with one unit spike at index 9, window 5:
        #   windows ending before the spike  -> constant -> 0 (std == 0 branch)
        #   window ending at the spike       -> sqrt(w-1) = 2
        #   windows containing the spike but ending on a zero:
        #       mean 0.2, std sqrt(0.2*0.8) = 0.4 -> z = -0.2/0.4 = -0.5
        #   windows after the spike          -> 0
        n, w = 20, 5
        data = numpy.zeros(n)
        data[9] = 1.0
        t, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
        expected = numpy.concatenate([numpy.zeros(5), [2.0], numpy.full(4, -0.5), numpy.zeros(6)])
        assert_allclose(z, expected, atol=1e-12)
        assert t[5] == 9.0

    def test_sinusoid_with_window_equal_to_period(self):
        # x_k = B + A sin(2πk/w). Over any w consecutive samples the mean is
        # exactly B and the population std is exactly A/sqrt(2), so
        # z[i] = sqrt(2) * sin(2π(i+w-1)/w) — independent of A and B.
        w, n, A, B = 24, 240, 3.0, 100.0
        k = numpy.arange(n, dtype=float)
        x = B + A * numpy.sin(2.0 * numpy.pi * k / w)
        _, z = compute_zscore(k, x, w)
        expected = math.sqrt(2.0) * numpy.sin(2.0 * numpy.pi * k[w - 1:] / w)
        assert_allclose(z, expected, atol=1e-10)

    def test_matches_pandas_rolling(self):
        n, w = 400, 20
        data = numpy.random.normal(2.0, 0.5, n)
        _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
        assert_allclose(z, _pandas_rolling_zscore(data, w), rtol=1e-10)

    def test_matches_vectorised_lib_stats_implementation(self):
        # lib.data.impl.stats.compute_zscore goes through lib.stats.moving_avg /
        # moving_std, which use cumulative sums; the per-window loop here is an
        # independent code path and must agree on every window with std > 0.
        n, w = 500, 30
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        t_loop, z_loop = compute_zscore(time, data, w)
        t_vec, z_vec = dstats.compute_zscore(time, data, w)
        assert_array_equal(t_loop, t_vec)
        assert_allclose(z_loop, z_vec, rtol=1e-8, atol=1e-8)

    def test_sign_follows_deviation_from_window_mean(self):
        # The reference deviation comes from pandas, not from a numpy.mean over
        # the same slice: recomputing the library's own expression would make
        # the assertion "dividing by a positive float preserves the sign",
        # which no implementation can fail.
        n, w = 300, 15
        data = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))  # random walk
        _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
        deviation = _pandas_rolling_mean_deviation(data, w)
        # a random walk never lands exactly on its rolling mean, so no window is
        # at risk of a sign flip from the two paths' last-bit disagreement
        assert numpy.all(numpy.abs(deviation) > 1e-9)
        assert_array_equal(numpy.sign(z), numpy.sign(deviation))

    def test_bounded_by_population_maximum(self):
        # The Samuelson/Cauchy-Schwarz bound holds for every real input, so on
        # its own it cannot fail; the second assertion is the one with teeth —
        # heavy tails must actually drive |z| towards the bound, catching a
        # z-score that is systematically too small (or one flattened to 0 by
        # the degenerate branch).
        n, w = 500, 10
        data = numpy.random.standard_t(2, n)  # heavy tails
        _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
        bound = math.sqrt(w - 1)
        assert numpy.all(numpy.abs(z) <= bound + 1e-12)
        assert numpy.max(numpy.abs(z)) > 0.5 * bound

    def test_window_one_is_identically_zero(self):
        # a one-point window has zero std, so every value hits the guard
        n = 30
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        t, z = compute_zscore(time, data, 1)
        assert_array_equal(t, time)
        assert_array_equal(z, numpy.zeros(n))

    def test_window_equal_to_length_gives_single_value(self):
        time = numpy.array([10.0, 11.0, 12.0, 13.0, 14.0])
        data = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, z = compute_zscore(time, data, 5)
        assert_array_equal(t, [14.0])
        assert_allclose(z, [math.sqrt(2.0)])

    def test_window_longer_than_series_returns_empty(self):
        time = numpy.arange(5, dtype=float)
        data = numpy.ones(5)
        t, z = compute_zscore(time, data, 8)
        assert t.shape == (0,)
        assert z.shape == (0,)

    def test_window_one_longer_than_series_is_the_exact_npts_zero_boundary(self):
        # len(data) == window - 1 makes npts exactly 0 — the last window size
        # that still produces no output, one step from the single-window case in
        # test_window_equal_to_length_gives_single_value. The tests above only
        # exercise npts < 0, where range() is empty for a different reason.
        t, z = compute_zscore(numpy.arange(5, dtype=float), numpy.arange(5, dtype=float), 6)
        assert t.shape == (0,)
        assert z.shape == (0,)
        assert z.dtype == numpy.float64

    @pytest.mark.parametrize("window", [0, -1], ids=["zero", "negative"])
    def test_non_positive_window_raises(self, window):
        # compute_zscore rejects a non-positive window, though only by accident:
        # nothing validates `window`, but the slices data[i:i+window] go empty
        # (immediately for 0, from the second window for -1) and zscore's
        # verify_condition guard fires. The raise is asserted without matching
        # the message — "No samples to compute z-score" today — so that adding a
        # real window check with its own message keeps this test green.
        # compute_std, which has no such guard, silently returns a mismatched
        # (t, values) pair of NaNs for the same input; see its xfail below.
        with pytest.raises(Exception):
            compute_zscore(numpy.arange(10, dtype=float), numpy.random.normal(0.0, 1.0, 10), window)

    def test_empty_data_returns_empty_instead_of_raising(self):
        # n = 0 drives npts negative, range() is then empty and the function
        # returns empty arrays rather than reporting that there is nothing to
        # compute — the mirror image of the window = 0 guard above.
        t, z = compute_zscore(numpy.array([]), numpy.array([]), 5)
        assert t.shape == (0,)
        assert z.shape == (0,)
        assert z.dtype == numpy.float64

    def test_integer_data_gives_float_zscores(self):
        n, w = 20, 4
        data = numpy.random.randint(0, 100, n)
        _, z = compute_zscore(numpy.arange(n), data, w)
        assert z.dtype == numpy.float64
        assert_allclose(z, _pandas_rolling_zscore(data.astype(float), w), rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_std
# ---------------------------------------------------------------------------

class TestComputeStd:

    def test_output_shapes_and_time_alignment(self):
        n, w = 64, 9
        time = numpy.linspace(100.0, 163.0, n)
        data = numpy.random.normal(0.0, 1.0, n)
        result = compute_std(time, data, w)
        assert isinstance(result, tuple) and len(result) == 2
        t, s = result
        assert isinstance(t, numpy.ndarray) and isinstance(s, numpy.ndarray)
        assert t.shape == s.shape == (n - w + 1,)
        assert s.dtype == numpy.float64
        assert_array_equal(t, numpy.arange(100.0 + w - 1, 164.0))
        assert t[-1] == time[-1]
        assert numpy.all(s >= 0.0)

    @pytest.mark.parametrize("slope, window", [(1.0, 2), (0.5, 5), (-2.0, 10), (3.0, 25)])
    def test_linear_ramp_closed_form(self, slope, window):
        n = 60
        time = numpy.arange(n, dtype=float)
        _, s = compute_std(time, slope * time - 4.0, window)
        assert_allclose(s, numpy.full(n - window + 1, abs(slope) * _ramp_std(window)), rtol=1e-10)

    def test_single_spike_hand_computed_vector(self):
        # zeros with a unit spike: every window containing the spike has
        # population variance (w-1)/w², i.e. std sqrt(w-1)/w; all others are 0.
        n, w = 20, 5
        data = numpy.zeros(n)
        data[9] = 1.0
        _, s = compute_std(numpy.arange(n, dtype=float), data, w)
        spike_std = math.sqrt(w - 1) / w
        expected = numpy.concatenate([numpy.zeros(5), numpy.full(5, spike_std), numpy.zeros(6)])
        assert_allclose(s, expected, atol=1e-12)

    def test_sinusoid_with_window_equal_to_period(self):
        # population std over one full period of A sin(.) is exactly A/sqrt(2)
        w, n, A = 24, 240, 3.0
        k = numpy.arange(n, dtype=float)
        x = 50.0 + A * numpy.sin(2.0 * numpy.pi * k / w)
        _, s = compute_std(k, x, w)
        assert_allclose(s, numpy.full(n - w + 1, A / math.sqrt(2.0)), rtol=1e-10)

    def test_matches_pandas_rolling_population_std(self):
        n, w = 400, 20
        data = numpy.random.normal(-1.0, 4.0, n)
        _, s = compute_std(numpy.arange(n, dtype=float), data, w)
        assert_allclose(s, _pandas_rolling_std(data, w), rtol=1e-10)

    def test_matches_vectorised_lib_stats_implementation(self):
        # cumulative-sum moving_std (lib.stats) vs the per-window loop here
        n, w = 500, 30
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        t_loop, s_loop = compute_std(time, data, w)
        t_vec, s_vec = dstats.compute_moving_std(time, data, w)
        assert_array_equal(t_loop, t_vec)
        assert_allclose(s_loop, s_vec, rtol=1e-8, atol=1e-8)

    def test_reversed_series_matches_the_reversed_pandas_reference(self):
        # Self-symmetry (compute_std(data[::-1]) == compute_std(data)[::-1]) is
        # an identity of numpy.std's order invariance and cannot fail for any
        # implementation, so the reversed series is checked against an OUTSIDE
        # rolling reference instead: that does catch a window-alignment error,
        # which reverses into a different offset. The self-symmetry is asserted
        # alongside only because it is free once the reference is in hand.
        n, w = 200, 12
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        _, forward = compute_std(time, data, w)
        _, backward = compute_std(time, data[::-1], w)
        assert_allclose(backward, _pandas_rolling_std(data[::-1], w), rtol=1e-10, atol=1e-12)
        assert_allclose(backward, forward[::-1], rtol=1e-12)

    def test_pandas_series_data(self):
        # compute_std accepts a Series (numpy.flip/numpy.std cope with it) where
        # compute_zscore raises KeyError: -1 — see TestComputeZscore's xfail.
        n, w = 40, 6
        time = numpy.arange(n, dtype=float)
        values = numpy.random.normal(0.0, 1.0, n)
        t, s = compute_std(time, pandas.Series(values), w)
        assert_array_equal(t, time[w - 1:])
        assert_allclose(numpy.asarray(s), _pandas_rolling_std(values, w), rtol=1e-10, atol=1e-12)

    def test_window_one_is_identically_zero(self):
        n = 30
        time = numpy.arange(n, dtype=float)
        t, s = compute_std(time, numpy.random.normal(0.0, 1.0, n), 1)
        assert_array_equal(t, time)
        assert_array_equal(s, numpy.zeros(n))

    def test_window_longer_than_series_returns_empty(self):
        t, s = compute_std(numpy.arange(5, dtype=float), numpy.ones(5), 9)
        assert t.shape == (0,)
        assert s.shape == (0,)

    def test_window_one_longer_than_series_is_the_exact_npts_zero_boundary(self):
        # len(data) == window - 1: npts is exactly 0, the boundary between the
        # single-window case and the negative-npts cases covered above.
        t, s = compute_std(numpy.arange(5, dtype=float), numpy.arange(5, dtype=float), 6)
        assert t.shape == (0,)
        assert s.shape == (0,)
        assert s.dtype == numpy.float64

    @pytest.mark.parametrize("window", [0, -1], ids=["zero", "negative"])
    @pytest.mark.xfail(strict=True,
                       reason="compute_std does not validate window > 0: numpy.std of the empty "
                              "slices is NaN, so on n points window = 0 yields n+1 NaNs against "
                              "the 1 stamp in time[-1:] and window = -1 yields n+2 NaNs against 2 "
                              "stamps — a (t, values) pair with mismatched lengths returned "
                              "silently instead of a raise. compute_zscore rejects both inputs, "
                              "but only by accident, from its empty-sample guard")
    def test_non_positive_window_is_rejected(self, window):
        # There is no sane (t, values) pair to return for a non-positive window,
        # so the fix is to validate and raise — which is what is asserted, so
        # that the marker XPASSes once the guard lands. (Asserting
        # len(t) == len(s) instead would keep xfailing forever: a raise inside
        # an xfail body still counts as an xfail.)
        n = 10
        time = numpy.arange(n, dtype=float)
        data = numpy.random.normal(0.0, 1.0, n)
        with numpy.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(Exception):
                compute_std(time, data, window)

    def test_empty_data_returns_empty_instead_of_raising(self):
        # n = 0 drives npts negative, so the loop never runs and empty arrays
        # come back rather than an error — same entry point as compute_zscore's.
        t, s = compute_std(numpy.array([]), numpy.array([]), 5)
        assert t.shape == (0,)
        assert s.shape == (0,)
        assert s.dtype == numpy.float64

    def test_white_noise_round_trip(self):
        # Rolling population std of w iid N(0, σ²) draws has expectation
        # σ·sqrt((w-1)/w)·c4(w) ≈ 0.9925σ for w = 100. Each window's std has
        # relative SD ≈ 1/sqrt(2w) ≈ 7 %; averaging ~2900 overlapping windows
        # (~29 independent) gives SE ≈ 1.3 %, so 5 % is ~3.5 SE plus bias.
        sigma, n, w = 2.0, 3000, 100
        data = numpy.random.normal(0.0, sigma, n)
        _, s = compute_std(numpy.arange(n, dtype=float), data, w)
        assert s.mean() == pytest.approx(sigma, rel=0.05)
        # The spread of the individual windows is pinned with a mid quantile
        # rather than the maximum: the max over ~2900 overlapping (≈29
        # effectively independent) right-skewed chi variates has a fat upper
        # tail, so any max bound — and even the 99th percentile, which only a
        # couple of independent windows determine — is seed-dependent. The
        # per-window relative SD is 1/sqrt(2w) = 0.0707, so the 90th percentile
        # of |s/σ - 1| is ≈ 1.645·0.0707 = 0.116; measured over 2000 seeds it is
        # 0.117 ± 0.013 (range 0.075 - 0.168), so the two-sided band below is
        # ~5-6 SD on each side and catches both an over- and an under-dispersed
        # rolling estimate (e.g. the wrong window length).
        spread = numpy.quantile(numpy.abs(s / sigma - 1.0), 0.90)
        assert 0.05 < spread < 0.20
        assert numpy.all(numpy.isfinite(s))

    def test_detects_volatility_regime_change(self):
        # σ jumps from 1 to 3 at the midpoint. Windows entirely inside one
        # regime should report that regime's σ; per-window relative SD ≈ 7 %
        # for w = 100, so the median over each regime is well within 15 %.
        n_half, w = 600, 100
        data = numpy.concatenate([numpy.random.normal(0.0, 1.0, n_half),
                                  numpy.random.normal(0.0, 3.0, n_half)])
        _, s = compute_std(numpy.arange(2 * n_half, dtype=float), data, w)
        first = s[: n_half - w + 1]         # windows ending before the switch
        second = s[n_half:]                 # windows starting at/after the switch
        assert numpy.median(first) == pytest.approx(1.0, rel=0.15)
        assert numpy.median(second) == pytest.approx(3.0, rel=0.15)
        assert numpy.median(second) > 2.0 * numpy.median(first)

    def test_stationary_ar1_round_trip(self):
        # AR(1) with known φ, σ has stationary std σ/sqrt(1-φ²) = 1.1547 for
        # φ = 0.5, σ = 1. The rolling estimator is biased low by roughly
        # (1 + 2Σρ_k)/(2w) = 3/(2·200) ≈ 0.75 %. The effective sample size is
        # n(1-φ)/(1+φ) = 1000, so SE(std) ≈ 1/sqrt(2·1000) ≈ 2.2 %.
        # rel = 0.08 is ~3.5 SE — not tuned to the seed.
        phi, sigma, n, w = 0.5, 1.0, 3000, 200
        data = _ar1(phi, sigma, n)
        _, s = compute_std(numpy.arange(n, dtype=float), data, w)
        expected = sigma / math.sqrt(1.0 - phi**2)
        assert s.mean() == pytest.approx(expected, rel=0.08)

    def test_integer_data_gives_float_std(self):
        n, w = 20, 4
        data = numpy.random.randint(0, 100, n)
        _, s = compute_std(numpy.arange(n), data, w)
        assert s.dtype == numpy.float64
        assert_allclose(s, _pandas_rolling_std(data.astype(float), w), rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-function behaviour and the mean-reversion signal
# ---------------------------------------------------------------------------

def test_zscore_times_std_recovers_deviation_from_window_mean():
    # By definition z_i · s_i = x_{i+w-1} - mean(window_i). The reference
    # deviation is built from pandas: rebuilding it with numpy.mean over the
    # same slice would just restate (val-mean)/std · std, an identity of the
    # library's own arithmetic. Against an outside reference this pins that the
    # two facades share a window convention and stamp the same times.
    n, w = 300, 20
    time = numpy.arange(n, dtype=float)
    data = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))
    t_z, z = compute_zscore(time, data, w)
    t_s, s = compute_std(time, data, w)
    assert_array_equal(t_z, t_s)
    deviation = _pandas_rolling_mean_deviation(data, w)
    assert_allclose(z * s, deviation, atol=1e-10)


def test_zscore_predicts_mean_reversion_of_a_stationary_ar1():
    # This is the strategy specification: for a mean-reverting series a high
    # z-score should precede a fall and a low z-score a rise. For AR(1),
    # Δx_{t+1} = (φ-1)x_t + σε, so
    #   corr(x_t, Δx_{t+1}) = (φ-1)γ0 / (σ_x σ_Δ) = -sqrt((1-φ)/2) = -0.5
    # exactly at φ = 0.5, and the z-score is (to the accuracy of a w = 200
    # window) an affine rescaling of x_t, which leaves the correlation alone.
    # The magnitude — not just the sign — is what the strategy trades on, so it
    # is asserted: the seed-to-seed spread at this n is ≈ 0.009, making
    # abs = 0.05 roughly 5 SE.
    phi, sigma, n, w = 0.5, 1.0, 3000, 200
    data = _ar1(phi, sigma, n)
    _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
    # z[i] is stamped at index i+w-1; the next increment is x[i+w] - x[i+w-1]
    increments = numpy.diff(data)[w - 1:]
    corr = numpy.corrcoef(z[:-1], increments)[0, 1]
    assert corr == pytest.approx(-math.sqrt((1.0 - phi) / 2.0), abs=0.05)
    # a random walk (φ = 1) has no such signal
    rw = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))
    _, z_rw = compute_zscore(numpy.arange(n, dtype=float), rw, w)
    corr_rw = numpy.corrcoef(z_rw[:-1], numpy.diff(rw)[w - 1:])[0, 1]
    assert abs(corr_rw) < 0.15


def test_compute_std_propagates_a_nan_window():
    # The correct half of the NaN divergence: numpy.std of a window containing
    # NaN is NaN and compute_std hands that on, so the strategies see a missing
    # value rather than a number. Windows clear of the NaN are untouched by
    # either facade.
    n, w = 20, 5
    data = numpy.random.normal(0.0, 1.0, n)
    data[9] = numpy.nan
    with numpy.errstate(invalid="ignore"):
        _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
        _, s = compute_std(numpy.arange(n, dtype=float), data, w)
    contaminated = slice(9 - w + 1, 10)     # the w windows containing index 9
    assert numpy.all(numpy.isnan(s[contaminated]))
    assert numpy.all(numpy.isfinite(z[10:]))
    assert numpy.all(numpy.isfinite(s[10:]))


@pytest.mark.xfail(strict=True,
                   reason="compute_zscore emits 0.0 for windows containing NaN instead of "
                          "propagating it: numpy.std of the window is NaN, `std > 0` is False for "
                          "NaN, so metrics.py:88 takes the degenerate branch. A 0.0 z-score means "
                          "'the price is exactly at its rolling mean' — a tradeable flat signal "
                          "manufactured out of missing data, while compute_std correctly reports "
                          "NaN for the very same windows. Fix: return NaN when std is not finite, "
                          "reserving the 0.0 branch for std == 0")
def test_compute_zscore_should_propagate_a_nan_window():
    n, w = 20, 5
    data = numpy.random.normal(0.0, 1.0, n)
    data[9] = numpy.nan
    with numpy.errstate(invalid="ignore"):
        _, z = compute_zscore(numpy.arange(n, dtype=float), data, w)
    contaminated = slice(9 - w + 1, 10)     # the w windows containing index 9
    assert numpy.all(numpy.isnan(z[contaminated]))


@pytest.mark.parametrize("compute", [compute_zscore, compute_std], ids=["zscore", "std"])
@pytest.mark.parametrize("ntime, ndata", [(4, 10), (20, 10)], ids=["time-shorter", "time-longer"])
@pytest.mark.xfail(strict=True,
                   reason="neither compute_ facade validates len(time) == len(data): the values "
                          "array is sized from len(data) while the time array is sliced from the "
                          "time argument, so a (t, values) pair with mismatched lengths is returned "
                          "silently — 2 vs 8 when time is shorter, 18 vs 8 when it is longer")
def test_mismatched_time_and_data_lengths_are_rejected(compute, ntime, ndata):
    # The strategies index the returned pair positionally, so a time array of a
    # different length from the data array must not be accepted quietly. There
    # is no aligned pair to hand back, so the fix is to raise — asserted here
    # rather than `len(t) == len(values)`, which would keep xfailing after the
    # fix because a raise inside an xfail body is still an xfail.
    time = numpy.arange(float(ntime))
    data = numpy.random.normal(0.0, 1.0, ndata)
    with pytest.raises(Exception):
        compute(time, data, 3)


@pytest.mark.parametrize("compute", [compute_zscore, compute_std], ids=["zscore", "std"])
def test_non_integer_window_is_not_silently_accepted(compute):
    # The mean-reversion notebooks estimate a half life as a float and feed it
    # in as the lookback, surviving only because they wrap it in int() by hand.
    # A caller who forgets gets a raise rather than a silently truncated (or
    # rounded) window — today an opaque TypeError from range(npts) inside the
    # facade, "'float' object cannot be interpreted as an integer", rather than
    # a message naming the window argument.
    time = numpy.arange(30.0)
    data = numpy.random.normal(0.0, 1.0, 30)
    with pytest.raises(Exception):
        compute(time, data, 20.0)
    # a fractional window is no different: nothing rounds or coerces it
    with pytest.raises(Exception):
        compute(time, data, 20.5)


def test_price_like_series_matches_pandas_rolling():
    # The strategies run these metrics on quotes: a large offset with a small
    # fluctuation (here ~1e5 ± 0.5), the regime where a variance computed from
    # raw cumulative sums cancels catastrophically. The per-window numpy.std
    # the metrics module uses is two-pass, so it must still track pandas to
    # near machine precision on such data.
    n, w = 500, 30
    time = numpy.arange(n, dtype=float)
    prices = 1.0e5 + numpy.cumsum(numpy.random.normal(0.0, 0.5, n))
    _, s = compute_std(time, prices, w)
    _, z = compute_zscore(time, prices, w)
    # rtol is 1e-6, not machine precision: the reference side is pandas' ONLINE
    # rolling variance, which itself loses digits on the 1e5 offset. Measured
    # over 500 seeds the two disagree by up to 2e-8 relative on the std and 4e-8
    # absolute on the z-score — so rtol=1e-8 fails on roughly 3 % of seeds
    # (a false alarm: it is the reference degrading, not the two-pass numpy.std
    # in metrics). The catastrophic cancellation this test exists to rule out —
    # the cumulative-sum path in lib.stats, xfailed below — is 2e-5 median and
    # 6e-4 max relative on the same data, two to three orders of magnitude
    # coarser, so 1e-6 keeps every bit of the discriminating power.
    assert_allclose(s, _pandas_rolling_std(prices, w), rtol=1e-6, atol=1e-9)
    # atol matters only for the rare window whose last price sits almost exactly
    # on its rolling mean: |z| ~ 1e-3 there, where a tiny absolute difference
    # between two float64 orderings is not a real disagreement.
    assert_allclose(z, _pandas_rolling_zscore(prices, w), rtol=1e-6, atol=1e-9)


def test_price_like_series_matches_vectorised_lib_stats_implementation():
    # Same cross-check as the N(0,1) ones above, moved to the data regime the
    # strategies actually see. At zero offset both paths agree to 1e-8; the
    # offset is the only thing that changes here.
    n, w = 500, 30
    time = numpy.arange(n, dtype=float)
    prices = 1.0e5 + numpy.cumsum(numpy.random.normal(0.0, 0.5, n))
    _, s_loop = compute_std(time, prices, w)
    _, s_vec = dstats.compute_moving_std(time, prices, w)
    _, z_loop = compute_zscore(time, prices, w)
    _, z_vec = dstats.compute_zscore(time, prices, w)
    assert_allclose(s_loop, s_vec, rtol=1e-8, atol=1e-8)
    assert_allclose(z_loop, z_vec, rtol=1e-8, atol=1e-8)


def test_package_reexports_metrics_functions():
    assert lib.trading.zscore is metrics.zscore
    assert lib.trading.compute_zscore is metrics.compute_zscore
    assert lib.trading.std is metrics.std
    assert lib.trading.compute_std is metrics.compute_std
