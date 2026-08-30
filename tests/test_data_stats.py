"""
Tests for ``lib.data.impl.stats`` — the ``compute_*`` facades over ``lib.stats``
together with the ``OLS`` regression enum and the Granger-causality report
wiring.

Three kinds of test, in priority order:

1. closed form: hand-computed values for ramps, quadratics, Parseval sums,
   Gaussian densities and window statistics;
2. round trip: simulate a process with known parameters (AR(1), Brownian
   motion, correlated normals), estimate them back, compare within a
   statistically justified tolerance;
3. contract: ``(x, values)`` tuple shape and alignment, required kwargs,
   defaults, exceptions, model wiring and serialisation.

Every simulation draws from numpy's global RNG, which ``conftest.py`` reseeds
before each test, so the numbers below are reproducible. Tolerances are stated
as multiples of the estimator's standard error so they also hold for other
seeds. The Granger tests assert hypothesis-test decisions, for which no
tolerance argument exists, so they are taken at a critical value far from every
p-value involved (the driven cell has p = 0, the null cells have p > 1e-3) and
are decided the same way for every seed rather than at the nominal 5% where a
null cell flips roughly one seed in ten.
"""

import json
import warnings

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from lib.data.hyp_test import GrangerCausalityTestReport, GrangerCausalityTestResult
from lib import stats as base_stats
from lib.data.impl import stats as dstats
from lib.data.impl.stats import OLS
from lib.data.param_est import EstModel, OLSParamType, ParamEst
from typing import cast
from pandas import DataFrame


# ---------------------------------------------------------------------------
# simulation helpers
# ---------------------------------------------------------------------------

def _ar1(phi: float, n: int, sigma: float = 1.0) -> numpy.ndarray:
    """AR(1) realization started at zero: x_t = phi*x_{t-1} + eps_t."""
    eps = numpy.random.normal(0.0, sigma, n)
    x = numpy.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _bm(n: int, sigma: float = 1.0) -> numpy.ndarray:
    """Brownian motion sampled at unit time steps with B_0 = 0."""
    dB = numpy.random.normal(0.0, sigma, n)
    dB[0] = 0.0
    return numpy.cumsum(dB)


def _window_stats(x: numpy.ndarray, window: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Per-window mean and population sd computed by an explicit loop.

    Independent of the cumulative-sum implementation used by ``lib.stats``.
    """
    means = numpy.array([x[i:i + window].mean() for i in range(len(x) - window + 1)])
    sds = numpy.array([x[i:i + window].std() for i in range(len(x) - window + 1)])
    return means, sds


# ---------------------------------------------------------------------------
# compute_pspec
# ---------------------------------------------------------------------------

class TestComputePspec:

    def test_returns_time_tail_and_matching_length(self):
        n = 64
        t = numpy.linspace(0.0, 10.0, n)
        x = numpy.random.normal(size=n)

        f, p = dstats.compute_pspec(t, x)

        assert_array_equal(f, t[1:])
        assert len(p) == n - 1
        assert p.dtype == numpy.float64
        assert numpy.all(p >= 0.0)

    @pytest.mark.parametrize("kind", ["noise", "ramp", "cosine"])
    def test_parseval_normalisation_is_exact(self, kind):
        # pspec returns |FFT(x - mean, zero padded to N = 2n-1)|^2 / (n * energy)
        # for padded frequencies 1..n-1. Parseval gives sum over all N bins =
        # N*energy/(n*energy) = N/n; the DC bin is zero because the mean is
        # removed, and the remaining N-1 = 2n-2 bins come in conjugate pairs,
        # so the returned half sums to exactly (2n-1)/(2n).
        n = 128
        t = numpy.arange(n, dtype=float)
        if kind == "noise":
            x = numpy.random.normal(size=n)
        elif kind == "ramp":
            x = numpy.arange(n, dtype=float)
        else:
            x = numpy.cos(2.0 * numpy.pi * 5.0 * t / n)

        _, p = dstats.compute_pspec(t, x)

        assert p.sum() == pytest.approx((2 * n - 1) / (2 * n), rel=1e-12)

    def test_peaks_at_the_frequency_of_a_pure_cosine(self):
        # A cosine of k0 cycles over n samples, zero padded to N = 2n-1, peaks
        # at padded bin k0*N/n. compute_pspec drops bin 0, so returned index j
        # is padded bin j+1.
        n = 128
        k0 = 8
        t = numpy.arange(n, dtype=float)
        x = numpy.cos(2.0 * numpy.pi * k0 * t / n)

        _, p = dstats.compute_pspec(t, x)

        padded_peak = int(numpy.argmax(p)) + 1
        assert padded_peak == pytest.approx(k0 * (2 * n - 1) / n, abs=1.0)


# ---------------------------------------------------------------------------
# compute_acf
# ---------------------------------------------------------------------------

class TestComputeAcf:

    def test_nlags_is_required(self):
        with pytest.raises(Exception, match="nlags parameter is required"):
            dstats.compute_acf(numpy.arange(10.0), numpy.random.normal(size=10))

    def test_shape_alignment_and_unit_lag_zero(self):
        n, nlags = 200, 12
        t = numpy.linspace(0.0, 1.0, n)
        x = numpy.random.normal(size=n)

        lags, ac = dstats.compute_acf(t, x, nlags=nlags)

        assert_array_equal(lags, t[:nlags + 1])
        assert len(ac) == nlags + 1
        assert ac[0] == pytest.approx(1.0)

    def test_ar1_acf_matches_phi_to_the_k(self):
        # rho(k) = phi^k for an AR(1). The sample acf has standard error
        # ~sqrt((1 + 2*sum phi^2i)/n) = 0.016 for n=8000, phi=0.6; 0.08 is 5 se.
        n, phi = 8000, 0.6
        x = _ar1(phi, n)
        t = numpy.arange(n, dtype=float)

        _, ac = dstats.compute_acf(t, x, nlags=5)

        expected = phi ** numpy.arange(6)
        assert_allclose(ac, expected, atol=0.08)

    def test_white_noise_acf_lies_inside_the_confidence_band(self):
        # For iid noise the sample acf at lag>0 has sd 1/sqrt(n). A 4 sd band
        # over 10 lags fails for roughly one seed in 15000.
        n = 4000
        x = numpy.random.normal(size=n)
        t = numpy.arange(n, dtype=float)

        _, ac = dstats.compute_acf(t, x, nlags=10)

        assert numpy.all(numpy.abs(ac[1:]) < 4.0 / numpy.sqrt(n))


# ---------------------------------------------------------------------------
# compute_diff / compute_ndiff
# ---------------------------------------------------------------------------

class TestDifferences:

    def test_diff_of_quadratic_is_the_linear_ramp(self):
        n = 20
        t = numpy.arange(n, dtype=float)
        x = t ** 2

        tv, dx = dstats.compute_diff(t, x)

        # (i+1)^2 - i^2 = 2i + 1
        assert_array_equal(tv, t[:-1])
        assert_allclose(dx, 2.0 * numpy.arange(n - 1) + 1.0)

    def test_ndiff_defaults_to_a_single_difference(self):
        n = 15
        t = numpy.arange(n, dtype=float)
        x = numpy.random.normal(size=n)

        t_default, d_default = dstats.compute_ndiff(t, x)
        t_one, d_one = dstats.compute_ndiff(t, x, ndiff=1)

        assert_array_equal(t_default, t_one)
        assert_allclose(d_default, d_one)
        assert len(d_default) == n - 1

    def test_second_difference_of_a_quadratic_is_constant(self):
        n = 20
        t = numpy.arange(n, dtype=float)
        x = 3.0 * t ** 2

        tv, d2 = dstats.compute_ndiff(t, x, ndiff=2)

        assert_array_equal(tv, t[:-2])
        assert_allclose(d2, numpy.full(n - 2, 6.0))

    @pytest.mark.parametrize("ndiff", [1, 2, 3])
    def test_time_and_values_stay_aligned(self, ndiff):
        n = 30
        t = numpy.arange(n, dtype=float)
        x = numpy.random.normal(size=n)

        tv, d = dstats.compute_ndiff(t, x, ndiff=ndiff)

        assert len(tv) == len(d) == n - ndiff


# ---------------------------------------------------------------------------
# cumulative statistics
# ---------------------------------------------------------------------------

class TestCumulativeStatistics:

    def test_cumu_mean_of_a_ramp(self):
        n = 25
        t = numpy.linspace(0.0, 5.0, n)
        x = numpy.arange(n, dtype=float)

        tv, mean = dstats.compute_cumu_mean(t, x)

        # mean of 0..i is i/2
        assert_array_equal(tv, t)
        assert_allclose(mean, numpy.arange(n) / 2.0)

    def test_cumu_var_of_a_ramp(self):
        n = 25
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        _, var = dstats.compute_cumu_var(t, x)

        # population variance of the integers 0..i is (i^2 + 2i)/12
        i = numpy.arange(n, dtype=float)
        assert_allclose(var, (i ** 2 + 2.0 * i) / 12.0, atol=1e-9)

    def test_cumu_sd_is_the_square_root_of_cumu_var(self):
        n = 25
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        _, sd = dstats.compute_cumu_sd(t, x)

        i = numpy.arange(n, dtype=float)
        assert_allclose(sd, numpy.sqrt((i ** 2 + 2.0 * i) / 12.0), atol=1e-9)

    @pytest.mark.parametrize("dt", [0.25, 2.0])
    def test_delta_t_divides_the_cumulative_variance(self, dt):
        n = 25
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        _, var = dstats.compute_cumu_var(t, x, **{"Δt": dt})
        _, sd = dstats.compute_cumu_sd(t, x, **{"Δt": dt})

        i = numpy.arange(n, dtype=float)
        expected = (i ** 2 + 2.0 * i) / 12.0 / dt
        assert_allclose(var, expected, atol=1e-9)
        assert_allclose(sd, numpy.sqrt(expected), atol=1e-9)

    def test_cumulative_estimates_converge_to_the_simulated_parameters(self):
        # mean has se sigma/sqrt(n) = 0.042 and the variance se
        # sigma^2*sqrt(2/n) = 0.18 for n=5000, sigma=3; tolerances are ~4 se.
        n, mu, sigma = 5000, 2.0, 3.0
        x = numpy.random.normal(mu, sigma, n)
        t = numpy.arange(n, dtype=float)

        _, mean = dstats.compute_cumu_mean(t, x)
        _, var = dstats.compute_cumu_var(t, x)

        assert mean[-1] == pytest.approx(mu, abs=0.17)
        assert var[-1] == pytest.approx(sigma ** 2, abs=0.75)

    def test_cumu_cov_of_proportional_ramps(self):
        n = 25
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)
        y = 2.0 * x

        tv, cov = dstats.compute_cumu_cov(t, x, y)

        # cov(x, 2x) = 2*var(x)
        i = numpy.arange(n, dtype=float)
        assert_array_equal(tv, t)
        assert_allclose(cov, 2.0 * (i ** 2 + 2.0 * i) / 12.0, atol=1e-9)

    def test_cumu_cov_recovers_a_known_correlation(self):
        # cov = rho*sx*sy = 0.7*2*3 = 4.2; se ~ sx*sy*sqrt((1+rho^2)/n) = 0.10
        # for n=5000, so 0.45 is ~4 se.
        n, rho, sx, sy = 5000, 0.7, 2.0, 3.0
        z1 = numpy.random.normal(size=n)
        z2 = numpy.random.normal(size=n)
        x = sx * z1
        y = sy * (rho * z1 + numpy.sqrt(1.0 - rho ** 2) * z2)
        t = numpy.arange(n, dtype=float)

        _, cov = dstats.compute_cumu_cov(t, x, y)

        assert cov[-1] == pytest.approx(rho * sx * sy, abs=0.45)


# ---------------------------------------------------------------------------
# moving window statistics and z-score
# ---------------------------------------------------------------------------

class TestMovingWindowStatistics:

    def test_moving_avg_of_a_ramp(self):
        n, window = 20, 5
        t = numpy.linspace(0.0, 2.0, n)
        x = numpy.arange(n, dtype=float)

        tv, ma = dstats.compute_moving_avg(t, x, window)

        # the mean of the window ending at index i is i - (window-1)/2
        idx = numpy.arange(window - 1, n, dtype=float)
        assert_array_equal(tv, t[window - 1:])
        assert_allclose(ma, idx - (window - 1) / 2.0)

    @pytest.mark.parametrize("window", [3, 5, 8])
    def test_moving_var_of_a_ramp_is_constant(self, window):
        n = 30
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        tv, mv = dstats.compute_moving_var(t, x, window)

        # population variance of any window consecutive integers is (w^2-1)/12
        assert_array_equal(tv, t[window - 1:])
        assert len(mv) == n - window + 1
        assert_allclose(mv, numpy.full(n - window + 1, (window ** 2 - 1) / 12.0))

    def test_moving_std_matches_an_explicit_window_calculation(self):
        n, window = 60, 7
        t = numpy.arange(n, dtype=float)
        x = numpy.random.normal(size=n)

        tv, ms = dstats.compute_moving_std(t, x, window)

        _, expected = _window_stats(x, window)
        assert_array_equal(tv, t[window - 1:])
        assert_allclose(ms, expected, atol=1e-10)

    def test_moving_avg_matches_an_explicit_window_calculation(self):
        n, window = 60, 7
        t = numpy.arange(n, dtype=float)
        x = numpy.random.normal(size=n)

        _, ma = dstats.compute_moving_avg(t, x, window)

        expected, _ = _window_stats(x, window)
        assert_allclose(ma, expected, atol=1e-10)

    def test_zscore_of_a_ramp_is_constant(self):
        n, window = 20, 5
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        tv, z = dstats.compute_zscore(t, x, window)

        # x_i - mean = (w-1)/2 and the population sd is sqrt((w^2-1)/12)
        expected = ((window - 1) / 2.0) / numpy.sqrt((window ** 2 - 1) / 12.0)
        assert_array_equal(tv, t[window - 1:])
        assert len(z) == n - window + 1
        assert_allclose(z, numpy.full(n - window + 1, expected))

    def test_zscore_standardises_against_explicit_window_statistics(self):
        n, window = 80, 10
        t = numpy.arange(n, dtype=float)
        x = numpy.random.normal(2.0, 4.0, n)

        _, z = dstats.compute_zscore(t, x, window)

        means, sds = _window_stats(x, window)
        assert_allclose(z, (x[window - 1:] - means) / sds, atol=1e-9)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

class TestAggregation:

    def test_agg_averages_consecutive_bins(self):
        n, m = 12, 3
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        tv, a = dstats.compute_agg(t, x, m=m)

        # bin k holds 3k..3k+2 whose mean is 3k+1
        assert_allclose(a, numpy.array([1.0, 4.0, 7.0, 10.0]))
        # the times are asserted by their semantics rather than by restating the
        # library's expression: one time per bin, each at the centre of the
        # values that bin averages
        assert len(tv) == n // m
        assert tv[0] == t[0:m].mean()

    def test_agg_times_label_the_bins_they_summarise(self):
        n, m = 12, 3
        t = numpy.arange(n, dtype=float)
        x = numpy.arange(n, dtype=float)

        tv, _ = dstats.compute_agg(t, x, m=m)

        centres = numpy.array([t[k * m:(k + 1) * m].mean() for k in range(n // m)])
        assert_allclose(tv, centres)

    def test_agg_requires_m(self):
        with pytest.raises(Exception, match="m parameter is required"):
            dstats.compute_agg(numpy.arange(10.0), numpy.arange(10.0))

    def test_agg_var_of_a_ramp(self):
        # aggregating a ramp of n points into bins of m gives an arithmetic
        # sequence of step m, whose ddof=1 variance is m^2*d*(d+1)/12 for
        # d = n//m bins.
        n = 60
        x = numpy.arange(n, dtype=float)

        m_vals, var = dstats.compute_agg_var(x, npts=3, m_max=5, m_min=1)

        assert_allclose(m_vals, numpy.array([1.0, 3.0, 5.0]))
        expected = [m ** 2 * (n // m) * (n // m + 1) / 12.0 for m in (1, 3, 5)]
        assert_allclose(var, expected)

    @pytest.mark.parametrize("kwargs,missing", [({"m_max": 8}, "npts"), ({"npts": 4}, "m_max")])
    def test_agg_var_required_kwargs(self, kwargs, missing):
        with pytest.raises(Exception, match=f"{missing} parameter is required"):
            dstats.compute_agg_var(numpy.arange(20.0), **kwargs)

    def test_agg_var_m_min_defaults_to_one(self):
        x = numpy.arange(40.0)

        m_default, _ = dstats.compute_agg_var(x, npts=4, m_max=10)
        m_explicit, _ = dstats.compute_agg_var(x, npts=4, m_max=10, m_min=1)

        assert m_default[0] == 1.0
        assert_allclose(m_default, m_explicit)

    def test_agg_var_returns_the_bin_sizes_it_actually_used(self):
        x = numpy.random.normal(size=1024)

        m_vals, var = dstats.compute_agg_var(x, npts=10, m_max=64, m_min=2)

        assert len(m_vals) == len(var) == 10
        assert_allclose(m_vals, numpy.trunc(m_vals))

    def test_white_noise_aggregated_variance_scales_as_one_over_m(self):
        # Var(mean of m iid samples) = sigma^2/m, so the log-log slope is -1
        # (i.e. 2H-2 for H=1/2). The aggregated variance at bin size m has
        # relative se sqrt(2m/n) <= 0.13 here, which propagates to roughly
        # +/-0.05 on the fitted slope; 0.15 is ~3x that. The regression runs
        # against the fractional m values the facade returns rather than the
        # truncated ones it aggregated at, which is the defect xfailed above.
        n = 8192
        x = numpy.random.normal(0.0, 2.0, n)

        m_vals, var = dstats.compute_agg_var(x, npts=10, m_max=64, m_min=2)

        _, ols = OLS.LOG.single_variable_estimate(var, m_vals)
        assert ols.params[0].est == pytest.approx(-1.0, abs=0.15)
        # intercept is log10(sigma^2) = log10(4) = 0.602
        assert ols.const.est == pytest.approx(numpy.log10(4.0), abs=0.12)
        assert ols.r2 > 0.98


# ---------------------------------------------------------------------------
# lagged variance
# ---------------------------------------------------------------------------

class TestLagVariance:

    def test_lag_var_of_brownian_motion_grows_linearly_with_lag(self):
        # For a random walk with iid N(0, sigma^2) increments the Lo-MacKinlay
        # lagged variance estimator has expectation s*sigma^2. Its relative se
        # is roughly sqrt(4s/(3n)) <= 0.06 for s<=16, n=8000; 0.15 is ~2.5 se.
        n, sigma = 8000, 1.5
        x = _bm(n, sigma)

        s_vals, var = dstats.compute_lag_var(x, svals=[1, 2, 4, 8, 16])

        assert s_vals == [1, 2, 4, 8, 16]
        ratio = var / (numpy.array(s_vals, dtype=float) * sigma ** 2)
        assert_allclose(ratio, numpy.ones(len(s_vals)), atol=0.15)

    def test_lag_var_requires_svals_or_a_scan_range(self):
        with pytest.raises(Exception, match="smax and npts or svals is required"):
            dstats.compute_lag_var(numpy.random.normal(size=50))

    def test_lag_var_scan_range_produces_integer_lags(self):
        x = _bm(500)

        s_vals, var = dstats.compute_lag_var(x, npts=5, smax=16)

        assert all(isinstance(s, int) for s in s_vals)
        assert len(s_vals) == len(var) == 5
        assert s_vals == sorted(s_vals)
        assert s_vals[0] >= 1 and s_vals[-1] <= 16

    def test_lag_var_of_white_noise_is_flat_in_s(self):
        # Differencing a random walk leaves iid increments. For a stationary
        # series x_i - x_{i-s} has variance 2*sigma^2 at every lag, and the
        # estimator divides by (t-s+1)(1-s/t) ~ t-s+1, so its lagged variance is
        # FLAT at 2*sigma^2 rather than growing like s as it does for the walk
        # itself. Each point has relative se ~sqrt(2/n) = 0.02 plus the drift
        # correction, so 15% is a wide margin.
        n, sigma = 4000, 1.0
        dx = numpy.diff(_bm(n, sigma))

        s_vals, var = dstats.compute_lag_var(dx, svals=[1, 2, 4, 8])

        assert s_vals == [1, 2, 4, 8]
        assert_allclose(var, numpy.full(4, 2.0 * sigma ** 2), rtol=0.15)

    def test_lag_var_linear_scan_spans_the_requested_range(self):
        # linear=True routes through create_space, i.e. numpy.linspace(smin,
        # smax, npts), which does reach smax: linspace(4, 64, 5) is exact.
        x = _bm(2000)

        s_vals, var = dstats.compute_lag_var(x, npts=5, smax=64, smin=4, linear=True)

        assert s_vals == [4, 19, 34, 49, 64]
        assert len(var) == 5
        assert numpy.all(var > 0.0)
        # the lagged variance of a walk grows with the lag, so the s=64 estimate
        # is far above the s=4 one (expected ratio 16)
        assert var[-1] > 4.0 * var[0]

    def test_lag_var_logarithmic_scan_honours_smin_and_smax(self):
        x = _bm(2000)

        s_vals, var = dstats.compute_lag_var(x, npts=5, smax=64, smin=4)

        assert len(s_vals) == len(var) == 5
        assert s_vals[0] == 4
        assert s_vals[-1] == 64


# ---------------------------------------------------------------------------
# ensemble statistics
# ---------------------------------------------------------------------------

class TestEnsembleStatistics:

    def test_ensemble_mean_hand_computed(self):
        t = numpy.array([0.0, 1.0, 2.0])
        ensemble = [numpy.array([1.0, 2.0, 3.0]), numpy.array([3.0, 4.0, 5.0])]

        tv, mean = dstats.compute_ensemble_mean(t, ensemble)

        assert_array_equal(tv, t)
        assert_allclose(mean, numpy.array([2.0, 3.0, 4.0]))

    def test_ensemble_var_and_sd_hand_computed(self):
        t = numpy.array([0.0, 1.0, 2.0])
        ensemble = [numpy.array([1.0, 2.0, 3.0]), numpy.array([3.0, 4.0, 5.0])]

        _, var = dstats.compute_ensemble_var(t, ensemble)
        _, sd = dstats.compute_ensemble_sd(t, ensemble)

        # the estimator normalises by nsim, so ((1-2)^2 + (3-2)^2)/2 = 1
        assert_allclose(var, numpy.ones(3))
        assert_allclose(sd, numpy.ones(3))

    @pytest.mark.parametrize("dt", [0.5, 4.0])
    def test_ensemble_delta_t_divides_the_variance(self, dt):
        t = numpy.array([0.0, 1.0, 2.0])
        ensemble = [numpy.array([1.0, 2.0, 3.0]), numpy.array([3.0, 4.0, 5.0])]

        _, var = dstats.compute_ensemble_var(t, ensemble, **{"Δt": dt})
        _, sd = dstats.compute_ensemble_sd(t, ensemble, **{"Δt": dt})

        assert_allclose(var, numpy.full(3, 1.0 / dt))
        assert_allclose(sd, numpy.full(3, numpy.sqrt(1.0 / dt)))

    @pytest.mark.parametrize("func", [dstats.compute_ensemble_mean,
                                      dstats.compute_ensemble_var,
                                      dstats.compute_ensemble_sd,
                                      dstats.compute_ensemble_acf])
    def test_one_dimensional_input_is_rejected(self, func):
        t = numpy.arange(5.0)
        with pytest.raises(Exception, match="two dimensional"):
            func(t, numpy.arange(5.0))

    @pytest.mark.parametrize("func", [dstats.compute_ensemble_cov,
                                      dstats.compute_ensemble_correlation_coefficient])
    def test_one_dimensional_input_is_rejected_by_the_two_sample_functions(self, func):
        t = numpy.arange(5.0)
        with pytest.raises(Exception, match="two dimensional"):
            func(t, numpy.arange(5.0), numpy.arange(5.0))

    def test_brownian_ensemble_mean_and_variance(self):
        # BM has E[B_t] = 0 and Var(B_t) = sigma^2*t. The ensemble mean at t has
        # sd sigma*sqrt(t/nsim), so |mean| < 0.25*sigma*sqrt(t) is ~5 sd for
        # nsim=400; the variance estimator has relative sd sqrt(2/nsim)=0.07 and
        # its values are strongly correlated across t, so 0.2 on the average
        # ratio is ~3 sd.
        nsim, npts, sigma = 400, 80, 1.0
        ensemble = [_bm(npts, sigma) for _ in range(nsim)]
        t = numpy.arange(npts, dtype=float)

        _, mean = dstats.compute_ensemble_mean(t, ensemble)
        _, var = dstats.compute_ensemble_var(t, ensemble)

        assert mean[0] == 0.0
        assert var[0] == 0.0
        assert numpy.all(numpy.abs(mean[1:]) < 0.25 * sigma * numpy.sqrt(t[1:]))
        ratio = var[1:] / (sigma ** 2 * t[1:])
        assert ratio.mean() == pytest.approx(1.0, abs=0.2)

    def test_ensemble_acf_of_an_ar1_ensemble(self):
        # The ensemble average of nsim sample acfs estimates phi^k. The residual
        # error is dominated by the O(1/npts) small-sample bias of each sample
        # acf (~0.01 for npts=600), so 0.05 is a comfortable bound.
        nsim, npts, phi = 60, 600, 0.6
        ensemble = [_ar1(phi, npts) for _ in range(nsim)]
        t = numpy.arange(npts, dtype=float)

        _, ac = dstats.compute_ensemble_acf(t, ensemble)

        assert ac[0] == pytest.approx(1.0)
        assert_allclose(ac[:5], phi ** numpy.arange(5), atol=0.05)

    def test_ensemble_acf_defaults_to_the_full_sample_length(self):
        nsim, npts = 6, 100
        ensemble = [numpy.random.normal(size=npts) for _ in range(nsim)]
        t = numpy.arange(npts, dtype=float)

        tv, ac = dstats.compute_ensemble_acf(t, ensemble)

        assert len(ac) == npts
        assert len(tv) == npts

    def test_ensemble_acf_returns_the_requested_number_of_lags(self):
        # nlags <= nsim is the branch lib.stats.ensemble_acf handles correctly
        # (see the xfail below for nlags > nsim). The ensemble averaged acf of
        # nsim=30 AR(1) paths of npts=200 estimates phi^k with se
        # ~sqrt((1+phi^2)/(1-phi^2)/(npts*nsim)) = 0.017 plus the O(1/npts)
        # small-sample bias, so 0.06 is ~3 se.
        nsim, npts, nlags, phi = 30, 200, 10, 0.5
        ensemble = [_ar1(phi, npts) for _ in range(nsim)]
        t = numpy.arange(npts, dtype=float)

        tv, ac = dstats.compute_ensemble_acf(t, ensemble, nlags=nlags)

        assert len(tv) == len(ac) == nlags
        assert_array_equal(tv, t[:nlags])
        assert ac[0] == pytest.approx(1.0)
        assert_allclose(ac[:4], phi ** numpy.arange(4), atol=0.06)

    def test_ensemble_acf_honours_the_requested_number_of_lags(self):
        nsim, npts, nlags = 5, 200, 20
        ensemble = [numpy.random.normal(size=npts) for _ in range(nsim)]
        t = numpy.arange(npts, dtype=float)

        tv, ac = dstats.compute_ensemble_acf(t, ensemble, nlags=nlags)

        assert len(tv) == nlags
        assert len(ac) == nlags

    def test_ensemble_cov_hand_computed(self):
        t = numpy.array([0.0, 1.0])
        x = [numpy.array([1.0, 2.0]), numpy.array([3.0, 4.0])]
        y = [numpy.array([2.0, 6.0]), numpy.array([6.0, 10.0])]

        tv, cov = dstats.compute_ensemble_cov(t, x, y)

        # means are (2,3) and (4,8): ((1-2)(2-4) + (3-2)(6-4))/2 = 2 at t=0 and
        # ((2-3)(6-8) + (4-3)(10-8))/2 = 2 at t=1
        assert_array_equal(tv, t)
        assert_allclose(cov, numpy.array([2.0, 2.0]))

    def test_ensemble_cov_of_independent_ensembles_vanishes(self):
        # cov estimator sd is sigma_x*sigma_y/sqrt(nsim) = 0.05 for nsim=400
        nsim, npts = 400, 20
        x = numpy.random.normal(size=(nsim, npts))
        y = numpy.random.normal(size=(nsim, npts))
        t = numpy.arange(npts, dtype=float)

        _, cov = dstats.compute_ensemble_cov(t, x, y)

        assert numpy.abs(cov).max() < 0.25

    def test_ensemble_correlation_coefficient_recovers_rho(self):
        # each time slice is an independent estimate of rho with sd
        # (1-rho^2)/sqrt(nsim) = 0.026, so the average over 39 slices is
        # accurate to ~0.004; 0.03 on the mean and 0.12 pointwise are safe.
        nsim, npts, rho = 400, 40, 0.7
        x = numpy.random.normal(size=(nsim, npts))
        z = numpy.random.normal(size=(nsim, npts))
        y = rho * x + numpy.sqrt(1.0 - rho ** 2) * z
        t = numpy.arange(npts, dtype=float)

        tv, cc = dstats.compute_ensemble_correlation_coefficient(t, x, y)

        assert_array_equal(tv, t)
        assert cc[1:].mean() == pytest.approx(rho, abs=0.03)
        assert_allclose(cc[1:], numpy.full(npts - 1, rho), atol=0.12)

    def test_ensemble_correlation_coefficient_leaves_the_first_point_unscaled(self):
        # lib.stats normalises from index 1 onward, guarding against the zero
        # ensemble sd of processes with a deterministic initial value, so the
        # t=0 entry is the raw covariance.
        t = numpy.array([0.0, 1.0])
        x = [numpy.array([10.0, 1.0]), numpy.array([-10.0, -1.0])]
        y = [numpy.array([4.0, 2.0]), numpy.array([-4.0, -2.0])]

        _, cc = dstats.compute_ensemble_correlation_coefficient(t, x, y)

        # covariance is (10*4 + 10*4)/2 = 40 at t=0 and (1*2 + 1*2)/2 = 2 at t=1
        # while the sds are (10, 4) and (1, 2)
        assert cc[0] == pytest.approx(40.0)
        assert cc[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# histograms
# ---------------------------------------------------------------------------

class TestHistograms:

    def test_pdf_hist_density_and_bin_centres(self):
        data = numpy.array([0.5, 1.5, 1.5, 2.5])

        x, pdf = dstats.compute_pdf_hist(data, xmin=0.0, xmax=3.0, nbins=3)

        # density = count/(n*width) with n=4 and width=1
        assert_allclose(x, numpy.array([0.5, 1.5, 2.5]))
        assert_allclose(pdf, numpy.array([0.25, 0.5, 0.25]))

    def test_pdf_hist_defaults_to_fifty_bins_over_the_data_range(self):
        data = numpy.random.normal(size=2000)

        x, pdf = dstats.compute_pdf_hist(data)

        width = x[1] - x[0]
        assert len(pdf) == len(x) == 50
        assert (pdf * width).sum() == pytest.approx(1.0)
        assert x[0] == pytest.approx(data.min() + width / 2.0)
        assert x[-1] == pytest.approx(data.max() - width / 2.0)

    def test_pdf_hist_honours_a_half_specified_range(self):
        data = numpy.random.normal(size=500)

        x, _ = dstats.compute_pdf_hist(data, xmin=-1.0, nbins=10)

        width = x[1] - x[0]
        assert x[0] - width / 2.0 == pytest.approx(-1.0)

    def test_pdf_hist_of_normal_samples_matches_the_density(self):
        # A histogram estimates the AVERAGE of the density over each bin, not
        # its value at the centre, so the comparison is against
        # (F(hi) - F(lo))/width. Each bin count is binomial, so the density
        # estimate has se sqrt(f/(n*width)); near the mode f = 0.199, n = 20000
        # and width = 10*sigma/100 = 0.2 give 0.0071. The bound is 5 se, which
        # covers the maximum taken over the ~20 near-mode bins.
        n, mu, sigma = 20000, 1.0, 2.0
        data = numpy.random.normal(mu, sigma, n)

        x, pdf = dstats.compute_pdf_hist(data, xmin=mu - 5 * sigma, xmax=mu + 5 * sigma, nbins=100)

        width = x[1] - x[0]
        peak = 1.0 / (sigma * numpy.sqrt(2.0 * numpy.pi))
        atol = 5.0 * numpy.sqrt(peak / (n * width))
        assert width == pytest.approx(10.0 * sigma / 100.0)
        bin_avg = (norm.cdf(x + width / 2.0, mu, sigma)
                   - norm.cdf(x - width / 2.0, mu, sigma)) / width

        assert_allclose(pdf, bin_avg, atol=atol)
        assert pdf.max() == pytest.approx(peak, abs=atol)

    def test_cdf_hist_is_a_left_riemann_sum(self):
        x = numpy.linspace(0.05, 0.95, 10)
        pdf = numpy.ones(10)

        xv, cdf = dstats.compute_cdf_hist(x, pdf)

        # cdf[i] = sum(pdf[:i])*dx excludes bin i, so it starts at zero
        assert_array_equal(xv, x)
        assert_allclose(cdf, numpy.arange(10) * 0.1)

    def test_cdf_hist_of_normal_samples_matches_the_gaussian_cdf(self):
        n, mu, sigma = 20000, 0.0, 1.0
        data = numpy.random.normal(mu, sigma, n)
        x, pdf = dstats.compute_pdf_hist(data, xmin=-5.0, xmax=5.0, nbins=100)

        _, cdf = dstats.compute_cdf_hist(x, pdf)

        # cdf[i] accumulates the bins strictly below bin i, i.e. the mass below
        # the lower edge of bin i.
        dx = x[1] - x[0]
        assert_allclose(cdf, norm.cdf(x - dx / 2.0, mu, sigma), atol=0.02)
        assert cdf[len(cdf) // 2] == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# multivariate normal
# ---------------------------------------------------------------------------

class TestMultivariateNormal:

    @pytest.mark.parametrize("n", [10, 50, 64])
    def test_bivariate_pdf_grid_shapes_and_peak(self, n):
        mu = numpy.zeros(2)
        omega = numpy.eye(2)

        vals, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        # numpy.mgrid[-3s : 3s + delta : delta] with delta = 6s/(n-1) yields n
        # points, except when the floating point arange overshoots the stop and
        # appends one more: n=50 does, n=10 and n=64 do not. The contract is the
        # square grid of coordinate pairs and the pdf laid out over it, not the
        # exact point count, so both counts are accepted.
        assert vals.shape[0] == 2
        assert vals.shape[1] == vals.shape[2]
        assert vals.shape[1] in (n, n + 1)
        assert pdf.shape == vals.shape[1:]
        # the grid need not contain the mode: for n = 10 the nearest lines sit
        # delta/2 = 1/3 from the origin in each coordinate, so the largest
        # density on the grid is the standard bivariate normal evaluated there,
        # exp(-(dx^2 + dy^2)/2)/(2 pi), which is 1/(2 pi) only in the limit
        dx = numpy.abs(vals[0]).min()
        dy = numpy.abs(vals[1]).min()
        expected_peak = numpy.exp(-(dx ** 2 + dy ** 2) / 2.0) / (2.0 * numpy.pi)
        assert pdf.max() == pytest.approx(expected_peak, rel=1e-9)
        assert expected_peak <= 1.0 / (2.0 * numpy.pi)

    def test_bivariate_pdf_integrates_to_one_over_the_grid(self):
        n = 60
        mu = numpy.zeros(2)
        omega = numpy.eye(2)

        vals, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        # the spacing is read off the returned grid rather than recomputed, so
        # the mass is the only quantity under test
        delta = vals[0][1, 0] - vals[0][0, 0]
        # a +/-3 sigma square holds 0.9973^2 = 0.9946 of the mass
        assert pdf.sum() * delta ** 2 == pytest.approx(0.9946, abs=0.01)

    def test_grid_covers_three_standard_deviations_for_a_non_unit_variance(self):
        n, variance = 60, 0.25
        mu = numpy.zeros(2)
        omega = variance * numpy.eye(2)

        vals, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        delta = vals[0][1, 0] - vals[0][0, 0]
        assert vals[0].max() == pytest.approx(3.0 * numpy.sqrt(variance), abs=delta)
        assert pdf.sum() * delta ** 2 == pytest.approx(0.9946, abs=0.01)

    def test_correlated_bivariate_peak_scales_with_the_determinant(self):
        n, rho = 60, 0.5
        mu = numpy.zeros(2)
        omega = numpy.array([[1.0, rho], [rho, 1.0]])

        _, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        expected = 1.0 / (2.0 * numpy.pi * numpy.sqrt(1.0 - rho ** 2))
        assert pdf.max() == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("nvars", [
        pytest.param(0, marks=pytest.mark.xfail(
            strict=True,
            reason="the guard reads 'nvars == 1 or nvars > 3', so an empty mean falls "
                   "through it and dies in min(numpy.diag(omega)) with "
                   "ValueError('min() iterable argument is empty') instead of raising the "
                   "documented 'Number of variables must be between 2 or 3'")),
        1,
        4,
    ])
    def test_rejects_unsupported_dimensions(self, nvars):
        with pytest.raises(Exception, match="Number of variables"):
            dstats.compute_multivariate_normal_pdf(numpy.zeros(nvars), numpy.eye(nvars), 10)

    def test_trivariate_pdf(self):
        n = 10
        mu = numpy.zeros(3)
        omega = numpy.eye(3)

        vals, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        assert vals.shape[0] == 3
        assert pdf.max() > 0.0

    def test_bivariate_pdf_with_distinct_means_still_carries_the_mass(self):
        n = 60
        mu = numpy.array([5.0, -5.0])
        omega = numpy.eye(2)

        _, pdf = dstats.compute_multivariate_normal_pdf(mu, omega, n)

        delta = 6.0 / (n - 1)
        assert pdf.sum() * delta ** 2 == pytest.approx(0.9946, abs=0.01)

    def test_sample_source_returns_variables_by_row_and_recovers_the_moments(self):
        # covariance entries have se ~ sqrt((s_i^2 s_j^2 + c_ij^2)/n) <= 0.09 for
        # n=4000 and these scales; 0.35 is ~4 se. The mean se is s_i/sqrt(n).
        n = 4000
        mu = numpy.array([1.0, -2.0])
        omega = numpy.array([[4.0, 1.5], [1.5, 1.0]])

        samples = dstats.create_multivariate_normal_samples_source(mu, omega, n)

        assert samples.shape == (2, n)
        assert_allclose(samples.mean(axis=1), mu, atol=0.15)
        assert_allclose(numpy.cov(samples, bias=True), omega, atol=0.35)


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------

class TestErrorMetrics:

    def test_bias_is_the_mean_signed_error(self):
        pred = numpy.array([1.0, 2.0, 3.0, 4.0])
        obs = numpy.array([0.0, 3.0, 3.0, 8.0])

        # errors are 1, -1, 0, -4 -> mean -1
        assert dstats.compute_bias(pred, obs) == pytest.approx(-1.0)

    def test_mae_is_the_mean_absolute_error(self):
        pred = numpy.array([1.0, 2.0, 3.0, 4.0])
        obs = numpy.array([0.0, 3.0, 3.0, 8.0])

        # |errors| are 1, 1, 0, 4 -> mean 1.5
        assert dstats.compute_mae(pred, obs) == pytest.approx(1.5)

    def test_rmse_is_the_root_mean_squared_error(self):
        pred = numpy.array([1.0, 2.0, 3.0, 4.0])
        obs = numpy.array([0.0, 3.0, 3.0, 8.0])

        # squared errors are 1, 1, 0, 16 -> sqrt(18/4)
        assert dstats.compute_rmse(pred, obs) == pytest.approx(numpy.sqrt(4.5))

    def test_metrics_vanish_for_a_perfect_prediction(self):
        obs = numpy.random.normal(size=100)

        assert dstats.compute_bias(obs, obs) == pytest.approx(0.0)
        assert dstats.compute_mae(obs, obs) == pytest.approx(0.0)
        assert dstats.compute_rmse(obs, obs) == pytest.approx(0.0)

    def test_metrics_recover_a_known_noise_level(self):
        # pred = obs + N(shift, sigma^2): bias -> shift, rmse -> sqrt(shift^2 +
        # sigma^2), and mae -> the folded normal mean
        #   E|N(m, s)| = s*sqrt(2/pi)*exp(-m^2/(2s^2)) + m*(2*Phi(m/s) - 1),
        # which is 0.8956 for m=0.5, s=1. n=4000 gives a mean se of
        # sigma/sqrt(n) = 0.008, so 0.05 is comfortable.
        n, shift, sigma = 4000, 0.5, 1.0
        obs = numpy.random.normal(size=n)
        pred = obs + numpy.random.normal(shift, sigma, n)

        folded_mean = (sigma * numpy.sqrt(2.0 / numpy.pi) * numpy.exp(-shift ** 2 / (2.0 * sigma ** 2))
                       + shift * (2.0 * norm.cdf(shift / sigma) - 1.0))
        assert folded_mean == pytest.approx(0.8956, abs=1e-4)

        assert dstats.compute_bias(pred, obs) == pytest.approx(shift, abs=0.05)
        assert dstats.compute_rmse(pred, obs) == pytest.approx(numpy.sqrt(shift ** 2 + sigma ** 2), abs=0.05)
        assert dstats.compute_mae(pred, obs) == pytest.approx(folded_mean, abs=0.05)


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def _causal_pair(n: int = 300, phi: float = 0.8) -> numpy.ndarray:
    """x is white noise, y is driven by the previous value of x."""
    x = numpy.random.normal(size=n)
    y = numpy.zeros(n)
    for i in range(1, n):
        y[i] = phi * x[i - 1] + 0.3 * numpy.random.normal()
    return numpy.array([x, y])


class TestCausalityMatrix:

    def test_detects_the_known_driver(self):
        # The decision is taken at 1e-3, not at the nominal 5%. causality_matrix
        # reports min(p) over lags 1..nlags, which inflates the per-cell type-I
        # rate well past 5%, so a null cell is "significant" at 0.05 for roughly
        # one seed in ten. At 1e-3 the driven cell (p = 0) and the null cell
        # (smallest p observed across seeds 2.7e-3) fall on opposite sides for
        # every seed, so the assertions are decisions rather than coin flips.
        samples = _causal_pair()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix, report = dstats.compute_causality_matrix(samples, nlags=2, critical_value=1e-3)

        assert list(matrix.columns) == ["pvalue", "critical_value", "result",
                                        "dependent_var", "causal_var"]
        assert len(matrix) == 4

        driven = cast(DataFrame, matrix[(matrix["dependent_var"] == 2) & (matrix["causal_var"] == 1)])
        not_driven = cast(DataFrame, matrix[(matrix["dependent_var"] == 1) & (matrix["causal_var"] == 2)])
        assert bool(driven["result"].iloc[0]) is True
        assert float(driven["pvalue"].iloc[0]) < 1e-4
        assert bool(not_driven["result"].iloc[0]) is False
        assert float(not_driven["pvalue"].iloc[0]) > 1e-3

    def test_report_wiring_and_rank(self):
        # critical_value=1e-3 for the same reason as above: rank counts the
        # dependent variables with an incoming True, so one null cell decided at
        # 5% would flip it from 1 to 2.
        samples = _causal_pair()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix, report = dstats.compute_causality_matrix(samples, nlags=2, critical_value=1e-3)

        assert isinstance(report, GrangerCausalityTestReport)
        assert all(isinstance(r, GrangerCausalityTestResult) for r in report.results)
        assert len(report.results) == len(matrix)
        # only the second variable has an incoming causal relation
        assert report.rank == 1
        # every result shares the report identifier
        assert all(report.test_id in repr(r) for r in report.results)

        payload = json.loads(report.to_json())
        assert payload["rank"] == 1
        assert payload["test_id"] == report.test_id
        assert len(payload["results"]) == 4

    def test_critical_value_defaults_to_five_percent(self):
        samples = _causal_pair()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix, _ = dstats.compute_causality_matrix(samples, nlags=2)

        assert set(matrix["critical_value"]) == {0.05}

    def test_critical_value_override_changes_the_decisions(self):
        # Two thresholds over the SAME samples. No series Granger causes itself,
        # and the F test of a series against itself returns p = 1.0 exactly, so
        # the two diagonal cells are rejected at 1e-3 and accepted at 1.0 for
        # every seed. The driven cell has p = 0 and is accepted at both. The
        # counts below are therefore deterministic and the loose threshold must
        # produce strictly more positives than the strict one.
        samples = _causal_pair()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strict, _ = dstats.compute_causality_matrix(samples, nlags=2, critical_value=1e-3)
            loose, _ = dstats.compute_causality_matrix(samples, nlags=2, critical_value=1.0)

        assert set(strict["critical_value"]) == {1e-3}
        assert set(loose["critical_value"]) == {1.0}

        diagonal = (strict["dependent_var"] == strict["causal_var"]).to_numpy()
        assert_allclose(strict["pvalue"].to_numpy(dtype=float)[diagonal], numpy.ones(2))
        assert not strict["result"].to_numpy()[diagonal].any()
        assert loose["result"].to_numpy()[diagonal].all()

        driven = ((strict["dependent_var"] == 2) & (strict["causal_var"] == 1)).to_numpy()
        assert bool(strict["result"].to_numpy()[driven][0]) is True
        assert bool(loose["result"].to_numpy()[driven][0]) is True

        assert int(strict["result"].to_numpy().sum()) == 1
        assert int(loose["result"].to_numpy().sum()) == 4

    @pytest.mark.xfail(strict=True,
                       reason="statsmodels' grangercausalitytests raises a bare "
                              "NotImplementedError for addconst=False, so True is the only "
                              "model it can build and no value of add_const can change the "
                              "result. causality_matrix now forwards the kwarg and rejects "
                              "False with a clear message rather than silently ignoring it, "
                              "but the parameter cannot become meaningful without upstream "
                              "support")
    def test_add_const_changes_the_model(self):
        samples = _causal_pair()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            without, _ = dstats.compute_causality_matrix(samples, nlags=2, add_const=False)
            with_const, _ = dstats.compute_causality_matrix(samples, nlags=2, add_const=True)

        assert not numpy.allclose(without["pvalue"].to_numpy(dtype=float),
                                  with_const["pvalue"].to_numpy(dtype=float))

    def test_independent_series_show_no_causality(self):
        # decided at 1e-3, see test_detects_the_known_driver for why
        n = 300
        samples = numpy.array([numpy.random.normal(size=n), numpy.random.normal(size=n)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix, report = dstats.compute_causality_matrix(samples, nlags=2, critical_value=1e-3)

        off_diagonal = cast(DataFrame, matrix[matrix["dependent_var"] != matrix["causal_var"]])
        assert numpy.all(off_diagonal["pvalue"].to_numpy(dtype=float) > 1e-3)
        assert not off_diagonal["result"].to_numpy().any()
        assert report.rank == 0


# ---------------------------------------------------------------------------
# OLS regression enum
# ---------------------------------------------------------------------------

class TestOLS:

    def test_linear_single_variable_recovers_slope_and_intercept(self):
        x = numpy.linspace(1.0, 10.0, 400)
        y = 3.0 * x + 2.0 + numpy.random.normal(0.0, 0.01, 400)

        report, result = OLS.LINEAR.single_variable_estimate(y, x)

        assert result.const.est == pytest.approx(2.0, abs=0.01)
        assert result.params[0].est == pytest.approx(3.0, abs=0.01)
        assert result.r2 > 0.999
        # the report is the raw statsmodels fit the model object was built from
        assert report.params[1] == pytest.approx(result.params[0].est)
        assert report.bse[0] == pytest.approx(result.const.err)

    def test_single_variable_result_structure(self):
        x = numpy.linspace(1.0, 10.0, 100)
        y = -1.5 * x + 4.0

        _, result = OLS.LINEAR.single_variable_estimate(y, x)

        assert result.est_model == EstModel.OLS
        assert isinstance(result.const, ParamEst)
        assert result.const.param_type == OLSParamType.OLS_CONST.value
        assert len(result.params) == 1
        assert result.params[0].param_type == OLSParamType.OLS_PARAM.value
        assert result.params[0].column == 1
        assert result.const.est_label == r"$\beta$"
        assert result.params[0].est_label == r"$\alpha_1$"
        assert result.params[0].err_label == r"$\sigma_{\alpha_1}$"
        # const and every parameter share the estimate id of the result
        assert result.const.est_id == result.est_id
        assert result.params[0].est_id == result.est_id
        assert "est_id=" in repr(result)

    def test_log_model_recovers_a_power_law(self):
        # y = b*x^a becomes log10(y) = a*log10(x) + log10(b)
        a, b = 1.5, 2.0
        x = numpy.linspace(1.0, 100.0, 300)
        y = b * x ** a

        _, result = OLS.LOG.single_variable_estimate(y, x)

        assert result.params[0].est == pytest.approx(a, abs=1e-9)
        assert result.const.est == pytest.approx(numpy.log10(b), abs=1e-9)
        assert result.r2 == pytest.approx(1.0)

    def test_linear_and_log_dispatch_differ(self):
        a, b = 1.5, 2.0
        x = numpy.linspace(1.0, 100.0, 300)
        y = b * x ** a

        _, linear = OLS.LINEAR.single_variable_estimate(y, x)
        _, log = OLS.LOG.single_variable_estimate(y, x)

        # the untransformed fit of a power law is not exact
        assert linear.r2 < 0.99
        assert log.r2 > linear.r2

    def test_two_variable_estimate(self):
        n = 500
        x1 = numpy.random.normal(size=n)
        x2 = numpy.random.normal(size=n)
        y = 1.0 + 2.0 * x1 - 3.0 * x2 + numpy.random.normal(0.0, 0.01, n)

        report, result = OLS.LINEAR.two_variable_estimate(y, x1, x2)

        assert len(result.params) == 2
        assert result.const.est == pytest.approx(1.0, abs=0.01)
        assert result.params[0].est == pytest.approx(2.0, abs=0.01)
        assert result.params[1].est == pytest.approx(-3.0, abs=0.01)
        assert [p.column for p in result.params] == [1, 2]
        assert result.r2 > 0.999
        assert len(report.params) == 3

    def test_multi_variable_estimate(self):
        n = 500
        x = numpy.random.normal(size=(3, n))
        y = 0.5 + 1.0 * x[0] - 2.0 * x[1] + 3.0 * x[2] + numpy.random.normal(0.0, 0.01, n)

        report, result = OLS.LINEAR.multi_variable_estimate(y, x)

        assert len(result.params) == 3
        assert result.const.est == pytest.approx(0.5, abs=0.01)
        assert_allclose([p.est for p in result.params], [1.0, -2.0, 3.0], atol=0.01)
        assert [p.column for p in result.params] == [1, 2, 3]
        # the formula fit names the terms
        assert list(report.params.index) == ["Intercept", "x1", "x2", "x3"]
        assert all(p.err > 0.0 for p in result.params)

    def test_r2_matches_an_independent_calculation(self):
        n = 300
        x = numpy.linspace(0.0, 5.0, n)
        y = 2.0 * x + 1.0 + numpy.random.normal(0.0, 0.5, n)

        _, result = OLS.LINEAR.single_variable_estimate(y, x)

        fitted = result.const.est + result.params[0].est * x
        ssr = numpy.sum((y - fitted) ** 2)
        sst = numpy.sum((y - y.mean()) ** 2)
        assert result.r2 == pytest.approx(1.0 - ssr / sst, rel=1e-9)

    def test_result_serialises_to_json(self):
        x = numpy.linspace(1.0, 10.0, 50)
        y = 2.0 * x + 1.0

        _, result = OLS.LINEAR.single_variable_estimate(y, x)
        payload = json.loads(result.to_json())

        assert payload["est_model"] == EstModel.OLS.value
        assert payload["est_id"] == result.est_id
        assert payload["const"]["est"] == pytest.approx(result.const.est)
        assert payload["const"]["param_type"] == OLSParamType.OLS_CONST.value
        assert len(payload["params"]) == 1
        assert payload["params"][0]["est"] == pytest.approx(result.params[0].est)
        assert payload["r2"] == pytest.approx(result.r2)
        assert json.loads(result.to_json(pretty=True)) == payload

    @pytest.mark.parametrize("member", list(OLS))
    def test_every_enum_member_runs_a_regression(self, member):
        # noisy so the residual variance, and with it every standard error, is
        # strictly positive whichever branch the member dispatches to
        x = numpy.linspace(1.0, 10.0, 100)
        y = 2.0 * x + 3.0 + numpy.random.normal(0.0, 0.2, 100)

        _, result = member.single_variable_estimate(y, x)

        assert len(result.params) == 1
        assert 0.0 <= result.r2 <= 1.0
        assert numpy.isfinite(result.const.est)
        assert numpy.isfinite(result.params[0].est)
        assert result.const.err > 0.0
        assert result.params[0].err > 0.0

    @pytest.mark.parametrize("member", list(OLS))
    def test_every_enum_member_runs_the_two_and_multi_variable_paths(self, member):
        # positive regressors and a multiplicative error keep y > 0 so the LOG
        # branch's log10 is defined for every member
        n = 200
        x1 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        x2 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        x3 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        y = (2.0 * x1 ** 1.5 * x2 ** 0.5 * x3 ** 0.25
             * numpy.exp(numpy.random.normal(0.0, 0.1, n)))

        _, two = member.two_variable_estimate(y, x1, x2)
        report, multi = member.multi_variable_estimate(y, numpy.array([x1, x2, x3]))

        assert len(two.params) == 2
        assert [p.column for p in two.params] == [1, 2]
        assert len(multi.params) == 3
        assert [p.column for p in multi.params] == [1, 2, 3]
        assert list(report.params.index) == ["Intercept", "x1", "x2", "x3"]
        assert 0.0 <= two.r2 <= 1.0
        assert 0.0 <= multi.r2 <= 1.0
        assert all(p.err > 0.0 for p in two.params)
        assert all(p.err > 0.0 for p in multi.params)
        assert all(numpy.isfinite(p.est) for p in multi.params)

    def test_log_two_variable_estimate_recovers_a_multiplicative_power_law(self):
        # y = b*x1^a1*x2^a2 is exactly linear in log10, so the LOG branch of
        # __OLS_fit recovers both exponents from noiseless data to machine
        # precision. The regressors are independent, so the exponents are
        # separately identified rather than only their sum.
        a1, a2, b, n = 1.5, 0.5, 2.0, 200
        x1 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        x2 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        y = b * x1 ** a1 * x2 ** a2

        _, result = OLS.LOG.two_variable_estimate(y, x1, x2)

        assert result.r2 == pytest.approx(1.0)
        assert result.params[0].est == pytest.approx(a1, abs=1e-9)
        assert result.params[1].est == pytest.approx(a2, abs=1e-9)
        assert result.const.est == pytest.approx(numpy.log10(b), abs=1e-9)

    def test_log_multi_variable_estimate_recovers_a_multiplicative_power_law(self):
        a1, a2, b, n = 1.5, 0.5, 2.0, 200
        x1 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        x2 = numpy.exp(numpy.random.uniform(0.1, 2.0, n))
        y = b * x1 ** a1 * x2 ** a2

        _, result = OLS.LOG.multi_variable_estimate(y, numpy.array([x1, x2]))

        assert result.r2 == pytest.approx(1.0)
        assert result.params[0].est == pytest.approx(a1, abs=1e-9)
        assert result.params[1].est == pytest.approx(a2, abs=1e-9)
        assert result.const.est == pytest.approx(numpy.log10(b), abs=1e-9)

    def test_xlog_linearises_an_exponential_relation(self):
        # XLOG documents y = b*exp(a*x), which is exactly linear once y is
        # log transformed, so the fit of noiseless data must have r2 == 1.
        x = numpy.linspace(0.0, 10.0, 200)
        y = 2.0 * numpy.exp(0.5 * x)

        _, result = OLS.XLOG.single_variable_estimate(y, x)

        assert result.r2 == pytest.approx(1.0, abs=1e-6)

    def test_ylog_linearises_a_logarithmic_relation(self):
        # YLOG documents y = b*ln(a*x), which is exactly linear once x is
        # log transformed, so the fit of noiseless data must have r2 == 1.
        x = numpy.linspace(1.0, 100.0, 200)
        y = 3.0 * numpy.log(2.0 * x)

        _, result = OLS.YLOG.single_variable_estimate(y, x)

        assert result.r2 == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# facade contract shared by the compute_* functions
# ---------------------------------------------------------------------------

class TestFractionalPriceChange:
    """lib.stats.fractional_purchase / compute_fractional_price_change.

    Deleted by accident during a type-annotation sweep (navi 2d34c07) and only
    noticed when a notebook that calls them was re-run months later. The sole
    caller is an .ipynb, so a code-wide grep for the symbol finds nothing --
    which is exactly why these tests exist.
    """

    def test_is_the_one_step_ahead_short_return(self):
        x = numpy.array([100.0, 110.0, 99.0, 99.0, 50.0])
        got = base_stats.fractional_purchase(x, 1)
        assert_allclose(got, [(100 - 110) / 110, (110 - 99) / 99, 0.0, (99 - 50) / 50])

    def test_constant_series_has_zero_change(self):
        assert_allclose(base_stats.fractional_purchase(numpy.full(12, 7.5), 4), numpy.zeros(8))

    @pytest.mark.parametrize("window", [1, 2, 5, 23])
    def test_time_axis_and_values_are_the_same_length(self, window):
        n = 40
        t = numpy.arange(float(n))
        x = 100.0 + numpy.arange(float(n))

        ft, fp = dstats.compute_fractional_price_change(t, x, window)

        assert len(ft) == len(fp) == n - window
        assert ft[0] == t[window - 1]
        assert ft[-1] == t[-2]

    def test_facade_shares_its_window_origin_with_zscore(self):
        t = numpy.arange(60.0)
        x = 100.0 + numpy.sin(t / 5.0)
        ft, _ = dstats.compute_fractional_price_change(t, x, 10)
        zt, _ = dstats.compute_zscore(t, x, 10)
        assert ft[0] == zt[0]

    def test_facade_matches_the_leaf(self):
        t = numpy.arange(30.0)
        x = 50.0 + numpy.arange(30.0) ** 1.3
        _, fp = dstats.compute_fractional_price_change(t, x, 6)
        assert_allclose(fp, base_stats.fractional_purchase(x, 6))


class TestFacadeContract:

    @pytest.mark.parametrize("call", [
        lambda t, x: dstats.compute_pspec(t, x),
        lambda t, x: dstats.compute_acf(t, x, nlags=5),
        lambda t, x: dstats.compute_ndiff(t, x),
        lambda t, x: dstats.compute_diff(t, x),
        lambda t, x: dstats.compute_cumu_mean(t, x),
        lambda t, x: dstats.compute_cumu_var(t, x),
        lambda t, x: dstats.compute_cumu_sd(t, x),
        lambda t, x: dstats.compute_moving_avg(t, x, 5),
        lambda t, x: dstats.compute_moving_var(t, x, 5),
        lambda t, x: dstats.compute_moving_std(t, x, 5),
        lambda t, x: dstats.compute_zscore(t, x, 5),
        lambda t, x: dstats.compute_cumu_cov(t, x, x),
        lambda t, x: dstats.compute_agg(t, x, m=4),
    ])
    def test_returns_aligned_x_and_value_arrays(self, call):
        n = 40
        t = numpy.linspace(0.0, 4.0, n)
        x = numpy.random.normal(size=n) + numpy.arange(n)

        out = call(t, x)

        assert isinstance(out, tuple) and len(out) == 2
        xs, values = out
        assert len(xs) == len(values)
        assert numpy.asarray(values).dtype == numpy.float64
