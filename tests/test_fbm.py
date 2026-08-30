"""Tests for navi's fractional Brownian motion module group.

Model layer  : ``lib.models.fbm``    -- closed forms, Cholesky/FFT generators,
                                       Lo & MacKinlay variance-ratio test.
Facade layer : ``lib.data.impl.fbm`` -- the ``**kwargs`` API returning
                                       ``(t, values)`` tuples that notebooks call.

Every simulation draws from numpy's global RNG, which ``tests/conftest.py``
reseeds before each test, so the ensemble and round-trip checks below are
deterministic. Their tolerances are nevertheless chosen from the sampling
distribution (see the comments) so they hold for most seeds, not just this one.
"""
import json
import uuid

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import toeplitz
from scipy.stats import norm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib import stats
from lib.models import fbm
from lib.data.impl import fbm as fbm_data
from lib.data.hyp_test import (HypothesisTestStatus, HypothesisTestType, HypothesisType,
                               StatisticalTestData, StatisticalTestReport)
from lib.data.param_est import OLSParamType, OLSResult
from lib.data.reports import VarianceRatioTestReport
from helpers import present


# ---------------------------------------------------------------------------
# Helpers (independent of the module's private internals)
# ---------------------------------------------------------------------------

def fgn_cov(H, n):
    """Covariance of n consecutive unit fractional Gaussian noise increments,
    built with scipy's toeplitz rather than the module's private matrix builder."""
    return toeplitz(fbm.acf(H, numpy.arange(n, dtype=float)))


def linear_map(func, dim_in):
    """Matrix of a linear map R^dim_in -> R^m, recovered by feeding unit vectors."""
    cols = []
    for k in range(dim_in):
        e = numpy.zeros(dim_in)
        e[k] = 1.0
        cols.append(numpy.asarray(func(e)))
    return numpy.column_stack(cols)


def random_walk(n):
    """Gaussian random walk from 0: the H = 1/2 reference process, generated
    without going through either fBm generator."""
    return numpy.concatenate(([0.0], numpy.cumsum(numpy.random.normal(size=n - 1))))


def expected_report_status(hyp_test_type, status_vals):
    """Roll-up rule for a variance-ratio report, written from the documented
    semantics rather than by calling ``HypothesisTestType.status``.

    ``BM`` accepts the null (Brownian motion) when at least one of the lag
    statistics falls inside the band. The two fBm tests want the *alternative*, so
    they pass when at least one lag rejects, i.e. when some status is False.
    """
    if hyp_test_type is HypothesisTestType.BM:
        return HypothesisTestStatus.PASSED if any(status_vals) else HypothesisTestStatus.FAILED
    if hyp_test_type in (HypothesisTestType.FBM_AUTO_CORR, HypothesisTestType.FBM_NEG_AUTO_CORR):
        return HypothesisTestStatus.FAILED if all(status_vals) else HypothesisTestStatus.PASSED
    raise AssertionError(f"no expectation encoded for {hyp_test_type}")


def lo_mackinlay_theta(s, t):
    """Homoscedastic asymptotic variance of VR(s) - 1, Lo & MacKinlay (1988)."""
    return 2.0 * (2.0 * s - 1.0) * (s - 1.0) / (3.0 * s * t)


def lo_mackinlay_theta_hetero(samples, s, drop_last=False, small_sample_norm=False):
    """Heteroscedasticity-consistent asymptotic variance of VR(s) - 1, written
    straight from Lo & MacKinlay (1988) eq. (18) with numpy slicing rather than
    through the library's private ``__delta_factor``/``__theta_factor``::

        delta(j) = sum_{k=j+1}^{t} (dX_k - mu)^2 (dX_{k-j} - mu)^2
                   / [ sum_{k=1}^{t} (dX_k - mu)^2 ]^2
        theta(s) = sum_{j=1}^{s-1} [ 2(s - j)/s ]^2 delta(j)

    ``drop_last`` and ``small_sample_norm`` switch on the two deviations navi's
    implementation carries (see the tests below): the delta sum stopping at
    k = t - 1 instead of k = t, and an extra (t-1)^2/t^2 factor left over from
    dividing by ``stats.lag_var(samples, 1)**2`` and then by t^2.
    """
    t = len(samples) - 1
    mu = (samples[t] - samples[0]) / t
    d = numpy.diff(samples) - mu            # d[k-1] = dX_k - mu for k = 1..t
    denom = numpy.sum(d ** 2) ** 2
    stop = t - 1 if drop_last else t
    theta = 0.0
    for j in range(1, s):
        delta = numpy.sum(d[j:stop] ** 2 * d[0:stop - j] ** 2) / denom
        theta += (2.0 * (s - j) / s) ** 2 * delta
    if small_sample_norm:
        theta *= (t - 1.0) ** 2 / t ** 2
    return theta


# ===========================================================================
# Model layer: closed forms
# ===========================================================================

class TestClosedForms:

    def test_var_reduces_to_brownian_motion_at_half(self):
        t = numpy.linspace(0.0, 10.0, 21)
        assert_allclose(fbm.var(0.5, t), t)

    @pytest.mark.parametrize("H, expected", [
        (0.25, [1.0, 2.0, 3.0]),       # t^(1/2)
        (1.0, [1.0, 16.0, 81.0]),      # t^2
        # 4^1.6 = (2^2)^1.6 = 2^3.2 and 9^1.6 = 3^3.2, reduced by hand so the
        # expectation is not the implementation's own t**(2H) expression.
        (0.8, [1.0, 2.0 ** 3.2, 3.0 ** 3.2]),
    ])
    def test_var_is_t_to_the_2H(self, H, expected):
        assert_allclose(fbm.var(H, numpy.array([1.0, 4.0, 9.0])), expected)

    def test_cov_is_min_of_times_for_brownian_motion(self):
        t = numpy.arange(7, dtype=float)
        assert_allclose(fbm.cov(0.5, 3.0, t), [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0])

    @pytest.mark.parametrize("H", [0.3, 0.5, 0.8])
    def test_cov_diagonal_symmetry_and_zero_at_origin(self, H):
        t = numpy.array([0.5, 2.0, 7.0])
        # Cov(B_t, B_t) = Var(B_t)
        assert_allclose(fbm.cov(H, 2.0, numpy.array([2.0])), fbm.var(H, numpy.array([2.0])))
        # Cov(B_s, B_t) = Cov(B_t, B_s)
        assert_allclose(fbm.cov(H, 2.0, t), [fbm.cov(H, ti, numpy.array([2.0]))[0] for ti in t])
        # B_0 = 0 so Cov(B_0, B_t) = 0
        assert_allclose(fbm.cov(H, 0.0, t), 0.0, atol=1e-12)

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.8, 0.95])
    def test_acf_is_one_at_lag_zero(self, H):
        assert fbm.acf(H, 0.0) == pytest.approx(1.0)

    def test_acf_vanishes_for_brownian_increments(self):
        lags = numpy.arange(1.0, 20.0)
        assert_allclose(fbm.acf(0.5, lags), 0.0, atol=1e-12)

    @pytest.mark.parametrize("H, expected", [
        (1.0, 1.0),                        # 2^(2H-1) - 1 = 2 - 1
        (0.25, 2.0 ** -0.5 - 1.0),
        (0.75, 2.0 ** 0.5 - 1.0),
    ])
    def test_acf_lag_one_closed_form(self, H, expected):
        assert fbm.acf(H, 1.0) == pytest.approx(expected)

    def test_acf_sign_follows_persistence(self):
        lags = numpy.arange(1.0, 50.0)
        assert numpy.all(fbm.acf(0.8, lags) > 0.0)
        assert numpy.all(fbm.acf(0.3, lags) < 0.0)

    @pytest.mark.parametrize("H", [0.3, 0.7, 0.9])
    def test_acf_large_lag_power_law(self, H):
        # acf(k) ~ H(2H-1) k^(2H-2); the next term is O(k^-2) smaller, ~1e-6 at k=1000.
        k = 1000.0
        assert fbm.acf(H, k) / (H * (2.0 * H - 1.0) * k ** (2.0 * H - 2.0)) == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("H", [0.3, 0.5, 0.8])
    def test_acf_matrix_sums_to_motion_variance(self, H):
        # Var(B_H(n)) = Var(sum of n increments) = sum of the n x n increment covariance.
        n = 10
        assert fgn_cov(H, n).sum() == pytest.approx(fbm.var(H, float(n)), rel=1e-10)


# ===========================================================================
# Model layer: Cholesky generator
# ===========================================================================

class TestCholeskyGenerator:

    @pytest.mark.parametrize("H", [0.3, 0.5, 0.8])
    def test_cholesky_decompose_factors_fgn_covariance(self, H):
        n = 12
        L = numpy.asarray(fbm.cholesky_decompose(H, n))
        assert L.shape == (n + 1, n + 1)
        assert_allclose(L, numpy.tril(L))
        assert_allclose(L @ L.T, fgn_cov(H, n + 1), atol=1e-12)
        assert_allclose(L, numpy.linalg.cholesky(fgn_cov(H, n + 1)), atol=1e-12)

    def test_cholesky_decompose_is_identity_for_brownian_motion(self):
        assert_array_equal(numpy.asarray(fbm.cholesky_decompose(0.5, 8)), numpy.eye(9))

    def test_cholesky_noise_is_identity_map_for_brownian_motion(self):
        dB = numpy.random.normal(size=9)
        assert_allclose(fbm.cholesky_noise(0.5, 8, dB), dB)

    def test_cholesky_noise_validates_input_lengths(self):
        with pytest.raises(Exception, match="dB should have length 9"):
            fbm.cholesky_noise(0.7, 8, numpy.zeros(8))
        with pytest.raises(Exception, match="L should have length 9"):
            fbm.cholesky_noise(0.7, 8, numpy.zeros(9), L=numpy.eye(5))

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_cholesky_noise_has_fgn_covariance(self, H):
        # cholesky_noise is linear in dB; its matrix A must satisfy A A^T = Cov(fGn).
        n = 12
        A = linear_map(lambda e: fbm.cholesky_noise(H, n, e), n + 1)
        assert A.shape == (n + 1, n + 1)
        assert_allclose(A @ A.T, fgn_cov(H, n + 1), atol=1e-12)
        # Supplying the module's own decomposition must give the same map.
        L = fbm.cholesky_decompose(H, n)
        A_L = linear_map(lambda e: fbm.cholesky_noise(H, n, e, L), n + 1)
        assert_allclose(A_L, A, atol=1e-12)

    def test_generate_cholesky_is_cumulative_sum_from_zero(self):
        n = 16
        dB = numpy.random.normal(size=n + 1)
        Z = fbm.generate_cholesky(0.5, n, dB)
        assert len(Z) == n + 1
        assert Z[0] == 0.0
        # H = 1/2: increments are dB itself, the first one being discarded.
        assert_allclose(Z, numpy.cumsum(dB) - dB[0])
        with pytest.raises(Exception, match="dB should have length 17"):
            fbm.generate_cholesky(0.5, n, numpy.zeros(n))

    def test_generate_cholesky_accumulates_its_noise(self):
        n, H = 10, 0.7
        dB = numpy.random.normal(size=n + 1)
        L = fbm.cholesky_decompose(H, n)
        Z = fbm.generate_cholesky(H, n, dB, L)
        assert Z[0] == 0.0
        assert_allclose(numpy.diff(Z), fbm.cholesky_noise(H, n, dB, L)[1:])

    def test_generate_cholesky_variance_grows_as_t_2H(self):
        # Sample variance of nsim Gaussian draws has relative sd sqrt(2/nsim) ~ 0.032,
        # so a 20% band is >6 sigma at every time point.
        H, n, nsim = 0.3, 15, 2000
        L = numpy.linalg.cholesky(fgn_cov(H, n + 1))
        ensemble = numpy.array([fbm.generate_cholesky(H, n, L=L) for _ in range(nsim)])
        t = numpy.arange(1, n + 1, dtype=float)
        assert_allclose(ensemble.var(axis=0)[1:], fbm.var(H, t), rtol=0.2)
        assert_allclose(ensemble.mean(axis=0), 0.0, atol=0.2)


# ===========================================================================
# Model layer: FFT (circulant embedding) generator
# ===========================================================================

class TestFFTGenerator:

    @pytest.mark.parametrize("H", [0.3, 0.5, 0.8])
    def test_fft_noise_has_fgn_covariance(self, H):
        # fft_noise is linear in the 2n driving normals; the first n outputs
        # must have exactly the fGn covariance (top-left block of the circulant).
        n = 16
        A = linear_map(lambda e: fbm.fft_noise(H, n, e), 2 * n)
        assert A.shape == (n, 2 * n)
        assert_allclose(A @ A.T, fgn_cov(H, n), atol=1e-12)

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_fft_and_cholesky_generators_agree_in_distribution(self, H):
        # Two zero-mean Gaussian generators agree in law iff their Gram matrices agree.
        n = 16
        A_fft = linear_map(lambda e: fbm.fft_noise(H, n, e), 2 * n)
        L = numpy.asarray(fbm.cholesky_decompose(H, n - 1))
        assert_allclose(A_fft @ A_fft.T, L @ L.T, atol=1e-12)

    def test_fft_noise_validates_input_length(self):
        with pytest.raises(Exception, match="dB should have length 32"):
            fbm.fft_noise(0.7, 16, numpy.zeros(16))
        with pytest.raises(Exception, match="dB should have length 32"):
            fbm.generate_fft(0.7, 16, numpy.zeros(33))

    def test_fft_noise_supports_strongly_persistent_H(self):
        H, n = 0.95, 64
        A = linear_map(lambda e: fbm.fft_noise(H, n, e), 2 * n)
        assert_allclose(A @ A.T, fgn_cov(H, n), atol=1e-10)

    def test_generate_fft_starts_at_zero_and_accumulates_noise(self):
        n, H = 32, 0.7
        dB = numpy.random.normal(size=2 * n)
        Z = fbm.generate_fft(H, n, dB)
        assert len(Z) == n
        assert Z[0] == 0.0
        assert_allclose(numpy.diff(Z), fbm.fft_noise(H, n, dB)[1:])

    def test_generate_fft_variance_grows_as_t_2H(self):
        # Relative sd of a sample variance over nsim=2000 draws is ~0.032; 20% band is >6 sigma.
        H, n, nsim = 0.8, 16, 2000
        ensemble = numpy.array([fbm.generate_fft(H, n) for _ in range(nsim)])
        t = numpy.arange(1, n, dtype=float)
        assert_allclose(ensemble.var(axis=0)[1:], fbm.var(H, t), rtol=0.2)
        # The ensemble mean at time k has sd sqrt(k^(2H)/nsim), which grows with k,
        # so a flat atol is only ~2.4 sigma at the last point. Scale the bound with
        # the process sd instead: 6 sigma pointwise (worst observed over 60 fresh
        # seeds was 0.50 of this bound).
        bound = 6.0 * numpy.sqrt(fbm.var(H, numpy.arange(n, dtype=float)) / nsim) + 1e-12
        assert numpy.all(numpy.abs(ensemble.mean(axis=0)) < bound)


# ===========================================================================
# Model layer: variance ratio statistics
# ===========================================================================

class TestVarianceRatio:

    def test_vr_matches_hand_computed_overlapping_estimator(self):
        # Deterministic anchor for the overlapping Lo & MacKinlay estimator and the
        # m = (t - s + 1)(1 - s/t) normalisation stats.lag_var applies.
        #
        #   x  = [0, 1, 3, 2, 5, 4, 8],  t = 6,  mu = (x[6] - x[0])/6 = 4/3
        #   s = 1: m = 6(1 - 1/6) = 5
        #          sum (dx - 4/3)^2 = (1/9)(1 + 4 + 49 + 25 + 49 + 64) = 192/9
        #          lag_var = 192/45
        #   s = 2: m = 5(1 - 2/6) = 10/3
        #          sum (x_i - x_{i-2} - 8/3)^2 = (1/9)(1 + 25 + 4 + 4 + 1) = 35/9
        #          lag_var = (35/9)(3/10) = 7/6
        #   VR(2) = (7/6) / (2 * 192/45) = 315/2304 = 0.13671875
        x = numpy.array([0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 8.0])
        assert stats.lag_var(x, 1) == pytest.approx(192.0 / 45.0)
        assert stats.lag_var(x, 2) == pytest.approx(7.0 / 6.0)
        assert fbm.vr_scan(x, [2])[0] == pytest.approx(0.13671875)

        # s = 3: m = 4(1 - 3/6) = 2; x_i - x_{i-3} = 2, 4, 1, 6 so the centred
        # values are -2, 0, -3, 2 -> (4 + 0 + 9 + 4)/2 = 17/2
        # VR(3) = (17/2) / (3 * 192/45) = 765/1152 = 0.6640625
        assert stats.lag_var(x, 3) == pytest.approx(8.5)
        assert fbm.vr_scan(x, [3])[0] == pytest.approx(0.6640625)

    def test_vr_is_exactly_one_at_lag_one(self):
        # VR(1) = lag_var(x, 1) / (1 * lag_var(x, 1)) is an identity; assert it as a
        # documented contract only, the numeric anchor being the test above.
        x = random_walk(256)
        assert fbm.vr_scan(x, [1])[0] == 1.0

    def test_vr_scan_is_one_for_random_walk(self):
        # sd of VR(s)-1 is sqrt(theta) <= 0.069 for s <= 16 at t = 4095; 0.25 is >3.6 sigma.
        x = random_walk(4096)
        s_vals = [2, 4, 8, 16]
        vr = fbm.vr_scan(x, s_vals)
        assert vr.shape == (4,)
        assert_allclose(vr, 1.0, atol=0.25)

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_vr_scan_follows_s_to_2H_minus_1(self, H):
        # Population VR(s) = s^(2H-1). A single path is far too noisy to pin this
        # (the one-path relative error at s = 16 exceeds 30% for some seeds), so
        # average VR(s) over an ensemble: the per-seed sd of the mean then drops to
        # <= 1.6%, leaving only the estimator's finite-sample bias, which for the
        # persistent case is systematically negative and grows with the lag
        # (-1.2%, -2.8%, -4.9%, -7.6% at s = 2, 4, 8, 16). rtol = 0.15 is bias plus
        # ~4.5 sd; worst observed over 40 fresh seeds was 0.104 (H = 0.8) and
        # 0.037 (H = 0.3).
        s_vals = [2, 4, 8, 16]
        nsim, n = 30, 4096
        vr = numpy.mean([fbm.vr_scan(fbm.generate_fft(H, n), s_vals) for _ in range(nsim)], axis=0)
        expected = numpy.array(s_vals, dtype=float) ** (2.0 * H - 1.0)
        assert_allclose(vr, expected, rtol=0.15)
        # VR(s) is monotone in s with the sign of 2H - 1, which is never in doubt
        if H > 0.5:
            assert numpy.all(numpy.diff(vr) > 0.0)
        else:
            assert numpy.all(numpy.diff(vr) < 0.0)

    def test_homo_stat_matches_lo_mackinlay_formula(self):
        x = random_walk(1024)
        t = len(x) - 1
        s_vals = [1, 3, 7, 12]
        vr = fbm.vr_scan(x, s_vals)
        stat = fbm.vr_stat_homo_scan(x, s_vals)
        assert stat[0] == 0.0  # s = 1 is defined to be 0
        for i in range(1, len(s_vals)):
            assert stat[i] == pytest.approx((vr[i] - 1.0) / numpy.sqrt(lo_mackinlay_theta(s_vals[i], t)))

    def test_hetero_stat_agrees_with_homo_stat_for_gaussian_random_walk(self):
        # Both statistics share the (VR - 1) numerator, so this ratio is exactly
        # sqrt(theta_homo / theta_hetero) and measures nothing but the two theta
        # estimators. For iid Gaussian increments they coincide asymptotically; the
        # observed spread over 80 fresh seeds at n = 2048 had max 0.043, so 0.06 is
        # a real constraint (a 12% error in theta_hetero) rather than a 30% one.
        x = random_walk(2048)
        s_vals = [4, 6, 10, 16, 24]
        homo = fbm.vr_stat_homo_scan(x, s_vals)
        hetero = fbm.vr_stat_hetero_scan(x, s_vals)
        assert hetero.shape == (5,)
        assert_allclose(hetero / homo, 1.0, atol=0.06)

    @pytest.mark.parametrize("n, s", [(32, 8), (129, 6)])
    def test_hetero_stat_matches_lo_mackinlay_closed_form(self, n, s):
        x = random_walk(n)
        vr = fbm.vr_scan(x, [s])[0]
        expected = (vr - 1.0) / numpy.sqrt(lo_mackinlay_theta_hetero(x, s))
        assert fbm.vr_stat_hetero_scan(x, [s])[0] == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("n, s", [(32, 8), (129, 6), (513, 24)])
    def test_hetero_theta_is_the_closed_form(self, n, s):
        # Deterministic anchor for __theta_factor/__delta_factor: the closed form of
        # Lo & MacKinlay eq. (18) computed in the test must reproduce the shipped
        # statistic exactly. Anything in the summation (the [2(s-j)/s]^2 weights, the
        # mean adjustment, the squared-sum denominator, the k = t term) is pinned by
        # this equality. The drop_last/small_sample_norm switches on the helper are
        # the two deviations the library used to carry; it no longer needs them.
        x = random_walk(n)
        vr = fbm.vr_scan(x, [s])[0]
        theta = lo_mackinlay_theta_hetero(x, s)
        assert theta > 0.0
        assert fbm.vr_stat_hetero_scan(x, [s])[0] == pytest.approx((vr - 1.0) / numpy.sqrt(theta), rel=1e-12)

    def test_hetero_theta_scale_matches_the_homoscedastic_theta(self):
        # Recover the library's theta from the statistic it returns, theta =
        # ((VR - 1)/Z)^2, and pin its absolute scale against the closed-form
        # homoscedastic theta 2(2s-1)(s-1)/(3st), which is its iid-Gaussian limit.
        # This is an absolute anchor, not the ratio of the two statistics.
        x = random_walk(4096)
        t = len(x) - 1
        s_vals = [4, 10, 24]
        vr = fbm.vr_scan(x, s_vals)
        stat = fbm.vr_stat_hetero_scan(x, s_vals)
        theta = ((vr - 1.0) / stat) ** 2
        assert_allclose(theta, [lo_mackinlay_theta(s, t) for s in s_vals], rtol=0.15)

    def test_hetero_stat_is_zero_at_lag_one(self):
        x = random_walk(256)
        with numpy.errstate(invalid="ignore"):
            stat = fbm.vr_stat_hetero_scan(x, [1, 4])
        assert numpy.isfinite(stat[0])
        assert stat[0] == 0.0


# ===========================================================================
# Model layer: variance ratio hypothesis tests
# ===========================================================================

class TestVarianceRatioTests:

    def test_two_tail_report(self):
        x = random_walk(1024)
        s_vals = [4, 6, 10, 16, 24]
        report = fbm.vr_homo_test(x, s_vals, 0.1, HypothesisType.TWO_TAIL)
        assert isinstance(report, VarianceRatioTestReport)
        assert report.hyp_type is HypothesisType.TWO_TAIL
        assert report.hyp_test_type is HypothesisTestType.BM
        assert report.sig_level == pytest.approx(0.1)
        assert report.s_vals == s_vals
        assert_allclose(report.stats, fbm.vr_stat_homo_scan(x, s_vals))
        assert_allclose([present(v) for v in report.critical_values], [norm.ppf(0.05), norm.ppf(0.95)])
        assert_allclose(report.p_vals, 2.0 * (1.0 - norm.cdf(numpy.abs(report.stats))))
        expected_status = [bool(norm.ppf(0.05) < z < norm.ppf(0.95)) for z in report.stats]
        assert list(map(bool, report.status_vals)) == expected_status

    def test_upper_tail_report(self):
        x = random_walk(1024)
        report = fbm.vr_homo_test(x, hyp_type=HypothesisType.UPPER_TAIL)
        assert report.hyp_test_type is HypothesisTestType.FBM_AUTO_CORR
        assert report.s_vals == [4, 6, 10, 16, 24]
        assert report.critical_values[0] is None
        assert report.critical_values[1] == pytest.approx(norm.ppf(0.9))
        assert_allclose(report.p_vals, 1.0 - norm.cdf(report.stats))
        assert list(map(bool, report.status_vals)) == [bool(z < norm.ppf(0.9)) for z in report.stats]

    def test_lower_tail_report(self):
        x = random_walk(1024)
        report = fbm.vr_homo_test(x, hyp_type=HypothesisType.LOWER_TAIL)
        assert report.hyp_test_type is HypothesisTestType.FBM_NEG_AUTO_CORR
        assert report.critical_values[0] == pytest.approx(norm.ppf(0.1))
        assert report.critical_values[1] is None
        assert_allclose(report.p_vals, norm.cdf(report.stats))
        assert list(map(bool, report.status_vals)) == [bool(z > norm.ppf(0.1)) for z in report.stats]

    def test_custom_significance_and_lags(self):
        x = random_walk(512)
        report = fbm.vr_homo_test(x, s_vals=[2, 8], sig_level=0.05)
        assert report.s_vals == [2, 8]
        assert len(report.stats) == 2 and len(report.p_vals) == 2
        assert report.sig_level == pytest.approx(0.05)
        assert_allclose([present(v) for v in report.critical_values], [-1.959963984540054, 1.959963984540054])

    def test_hetero_test_uses_hetero_statistics(self):
        x = random_walk(1024)
        s_vals = [4, 8]
        report = fbm.vr_hetero_test(x, s_vals, 0.1, HypothesisType.UPPER_TAIL)
        assert_allclose(report.stats, fbm.vr_stat_hetero_scan(x, s_vals))
        assert report.hyp_test_type is HypothesisTestType.FBM_AUTO_CORR
        assert report.critical_values == [None, pytest.approx(norm.ppf(0.9))]

    @pytest.mark.parametrize("test_fn", [fbm.vr_homo_test, fbm.vr_hetero_test])
    def test_unknown_hypothesis_type_is_rejected(self, test_fn):
        # __vr_test dispatches on hyp_type.value; anything outside
        # TWO_TAIL/UPPER_TAIL/LOWER_TAIL falls through to the raise. Passing a member
        # of the sibling str enum reaches it with a valid .value attribute.
        x = random_walk(256)
        with pytest.raises(Exception, match="Hypothesis test type is invalid"):
            test_fn(x, [4, 8], 0.1, HypothesisTestType.STATIONARITY)

    def test_report_summary_and_repr(self, capsys):
        x = random_walk(512)
        report = fbm.vr_homo_test(x)
        report.summary()
        out = capsys.readouterr().out
        assert "Lower Critical Value" in out and "Upper Critical Value" in out
        assert "10%" in out and "Z(s)" in out and "pvalue" in out
        assert repr(report).startswith("VarianceRatioTestReport(status_vals=")
        assert "hyp_test_type=BM" in repr(report)


# ===========================================================================
# Facade: theoretical curves
# ===========================================================================

class TestFacadeTheory:

    def test_compute_mean_defaults(self):
        t, mean = fbm_data.compute_mean()
        assert_allclose(t, numpy.arange(11, dtype=float))
        assert_array_equal(mean, numpy.zeros(11))

    def test_compute_mean_custom_grid_and_level(self):
        t, mean = fbm_data.compute_mean(npts=5, Δt=0.5, μ=2.5)
        assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0])
        assert_array_equal(mean, numpy.full(5, 2.5))

    def test_compute_var_requires_H(self):
        with pytest.raises(Exception, match="H parameter is required"):
            fbm_data.compute_var(npts=5)

    def test_compute_var_requires_a_grid_when_t_absent(self):
        with pytest.raises(Exception, match="xmax or npts is required"):
            fbm_data.compute_var(H=0.5)

    def test_compute_var_uses_given_time(self):
        t_in = numpy.array([1.0, 4.0, 9.0])
        t, var = fbm_data.compute_var(t_in, H=0.25)
        assert t is t_in
        assert_allclose(var, [1.0, 2.0, 3.0])

    def test_compute_var_builds_grid_from_npts_or_tmax(self):
        t, var = fbm_data.compute_var(H=0.5, npts=4, Δt=0.5)
        assert_allclose(t, [0.0, 0.5, 1.0, 1.5])
        assert_allclose(var, t)
        t, var = fbm_data.compute_var(H=1.0, tmax=3.0)
        assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
        assert_allclose(var, [0.0, 1.0, 4.0, 9.0])

    def test_compute_sd_is_sqrt_of_var(self):
        t, sd = fbm_data.compute_sd(H=0.5, npts=5)
        assert_allclose(t, numpy.arange(5, dtype=float))
        assert_allclose(sd, numpy.sqrt(t))
        t, sd = fbm_data.compute_sd(H=0.8, npts=4)
        assert_allclose(sd, t ** 0.8)
        with pytest.raises(Exception, match="H parameter is required"):
            fbm_data.compute_sd(npts=4)

    def test_compute_acf_defaults_and_brownian_limit(self):
        # Lags start at 1: the acf formula 1/2[(n-1)^2H + (n+1)^2H - 2n^2H]
        # evaluates (-1)^2H at n = 0, which is complex for non-integer 2H, so
        # lag 0 is outside its domain (and is trivially 1 by definition anyway).
        t, acf = fbm_data.compute_acf(H=0.5)
        assert_allclose(t, numpy.arange(1, 12, dtype=float))
        # H = 0.5 is ordinary Brownian motion: its increments are uncorrelated
        # at every lag >= 1.
        assert_allclose(acf, numpy.zeros(11), atol=1e-12)

    def test_compute_acf_grid_and_model_agreement(self):
        t, acf = fbm_data.compute_acf(H=0.8, nlags=4, Δt=0.5)
        assert_allclose(t, [1.0, 1.5, 2.0, 2.5])
        assert_allclose(acf, fbm.acf(0.8, t))
        # lag 1 is the first entry now, and rho^H_1 = 2^(2H-1) - 1
        assert acf[0] == pytest.approx(2.0 ** 0.6 - 1.0)
        with pytest.raises(Exception, match="H parameter is required"):
            fbm_data.compute_acf(nlags=4)

    def test_compute_cov_brownian_limit(self):
        t, cov = fbm_data.compute_cov(H=0.5, s=3.0, tmax=6.0)
        assert_allclose(t, numpy.arange(7, dtype=float))
        assert_allclose(cov, numpy.minimum(t, 3.0))

    def test_compute_cov_grid_and_required_params(self):
        t, cov = fbm_data.compute_cov(H=0.8, s=1.0, tmin=1.0, npts=3, Δt=2.0)
        assert_allclose(t, [1.0, 3.0, 5.0])
        assert_allclose(cov, fbm.cov(0.8, 1.0, t))
        assert cov[0] == pytest.approx(1.0)  # Cov(B_1, B_1) = Var(B_1) = 1
        with pytest.raises(Exception, match="s parameter is required"):
            fbm_data.compute_cov(H=0.8, npts=3)
        with pytest.raises(Exception, match="H parameter is required"):
            fbm_data.compute_cov(s=1.0, npts=3)

    def test_compute_vr_power_law(self):
        t_in = numpy.array([1.0, 2.0, 4.0, 8.0])
        t, vr = fbm_data.compute_vr(t_in, H=0.8)
        assert t is t_in
        assert_allclose(vr, t_in ** 0.6)
        t, vr = fbm_data.compute_vr(H=0.5, npts=4, Δt=0.5)
        assert_allclose(t, [0.0, 0.5, 1.0, 1.5])
        assert_allclose(vr, 1.0)
        with pytest.raises(Exception, match="H parameter is required"):
            fbm_data.compute_vr(t_in)

    def test_compute_vr_auto_grid_diverges_at_the_origin_for_antipersistent_H(self):
        # The auto-generated grid starts at t = 0 and VR(t) = t^(2H-1) has a negative
        # exponent for H < 1/2, so the first point is a 0**negative division: numpy
        # emits a RuntimeWarning and returns +inf. Pinned because notebooks plot this
        # curve on a log axis where the leading inf is silently dropped.
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            t, vr = fbm_data.compute_vr(H=0.3, npts=4)
        assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
        assert numpy.isinf(vr[0]) and vr[0] > 0.0
        assert_allclose(vr[1:], [1.0, 2.0 ** -0.4, 3.0 ** -0.4])

    def test_compute_vr_auto_grid_is_finite_for_persistent_H(self):
        # H > 1/2 has a positive exponent, so the same grid is finite and starts at 0.
        t, vr = fbm_data.compute_vr(H=0.8, npts=4)
        assert_allclose(vr, [0.0, 1.0, 2.0 ** 0.6, 3.0 ** 0.6])


# ===========================================================================
# Facade: lag scans
# ===========================================================================

class TestFacadeScans:

    def test_compute_vr_scan_with_explicit_lags(self):
        x = random_walk(512)
        s_vals, vr = fbm_data.compute_vr_scan(x, svals=[2.0, 4.0, 8.0])
        assert s_vals == [2, 4, 8]
        assert all(isinstance(s, int) for s in s_vals)
        assert_allclose(vr, fbm.vr_scan(x, [2, 4, 8]))

    def test_compute_vr_scan_generates_linear_and_log_lags(self):
        x = random_walk(512)
        s_vals, vr = fbm_data.compute_vr_scan(x, linear=True, smin=2, smax=10, npts=5)
        assert s_vals == [2, 4, 6, 8, 10]
        # the returned ratios must be the ratios at the returned lags
        assert_allclose(vr, fbm.vr_scan(x, s_vals))
        s_vals, vr = fbm_data.compute_vr_scan(x, smax=100, npts=3)
        assert s_vals == [1, 10, 100]
        assert_allclose(vr, fbm.vr_scan(x, s_vals))

    @pytest.mark.parametrize("scan, model_fn", [
        (fbm_data.compute_vr_scan, fbm.vr_scan),
        (fbm_data.compute_homo_vr_stat_scan, fbm.vr_stat_homo_scan),
        (fbm_data.compute_hetero_vr_stat_scan, fbm.vr_stat_hetero_scan),
    ])
    def test_log_scan_with_non_unit_smin_is_still_self_consistent(self, scan, model_fn):
        # smin is a documented kwarg on all three log-scale scans; whatever lags come
        # back, the values must belong to them.
        x = random_walk(1024)
        s_vals, vals = scan(x, smin=2, smax=100, npts=3)
        assert len(s_vals) == 3
        assert s_vals[0] == 2
        assert_allclose(vals, model_fn(x, s_vals))

    @pytest.mark.parametrize("scan", [
        fbm_data.compute_vr_scan,
        fbm_data.compute_homo_vr_stat_scan,
        fbm_data.compute_hetero_vr_stat_scan,
    ])
    def test_log_scan_with_non_unit_smin_reaches_smax(self, scan):
        x = random_walk(1024)
        s_vals, _ = scan(x, smin=2, smax=100, npts=3)
        assert s_vals[-1] == 100

    def test_compute_vr_scan_requires_lag_specification(self):
        with pytest.raises(Exception, match="smax and npts or svals is required"):
            fbm_data.compute_vr_scan(random_walk(64))

    def test_compute_homo_vr_stat_scan_agrees_with_model_layer(self):
        x = random_walk(1024)
        s_vals, stat = fbm_data.compute_homo_vr_stat_scan(x, svals=[4, 8, 16])
        assert s_vals == [4, 8, 16]
        assert_allclose(stat, fbm.vr_stat_homo_scan(x, [4, 8, 16]))

    def test_compute_hetero_vr_stat_scan_agrees_with_model_layer(self):
        x = random_walk(1024)
        s_vals, stat = fbm_data.compute_hetero_vr_stat_scan(x, linear=True, smin=4, smax=8, npts=2)
        assert s_vals == [4, 8]
        assert_allclose(stat, fbm.vr_stat_hetero_scan(x, [4, 8]))


# ===========================================================================
# Facade: simulation sources
# ===========================================================================

class TestFacadeSources:

    @pytest.mark.parametrize("source", [
        fbm_data.create_noise_cholesky_source,
        fbm_data.create_noise_fft_source,
        fbm_data.create_cholesky_source,
        fbm_data.create_fft_source,
    ])
    def test_sources_require_H(self, source):
        with pytest.raises(Exception, match="H parameter is required"):
            source(npts=16)

    def test_noise_fft_source_contract_and_default_npts(self):
        t, noise = fbm_data.create_noise_fft_source(H=0.7)
        assert len(t) == len(noise) == 1024
        assert_allclose(t, numpy.arange(1024, dtype=float))
        # fGn is unit variance by construction, and 1024 samples put the sample
        # variance's sd at sqrt(2/1024) ~ 0.044 before the long-memory inflation, so
        # a 15% band is a real check. The lag-1 sample autocorrelation of a 1024
        # point H = 0.7 path concentrates within ~0.05 of acf(0.7, 1) = 0.3195.
        assert noise.var() == pytest.approx(1.0, rel=0.15)
        assert numpy.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(fbm.acf(0.7, 1.0), abs=0.1)

    def test_cholesky_source_default_npts(self):
        # The only exercise of create_cholesky_source's npts=1024 default and of the
        # L=None fallback at that size (a pure-Python O(npts^2) ACF matrix build).
        t, Z = fbm_data.create_cholesky_source(H=0.7)
        # generate_cholesky(H, npts) returns npts + 1 points, and the façade's
        # time axis matches it.
        assert len(t) == 1025
        assert_allclose(t, numpy.arange(1025, dtype=float))
        assert Z[0] == 0.0
        assert numpy.all(numpy.isfinite(Z))
        # increments of the generated motion are unit-variance fGn with the H = 0.7 acf
        dZ = numpy.diff(Z)
        assert dZ.var() == pytest.approx(1.0, rel=0.2)
        assert numpy.corrcoef(dZ[:-1], dZ[1:])[0, 1] == pytest.approx(fbm.acf(0.7, 1.0), abs=0.12)

    def test_noise_cholesky_source_default_npts(self):
        t, noise = fbm_data.create_noise_cholesky_source(H=0.7)
        assert len(t) == 1025
        assert noise.var() == pytest.approx(1.0, rel=0.2)
        assert numpy.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(fbm.acf(0.7, 1.0), abs=0.12)

    def test_noise_fft_source_agrees_with_model_layer(self):
        dB = numpy.random.normal(size=64)
        t, noise = fbm_data.create_noise_fft_source(H=0.3, npts=32, Δt=0.25, dB=dB)
        assert_allclose(t, 0.25 * numpy.arange(32))
        assert_allclose(noise, 0.25**0.3 * fbm.fft_noise(0.3, 32, dB))
        with pytest.raises(Exception, match="dB should have length 64"):
            fbm_data.create_noise_fft_source(H=0.3, npts=32, dB=dB[:10])

    def test_fft_source_agrees_with_model_layer(self):
        dB = numpy.random.normal(size=64)
        t, Z = fbm_data.create_fft_source(H=0.8, npts=32, Δt=2.0, dB=dB)
        assert len(t) == len(Z) == 32
        assert_allclose(t, 2.0 * numpy.arange(32))
        assert Z[0] == 0.0
        assert_allclose(Z, 2.0**0.8 * fbm.generate_fft(0.8, 32, dB))

    def test_noise_cholesky_source_agrees_with_model_layer(self):
        npts = 16
        dB = numpy.random.normal(size=npts + 1)
        L = fbm.cholesky_decompose(0.7, npts)
        t, noise = fbm_data.create_noise_cholesky_source(H=0.7, npts=npts, Δt=0.5, dB=dB, L=L)
        assert t[0] == 0.0 and t[1] - t[0] == pytest.approx(0.5)
        assert_allclose(noise, 0.5**0.7 * fbm.cholesky_noise(0.7, npts, dB, L))
        # Without L the facade must fall back to the numpy Cholesky of the ACF matrix.
        # It also defaults to Δt = 1, so the Δt**H factor above has to be divided out.
        _, noise_default = fbm_data.create_noise_cholesky_source(H=0.7, npts=npts, dB=dB)
        assert_allclose(noise_default, noise / 0.5**0.7, atol=1e-12)

    def test_cholesky_source_agrees_with_model_layer(self):
        npts = 16
        dB = numpy.random.normal(size=npts + 1)
        t, Z = fbm_data.create_cholesky_source(H=0.3, npts=npts, Δt=0.5, dB=dB)
        assert t[0] == 0.0 and t[1] - t[0] == pytest.approx(0.5)
        assert Z[0] == 0.0
        assert_allclose(Z, 0.5**0.3 * fbm.generate_cholesky(0.3, npts, dB))

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_motion_sources_are_self_consistent_on_the_unit_grid(self, H):
        # Baseline for the Δt tests below: at Δt = 1 the returned (t, values) pair is
        # a genuine fBm on that grid, Var(Z_k) = t_k^(2H). Sample variance over 400
        # draws has relative sd sqrt(2/400) = 0.07, so 30% is >4 sigma.
        npts, nsim = 16, 400
        t, _ = fbm_data.create_fft_source(H=H, npts=npts)
        ensemble = numpy.array([fbm_data.create_fft_source(H=H, npts=npts)[1] for _ in range(nsim)])
        assert_allclose(ensemble.var(axis=0)[1:], fbm.var(H, t[1:]), rtol=0.3)

    @pytest.mark.parametrize("source", [fbm_data.create_fft_source, fbm_data.create_cholesky_source])
    def test_motion_sources_are_self_consistent_on_a_non_unit_grid(self, source):
        H, npts, Δt, nsim = 0.5, 16, 4.0, 400
        L = numpy.linalg.cholesky(fgn_cov(H, npts + 1))
        kwargs = {"H": H, "npts": npts, "Δt": Δt}
        if source is fbm_data.create_cholesky_source:
            kwargs["L"] = L
        t, _ = source(**kwargs)
        # the two sources differ in length by one (fft returns npts, cholesky
        # npts + 1); what is pinned here is the spacing, not that convention
        assert_allclose(t, Δt * numpy.arange(len(t), dtype=float))
        ensemble = numpy.array([source(**kwargs)[1] for _ in range(nsim)])
        assert_allclose(ensemble.var(axis=0)[1:], fbm.var(H, t[1:]), rtol=0.3)

    def test_noise_source_variance_scales_with_the_time_step(self):
        H, npts, Δt, nsim = 0.7, 16, 4.0, 400
        ensemble = numpy.array([fbm_data.create_noise_fft_source(H=H, npts=npts, Δt=Δt)[1]
                                for _ in range(nsim)])
        assert ensemble.var() == pytest.approx(Δt ** (2.0 * H), rel=0.2)

    def test_noise_cholesky_source_time_and_values_have_same_length(self):
        t, noise = fbm_data.create_noise_cholesky_source(H=0.7, npts=16)
        assert len(t) == len(noise)

    def test_cholesky_source_time_and_values_have_same_length(self):
        t, Z = fbm_data.create_cholesky_source(H=0.7, npts=16)
        assert len(t) == len(Z)


# ===========================================================================
# Facade: Hurst parameter estimation
# ===========================================================================

class TestHurstEstimation:

    def test_periodogram_estimate_recovers_exact_power_law(self):
        H, C = 0.7, 3.0
        freq = numpy.arange(1.0, 200.0)
        pspec = C * freq ** (1.0 - 2.0 * H)
        report, result = fbm_data.compute_H_estimate_periodogram(freq, pspec)
        assert isinstance(report, RegressionResultsWrapper)
        assert isinstance(result, OLSResult)
        assert report.rsquared == pytest.approx(1.0)
        H_param = present(result.param_transforms)[0].param
        C_param = present(result.const_transform).param
        assert H_param.est == pytest.approx(H, abs=1e-10)
        assert H_param.err == pytest.approx(0.0, abs=1e-8)
        assert C_param.est == pytest.approx(C, rel=1e-10)
        assert C_param.err == pytest.approx(0.0, abs=1e-8)
        assert H_param.param_type == OLSParamType.TRANS_PARAM.value
        assert C_param.param_type == OLSParamType.TRANS_CONST.value
        assert H_param.est_id == C_param.est_id == result.est_id
        assert "1 - 2H" in present(result.model)
        # the transform is built from the raw OLS slope/const, which must also be exact
        assert result.params[0].est == pytest.approx(1.0 - 2.0 * H, abs=1e-10)
        assert result.const.est == pytest.approx(numpy.log10(C), abs=1e-10)

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_periodogram_estimate_round_trip(self, H):
        # Mirrors fbm_estimation_periodogram.ipynb. Over 12 seeds at n = 2048 the
        # estimate stayed within 0.08 of H (sd ~0.02 plus a small high-frequency bias).
        t, noise = fbm_data.create_noise_fft_source(H=H, npts=2048)
        freq, pspec = t[1:], stats.pspec(noise)
        _, result = fbm_data.compute_H_estimate_periodogram(freq, pspec)
        assert present(result.param_transforms)[0].param.est == pytest.approx(H, abs=0.12)
        assert present(result.param_transforms)[0].param.err < 0.05

    def test_variance_aggregation_estimate_recovers_exact_power_law(self):
        H, sigma2 = 0.6, 2.0
        m_vals = numpy.arange(1.0, 50.0)
        agg_var = sigma2 * m_vals ** (2.0 * (H - 1.0))
        report, result = fbm_data.compute_H_estimate_variance_aggregation(m_vals, agg_var)
        assert isinstance(report, RegressionResultsWrapper)
        assert report.rsquared == pytest.approx(1.0)
        H_param = present(result.param_transforms)[0].param
        s2_param = present(result.const_transform).param
        assert H_param.est == pytest.approx(H, abs=1e-10)
        assert H_param.err == pytest.approx(0.0, abs=1e-8)
        assert s2_param.est == pytest.approx(sigma2, rel=1e-10)
        assert H_param.param_type == OLSParamType.TRANS_PARAM.value
        assert s2_param.param_type == OLSParamType.TRANS_CONST.value
        assert "H-1" in present(result.model)
        assert result.params[0].est == pytest.approx(2.0 * (H - 1.0), abs=1e-10)

    @pytest.mark.parametrize("H", [0.3, 0.8])
    def test_variance_aggregation_estimate_round_trip(self, H):
        # Mirrors fbm_estimation_variance_aggregation.ipynb (m = 1..100). Over 12 seeds
        # at n = 2048 the estimate stayed within 0.11 of H (sd ~0.035, downward bias ~0.04).
        _, noise = fbm_data.create_noise_fft_source(H=H, npts=2048)
        m_vals = numpy.linspace(1.0, 100.0, 100)
        agg_var = stats.agg_var(noise, m_vals)
        _, result = fbm_data.compute_H_estimate_variance_aggregation(m_vals, agg_var)
        assert present(result.param_transforms)[0].param.est == pytest.approx(H, abs=0.2)


# ===========================================================================
# Facade: variance ratio hypothesis tests and their report models
# ===========================================================================

class TestFacadeVarianceRatioTests:

    @pytest.mark.parametrize("hyp_test_type, hyp_type, has_lower, has_upper, desc", [
        (HypothesisTestType.BM, HypothesisType.TWO_TAIL, True, True, "Brownian Motion Test"),
        (HypothesisTestType.FBM_AUTO_CORR, HypothesisType.UPPER_TAIL, False, True, "Autocorrelation Test"),
        (HypothesisTestType.FBM_NEG_AUTO_CORR, HypothesisType.LOWER_TAIL, True, False, "Negative Autocorrelation Test"),
    ])
    def test_compute_vr_test_dispatch_and_report_model(self, hyp_test_type, hyp_type, has_lower, has_upper, desc):
        x = random_walk(1024)
        report, model = fbm_data.compute_vr_test(x, hyp_test_type)
        assert isinstance(report, VarianceRatioTestReport)
        assert isinstance(model, StatisticalTestReport)
        assert report.hyp_type is hyp_type
        assert report.hyp_test_type is hyp_test_type
        assert_allclose(report.stats, fbm.vr_stat_homo_scan(x, [4, 6, 10, 16, 24]))

        assert model.hyp_type is hyp_type
        assert model.hyp_test_type is hyp_test_type
        assert model.desc == desc
        assert model.status is expected_report_status(hyp_test_type, report.status_vals)
        uuid.UUID(model.test_id)
        assert len(model.test_data) == 5
        for i, row in enumerate(model.test_data):
            assert row.test_id == model.test_id
            assert row.status is HypothesisTestStatus.from_bool(report.status_vals[i])
            assert present(row.stat).value == pytest.approx(report.stats[i])
            assert present(row.pval).value == pytest.approx(report.p_vals[i])
            assert [p.value for p in row.params] == [report.s_vals[i]]
            assert present(row.sig).label == "10%" and present(row.sig).value == pytest.approx(0.1)
            assert (row.lower is not None) is has_lower
            assert (row.upper is not None) is has_upper
            if has_lower:
                assert present(row.lower).value == pytest.approx(report.critical_values[0])
            if has_upper:
                assert present(row.upper).value == pytest.approx(report.critical_values[1])

    def test_compute_vr_test_kwargs(self):
        x = random_walk(512)
        report, model = fbm_data.compute_vr_test(x, HypothesisTestType.BM, sig_level=0.05, s=[2, 8])
        assert report.s_vals == [2, 8]
        assert_allclose([present(v) for v in report.critical_values], [norm.ppf(0.025), norm.ppf(0.975)])
        assert len(model.test_data) == 2
        assert present(model.test_data[0].sig).label == "5%"
        assert model.test_data[1].params[0].value == 8

    def test_compute_vr_test_rejects_non_list_lags_and_unknown_test(self):
        x = random_walk(256)
        with pytest.raises(Exception, match="Expected <class 'list'>"):
            fbm_data.compute_vr_test(x, HypothesisTestType.BM, s=(4, 8))
        with pytest.raises(Exception, match="Hypothesis test type is invalid"):
            fbm_data.compute_vr_test(x, HypothesisTestType.STATIONARITY)

    @pytest.mark.parametrize("hyp_test_type, hyp_type", [
        (HypothesisTestType.BM, HypothesisType.TWO_TAIL),
        (HypothesisTestType.FBM_AUTO_CORR, HypothesisType.UPPER_TAIL),
        (HypothesisTestType.FBM_NEG_AUTO_CORR, HypothesisType.LOWER_TAIL),
    ])
    def test_compute_hetero_vr_test_dispatch(self, hyp_test_type, hyp_type):
        x = random_walk(1024)
        report, model = fbm_data.compute_hetero_vr_test(x, hyp_test_type, s=[4, 10])
        assert report.hyp_type is hyp_type
        assert report.hyp_test_type is hyp_test_type
        assert_allclose(report.stats, fbm.vr_stat_hetero_scan(x, [4, 10]))
        assert model.hyp_test_type is hyp_test_type
        assert [row.params[0].value for row in model.test_data] == [4, 10]
        assert model.status is expected_report_status(hyp_test_type, report.status_vals)

    def test_compute_hetero_vr_test_rejects_unknown_test(self):
        with pytest.raises(Exception, match="Hypothesis test type is invalid"):
            fbm_data.compute_hetero_vr_test(random_walk(256), HypothesisTestType.STATIONARITY_DRIFT)

    # -- round trips: simulate with known H, run the test the notebook runs --

    @pytest.mark.parametrize("hyp_test_type", [HypothesisTestType.BM,
                                               HypothesisTestType.FBM_AUTO_CORR,
                                               HypothesisTestType.FBM_NEG_AUTO_CORR])
    def test_report_status_rollup_rules(self, hyp_test_type):
        # Pin the enum's roll-up over all 32 patterns of five lag outcomes against the
        # rule written independently in expected_report_status.
        for bits in range(32):
            status_vals = [bool(bits >> i & 1) for i in range(5)]
            assert hyp_test_type.status(status_vals) is expected_report_status(hyp_test_type, status_vals)
        # the boundaries are what the enum docstrings claim
        all_inside = HypothesisTestStatus.PASSED if hyp_test_type is HypothesisTestType.BM \
            else HypothesisTestStatus.FAILED
        assert hyp_test_type.status([True] * 5) is all_inside
        assert hyp_test_type.status([False] * 5) is (
            HypothesisTestStatus.FAILED if hyp_test_type is HypothesisTestType.BM
            else HypothesisTestStatus.PASSED)

    def test_random_walk_passes_brownian_motion_test(self):
        # A random walk's five Z(s) are strongly correlated, so at the 10% default a
        # mildly drifting path can push all five outside the band at once and flip the
        # BM status (5/400 fresh seeds do). Test the null at sig_level = 0.01, where
        # the band is wide enough: 0/400 fresh seeds failed either the homoscedastic
        # or the heteroscedastic roll-up.
        x = random_walk(4096)
        _, homo = fbm_data.compute_vr_test(x, HypothesisTestType.BM, sig_level=0.01)
        _, hetero = fbm_data.compute_hetero_vr_test(x, HypothesisTestType.BM, sig_level=0.01)
        assert homo.status is HypothesisTestStatus.PASSED
        assert hetero.status is HypothesisTestStatus.PASSED
        # Z(s) is asymptotically standard normal under the null; max |Z| over the five
        # lags stayed below 3.33 across those same 400 seeds.
        assert numpy.all(numpy.abs([present(row.stat).value for row in homo.test_data]) < 4.5)
        assert numpy.all(numpy.abs([present(row.stat).value for row in hetero.test_data]) < 4.5)

    def test_persistent_fbm_is_detected_as_positively_autocorrelated(self):
        # H = 0.8: VR(4) ~ 4^0.6 = 2.3 gives Z ~ 40 at n = 4096; the sign is never in doubt.
        _, Z = fbm_data.create_fft_source(H=0.8, npts=4096)
        report, auto = fbm_data.compute_vr_test(Z, HypothesisTestType.FBM_AUTO_CORR)
        assert numpy.all(report.stats > 10.0)
        assert auto.status is HypothesisTestStatus.PASSED
        _, bm_test = fbm_data.compute_vr_test(Z, HypothesisTestType.BM)
        assert bm_test.status is HypothesisTestStatus.FAILED
        _, neg = fbm_data.compute_vr_test(Z, HypothesisTestType.FBM_NEG_AUTO_CORR)
        assert neg.status is HypothesisTestStatus.FAILED
        _, hetero = fbm_data.compute_hetero_vr_test(Z, HypothesisTestType.FBM_AUTO_CORR)
        assert hetero.status is HypothesisTestStatus.PASSED

    def test_antipersistent_fbm_is_detected_as_negatively_autocorrelated(self):
        # H = 0.3: VR(4) ~ 4^-0.4 = 0.57 gives Z ~ -14 at n = 4096 (observed [-15, -8]).
        _, Z = fbm_data.create_fft_source(H=0.3, npts=4096)
        report, neg = fbm_data.compute_vr_test(Z, HypothesisTestType.FBM_NEG_AUTO_CORR)
        assert numpy.all(report.stats < -4.0)
        assert neg.status is HypothesisTestStatus.PASSED
        _, bm_test = fbm_data.compute_vr_test(Z, HypothesisTestType.BM)
        assert bm_test.status is HypothesisTestStatus.FAILED
        _, auto = fbm_data.compute_vr_test(Z, HypothesisTestType.FBM_AUTO_CORR)
        assert auto.status is HypothesisTestStatus.FAILED


# ===========================================================================
# Facade: report serialization
# ===========================================================================

class TestReportSerialization:

    def test_statistical_report_to_json_and_row_round_trip(self):
        x = random_walk(1024)
        report, model = fbm_data.compute_vr_test(x, HypothesisTestType.BM)
        data = json.loads(model.to_json())
        assert data["test_id"] == model.test_id
        assert data["status"] == model.status.value
        assert data["hyp_type"] == "TWO_TAIL"
        assert data["hyp_test_type"] == "BM"
        assert data["desc"] == "Brownian Motion Test"
        assert len(data["test_data"]) == 5
        row = StatisticalTestData.from_dict(data["test_data"][2])
        assert row.test_id == model.test_id
        assert row.status == model.test_data[2].status
        assert present(row.stat).value == pytest.approx(report.stats[2])
        assert present(row.pval).value == pytest.approx(report.p_vals[2])
        assert row.params[0].value == 10
        assert present(row.sig).value == pytest.approx(0.1)
        assert present(row.lower).value == pytest.approx(norm.ppf(0.05))
        assert present(row.upper).value == pytest.approx(norm.ppf(0.95))
        assert "TestReport(test_id=(" in repr(model)

    def test_statistical_report_from_dict_round_trip(self):
        x = random_walk(512)
        _, model = fbm_data.compute_vr_test(x, HypothesisTestType.BM)
        restored = StatisticalTestReport.from_dict(json.loads(model.to_json()))
        assert restored.test_id == model.test_id
        assert restored.status == model.status
        assert restored.hyp_test_type == HypothesisTestType.BM
        assert len(restored.test_data) == len(model.test_data)


    @pytest.mark.parametrize("hyp_test_type", [HypothesisTestType.FBM_AUTO_CORR,
                                               HypothesisTestType.FBM_NEG_AUTO_CORR])
    def test_one_tailed_row_round_trip(self, hyp_test_type):
        x = random_walk(512)
        report, model = fbm_data.compute_vr_test(x, hyp_test_type)
        data = json.loads(model.to_json())
        # exactly one of the two bounds is serialised as null for a one-tailed test
        assert (data["test_data"][0]["lower"] is None) != (data["test_data"][0]["upper"] is None)
        row = StatisticalTestData.from_dict(data["test_data"][0])
        assert row.test_id == model.test_id
        assert present(row.stat).value == pytest.approx(report.stats[0])
        assert row.params[0].value == report.s_vals[0]
