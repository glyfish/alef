"""
Tests for navi's Dickey-Fuller simulation and ADF test wrappers.

  lib.models.adf     -- model layer: Brownian-noise simulation, the stochastic
                        integrals that define the Dickey-Fuller distribution,
                        the raw DF statistic and the statsmodels ADF wrappers.
  lib.data.impl.adf  -- **kwargs facade returning (t, values) / report tuples;
                        this is what the notebooks call.

Model-layer functions are checked against closed forms (the Ito identity
int B dB = (B(1)^2 - 1)/2, the chi-squared transform behind the modified
chi-squared pdf, exact moments of the discretised integrals, the Dickey-Fuller
regression as computed independently by statsmodels). The facade is checked
for its contract and for agreeing with the model layer under the same seed.

Every simulation draws from numpy's global RNG, which tests/conftest.py reseeds
before each test; tests that need to replay a stream twice reseed explicitly.
"""
import json
import re
import uuid

import numpy
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats
from scipy.signal import lfilter

from lib.models import adf
from lib.data.impl import adf as fadf
from lib.data.reports import ADFTestReport
from lib.data.hyp_test import (HypothesisTestStatus, HypothesisTestType, HypothesisType,
                               StatisticalTestParam, StatisticalTestReport)

SEED = 20260820

# MacKinnon 5% critical values of the ADF t-statistic for ~1000 observations,
# by statsmodels regression code: no constant, constant, constant + trend.
MACKINNON_5PCT = {"n": -1.95, "c": -2.86, "ct": -3.41}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def random_walk(n: int, drift: float = 0.0, σ: float = 1.0) -> numpy.ndarray:
    return numpy.cumsum(drift + numpy.random.normal(0.0, σ, n))


def ar1(n: int, φ: float, offset: float = 0.0, σ: float = 1.0) -> numpy.ndarray:
    """x_t = offset + φ x_{t-1} + ε_t, started from zero."""
    ε = numpy.random.normal(0.0, σ, n)
    return lfilter([1.0], [1.0, -φ], offset + ε)


def hand_noise() -> numpy.ndarray:
    """Tiny 'noise' vector whose partial sums (0, 1, 3, 6, 10) are easy to hand-check."""
    return numpy.array([1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# scaled_brownian_noise / brownian_motion
# ---------------------------------------------------------------------------

def test_scaled_brownian_noise_shape_and_moments():
    n = 4000
    bn = adf.scaled_brownian_noise(n)

    assert bn.shape == (n,)
    assert bn.dtype == numpy.float64
    # Var = 1/n. Sample variance has relative std error sqrt(2/n) ~ 2.2%,
    # so a 12% relative tolerance is >5 sigma. Mean has std error 1/n = 2.5e-4.
    assert bn.var() == pytest.approx(1.0 / n, rel=0.12)
    assert bn.mean() == pytest.approx(0.0, abs=2e-3)


def test_scaled_brownian_noise_is_gaussian_not_merely_mean_zero_variance_one_over_n():
    # The moment tests above are satisfied by ANY mean-0, variance-1/n generator
    # (a scaled uniform passes them, and the sum test below passes by the CLT),
    # so pin the shape of the law itself. The sample excess kurtosis of n draws
    # has std error sqrt(24/n) = 0.011 at n = 200000, so abs=0.25 is >20 sigma
    # for a Gaussian while separating it from a uniform (-1.2) or a Laplace (+3).
    bn = adf.scaled_brownian_noise(200000)
    z = (bn - bn.mean()) / bn.std()
    assert (z ** 4).mean() - 3.0 == pytest.approx(0.0, abs=0.25)
    # ... and the third moment is 0 too (std error sqrt(6/n) = 0.005).
    assert (z ** 3).mean() == pytest.approx(0.0, abs=0.1)


def test_scaled_brownian_noise_sums_to_standard_normal():
    # B(1) = sum of n scaled increments has variance n * (1/n) = 1 regardless of n.
    n, npaths = 50, 2000
    b1 = numpy.array([adf.scaled_brownian_noise(n).sum() for _ in range(npaths)])
    # relative std error of the variance is sqrt(2/npaths) ~ 3.2%; 0.15 is ~5 sigma.
    assert b1.var() == pytest.approx(1.0, abs=0.15)
    assert b1.mean() == pytest.approx(0.0, abs=0.1)


def test_brownian_motion_is_partial_sum_up_to_t():
    bn = hand_noise()
    assert [adf.brownian_motion(bn, t) for t in range(5)] == [0.0, 1.0, 3.0, 6.0, 10.0]

    rnd = numpy.random.normal(0.0, 1.0, 30)
    expected = numpy.concatenate([[0.0], numpy.cumsum(rnd)])
    for t in (0, 1, 7, 29, 30):
        assert adf.brownian_motion(rnd, t) == pytest.approx(expected[t])


def test_brownian_motion_out_of_range_t_is_silently_clipped_by_slicing():
    # brownian_motion is sum(bn[:t]), so an out-of-range t is neither clamped
    # nor rejected -- it goes through Python slicing semantics. Pinned here
    # because a plotting loop that walks t past the end, or hands over a
    # negative index, gets a plausible-looking number instead of an error.
    bn = hand_noise()                                   # partial sums 0, 1, 3, 6, 10

    # t at or beyond the end saturates at B(1).
    assert adf.brownian_motion(bn, len(bn)) == 10.0
    assert adf.brownian_motion(bn, 99) == 10.0

    # a negative t counts from the END: t = -1 drops the last increment (10 - 4)
    # rather than returning B(0) = 0, and t <= -len(bn) empties the slice.
    assert adf.brownian_motion(bn, -1) == 6.0
    assert adf.brownian_motion(bn, -4) == 0.0

    # t = 0 sums an empty slice, which is Python's int 0 rather than the
    # annotated float; it still compares and casts as a float would.
    origin = adf.brownian_motion(bn, 0)
    assert origin == 0.0
    assert float(origin) == 0.0


def test_brownian_motion_variance_grows_linearly_in_time():
    # With n scaled increments B at step k has Var = k/n (B(1) has Var 1).
    n, k, npaths = 20, 10, 2000
    bk = numpy.array([adf.brownian_motion(adf.scaled_brownian_noise(n), k) for _ in range(npaths)])
    # std error of sample variance ~ 0.5*sqrt(2/2000) = 0.016; tolerance 0.1 is ~6 sigma.
    assert bk.var() == pytest.approx(k / n, abs=0.1)


# ---------------------------------------------------------------------------
# modified_chi_squared / stochastic_integral_solution_1
# ---------------------------------------------------------------------------

def test_modified_chi_squared_is_pdf_of_half_chi2_minus_one():
    # If X ~ chi2(1) then Y = (X - 1)/2 has density f_Y(y) = 2 f_X(2y + 1).
    y = numpy.linspace(-0.49, 8.0, 200)
    assert_allclose(adf.modified_chi_squared(y), 2.0 * stats.chi2.pdf(2.0 * y + 1.0, df=1), rtol=1e-12)


def test_modified_chi_squared_normalised_with_zero_mean_and_half_variance():
    f = adf.modified_chi_squared
    total, _ = integrate.quad(f, -0.5, numpy.inf)
    mean, _ = integrate.quad(lambda y: y * f(y), -0.5, numpy.inf)
    second, _ = integrate.quad(lambda y: y * y * f(y), -0.5, numpy.inf)
    # E[chi2(1)] = 1, Var[chi2(1)] = 2  ->  E[Y] = 0, Var[Y] = 2/4 = 1/2.
    assert total == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(0.0, abs=1e-8)
    assert second == pytest.approx(0.5, abs=1e-8)


def test_modified_chi_squared_outside_its_support_returns_inf_and_nan():
    # The density is supported on y > -1/2 (its argument is 2y + 1 > 0). Outside
    # that the function does not raise: at the boundary the normalisation
    # sqrt(2 pi (2y + 1)) is exactly 0 so the value is +inf, and below it the
    # sqrt is taken of a negative number so the value is nan. Both only emit a
    # numpy RuntimeWarning, which is why a notebook plotting over a symmetric
    # grid silently gets a poisoned curve rather than an error.
    f = adf.modified_chi_squared

    with pytest.warns(RuntimeWarning):
        boundary = f(numpy.array([-0.5]))
    assert numpy.isposinf(boundary).all()

    with pytest.warns(RuntimeWarning):
        below = f(numpy.array([-0.5001, -1.0, -10.0]))
    assert numpy.isnan(below).all()

    # just inside the support the density is finite (and large).
    inside = f(numpy.array([-0.4999]))
    assert numpy.isfinite(inside).all() and (inside > 1.0).all()


def test_stochastic_integral_solution_1_moments():
    n = 5000
    vals = adf.stochastic_integral_solution_1(n)

    assert vals.shape == (n,)
    # (Z^2 - 1)/2 is bounded below by -1/2.
    assert vals.min() >= -0.5
    # E = 0 (std error sqrt(0.5/n) = 0.01), Var = 1/2 (4th central moment of
    # (Z^2-1)/2 is 60/16 so the variance estimate has std error sqrt(3.5/n) = 0.026).
    assert vals.mean() == pytest.approx(0.0, abs=0.05)
    assert vals.var() == pytest.approx(0.5, abs=0.1)


def test_stochastic_integral_solution_1_cdf_matches_modified_chi_squared():
    # The samples and the pdf describe the same law: compare the empirical CDF
    # with the CDF of (chi2(1) - 1)/2 at a few points. ECDF std error <= 0.007.
    vals = adf.stochastic_integral_solution_1(5000)
    for y in (-0.4, -0.25, 0.0, 0.5, 1.0, 2.0):
        ecdf = numpy.mean(vals <= y)
        assert ecdf == pytest.approx(stats.chi2.cdf(2.0 * y + 1.0, df=1), abs=0.03)


# ---------------------------------------------------------------------------
# stochastic_integral_simulation_1 / ensemble_1  --  int_0^1 B dB
# ---------------------------------------------------------------------------

def test_stochastic_integral_simulation_1_hand_example():
    # sum_{i=1}^{3} B_{i-1} * bn[i] with B = (0, 1, 3, 6): 0*2 + 1*3 + 3*4 = 15
    assert adf.stochastic_integral_simulation_1(hand_noise()) == pytest.approx(15.0)


def test_stochastic_integral_simulation_1_is_exactly_the_left_riemann_sum():
    # Exactness on a real path, not just on the 4-element hand example: the
    # library's Python loop must reproduce sum_{i=1}^{n-1} B_{i-1} bn_i to the
    # last bit, computed here by an independent vectorised cumsum. This pins the
    # discrete sum itself, which the O(sqrt(1/n)) moment tests below cannot.
    bn = adf.scaled_brownian_noise(200)
    b = numpy.concatenate([[0.0], numpy.cumsum(bn)])          # b[k] = B(k)
    assert adf.stochastic_integral_simulation_1(bn) == pytest.approx((b[:len(bn) - 1] * bn[1:]).sum(), rel=1e-12)
    # same for the int B^2 ds sums: (1/n) sum_{k=0}^{n-1} B_k^2 and its square root.
    expected_2 = (b[:len(bn)] ** 2).sum() / len(bn)
    assert adf.stochastic_integral_simulation_2(bn) == pytest.approx(expected_2, rel=1e-12)
    assert adf.stochastic_integral_simulation_3(bn) == pytest.approx(numpy.sqrt(expected_2), rel=1e-12)


def test_stochastic_integral_simulation_1_satisfies_ito_identity():
    # Ito: int_0^1 B dB = (B(1)^2 - 1)/2. The discretisation error of the
    # simulated sum has std ~ sqrt(1.5/n) = 0.06 for n = 400, so 0.5 is ~8 sigma.
    n = 400
    for _ in range(5):
        bn = adf.scaled_brownian_noise(n)
        b1 = bn.sum()
        assert adf.stochastic_integral_simulation_1(bn) == pytest.approx(0.5 * (b1 * b1 - 1.0), abs=0.5)


def test_stochastic_integral_ensemble_1_moments():
    n, nsample = 50, 2000
    vals = adf.stochastic_integral_ensemble_1(n, nsample)

    assert vals.shape == (nsample,)
    # Var[int B dB] is APPROXIMATELY 1/2 (the discrete value is
    # (1/n^2) sum_{k=0}^{n-2} k = (n-1)(n-2)/(2 n^2) = 0.4704 at n = 50, only 6%
    # below the limit, and this test is far too coarse to tell the two apart --
    # the exact discrete sum is pinned deterministically by
    # test_stochastic_integral_simulation_1_is_exactly_the_left_riemann_sum).
    # The sample variance of a statistic with excess kurtosis ~12 is strongly
    # right-skewed: its measured sampling sd here is 8% relative, but the
    # observed spread over hundreds of alternate seeds reaches 33%, so rel=0.3
    # is an outright flake (it fails outright for some seeds) and rel=0.6 is the
    # smallest tolerance that is actually seed-robust.
    assert vals.var() == pytest.approx((n - 1) * (n - 2) / (2.0 * n * n), rel=0.6)
    # mean 0 with std error sqrt(0.47/2000) = 0.015
    assert vals.mean() == pytest.approx(0.0, abs=0.08)


# ---------------------------------------------------------------------------
# stochastic_integral_simulation_2 / 3, ensembles  --  int_0^1 B^2 ds
# ---------------------------------------------------------------------------

def test_stochastic_integral_simulation_2_and_3_hand_example():
    # (1/4) * sum of B_k^2 for B = (0, 1, 3, 6): (0 + 1 + 9 + 36)/4 = 11.5
    assert adf.stochastic_integral_simulation_2(hand_noise()) == pytest.approx(11.5)
    assert adf.stochastic_integral_simulation_3(hand_noise()) == pytest.approx(numpy.sqrt(11.5))


def test_stochastic_integral_simulation_3_is_sqrt_of_simulation_2():
    bn = adf.scaled_brownian_noise(60)
    assert adf.stochastic_integral_simulation_3(bn) == pytest.approx(numpy.sqrt(adf.stochastic_integral_simulation_2(bn)))


def test_stochastic_integral_ensemble_2_moments():
    n, nsample = 100, 1000
    vals = adf.stochastic_integral_ensemble_2(n, nsample)

    assert vals.shape == (nsample,)
    assert numpy.all(vals >= 0.0)
    # Left Riemann sum (1/n) sum_{k=0}^{n-1} B_k^2 with E[B_k^2] = k/n has
    # exact mean (n-1)/(2n) (-> int_0^1 s ds = 1/2). Var[int B^2] = 1/3, so the
    # mean has std error 0.577/sqrt(1000) = 0.018; 0.1 is ~5.5 sigma.
    assert vals.mean() == pytest.approx((n - 1) / (2.0 * n), abs=0.1)
    # Var[int_0^1 B^2 ds] = 2 * sum_j lambda_j^2 = 1/3 with lambda_j = 1/((j-1/2)^2 pi^2).
    # Its excess kurtosis is ~11.7 so the variance estimate has relative std
    # error sqrt(13.7/1000) ~ 12%; 45% is ~3.8 sigma.
    assert vals.var() == pytest.approx(1.0 / 3.0, rel=0.45)


def test_stochastic_integral_ensemble_3_is_sqrt_of_integral():
    n, nsample = 50, 1000
    vals = adf.stochastic_integral_ensemble_3(n, nsample)

    assert vals.shape == (nsample,)
    assert numpy.all(vals > 0.0)
    # Squaring recovers the int B^2 law, whose discrete mean is (n-1)/(2n).
    assert (vals ** 2).mean() == pytest.approx((n - 1) / (2.0 * n), abs=0.1)


def test_stochastic_integral_simulations_on_a_single_increment_are_zero():
    # The smallest nstep a notebook can pass is 1, and every integral degenerates:
    # the Ito sum runs over range(1, 1) (empty) and the int B^2 sums see only
    # B_0 = 0. No exception is raised, so the degenerate case has to be pinned.
    single = numpy.array([2.0])
    assert adf.stochastic_integral_simulation_1(single) == 0.0
    assert adf.stochastic_integral_simulation_2(single) == 0.0
    assert adf.stochastic_integral_simulation_3(single) == 0.0
    assert_array_equal(adf.stochastic_integral_ensemble_1(1, 4), numpy.zeros(4))
    assert_array_equal(adf.stochastic_integral_ensemble_2(1, 4), numpy.zeros(4))
    assert_array_equal(adf.stochastic_integral_ensemble_3(1, 4), numpy.zeros(4))


# ---------------------------------------------------------------------------
# dist_ensemble  --  the Dickey-Fuller distribution
# ---------------------------------------------------------------------------

def test_dist_ensemble_with_a_single_step_is_all_nan():
    # With n = 1 both integrals are exactly 0 (see the test above), so every
    # ensemble member is 0.0/0.0. numpy warns and yields nan instead of raising,
    # i.e. the smallest nstep produces a silently empty histogram.
    with pytest.warns(RuntimeWarning):
        vals = adf.dist_ensemble(1, 5)
    assert vals.shape == (5,)
    assert numpy.all(numpy.isnan(vals))

    # the facade inherits the degeneracy unchanged.
    with pytest.warns(RuntimeWarning):
        _, facade_vals = fadf.create_df_source(nstep=1, nsim=5)
    assert numpy.all(numpy.isnan(facade_vals))


def test_dist_ensemble_contract():
    n, nsim = 30, 200
    vals = adf.dist_ensemble(n, nsim)

    assert isinstance(vals, numpy.ndarray)
    assert vals.shape == (nsim,)          # ensemble size is nsim, not the step count
    assert vals.dtype == numpy.float64
    assert numpy.all(numpy.isfinite(vals))
    # The denominator sqrt(int B^2 ds) is strictly positive, so the sign of each
    # ratio is the sign of its numerator, the Riemann sum sum_i B_{i-1} bn_i.
    # That sum is a mean-zero martingale (it is NOT bounded below by -1/2 the way
    # the closed form (B(1)^2 - 1)/2 is), so it is genuinely two-sided and both
    # tails must be populated.
    assert (vals < 0).any() and (vals > 0).any()


def test_dist_ensemble_uses_one_brownian_path_for_numerator_and_denominator():
    # Structural, exact version of the distributional test below: each ensemble
    # member must be the two integrals of the SAME path. Drawing them from
    # independent paths gives each factor the right marginal (so the moment
    # tests still roughly pass) but the wrong ratio law, which is why this is
    # pinned bit-for-bit rather than statistically. Replaying the RNG stream
    # from the same seed reproduces the one path the ensemble consumed.
    numpy.random.seed(SEED)
    vals = adf.dist_ensemble(40, 1)

    numpy.random.seed(SEED)
    bn = adf.scaled_brownian_noise(40)
    assert vals[0] == pytest.approx(adf.stochastic_integral_simulation_1(bn)
                                    / adf.stochastic_integral_simulation_3(bn), rel=1e-12)
    # ... and exactly ONE path is consumed per ensemble member: replaying the
    # stream, the second member is the ratio built from the SECOND path. Had the
    # numerator and denominator drawn separately, two members would consume four
    # paths and this would land on the third.
    numpy.random.seed(SEED)
    two = adf.dist_ensemble(40, 2)
    assert two[0] == pytest.approx(vals[0], rel=1e-12)

    numpy.random.seed(SEED)
    adf.scaled_brownian_noise(40)                       # burn the first member's path
    second = adf.scaled_brownian_noise(40)
    assert two[1] == pytest.approx(adf.stochastic_integral_simulation_1(second)
                                   / adf.stochastic_integral_simulation_3(second), rel=1e-12)


def test_dist_ensemble_matches_dickey_fuller_distribution():
    vals = adf.dist_ensemble(100, 1000)
    # These are the FINITE-nstep values the code actually produces at nstep=100,
    # measured over 400000 simulations: mean -0.396, 5% quantile -1.874. They
    # converge to Fuller's asymptotic tau law (mean -0.42, 5% -1.95, 1% -2.58)
    # as nstep grows -- verified at nstep=2000, which gives -0.423 / -1.947.
    # Asserting the asymptotic numbers here would mis-centre the test by 0.08.
    # Sampling sd at nsim=1000: 0.031 for the mean, 0.060 for the quantile, so
    # these tolerances are ~4.8 and ~4.2 sigma.
    assert vals.mean() == pytest.approx(-0.394, abs=0.15)
    assert numpy.quantile(vals, 0.05) == pytest.approx(-1.873, abs=0.25)
    # ... and it still catches a revert to the independent-path bug, which puts
    # the 5% quantile near -1.55.
    assert numpy.quantile(vals, 0.05) < -1.65


# ---------------------------------------------------------------------------
# statistic  --  raw Dickey-Fuller statistic
# ---------------------------------------------------------------------------

def test_statistic_hand_example():
    # x = (1, 2, 4, 7): lagged x = (1, 2, 4), dx = (1, 2, 3)
    # numerator = 1*1 + 2*2 + 4*3 = 17, sum x_{t-1}^2 = 21
    samples = numpy.array([1.0, 2.0, 4.0, 7.0])
    assert adf.statistic(samples) == pytest.approx(17.0 / numpy.sqrt(21.0))


def test_statistic_matches_statsmodels_dickey_fuller_regression():
    # statsmodels' no-lag, no-constant DF t-statistic is beta_hat / (sigma_hat / sqrt(sum x^2)),
    # i.e. statistic(x) / sigma_hat where sigma_hat is the residual std of the
    # regression dx_t = beta x_{t-1} + e_t with nobs - 1 degrees of freedom.
    w = random_walk(500)
    sm_stat = sm.tsa.stattools.adfuller(w, regression="n", maxlag=0, autolag=None)[0]

    dx, x = numpy.diff(w), w[:-1]
    beta = (x @ dx) / (x @ x)
    resid = dx - beta * x
    sigma_hat = numpy.sqrt((resid @ resid) / (len(dx) - 1))

    assert adf.statistic(w) / sigma_hat == pytest.approx(sm_stat, rel=1e-9)


def test_statistic_limit_for_stationary_ar1():
    # For x_t = φ x_{t-1} + ε: sum x_{t-1} dx / sum x^2 -> φ-1 and sum x^2/N -> 1/(1-φ^2),
    # so statistic ~ (φ-1) sqrt(N/(1-φ^2)) = -25.8 for φ=0.5, N=2000. Since it is
    # a t-statistic its fluctuation is O(1); 5 is a >=5 sigma tolerance.
    φ, n = 0.5, 2000
    x = ar1(n, φ)
    assert adf.statistic(x) == pytest.approx((φ - 1.0) * numpy.sqrt(n / (1.0 - φ * φ)), abs=5.0)


def test_statistic_of_random_walks_follows_the_dickey_fuller_null():
    # The module's central claim: under a unit root, statistic(x)/sigma_hat is
    # distributed as the law dist_ensemble simulates. Assert it against the SAME
    # targets dist_ensemble is held to (and against the MacKinnon 5% entry for
    # the no-constant regression), which cross-checks the two halves of the
    # module against each other. A bound like |tau| < 4 on one draw does not:
    # returning beta_hat, or the numerator over var instead of sqrt(var), or half
    # the statistic, all satisfy it.
    #
    # Measured over 200000 replicates at n = 500: mean -0.423, 5% quantile
    # -1.942 (already the asymptotic values). With nrep = 400 the sampling sd is
    # 0.047 for the mean and 0.095 for the quantile, so these tolerances are
    # ~5.3 and ~4.2 sigma; no block of 400 out of 500 disjoint blocks breached them.
    nrep, n = 400, 500
    taus = numpy.empty(nrep)
    for i in range(nrep):
        w = random_walk(n)
        dx, x = numpy.diff(w), w[:-1]
        resid = dx - ((x @ dx) / (x @ x)) * x
        sigma_hat = numpy.sqrt((resid @ resid) / (len(dx) - 1))
        taus[i] = adf.statistic(w) / sigma_hat

    assert taus.mean() == pytest.approx(-0.42, abs=0.25)
    assert numpy.quantile(taus, 0.05) == pytest.approx(MACKINNON_5PCT["n"], abs=0.4)
    # The law is two-sided and left-skewed: a positive tau is common (P ~ 0.32,
    # so the proportion has sd 0.023 and this bound is ~9 sigma) even though the
    # mean and the whole lower tail sit well below zero -- a "t-statistic" whose
    # 5% point is -1.94 rather than the normal's -1.645.
    assert (taus > 0).mean() > 0.1
    assert numpy.quantile(taus, 0.5) < 0.0


def test_statistic_of_a_degenerate_series_is_nan_rather_than_an_error():
    # sum x_{t-1}^2 == 0 makes the statistic 0.0/0.0. numpy divides the Python
    # float by a numpy zero, so this is a nan plus a RuntimeWarning, never an
    # exception -- the same silent degeneracy the integrals and dist_ensemble
    # are pinned for above. It is reachable from a fewer-than-two-sample slice
    # or an all-zero series.
    for degenerate in (numpy.array([]), numpy.array([3.0]), numpy.zeros(5)):
        with pytest.warns(RuntimeWarning):
            assert numpy.isnan(adf.statistic(degenerate))

    # A constant NON-zero series is not degenerate in that sense: every delta is
    # 0 so the numerator vanishes while sum x_{t-1}^2 does not, giving exactly
    # 0.0 (no warning). Pinned to keep the nan case above from being read as
    # "any constant series".
    assert adf.statistic(numpy.full(6, 5.0)) == 0.0


def test_statistic_scales_with_the_amplitude_of_the_samples():
    # Fix-independent (it does not touch σ): the numerator sum x_{t-1} dx_t is
    # quadratic in x while the denominator sqrt(sum x_{t-1}^2) is |c|-homogeneous,
    # so statistic(c x) = |c| statistic(x) -- it grows with the amplitude of the
    # series and is blind to a sign flip. That growth is exactly what a σ
    # normalisation has to cancel, which is what the xfail below is about.
    w = random_walk(400)
    base = adf.statistic(w)
    for c in (-3.0, -1.0, 0.5, 3.0, 10.0):
        assert adf.statistic(c * w) == pytest.approx(abs(c) * base, rel=1e-12)


@pytest.mark.parametrize("σ", [0.5, 2.0, 3.0])
def test_statistic_sigma_divides_by_sigma(σ):
    # σ enters only as the divisor (lib/models/adf.py: delta_numerator /
    # (sqrt(var) * σ)), so it is a pure rescaling of the σ = 1 value and never
    # touches the samples. Dividing once is what makes the DF t-statistic
    # invariant under x -> c·x with σ -> c·σ (see the scale-invariance test).
    w = random_walk(400)
    base = adf.statistic(w)
    assert adf.statistic(w, σ=σ) == pytest.approx(base / σ, rel=1e-12)
    assert adf.statistic(w, σ) == pytest.approx(base / σ, rel=1e-12)


def test_statistic_is_scale_invariant_when_sigma_is_supplied():
    w = random_walk(500)
    c = 3.0
    assert adf.statistic(c * w, σ=c) == pytest.approx(adf.statistic(w))


# ---------------------------------------------------------------------------
# adf_test / adf_test_offset / adf_test_drift  --  statsmodels wrappers
# ---------------------------------------------------------------------------

ADF_VARIANTS = [
    (adf.adf_test, "n"),
    (adf.adf_test_offset, "c"),
    (adf.adf_test_drift, "ct"),
]


@pytest.mark.parametrize("func, regression", ADF_VARIANTS)
def test_adf_test_dispatches_regression_type(func, regression):
    w = random_walk(1000)
    report = func(w)
    expected = sm.tsa.stattools.adfuller(w, regression=regression, maxlag=12)

    assert isinstance(report, ADFTestReport)
    assert report.stat == pytest.approx(expected[0])
    assert report.pval == pytest.approx(expected[1])
    assert report.lags == expected[2]
    assert report.nobs == expected[3]
    # The critical values identify the regression variant independently of statsmodels' wiring.
    assert report.critical_vals[1] == pytest.approx(MACKINNON_5PCT[regression], abs=0.05)


def _hand_rolled_df_tstat(x: numpy.ndarray, regression: str) -> float:
    """The no-augmentation ADF t-statistic from a hand-built OLS, no statsmodels.

    Regresses dx_t on x_{t-1} plus the regression's deterministic columns and
    returns beta / se(beta) with se from s^2 (X'X)^{-1}, s^2 = RSS/(nobs - k).
    """
    dx, lagged = numpy.diff(x), x[:-1]
    nobs = len(dx)
    cols = [lagged]
    if regression in ("c", "ct"):
        cols.append(numpy.ones(nobs))
    if regression == "ct":
        cols.append(numpy.arange(1.0, nobs + 1.0))
    design = numpy.column_stack(cols)

    beta, *_ = numpy.linalg.lstsq(design, dx, rcond=None)
    resid = dx - design @ beta
    s2 = (resid @ resid) / (nobs - design.shape[1])
    cov = s2 * numpy.linalg.inv(design.T @ design)
    return float(beta[0] / numpy.sqrt(cov[0, 0]))


@pytest.mark.parametrize("func, regression", ADF_VARIANTS)
def test_adf_test_statistic_matches_a_hand_rolled_ols(func, regression):
    # The parametrized test above takes its expected statistic from the very
    # statsmodels call the wrapper makes, so it can only catch argument wiring.
    # This anchors all three statistics -- constant and constant+trend included
    # -- to an OLS built here from numpy alone, so a numerically wrong use of
    # statsmodels (a mis-specified design, the wrong degrees of freedom in s^2,
    # the wrong coefficient's t-statistic) fails independently of statsmodels.
    w = random_walk(500)
    assert func(w, max_lag=0).stat == pytest.approx(_hand_rolled_df_tstat(w, regression), rel=1e-9)

    # a stationary series exercises the same anchor away from the null.
    x = _stationary_series(regression, 500)
    assert func(x, max_lag=0).stat == pytest.approx(_hand_rolled_df_tstat(x, regression), rel=1e-9)


def test_adf_test_variants_give_distinct_statistics():
    w = random_walk(1000)
    stats_ = [func(w).stat for func, _ in ADF_VARIANTS]
    assert len({round(s, 6) for s in stats_}) == 3


def test_adf_report_fields_are_consistent():
    n = 600
    w = random_walk(n)
    report = adf.adf_test(w, max_lag=5)

    assert 0 <= report.lags <= 5
    assert report.nobs == n - report.lags - 1
    assert 0.0 <= report.pval <= 1.0
    assert report.sig_str == ["1%", "5%", "10%"]
    assert report.sig == [0.01, 0.05, 0.1]
    # lower-tail critical values become less negative as the significance level grows
    assert report.critical_vals[0] < report.critical_vals[1] < report.critical_vals[2]
    # the string column is just a rendering of the boolean column (the boolean
    # column itself is anchored to hand-known outcomes in the test below).
    for status, label in zip(report.status_vals, report.status_str):
        assert label == ("Passed" if status else "Failed")
    # the statuses are monotone in the significance level: the critical values
    # increase, so a pass at a looser level implies a pass at every tighter one
    # and the boolean column can only ever be True...True, False...False.
    assert report.status_vals == sorted(report.status_vals, reverse=True)


def test_adf_report_status_vals_are_anchored_at_both_extremes():
    # Rather than recomputing `stat >= critical_val` (which is exactly what the
    # report's constructor does), pin the two ends against series whose ADF
    # outcome is known in advance.

    # (a) A stationary AR(1) drives the statistic far BELOW every critical value
    # (phi = 0.5, N = 1000 gives tau ~ -18 +/- 0.7 against -2.57/-1.94/-1.62), so
    # the unit root is rejected at all three levels and every status is a failure.
    stationary = adf.adf_test(ar1(1000, 0.5), max_lag=0)
    assert stationary.stat < -10.0
    assert stationary.status_vals == [False, False, False]
    assert stationary.status_str == ["Failed", "Failed", "Failed"]

    # (b) A random walk with a large drift, regressed with no constant, drives it
    # far ABOVE every critical value (tau ~ +32 +/- 0.5 with no augmentation),
    # so nothing is rejected and every status is a pass.
    drifting = adf.adf_test(random_walk(400, drift=5.0), max_lag=0)
    assert drifting.stat > 10.0
    assert drifting.status_vals == [True, True, True]
    assert drifting.status_str == ["Passed", "Passed", "Passed"]


def test_adf_test_max_lag_zero_gives_plain_dickey_fuller():
    n = 400
    w = random_walk(n)
    report = adf.adf_test(w, max_lag=0)
    assert report.lags == 0
    assert report.nobs == n - 1
    # with no augmentation the statistic is the plain DF t-statistic
    assert report.stat == pytest.approx(sm.tsa.stattools.adfuller(w, regression="n", maxlag=0, autolag=None)[0])


def test_adf_test_max_lag_bounds_selected_lags():
    # ARIMA(1,1,0) differences have real autocorrelation so AIC wants >= 1 lag.
    dx = ar1(600, 0.6)
    x = numpy.cumsum(dx)
    assert adf.adf_test(x, max_lag=0).lags == 0
    assert 1 <= adf.adf_test(x, max_lag=12).lags <= 12
    assert adf.adf_test(x, max_lag=12).stat != pytest.approx(adf.adf_test(x, max_lag=0).stat)


def _long_lag_unit_root_series(n: int = 1000) -> numpy.ndarray:
    """Unit-root series whose differences follow dx_t = 0.6 dx_{t-15} + e_t.

    AIC needs 15 lags to whiten these differences, which is more than the
    wrappers' explicit default of 12 -- so the number of lags selected shows
    which cap was actually applied.
    """
    dx = lfilter([1.0], [1.0] + [0.0] * 14 + [-0.6], numpy.random.normal(0.0, 1.0, n))
    return numpy.cumsum(dx)


@pytest.mark.parametrize("func, regression", ADF_VARIANTS)
def test_adf_test_max_lag_none_uses_the_statsmodels_default_rule(func, regression):
    # The docstrings promise 12*(nobs/100)^{1/4} when max_lag is None. For
    # nobs = 999 that cap is ceil(12 * 9.99^{1/4}) = 22, well above the explicit
    # default of 12, so a series that genuinely wants 15 lags proves None was
    # forwarded rather than replaced by 12: AIC picks 13..22 lags, whereas the
    # explicit default truncates at 12 (and in practice settles far lower).
    x = _long_lag_unit_root_series(1000)
    nobs = len(x) - 1
    cap = int(numpy.ceil(12.0 * (nobs / 100.0) ** 0.25))
    assert cap == 22

    report = func(x, max_lag=None)
    assert 12 < report.lags <= cap
    assert report.nobs == len(x) - report.lags - 1
    # ... and the explicit 12-lag call cannot reach there.
    assert func(x, max_lag=12).lags <= 12


def _stationary_series(regression: str, n: int) -> numpy.ndarray:
    """A series that is stationary once the regression's deterministic terms are removed."""
    x = ar1(n, 0.5)
    if regression == "c":
        return x + 5.0
    if regression == "ct":
        return x + 0.1 * numpy.arange(n)
    return x


@pytest.mark.parametrize("func, regression", ADF_VARIANTS)
def test_adf_test_rejects_unit_root_for_stationary_series(func, regression):
    # Power: phi = 0.5, N = 1000 gives tau ~ -18 with no augmentation. The
    # wrappers let AIC pick up to 12 lags, and the lagged differences are
    # nearly collinear with x_{t-1} (their sum is x_{t-1} - x_{t-13}), which
    # inflates its standard error by ~sqrt(5) when all 12 are selected. Even
    # then tau ~ -8 +/- 1, still far below the 1% critical values
    # (-2.58 / -3.44 / -3.97), so every replicate rejects at every level.
    # (At N = 400 the 12-lag case sits near -5 and occasionally misses 1%.)
    for _ in range(20):
        report = func(_stationary_series(regression, 1000))
        assert report.status_vals == [False, False, False]
        assert report.pval < 0.01


@pytest.mark.parametrize("func, regression", ADF_VARIANTS)
def test_adf_test_size_under_unit_root(func, regression):
    # Size: the 10% test rejects a true random walk ~10% of the time. Over 30
    # replicates P(>= 11 rejections) ~ 1e-3 even if the finite-sample size is 15%.
    rejections = sum(not func(random_walk(400)).status_vals[2] for _ in range(30))
    assert rejections <= 10


# whole numeric tokens only, so the significance labels ("1%", "5%", "10%") and
# any partial prefix of a number are not read as values.
_FLOAT_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\d%.])")


def _numbers_on(line: str) -> list[float]:
    """Every numeric token on one line of a tabulate table."""
    return [float(tok) for tok in _FLOAT_TOKEN.findall(line)]


def _row_containing(out: str, label: str) -> str:
    """The single output line whose first cell is `label`."""
    rows = [line for line in out.splitlines() if label in line]
    assert len(rows) == 1, f"expected exactly one row for {label!r}, got {rows}"
    return rows[0]


def _assert_summary_reports(out: str, report: ADFTestReport) -> None:
    """The printed table carries every number the report holds, row by row."""
    assert _numbers_on(_row_containing(out, "Test Statistic")) == pytest.approx([report.stat], rel=1e-4)
    assert _numbers_on(_row_containing(out, "pvalue")) == pytest.approx([report.pval], rel=1e-4)
    assert _numbers_on(_row_containing(out, "Lags")) == [report.lags]
    assert _numbers_on(_row_containing(out, "Number Obs")) == [report.nobs]

    for i, sig in enumerate(report.sig_str):
        # each significance row pairs ITS OWN critical value with ITS OWN status;
        # transposing the two columns, or printing the statistic in place of a
        # critical value, has to fail here.
        row = _row_containing(out, sig)
        assert _numbers_on(row) == pytest.approx([report.critical_vals[i]], rel=1e-4)
        assert report.status_str[i] in row


def test_adf_test_report_summary_prints_every_field(capsys):
    report = adf.adf_test(random_walk(300))
    report.summary()
    out = capsys.readouterr().out

    assert "Test Statistic" in out and "Number Obs" in out
    assert "Significance" in out and "Critical Value" in out and "Result" in out
    _assert_summary_reports(out, report)


def test_adf_test_report_summary_honours_tablefmt(capsys):
    report = adf.adf_test(random_walk(300))
    report.summary(tablefmt="plain")
    plain = capsys.readouterr().out

    # "plain" drops the fancy_grid box drawing but keeps every value in place.
    assert not set("╒╤╕│├└┘─┼") & set(plain)
    _assert_summary_reports(plain, report)

    report.summary()
    fancy = capsys.readouterr().out
    assert "│" in fancy
    assert plain != fancy


# ---------------------------------------------------------------------------
# facade: create_df_source
# ---------------------------------------------------------------------------

def test_create_df_source_contract():
    nstep, nsim = 20, 50
    t, vals = fadf.create_df_source(nstep=nstep, nsim=nsim)

    assert isinstance(t, numpy.ndarray) and isinstance(vals, numpy.ndarray)
    assert vals.shape == (nsim,)
    assert numpy.all(numpy.isfinite(vals))
    # only the properties that hold whichever way the off-by-one below is
    # resolved: the axis starts at 0 and steps by 1. Its length and endpoint are
    # left entirely to the xfailed test.
    assert t[0] == 0.0
    assert_allclose(numpy.diff(t), 1.0)


def test_create_df_source_defaults():
    # defaults: nstep=100, nsim=1000. The nsim default shows up in the shape...
    _, vals = fadf.create_df_source()
    assert vals.shape == (1000,)

    # ... but nstep does not, so pin it exactly against the model layer under a
    # replayed RNG stream: nstep silently becoming anything but 100 changes
    # every value (and is invisible to every other test in this file, which all
    # pass nstep explicitly).
    numpy.random.seed(SEED)
    _, default_nstep = fadf.create_df_source(nsim=5)
    numpy.random.seed(SEED)
    assert_array_equal(default_nstep, adf.dist_ensemble(100, 5))


def test_create_df_source_time_axis_matches_values():
    nsim = 50
    t, vals = fadf.create_df_source(nstep=20, nsim=nsim)
    assert len(t) == len(vals)
    assert t[-1] == nsim - 1


@pytest.mark.parametrize("nstep, nsim", [(20, 40), (35, 25)])
def test_create_df_source_agrees_with_model_layer(nstep, nsim):
    numpy.random.seed(SEED)
    _, facade_vals = fadf.create_df_source(nstep=nstep, nsim=nsim)
    numpy.random.seed(SEED)
    model_vals = adf.dist_ensemble(nstep, nsim)
    assert_array_equal(facade_vals, model_vals)


# ---------------------------------------------------------------------------
# facade: compute_adf_test / compute_adf_offset_test / compute_adf_drift_test
# ---------------------------------------------------------------------------

# (facade, model-layer function, statsmodels regression code, HypothesisTestType, desc)
FACADE_VARIANTS = [
    (fadf.compute_adf_test, adf.adf_test, "n",
     HypothesisTestType.STATIONARITY, "Stationarity Test"),
    (fadf.compute_adf_offset_test, adf.adf_test_offset, "c",
     HypothesisTestType.STATIONARITY_OFFSET, "Stationarity Test with Constant Offset."),
    (fadf.compute_adf_drift_test, adf.adf_test_drift, "ct",
     HypothesisTestType.STATIONARITY_DRIFT, "Stationarity Test with Drift."),
]


@pytest.mark.parametrize("facade, model, regression, hyp_test_type, desc", FACADE_VARIANTS)
def test_compute_adf_test_contract(facade, model, regression, hyp_test_type, desc):
    w = random_walk(500)
    result = facade(w)

    assert isinstance(result, tuple) and len(result) == 2
    report, test_report = result
    assert isinstance(report, ADFTestReport)
    assert isinstance(test_report, StatisticalTestReport)

    # the ADF report is the model layer's (the test is deterministic given samples)
    expected = model(w)
    assert report.stat == pytest.approx(expected.stat)
    assert report.pval == pytest.approx(expected.pval)
    assert report.lags == expected.lags

    assert test_report.hyp_test_type is hyp_test_type
    assert test_report.hyp_type is HypothesisType.LOWER_TAIL
    assert test_report.desc == desc
    assert uuid.UUID(test_report.test_id).version == 4
    assert len(test_report.test_data) == 3

    for i, data in enumerate(test_report.test_data):
        assert data.test_id == test_report.test_id
        assert data.sig.label == report.sig_str[i]
        assert data.sig.value == report.sig[i]
        assert data.lower.label == r"$t_L$"
        assert data.lower.value == pytest.approx(report.critical_vals[i])
        assert data.stat.label == r"$t$"
        assert data.stat.value == pytest.approx(report.stat)
        assert data.pval.label == r"$p-value$"
        assert data.pval.value == pytest.approx(report.pval)
        assert data.upper is None
        assert data.params == []
        # the per-level status spells out the lower-tail rule on the numbers
        # themselves: the ADF null (a unit root) is rejected when the statistic
        # falls below the critical value, and the ADF report calls a rejection
        # a FAILED test.
        rejected = report.stat < report.critical_vals[i]
        assert data.status is (HypothesisTestStatus.FAILED if rejected else HypothesisTestStatus.PASSED)
        assert data.stat.test_id == data.sig.test_id == data.lower.test_id == test_report.test_id

    # stationarity "passes" exactly when the ADF test rejects the unit root at 10%
    rejected_at_10pct = report.stat < report.critical_vals[2]
    assert test_report.status is (HypothesisTestStatus.PASSED if rejected_at_10pct
                                  else HypothesisTestStatus.FAILED)


@pytest.mark.parametrize("facade, model, regression, hyp_test_type, desc", FACADE_VARIANTS)
def test_compute_adf_test_max_lag_kwarg(facade, model, regression, hyp_test_type, desc):
    x = numpy.cumsum(ar1(600, 0.6))       # AIC selects >= 1 lag when allowed

    report0, _ = facade(x, max_lag=0)
    assert report0.lags == 0
    assert report0.stat == pytest.approx(model(x, max_lag=0).stat)

    report3, _ = facade(x, max_lag=3)
    assert 1 <= report3.lags <= 3

    report_default, _ = facade(x)
    assert report_default.lags == model(x, max_lag=12).lags
    assert report_default.stat == pytest.approx(model(x, max_lag=12).stat)


@pytest.mark.parametrize("facade, model, regression, hyp_test_type, desc", FACADE_VARIANTS)
def test_compute_adf_test_max_lag_none_is_forwarded(facade, model, regression, hyp_test_type, desc):
    # max_lag=None must reach statsmodels as None (its 12*(nobs/100)^{1/4} rule),
    # not be treated as "missing" and replaced by the facade's default of 12.
    x = _long_lag_unit_root_series(1000)

    report_none, _ = facade(x, max_lag=None)
    assert 12 < report_none.lags <= int(numpy.ceil(12.0 * ((len(x) - 1) / 100.0) ** 0.25))
    assert report_none.lags == model(x, max_lag=None).lags

    report_default, _ = facade(x)
    assert report_default.lags <= 12
    assert report_none.stat != pytest.approx(report_default.stat)


@pytest.mark.parametrize("facade, model, regression, hyp_test_type, desc", FACADE_VARIANTS)
def test_compute_adf_test_silently_ignores_a_misspelled_max_lag(facade, model, regression, hyp_test_type, desc):
    # **kwargs + get_param_default_if_missing means an unknown key is swallowed
    # without warning: `max_lags=0` (note the plural) is NOT the same call as
    # `max_lag=0`, it is the default 12-lag call. Pinned deliberately because
    # this footgun silently changes what a notebook thinks it ran.
    x = numpy.cumsum(ar1(600, 0.6))       # AIC selects >= 1 lag when allowed

    misspelled, _ = facade(x, max_lags=0)
    default, _ = facade(x)
    intended, _ = facade(x, max_lag=0)

    assert misspelled.lags == default.lags
    assert misspelled.stat == pytest.approx(default.stat)
    assert misspelled.lags != intended.lags       # ... and not what was asked for
    assert intended.lags == 0


@pytest.mark.parametrize("facade, model, regression, hyp_test_type, desc", FACADE_VARIANTS)
def test_compute_adf_test_reports_stationary_series_as_passed(facade, model, regression, hyp_test_type, desc):
    _, test_report = facade(_stationary_series(regression, 400))
    assert test_report.status is HypothesisTestStatus.PASSED
    # every per-level ADF test rejected the unit root
    assert [d.status for d in test_report.test_data] == [HypothesisTestStatus.FAILED] * 3


def test_compute_adf_test_reports_drifting_random_walk_as_failed():
    # Without a constant in the regression a drifting random walk yields a
    # statistic far above every lower-tail critical value (dx ~ drift > 0
    # regressed on x_{t-1} > 0), so the unit root cannot be rejected at any
    # level. The statistic itself is NOT reliably positive: AIC is free to spend
    # up to 12 lagged differences absorbing the drift, which leaves it near zero
    # (mean 0.55, sd 0.25, occasionally slightly negative). Its distance from
    # the 1% critical value of -2.57 is ~10 sd, so that is what is asserted.
    w = random_walk(400, drift=5.0)
    report, test_report = fadf.compute_adf_test(w)
    assert report.stat > report.critical_vals[0]
    assert test_report.status is HypothesisTestStatus.FAILED
    assert [d.status for d in test_report.test_data] == [HypothesisTestStatus.PASSED] * 3


def test_compute_adf_test_generates_fresh_test_id_per_call():
    w = random_walk(300)
    _, first = fadf.compute_adf_test(w)
    _, second = fadf.compute_adf_test(w)
    assert first.test_id != second.test_id


# ---------------------------------------------------------------------------
# HypothesisTestType branches used by the facade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hyp_test_type", [HypothesisTestType.STATIONARITY,
                                           HypothesisTestType.STATIONARITY_OFFSET,
                                           HypothesisTestType.STATIONARITY_DRIFT])
def test_stationarity_status_depends_only_on_10pct_result(hyp_test_type):
    # only the 10% entry (index 2) decides; a rejection there is a PASS
    assert hyp_test_type.status([True, True, False]) is HypothesisTestStatus.PASSED
    assert hyp_test_type.status([False, False, False]) is HypothesisTestStatus.PASSED
    assert hyp_test_type.status([False, False, True]) is HypothesisTestStatus.FAILED
    assert hyp_test_type.status([True, True, True]) is HypothesisTestStatus.FAILED


def test_hypothesis_test_status_bool_round_trip():
    assert HypothesisTestStatus.from_bool(True) is HypothesisTestStatus.PASSED
    assert HypothesisTestStatus.from_bool(False) is HypothesisTestStatus.FAILED
    assert HypothesisTestStatus.from_bool(numpy.bool_(True)).to_bool() is True
    assert HypothesisTestStatus.FAILED.to_bool() is False


# ---------------------------------------------------------------------------
# serialisation of the facade's StatisticalTestReport
# ---------------------------------------------------------------------------

def test_statistical_test_report_to_json():
    w = random_walk(400)
    report, test_report = fadf.compute_adf_offset_test(w)

    data = json.loads(test_report.to_json())
    assert data["test_id"] == test_report.test_id
    assert data["status"] == test_report.status.value
    assert data["hyp_type"] == "LOWER_TAIL"
    assert data["hyp_test_type"] == "STATIONARITY_OFFSET"
    assert data["desc"] == "Stationarity Test with Constant Offset."
    assert len(data["test_data"]) == 3
    for i, entry in enumerate(data["test_data"]):
        assert entry["sig"] == {"test_id": test_report.test_id, "label": report.sig_str[i], "value": report.sig[i]}
        assert entry["lower"]["value"] == pytest.approx(report.critical_vals[i])
        assert entry["stat"]["value"] == pytest.approx(report.stat)
        assert entry["upper"] is None
        assert entry["params"] == []

    pretty = test_report.to_json(pretty=True)
    assert json.loads(pretty) == data
    assert pretty.count("\n") > 10

    assert "TestReport(" in repr(test_report) and test_report.test_id in repr(test_report)


def test_statistical_test_param_round_trip():
    param = StatisticalTestParam(test_id="abc", label=r"$t$", value=-1.5)
    restored = StatisticalTestParam.from_dict(json.loads(param.to_json()))
    assert (restored.test_id, restored.label, restored.value) == ("abc", r"$t$", -1.5)
    assert repr(param) == "TestParam(label=($t$), value=(-1.5), test_id=(abc))"


def test_statistical_test_report_from_dict_round_trip():
    _, test_report = fadf.compute_adf_test(random_walk(400))
    restored = StatisticalTestReport.from_dict(json.loads(test_report.to_json()))
    assert restored.test_id == test_report.test_id
    assert restored.status == test_report.status
    assert [d.lower.value for d in restored.test_data] == [d.lower.value for d in test_report.test_data]
