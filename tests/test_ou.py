"""Tests for the Ornstein-Uhlenbeck model group.

Two layers are covered:

* ``lib.models.ou`` — pure closed forms (mean, var, cov, pdf, cdf, limits,
  half-life) and the two samplers (``xt`` exact-in-distribution, ``ou`` Euler
  path). Closed forms are checked against hand-derived values and identities
  the formulas must satisfy; samplers are checked by simulating with known
  parameters and recovering the moments / parameters.
* ``lib.data.impl.ou`` — the ``**kwargs`` façade the notebooks call. Checked for
  its ``(t, values)`` contract, kwarg defaults and required-kwarg errors, and
  agreement with the model layer.

The process convention throughout is ``dX = λ(μ - X) dt + σ dW``, so
``E[X_t] = x0 e^{-λt} + μ(1 - e^{-λt})`` and ``Var[X_t] = σ²/(2λ) (1 - e^{-2λt})``.

The conftest autouse fixture seeds numpy's global RNG before each test; the
simulation tolerances below were calibrated over 40 independent seeds so they
are not tuned to that one seed.
"""

import json
import math

import numpy
import numpy.testing as npt
import pytest
from scipy.stats import ks_2samp, kstest, norm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from lib.data.impl import ou as ou_facade
from lib.data.param_est import EstModel, OLSParamType, OLSResult
from lib.models import ou
from lib.utils import create_ensemble

LN2 = math.log(2.0)

_COV_AXIS_REASON = (
    "compute_cov builds its axis as create_space(xmin=s, xmax=npts·Δt, Δx=Δt) while "
    "compute_cov_limit builds create_space(xmin=int(s/Δt), npts=npts, Δx=Δt), so the two "
    "grids agree only in the s = Δt = 1 coincidence the defaults sit on. Measured at "
    "s = 2, Δt = 1, npts = 10: compute_cov returns 9 points ending at 10.0, "
    "compute_cov_limit 10 points ending at 11.0 — the notebooks plot both on one axis."
)


# ---------------------------------------------------------------------------
# Model layer: closed forms
# ---------------------------------------------------------------------------


class TestMean:
    @pytest.mark.parametrize("μ, λ, x0", [(0.0, 1.0, 0.0), (2.0, 0.5, -1.0), (-3.0, 4.0, 7.5)])
    def test_starts_at_x0(self, μ, λ, x0):
        assert ou.mean(μ, λ, numpy.array([0.0]), x0)[0] == pytest.approx(x0)

    @pytest.mark.parametrize("μ, λ, x0", [(2.0, 0.5, -1.0), (-3.0, 4.0, 7.5)])
    def test_relaxes_to_mu(self, μ, λ, x0):
        # e^{-λt} < 1e-9 for λt > 21, so the mean has converged to μ
        assert ou.mean(μ, λ, numpy.array([50.0 / λ]), x0)[0] == pytest.approx(μ, abs=1e-8)

    def test_hand_computed_with_lambda_ln2(self):
        # λ = ln 2 makes e^{-λt} = 2^{-t}: mean(t) = x0 2^{-t} + μ (1 - 2^{-t}).
        t = numpy.array([0.0, 1.0, 2.0, 3.0])
        expected = numpy.array([4.0, 0.5 * 4.0 + 0.5 * 2.0, 0.25 * 4.0 + 0.75 * 2.0, 0.125 * 4.0 + 0.875 * 2.0])
        npt.assert_allclose(ou.mean(2.0, LN2, t, 4.0), expected, rtol=1e-12)

    def test_midpoint_at_half_life(self):
        # By definition of the half-life, the mean is midway between x0 and μ.
        μ, λ, x0 = 3.0, 0.7, -1.0
        t_half = numpy.array([ou.mean_halflife(λ)])
        assert ou.mean(μ, λ, t_half, x0)[0] == pytest.approx((x0 + μ) / 2.0)

    def test_x0_defaults_to_zero(self):
        t = numpy.linspace(0.0, 3.0, 7)
        npt.assert_array_equal(ou.mean(1.5, 0.8, t), ou.mean(1.5, 0.8, t, 0.0))


class TestVar:
    def test_zero_at_t0(self):
        assert ou.var(0.7, numpy.array([0.0]), 2.0)[0] == 0.0

    @pytest.mark.parametrize("λ, σ, expected", [(1.0, 1.0, 0.5), (0.5, 2.0, 4.0), (2.0, 1.0, 0.25), (0.25, 0.5, 0.5)])
    def test_var_limit_is_sigma2_over_2lambda(self, λ, σ, expected):
        assert ou.var_limit(λ, σ) == pytest.approx(expected)

    def test_var_limit_default_sigma_is_one(self):
        assert ou.var_limit(0.5) == pytest.approx(1.0)

    def test_hand_computed_with_lambda_half_ln2(self):
        # λ = ln2 / 2 makes e^{-2λt} = 2^{-t} and σ²/(2λ) = 1/ln2.
        t = numpy.array([0.0, 1.0, 2.0, 3.0])
        expected = (1.0 / LN2) * numpy.array([0.0, 0.5, 0.75, 0.875])
        npt.assert_allclose(ou.var(LN2 / 2.0, t, 1.0), expected, rtol=1e-12)

    def test_converges_to_var_limit(self):
        λ, σ = 0.3, 1.7
        assert ou.var(λ, numpy.array([50.0 / λ]), σ)[0] == pytest.approx(ou.var_limit(λ, σ), rel=1e-9)

    def test_scales_with_sigma_squared(self):
        t = numpy.linspace(0.0, 5.0, 11)
        npt.assert_allclose(ou.var(0.4, t, 3.0), 9.0 * ou.var(0.4, t, 1.0), rtol=1e-12)

    def test_monotone_increasing_in_t(self):
        v = ou.var(0.4, numpy.linspace(0.0, 10.0, 101), 1.0)
        assert numpy.all(numpy.diff(v) > 0)

    def test_sigma_defaults_to_one(self):
        t = numpy.linspace(0.0, 5.0, 11)
        npt.assert_array_equal(ou.var(0.4, t), ou.var(0.4, t, 1.0))


class TestCov:
    def test_hand_computed_with_lambda_ln2(self):
        # σ²/(2λ) (2^{-(t-s)} - 2^{-(t+s)}) with s = 1, t = 2: (1/2 - 1/8) / (2 ln 2)
        assert ou.cov(LN2, 1.0, numpy.array([2.0]), 1.0)[0] == pytest.approx(0.375 / (2.0 * LN2))

    def test_equals_var_at_coincident_times(self):
        s = numpy.array([0.5, 1.0, 2.0, 5.0])
        for si in s:
            assert ou.cov(0.6, si, numpy.array([si]), 1.3)[0] == pytest.approx(ou.var(0.6, numpy.array([si]), 1.3)[0])

    def test_markov_decay_from_var_at_s(self):
        # For t >= s the OU covariance factorises as Cov(X_s, X_t) = Var(X_s) e^{-λ(t-s)}.
        λ, σ, s = 0.6, 1.3, 1.5
        t = numpy.linspace(s, s + 8.0, 17)
        expected = ou.var(λ, numpy.array([s]), σ)[0] * numpy.exp(-λ * (t - s))
        npt.assert_allclose(ou.cov(λ, s, t, σ), expected, rtol=1e-12)

    def test_decays_to_zero(self):
        λ = 0.6
        assert ou.cov(λ, 1.0, numpy.array([1.0 + 60.0 / λ]), 1.0)[0] == pytest.approx(0.0, abs=1e-12)

    def test_scales_with_sigma_squared(self):
        t = numpy.linspace(1.0, 6.0, 11)
        npt.assert_allclose(ou.cov(0.6, 1.0, t, 2.0), 4.0 * ou.cov(0.6, 1.0, t, 1.0), rtol=1e-12)

    def test_sigma_defaults_to_one(self):
        # A value check on the default rather than only an equality with an explicit σ = 1.0:
        # with λ = ln2, s = 1, t = 2 the closed form is σ²/(2λ)(2^{-1} - 2^{-3}) = 0.375/(2 ln2).
        t = numpy.array([2.0])
        assert ou.cov(LN2, 1.0, t)[0] == pytest.approx(0.375 / (2.0 * LN2))
        npt.assert_array_equal(ou.cov(LN2, 1.0, t), ou.cov(LN2, 1.0, t, 1.0))

    @pytest.mark.xfail(
        strict=True,
        reason="lib/models/ou.py:96 computes exp(-λ(t - s)) with no abs() on (t - s), so the "
               "whole t < s half-plane blows up exponentially instead of decaying: with "
               "λ = σ = 1 and s = 1, ou.cov returns 1.175 at t = 0 where Cov(X_0, X_1) must "
               "be exactly 0 because Var(X_0) = 0.",
    )
    def test_matches_closed_form_for_t_below_s(self):
        # Hand-derived OU covariance, valid for either ordering of s and t:
        #   Cov(X_s, X_t) = σ²/(2λ) (e^{-λ|t-s|} - e^{-λ(t+s)}).
        # At t = 0 it is identically 0 for every s because X_0 = x0 is deterministic.
        λ, σ, s = 1.0, 1.0, 1.0
        t = numpy.array([0.0, 0.5, 1.0, 2.0])
        expected = (σ**2 / (2.0 * λ)) * (numpy.exp(-λ * numpy.abs(t - s)) - numpy.exp(-λ * (t + s)))
        npt.assert_allclose(ou.cov(λ, s, t, σ), expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.xfail(
        strict=True,
        reason="same missing abs() at lib/models/ou.py:96 — the covariance function of any "
               "process must satisfy Cov(X_s, X_t) = Cov(X_t, X_s), but ou.cov is symmetric "
               "only on the t >= s side.",
    )
    def test_is_symmetric_in_s_and_t(self):
        λ, σ = 0.6, 1.3
        below = ou.cov(λ, 3.0, numpy.array([1.0]), σ)[0]
        above = ou.cov(λ, 1.0, numpy.array([3.0]), σ)[0]
        assert below == pytest.approx(above)


class TestDistributions:
    μ, λ, σ, x0, t = 1.0, 0.5, 1.5, 4.0, 1.2

    @property
    def mean_t(self):
        return ou.mean(self.μ, self.λ, numpy.array([self.t]), self.x0)[0]

    @property
    def sd_t(self):
        return math.sqrt(ou.var(self.λ, numpy.array([self.t]), self.σ)[0])

    def test_pdf_integrates_to_one(self):
        x = numpy.linspace(-20.0, 20.0, 20001)
        p = ou.pdf(x, self.μ, self.λ, self.t, self.σ, self.x0)
        assert numpy.trapezoid(p, x) == pytest.approx(1.0, abs=1e-6)

    def test_pdf_is_gaussian_about_mean_t(self):
        # Peak height 1/(σ_t √(2π)) at the mean and e^{-1/2} relative height one sd away
        m, sd = self.mean_t, self.sd_t
        x = numpy.array([m - sd, m, m + sd])
        p = ou.pdf(x, self.μ, self.λ, self.t, self.σ, self.x0)
        assert p[1] == pytest.approx(1.0 / (sd * math.sqrt(2.0 * math.pi)))
        assert p[0] / p[1] == pytest.approx(math.exp(-0.5))
        assert p[2] / p[1] == pytest.approx(math.exp(-0.5))

    def test_cdf_is_half_at_mean_and_monotone(self):
        m, sd = self.mean_t, self.sd_t
        x = numpy.linspace(m - 8.0 * sd, m + 8.0 * sd, 1601)
        c = ou.cdf(x, self.μ, self.λ, self.t, self.σ, self.x0)
        assert c[800] == pytest.approx(0.5)
        assert c[0] == pytest.approx(0.0, abs=1e-12)
        assert c[-1] == pytest.approx(1.0, abs=1e-12)
        assert numpy.all(numpy.diff(c) >= 0)
        # one-sd quantile of a normal
        assert ou.cdf(numpy.array([m + sd]), self.μ, self.λ, self.t, self.σ, self.x0)[0] == pytest.approx(norm.cdf(1.0))

    def test_pdf_is_derivative_of_cdf(self):
        m, sd = self.mean_t, self.sd_t
        x = numpy.linspace(m - 5.0 * sd, m + 5.0 * sd, 4001)
        c = ou.cdf(x, self.μ, self.λ, self.t, self.σ, self.x0)
        p = ou.pdf(x, self.μ, self.λ, self.t, self.σ, self.x0)
        npt.assert_allclose(numpy.gradient(c, x)[5:-5], p[5:-5], rtol=1e-4, atol=1e-6)

    def test_pdf_defaults_sigma_one_x0_zero(self):
        x = numpy.linspace(-3.0, 3.0, 13)
        npt.assert_array_equal(ou.pdf(x, self.μ, self.λ, self.t), ou.pdf(x, self.μ, self.λ, self.t, 1.0, 0.0))
        npt.assert_array_equal(ou.cdf(x, self.μ, self.λ, self.t), ou.cdf(x, self.μ, self.λ, self.t, 1.0, 0.0))

    def test_pdf_limit_is_stationary_normal(self):
        sd = math.sqrt(ou.var_limit(self.λ, self.σ))
        x = numpy.array([self.μ - sd, self.μ, self.μ + sd])
        p = ou.pdf_limit(x, self.μ, self.λ, self.σ)
        assert p[1] == pytest.approx(1.0 / (sd * math.sqrt(2.0 * math.pi)))
        assert p[0] == pytest.approx(p[2])
        assert p[0] / p[1] == pytest.approx(math.exp(-0.5))

    def test_pdf_limit_integrates_to_one(self):
        x = numpy.linspace(-20.0, 20.0, 20001)
        assert numpy.trapezoid(ou.pdf_limit(x, self.μ, self.λ, self.σ), x) == pytest.approx(1.0, abs=1e-6)

    def test_cdf_limit_is_stationary_normal(self):
        sd = math.sqrt(ou.var_limit(self.λ, self.σ))
        c = ou.cdf_limit(numpy.array([self.μ - sd, self.μ, self.μ + sd]), self.μ, self.λ, self.σ)
        npt.assert_allclose(c, [norm.cdf(-1.0), 0.5, norm.cdf(1.0)])

    def test_pdf_limit_sigma_defaults_to_one(self):
        # λ = 0.5 with the default σ = 1 gives var_limit = σ²/(2λ) = 1 exactly, so the limiting
        # density is the standard normal centred on μ — hand values, not a re-call with σ = 1.0.
        x = numpy.array([self.μ - 1.0, self.μ, self.μ + 1.0])
        p = ou.pdf_limit(x, self.μ, 0.5)
        npt.assert_allclose(p, norm.pdf(numpy.array([-1.0, 0.0, 1.0])), rtol=1e-12)
        npt.assert_array_equal(p, ou.pdf_limit(x, self.μ, 0.5, 1.0))

    def test_cdf_limit_sigma_defaults_to_one(self):
        x = numpy.array([self.μ - 1.0, self.μ, self.μ + 1.0])
        c = ou.cdf_limit(x, self.μ, 0.5)
        npt.assert_allclose(c, [norm.cdf(-1.0), 0.5, norm.cdf(1.0)], rtol=1e-12)
        npt.assert_array_equal(c, ou.cdf_limit(x, self.μ, 0.5, 1.0))

    def test_cdf_limit_ignores_x0(self):
        x = numpy.linspace(-3.0, 5.0, 17)
        npt.assert_array_equal(ou.cdf_limit(x, self.μ, self.λ, self.σ, x0=0.0),
                               ou.cdf_limit(x, self.μ, self.λ, self.σ, x0=10.0))

    def test_time_dependent_distribution_converges_to_limit(self):
        x = numpy.linspace(-5.0, 7.0, 121)
        t_big = 60.0 / self.λ
        npt.assert_allclose(ou.pdf(x, self.μ, self.λ, t_big, self.σ, self.x0), ou.pdf_limit(x, self.μ, self.λ, self.σ), atol=1e-10)
        npt.assert_allclose(ou.cdf(x, self.μ, self.λ, t_big, self.σ, self.x0), ou.cdf_limit(x, self.μ, self.λ, self.σ), atol=1e-10)

    @pytest.mark.xfail(
        strict=True,
        reason="at t = 0 the OU variance is 0, so ou.cdf hands scale=0 to scipy's norm and "
               "gets nan back (verified: ou.cdf(x, 0, 1, 0.0) -> [nan, nan]) instead of the "
               "degenerate law X_0 = x0, whose CDF is the unit step at x0. Every mean/var "
               "test covers t = 0, so the distributions should too.",
    )
    def test_cdf_at_t0_is_the_unit_step_at_x0(self):
        x0 = 4.0
        c = ou.cdf(numpy.array([x0 - 1.0, x0 + 1.0]), self.μ, self.λ, 0.0, self.σ, x0)
        npt.assert_allclose(c, [0.0, 1.0])

    @pytest.mark.xfail(
        strict=True,
        reason="ou.pdf at t = 0 returns nan for the same scale=0 reason; the density of the "
               "point mass at x0 is 0 everywhere away from x0, never nan.",
    )
    def test_pdf_at_t0_is_zero_away_from_x0(self):
        x0 = 4.0
        p = ou.pdf(numpy.array([x0 - 1.0, x0 + 1.0]), self.μ, self.λ, 0.0, self.σ, x0)
        npt.assert_allclose(p, [0.0, 0.0])


class TestHalfLife:
    @pytest.mark.parametrize("λ, expected", [(1.0, LN2), (2.0, LN2 / 2.0), (0.1, 10.0 * LN2), (LN2, 1.0)])
    def test_ln2_over_lambda(self, λ, expected):
        assert ou.mean_halflife(λ) == pytest.approx(expected)


class TestDegenerateLambda:
    """λ ≤ 0 across both layers.

    Neither layer validates λ: it appears only inside divisions and exponentials, so λ = 0
    hits a bare division and λ < 0 silently produces a negative variance and a negative
    half-life. The notebooks scan λ, so these are the boundaries of the scanned range and
    are pinned here as characterisation — none of it is guarded, and a future guard should
    flip these tests rather than pass unnoticed.
    """

    def test_zero_lambda_divides_by_zero_in_the_variances(self):
        # σ²/(2λ) is a plain float division at lib/models/ou.py:54 and :73.
        with pytest.raises(ZeroDivisionError):
            ou.var_limit(0.0, 1.0)
        with pytest.raises(ZeroDivisionError):
            ou.var(0.0, numpy.array([1.0]), 1.0)
        with pytest.raises(ZeroDivisionError):
            ou_facade.compute_var(λ=0.0)
        with pytest.raises(ZeroDivisionError):
            ou_facade.compute_var_limit(λ=0.0)

    def test_zero_lambda_makes_cov_nan(self):
        # cov divides an ndarray instead, so 0/0 gives nan rather than an exception — the
        # limit of σ²/(2λ)(e^{-λ(t-s)} - e^{-λ(t+s)}) as λ -> 0 is the finite value σ²s.
        with numpy.errstate(invalid="ignore"):
            c = ou.cov(0.0, 1.0, numpy.array([2.0]), 1.0)
        assert numpy.isnan(c).all()

    def test_zero_lambda_half_life_is_infinite(self):
        # ln2/λ on a numpy scalar: inf, not an exception. This one is the correct limit.
        with numpy.errstate(divide="ignore"):
            assert math.isinf(ou.mean_halflife(0.0))
            assert math.isinf(ou_facade.compute_mean_half_life(λ=0.0))

    def test_negative_lambda_returns_a_negative_variance_and_half_life(self):
        assert ou.var_limit(-0.5, 1.0) == pytest.approx(-1.0)
        assert ou.mean_halflife(-1.0) == pytest.approx(-LN2)
        npt.assert_allclose(ou_facade.compute_var_limit(λ=-0.5, σ=1.0, npts=3)[1], numpy.full(3, -1.0))
        assert ou_facade.compute_mean_half_life(λ=-1.0) == pytest.approx(-LN2)

    def test_negative_lambda_makes_the_variance_diverge(self):
        # λ = -0.5 is explosive: var(t) = (1/(2λ))(1 - e^{-2λt}) = e^{t} - 1, unbounded and
        # nowhere near the (negative) var_limit it is supposed to approach.
        t = numpy.linspace(0.0, 10.0, 11)
        npt.assert_allclose(ou.var(-0.5, t, 1.0), numpy.exp(t) - 1.0, rtol=1e-12)
        assert ou.var(-0.5, t, 1.0)[-1] > 1e4


# ---------------------------------------------------------------------------
# Model layer: samplers (round trips)
# ---------------------------------------------------------------------------


class TestXt:
    μ, λ, t, σ, x0 = 1.0, 0.5, 1.0, 1.0, 3.0

    def test_shape_and_dtype(self):
        x = ou.xt(self.μ, self.λ, self.t, self.σ, self.x0, 17)
        assert x.shape == (17,)
        assert x.dtype == numpy.float64

    def test_default_n_is_one(self):
        assert ou.xt(self.μ, self.λ, self.t).shape == (1,)

    def test_defaults_sigma_one_x0_zero(self):
        # Byte-exact reseed-and-compare, the pattern every other default in the module gets
        # (TestOuPath.test_defaults_sigma_one_x0_zero, TestMean.test_x0_defaults_to_zero):
        # a swapped σ/x0 default, or an x0 dropped from the mean, changes these draws.
        numpy.random.seed(11)
        a = ou.xt(self.μ, self.λ, self.t, n=5)
        numpy.random.seed(11)
        b = ou.xt(self.μ, self.λ, self.t, 1.0, 0.0, 5)
        npt.assert_array_equal(a, b)
        # …and the defaults really are σ = 1, x0 = 0, not merely self-consistent: with those
        # values the law at t is N(μ(1 - e^{-λt}), (1 - e^{-2λt})/(2λ)), reconstructed here
        # from the same standard normals the sampler consumes.
        numpy.random.seed(11)
        ε = numpy.random.normal(0.0, 1.0, 5)
        m = self.μ * (1.0 - math.exp(-self.λ * self.t))
        sd = math.sqrt((1.0 - math.exp(-2.0 * self.λ * self.t)) / (2.0 * self.λ))
        npt.assert_allclose(a, m + sd * ε, rtol=1e-12)

    def test_zero_noise_collapses_to_mean(self):
        x = ou.xt(self.μ, self.λ, self.t, 0.0, self.x0, 5)
        npt.assert_allclose(x, ou.mean(self.μ, self.λ, numpy.array([self.t]), self.x0)[0])

    def test_sample_moments_match_closed_forms(self):
        n = 4000
        mean_t = ou.mean(self.μ, self.λ, numpy.array([self.t]), self.x0)[0]
        var_t = ou.var(self.λ, numpy.array([self.t]), self.σ)[0]
        x = ou.xt(self.μ, self.λ, self.t, self.σ, self.x0, n)
        # SE(mean) = sd/√n ≈ 0.0126 → 4 SE = 0.05; SE(var) ≈ var √(2/n) ≈ 2.2% → 4 SE = 9%.
        assert x.mean() == pytest.approx(mean_t, abs=0.05)
        assert x.var(ddof=1) == pytest.approx(var_t, rel=0.09)

    def test_samples_are_normal_with_closed_form_moments(self):
        mean_t = ou.mean(self.μ, self.λ, numpy.array([self.t]), self.x0)[0]
        sd_t = math.sqrt(ou.var(self.λ, numpy.array([self.t]), self.σ)[0])
        x = ou.xt(self.μ, self.λ, self.t, self.σ, self.x0, 4000)
        # KS against the closed-form normal, bounded by a fixed critical value rather than a
        # p-value: a p-value threshold fails for a fixed fraction of seeds by construction.
        # 1.95/√n is the ~1e-3 two-sided critical value (0.031 at n = 4000); measured
        # statistics run 0.010-0.020.
        assert kstest(x, norm(loc=mean_t, scale=sd_t).cdf).statistic < 1.95 / math.sqrt(4000)

    def test_large_t_variance_is_var_limit(self):
        x = ou.xt(self.μ, self.λ, 100.0, self.σ, self.x0, 4000)
        assert x.var(ddof=1) == pytest.approx(ou.var_limit(self.λ, self.σ), rel=0.09)
        assert x.mean() == pytest.approx(self.μ, abs=0.05)


class TestOuPath:
    def test_shape_and_initial_value(self):
        x = ou.ou(1.0, 0.5, 1.0, 25, 1.0, 2.5)
        assert x.shape == (25,)
        assert x[0] == 2.5

    def test_defaults_sigma_one_x0_zero(self):
        numpy.random.seed(7)
        a = ou.ou(1.0, 0.5, 1.0, 25)
        numpy.random.seed(7)
        b = ou.ou(1.0, 0.5, 1.0, 25, 1.0, 0.0)
        npt.assert_array_equal(a, b)
        assert a[0] == 0.0

    def test_zero_noise_follows_discrete_relaxation(self):
        # With σ = 0 the Euler recursion x_{i+1} = x_i + λ(μ - x_i)Δt has the closed
        # form x_i = μ + (x0 - μ)(1 - λΔt)^i.
        μ, λ, Δt, n, x0 = 2.0, 0.8, 0.25, 30, -1.0
        x = ou.ou(μ, λ, Δt, n, 0.0, x0)
        expected = μ + (x0 - μ) * (1.0 - λ * Δt) ** numpy.arange(n)
        npt.assert_allclose(x, expected, rtol=1e-12)

    def test_stationary_statistics_match_closed_forms_at_unit_dt(self):
        # λΔt = 0.1 keeps the Euler discretisation bias in the stationary variance
        # (factor 1/(1 - λΔt/2) ≈ 5%) small. n = 10 000 with correlation time 1/λ = 10
        # steps gives ~500 effective samples: SE(var) ≈ 6%, so 3 SE + bias ≈ 25%.
        μ, λ, Δt, σ = 0.0, 0.1, 1.0, 1.0
        x = ou.ou(μ, λ, Δt, 10_000, σ, 0.0)[200:]  # drop 20 correlation times of burn-in
        assert x.var() == pytest.approx(ou.var_limit(λ, σ), rel=0.25)
        # SE(mean) ≈ √(var · 2τ / n) ≈ 0.1; 5 SE
        assert x.mean() == pytest.approx(μ, abs=0.5)
        # The sampler is the Euler recursion x_{i+1} = (1 - λΔt)x_i + λμΔt + noise, i.e. an
        # AR(1) with φ = 1 - λΔt = 0.9 — not the exact process's e^{-λΔt} = 0.90484. Those two
        # predictions differ by 0.005 here, comparable to the ≈0.004 sampling SE of φ̂, so at
        # λΔt = 0.1 this assertion can only pin the strength of the autocorrelation.
        # test_lag1_autocorrelation_is_the_euler_coefficient separates them at λΔt = 0.5.
        acf1 = numpy.corrcoef(x[:-1], x[1:])[0, 1]
        assert acf1 == pytest.approx(1.0 - λ * Δt, abs=0.02)

    def test_lag1_autocorrelation_is_the_euler_coefficient(self):
        # λΔt = 0.5 separates the Euler AR(1) coefficient 1 - λΔt = 0.5 from the exact
        # process's e^{-λΔt} = 0.6065. SE(φ̂) ≈ √((1 - φ²)/n) ≈ 0.009 at n ≈ 9900, so
        # abs = 0.03 is > 3 SE and still excludes 0.6065 by more than three times that width.
        λ, Δt = 0.5, 1.0
        x = ou.ou(0.0, λ, Δt, 10_000, 1.0, 0.0)[100:]
        acf1 = numpy.corrcoef(x[:-1], x[1:])[0, 1]
        assert acf1 == pytest.approx(1.0 - λ * Δt, abs=0.03)
        assert abs(acf1 - math.exp(-λ * Δt)) > 0.05

    def test_single_step_path_is_just_x0(self):
        x = ou.ou(1.0, 0.5, 1.0, 1, 1.0, 2.5)
        assert x.shape == (1,)
        assert x[0] == 2.5

    @pytest.mark.xfail(
        strict=True,
        reason="ou.ou allocates numpy.zeros(n) then unconditionally assigns x[0] = x0, so "
               "n = 0 raises IndexError instead of returning the empty path.",
    )
    def test_zero_length_path_is_empty(self):
        assert ou.ou(1.0, 0.5, 1.0, 0, 1.0, 2.5).shape == (0,)

    @pytest.mark.xfail(
        strict=True,
        reason="ou.ou scales the noise as σ·Δt·ε instead of σ·√Δt·ε, so for Δt ≠ 1 the "
               "increment variance is σ²Δt² rather than the SDE's σ²Δt (paths are far too "
               "smooth for Δt < 1); all closed forms in lib.models.ou assume σ²Δt.",
    )
    def test_increment_variance_scales_as_sigma2_dt(self):
        # E[(dX)²] = σ² dt for dX = λ(μ - X)dt + σ dW. With Δt = 0.01 and λ = 1 the drift
        # contribution λ²Δt²Var(X) ≈ 5e-5 is negligible against σ²Δt = 0.01.
        # Sample variance of 2000 increments has SE ≈ √(2/2000) ≈ 3%.
        σ, Δt = 1.0, 0.01
        x = ou.ou(0.0, 1.0, Δt, 2001, σ, 0.0)
        assert numpy.diff(x).var() == pytest.approx(σ**2 * Δt, rel=0.15)


class TestSamplerConsistency:
    """``ou.ou`` and ``ou.xt`` are independent implementations of the same law.

    ``ou.xt`` draws the exact marginal at a time t; ``ou.ou`` walks an Euler path. Nothing
    else in this file compares the two directly — every other sampler test checks one of
    them against a closed form — yet the notebooks use them interchangeably.
    """

    μ, λ, σ, x0 = 1.0, 0.05, 1.0, 4.0

    def _compare(self, Δt, npts, nsim):
        _, ensemble = create_ensemble(ou_facade.create_source, nsim,
                                      μ=self.μ, λ=self.λ, Δt=Δt, σ=self.σ, x0=self.x0, npts=npts)
        path_end = numpy.array(ensemble)[:, -1]
        draws = ou.xt(self.μ, self.λ, (npts - 1) * Δt, self.σ, self.x0, nsim)
        return path_end, draws

    def test_path_marginal_agrees_with_xt_law_at_unit_dt(self):
        # λΔt = 0.05 keeps the Euler discretisation bias in the variance near 3% (the path's
        # AR(1) coefficient is 1 - λΔt = 0.95 against the exact e^{-0.05} = 0.9512). At
        # nsim = 1000 SE(var) ≈ 4.5%, so rtol = 0.2 is bias + ~3.7 SE — the measured ratio
        # spanned 0.97-1.13 over 12 seeds. SE of the difference of the two means is ≈ 0.14,
        # so abs = 0.5 is ~3.5 SE against a measured maximum of 0.25.
        nsim = 1000
        path_end, draws = self._compare(Δt=1.0, npts=41, nsim=nsim)
        assert path_end.mean() == pytest.approx(draws.mean(), abs=0.5)
        assert path_end.var(ddof=1) == pytest.approx(draws.var(ddof=1), rel=0.2)
        # Whole-distribution check with a deterministic bound: 1.95√(2/nsim) is the ~1e-3
        # two-sided critical value of the two-sample KS statistic at equal sample sizes.
        assert ks_2samp(path_end, draws).statistic < 1.95 * math.sqrt(2.0 / nsim)

    @pytest.mark.xfail(
        strict=True,
        reason="ou.ou scales its noise as σ·Δt·ε instead of σ·√Δt·ε (lib/models/ou.py:281), "
               "so the Euler path's marginal variance is ≈ Δt × the exact law ou.xt draws "
               "from. Measured at Δt = 0.5, λ = 0.05, t = 40, nsim = 600: path variance 4.32 "
               "against 9.72 from ou.xt, a ratio of 0.44 where 1.00 ± 0.20 is expected.",
    )
    def test_path_marginal_agrees_with_xt_law_at_non_unit_dt(self):
        nsim = 600
        path_end, draws = self._compare(Δt=0.5, npts=81, nsim=nsim)
        assert path_end.var(ddof=1) == pytest.approx(draws.var(ddof=1), rel=0.2)
        assert path_end.mean() == pytest.approx(draws.mean(), abs=0.5)


# ---------------------------------------------------------------------------
# Façade layer: (t, values) contracts and kwargs handling
# ---------------------------------------------------------------------------


def _assert_pair(result, n):
    assert isinstance(result, tuple) and len(result) == 2
    t, v = result
    assert isinstance(t, numpy.ndarray) and isinstance(v, numpy.ndarray)
    assert t.shape == (n,) and v.shape == (n,)
    assert numpy.issubdtype(t.dtype, numpy.floating) and numpy.issubdtype(v.dtype, numpy.floating)


class TestFacadeDefaults:
    def test_compute_mean_default_grid_and_values(self):
        result = ou_facade.compute_mean()
        _assert_pair(result, 11)
        t, m = result
        npt.assert_allclose(t, numpy.arange(11, dtype=float))
        npt.assert_array_equal(m, numpy.zeros(11))  # μ = x0 = 0 → identically zero

    def test_compute_mean_limit_default(self):
        result = ou_facade.compute_mean_limit()
        _assert_pair(result, 11)
        npt.assert_allclose(result[0], numpy.arange(11, dtype=float))
        npt.assert_array_equal(result[1], numpy.zeros(11))

    def test_compute_var_default_grid(self):
        result = ou_facade.compute_var()
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.arange(10, dtype=float))
        assert result[1][0] == 0.0
        assert result[1][-1] == pytest.approx(0.5, rel=1e-7)  # σ²/(2λ) = 0.5 at t = 9

    def test_compute_var_limit_default(self):
        result = ou_facade.compute_var_limit()
        _assert_pair(result, 10)
        npt.assert_array_equal(result[1], numpy.full(10, 0.5))

    def test_compute_cov_default_grid_starts_at_s(self):
        # The 10 points here are a coincidence of the defaults s = Δt = 1: compute_cov's npts
        # sets the END TIME (xmax = npts·Δt), not the point count — see
        # TestFacadeValues.test_compute_cov_npts_sets_the_end_time_not_the_point_count.
        result = ou_facade.compute_cov()
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.arange(1, 11, dtype=float))

    def test_compute_cov_limit_default(self):
        result = ou_facade.compute_cov_limit()
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.arange(1, 11, dtype=float))
        npt.assert_array_equal(result[1], numpy.zeros(10))

    @pytest.mark.parametrize("func", [ou_facade.compute_pdf, ou_facade.compute_cdf])
    def test_pdf_cdf_default_x_grid(self, func):
        result = func(t=1.0)
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.linspace(-5.0, 5.0, 10))

    @pytest.mark.parametrize("func", [ou_facade.compute_pdf_limit, ou_facade.compute_cdf_limit])
    def test_limit_pdf_cdf_default_x_grid(self, func):
        result = func()
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.linspace(-5.0, 5.0, 10))

    def test_compute_mean_half_life_default_lambda(self):
        assert ou_facade.compute_mean_half_life() == pytest.approx(LN2)

    def test_create_source_default_shape(self):
        result = ou_facade.create_source()
        _assert_pair(result, 10)
        npt.assert_allclose(result[0], numpy.arange(10, dtype=float))
        assert result[1][0] == 0.0

    def test_create_xt_source_default_npts(self):
        x = ou_facade.create_xt_source(t=1.0)
        assert isinstance(x, numpy.ndarray) and x.shape == (10,)

    def test_create_xt_source_defaults_produce_the_documented_law(self):
        # The defaults μ = 0, λ = 1, σ = 1, x0 = 0 fully determine the law at t = 1:
        # N(0, (1 - e^{-2})/2), variance 0.43233. At n = 4000, SE(mean) = 0.0104 (4 SE =
        # 0.042) and SE(var) = var√(2/n) = 2.2% (4 SE = 9%) — the tolerances TestXt uses.
        x = ou_facade.create_xt_source(t=1.0, npts=4000)
        var_t = ou.var(1.0, numpy.array([1.0]), 1.0)[0]
        assert var_t == pytest.approx(0.5 * (1.0 - math.exp(-2.0)))
        assert x.mean() == pytest.approx(0.0, abs=0.05)
        assert x.var(ddof=1) == pytest.approx(var_t, rel=0.09)
        # Fixed KS critical value, not a p-value threshold — see TestXt for the rationale.
        assert kstest(x, norm(loc=0.0, scale=math.sqrt(var_t)).cdf).statistic < 1.95 / math.sqrt(4000)


class TestFacadeRequiredKwargs:
    @pytest.mark.parametrize("func", [ou_facade.compute_pdf, ou_facade.compute_cdf, ou_facade.create_xt_source])
    def test_t_is_required(self, func):
        with pytest.raises(Exception, match="t parameter is required"):
            func(μ=0.0, λ=1.0, σ=1.0)


class TestFacadeValues:
    def test_compute_mean_hand_computed(self):
        # λ = ln2, Δt = 1: mean(k) = x0 2^{-k} + μ (1 - 2^{-k})
        t, m = ou_facade.compute_mean(μ=1.0, λ=LN2, x0=2.0, Δt=1.0, npts=4)
        npt.assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
        npt.assert_allclose(m, [2.0, 1.5, 1.25, 1.125], rtol=1e-12)

    def test_compute_mean_respects_dt(self):
        t, m = ou_facade.compute_mean(μ=0.0, λ=LN2, x0=8.0, Δt=0.5, npts=5)
        npt.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0])
        npt.assert_allclose(m, 8.0 * 2.0 ** -t, rtol=1e-12)

    def test_compute_mean_routes_kwargs_to_the_model(self):
        # Routing only — the expected values come from the very function the façade calls, so
        # this catches a μ/x0 swap or a dropped kwarg, NOT a wrong mean. The values are pinned
        # by hand in test_compute_mean_hand_computed / test_compute_mean_respects_dt above.
        t, m = ou_facade.compute_mean(μ=1.5, λ=0.3, x0=-2.0, Δt=0.25, npts=21)
        npt.assert_array_equal(m, ou.mean(1.5, 0.3, t, -2.0))

    def test_compute_mean_limit_is_constant_mu(self):
        t, m = ou_facade.compute_mean_limit(μ=3.5, npts=7, Δt=2.0)
        npt.assert_allclose(t, 2.0 * numpy.arange(7))
        npt.assert_array_equal(m, numpy.full(7, 3.5))

    def test_compute_var_hand_computed(self):
        # λ = ln2/2, σ = 1: var(k) = (1/ln2)(1 - 2^{-k})
        t, v = ou_facade.compute_var(λ=LN2 / 2.0, σ=1.0, Δt=1.0, npts=4)
        npt.assert_allclose(v, (1.0 / LN2) * numpy.array([0.0, 0.5, 0.75, 0.875]), rtol=1e-12)

    def test_compute_var_routes_kwargs_to_the_model(self):
        # Routing only, as above; the values are pinned by hand in test_compute_var_hand_computed.
        t, v = ou_facade.compute_var(λ=0.3, σ=1.7, Δt=0.25, npts=21)
        npt.assert_array_equal(v, ou.var(0.3, t, 1.7))

    def test_compute_var_limit_is_constant(self):
        t, v = ou_facade.compute_var_limit(λ=0.5, σ=2.0, npts=6, Δt=0.5)
        npt.assert_allclose(t, 0.5 * numpy.arange(6))
        npt.assert_array_equal(v, numpy.full(6, 4.0))

    def test_compute_cov_values_follow_the_markov_decay(self):
        # Values only. The grid is read off the returned t instead of being asserted, so this
        # test keeps working when compute_cov's point count is fixed — the point count itself
        # is characterised in the single test below, next to the xfail that contradicts it.
        λ, σ, s, Δt, npts = 0.6, 1.3, 2.0, 0.5, 10
        t, c = ou_facade.compute_cov(λ=λ, σ=σ, s=s, Δt=Δt, npts=npts)
        assert t[0] == pytest.approx(s)
        npt.assert_allclose(numpy.diff(t), Δt)
        # Cov(X_s, X_s) = Var(X_s), hand-computed from σ²/(2λ)(1 - e^{-2λs})
        assert c[0] == pytest.approx((σ**2 / (2.0 * λ)) * (1.0 - math.exp(-2.0 * λ * s)))
        npt.assert_allclose(c, c[0] * numpy.exp(-λ * (t - s)), rtol=1e-12)  # Markov decay
        npt.assert_array_equal(c, ou.cov(λ, s, t, σ))  # λ/σ/s reach the model unswapped

    def test_compute_cov_default_sigma_is_one(self):
        # Every other value assertion in this file passes σ explicitly. With λ = 0.5, s = 1
        # and the default σ = 1, Cov(X_1, X_1) = Var(X_1) = (1/(2·0.5))(1 - e^{-1}) = 1 - e^{-1}.
        λ, s = 0.5, 1.0
        t, c = ou_facade.compute_cov(λ=λ, s=s, Δt=1.0, npts=6)
        assert c[0] == pytest.approx(1.0 - math.exp(-1.0))
        npt.assert_allclose(c, c[0] * numpy.exp(-λ * (t - s)), rtol=1e-12)

    # This characterisation and the strict xfail immediately below it are a PAIR: they assert
    # opposite things about the same call on purpose. When compute_cov is fixed the xfail
    # XPASSes and this test must be updated in the same edit.
    def test_compute_cov_npts_sets_the_end_time_not_the_point_count(self):
        # compute_cov calls create_space(xmin=s, xmax=npts·Δt, Δx=Δt), so with s = 2, Δt = 1
        # and npts = 10 requested the grid is 2, 3, …, 10 — nine points, not ten.
        t, c = ou_facade.compute_cov(λ=0.6, σ=1.3, s=2.0, Δt=1.0, npts=10)
        npt.assert_allclose(t, numpy.arange(2.0, 11.0))
        assert len(t) == 9
        npt.assert_array_equal(c, ou.cov(0.6, 2.0, t, 1.3))

    @pytest.mark.xfail(
        strict=True,
        reason="compute_cov's npts is the only npts in lib/data/impl/ou.py that is not the "
               "number of returned points — lib/data/impl/ou.py:167-169 feeds it to "
               "create_space as xmax = npts·Δt — so s = 2, Δt = 1, npts = 10 returns 9 "
               "points. Every other compute_* in the module returns exactly npts.",
    )
    def test_compute_cov_npts_is_the_point_count(self):
        t, _ = ou_facade.compute_cov(s=2.0, Δt=1.0, npts=10)
        assert len(t) == 10

    def test_compute_cov_limit_values_are_zero(self):
        t, c = ou_facade.compute_cov_limit(s=3.0, Δt=1.0, npts=5)
        npt.assert_array_equal(c, numpy.zeros(5))
        npt.assert_allclose(numpy.diff(t), 1.0)

    @pytest.mark.parametrize("kw", [{}, {"λ": 7.0}, {"σ": 13.0}, {"λ": 0.01, "σ": 0.1}])
    def test_compute_cov_limit_output_is_independent_of_lambda_and_sigma(self, kw):
        # lib/data/impl/ou.py:192-199 never reads λ or σ — the t -> ∞ covariance at fixed s is
        # 0 for every mean-reverting parameterisation, so the constant zero is correct. Pinned
        # so a change that made the limit λ-dependent (or that silently reintroduced σ²/(2λ))
        # cannot slip through: both the values and the grid must be untouched.
        t, c = ou_facade.compute_cov_limit(s=3.0, Δt=1.0, npts=5, **kw)
        npt.assert_array_equal(c, numpy.zeros(5))
        npt.assert_allclose(t, numpy.arange(3.0, 8.0))

    @pytest.mark.parametrize(
        "s, Δt",
        [
            (3.0, 1.0),
            pytest.param(
                2.0, 0.5,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="compute_cov_limit starts its time axis at int(s/Δt) — the step index — "
                           "rather than at the time s, so for Δt ≠ 1 (or non-integer s) it is "
                           "misaligned with compute_cov, whose axis starts at s.",
                ),
            ),
        ],
    )
    def test_compute_cov_limit_axis_starts_at_s_like_compute_cov(self, s, Δt):
        # Start point only. A shared start is NOT a shared axis: the two grids still differ
        # in length and endpoint for every s ≠ Δt, including the Δt = 1 case that passes
        # here — that is what test_compute_cov_limit_axis_matches_compute_cov pins.
        t_cov, _ = ou_facade.compute_cov(s=s, Δt=Δt, npts=10)
        t_lim, _ = ou_facade.compute_cov_limit(s=s, Δt=Δt, npts=10)
        assert t_lim[0] == pytest.approx(s)
        assert t_lim[0] == pytest.approx(t_cov[0])

    @pytest.mark.parametrize(
        "s, Δt",
        [
            (1.0, 1.0),  # the s = Δt = 1 coincidence the defaults sit on: here they do agree
            pytest.param(2.0, 1.0, marks=pytest.mark.xfail(strict=True, reason=_COV_AXIS_REASON)),
            pytest.param(3.0, 1.0, marks=pytest.mark.xfail(strict=True, reason=_COV_AXIS_REASON)),
            pytest.param(2.0, 0.5, marks=pytest.mark.xfail(strict=True, reason=_COV_AXIS_REASON)),
        ],
    )
    def test_compute_cov_limit_axis_matches_compute_cov(self, s, Δt):
        # The notebooks draw the covariance and its t -> ∞ limit on one shared axis, so the
        # two functions have to return the same grid, not merely the same first point.
        t_cov, _ = ou_facade.compute_cov(s=s, Δt=Δt, npts=10)
        t_lim, _ = ou_facade.compute_cov_limit(s=s, Δt=Δt, npts=10)
        assert len(t_lim) == len(t_cov)
        assert t_lim[-1] == pytest.approx(t_cov[-1])
        npt.assert_allclose(t_lim, t_cov)

    def test_compute_pdf_grid_and_normalisation(self):
        x, p = ou_facade.compute_pdf(t=1.2, μ=1.0, λ=0.5, σ=1.5, x0=4.0, xmin=-15.0, xmax=15.0, npts=3001)
        npt.assert_allclose(x, numpy.linspace(-15.0, 15.0, 3001))
        assert numpy.trapezoid(p, x) == pytest.approx(1.0, abs=1e-6)
        mean_t = ou.mean(1.0, 0.5, numpy.array([1.2]), 4.0)[0]
        assert x[numpy.argmax(p)] == pytest.approx(mean_t, abs=0.01)
        npt.assert_array_equal(p, ou.pdf(x, 1.0, 0.5, 1.2, 1.5, 4.0))

    def test_compute_cdf_half_at_mean(self):
        # μ = x0 = 0 so the mean is 0, which the symmetric odd grid contains exactly.
        x, c = ou_facade.compute_cdf(t=1.2, μ=0.0, λ=0.5, σ=1.5, x0=0.0, xmin=-10.0, xmax=10.0, npts=201)
        assert x[100] == 0.0
        assert c[100] == pytest.approx(0.5)
        assert numpy.all(numpy.diff(c) >= 0)
        assert c[0] == pytest.approx(0.0, abs=1e-9) and c[-1] == pytest.approx(1.0, abs=1e-9)
        npt.assert_array_equal(c, ou.cdf(x, 0.0, 0.5, 1.2, 1.5, 0.0))

    def test_compute_pdf_limit_matches_large_t_pdf(self):
        kw = dict(μ=1.0, λ=0.5, σ=1.5, xmin=-8.0, xmax=10.0, npts=181)
        x_lim, p_lim = ou_facade.compute_pdf_limit(**kw)
        x_t, p_t = ou_facade.compute_pdf(t=200.0, x0=4.0, **kw)
        npt.assert_array_equal(x_lim, x_t)
        npt.assert_allclose(p_lim, p_t, atol=1e-10)
        assert numpy.trapezoid(p_lim, x_lim) == pytest.approx(1.0, abs=1e-4)
        assert x_lim[numpy.argmax(p_lim)] == pytest.approx(1.0, abs=0.1)

    def test_compute_cdf_limit_matches_large_t_cdf(self):
        kw = dict(μ=1.0, λ=0.5, σ=1.5, xmin=-8.0, xmax=10.0, npts=181)
        x_lim, c_lim = ou_facade.compute_cdf_limit(x0=99.0, **kw)  # x0 accepted and ignored
        x_t, c_t = ou_facade.compute_cdf(t=200.0, x0=4.0, **kw)
        npt.assert_array_equal(x_lim, x_t)
        npt.assert_allclose(c_lim, c_t, atol=1e-10)
        assert c_lim[90] == pytest.approx(0.5)  # x = μ = 1.0 sits at index 90

    def test_limit_functions_ignore_x0_although_only_one_of_them_reads_it(self):
        # compute_cdf_limit reads x0 (lib/data/impl/ou.py:352) and forwards it to
        # ou.cdf_limit, which ignores it; compute_pdf_limit never reads it at all, because
        # ou.pdf_limit has no x0 parameter. The observable contract is the same for both — the
        # t -> ∞ law forgets the initial condition — and that is what the notebooks rely on,
        # so pin it on both sides rather than only on compute_cdf_limit.
        kw = dict(μ=1.0, λ=0.5, σ=1.5, xmin=-8.0, xmax=10.0, npts=21)
        npt.assert_array_equal(ou_facade.compute_pdf_limit(**kw)[1],
                               ou_facade.compute_pdf_limit(x0=99.0, **kw)[1])
        npt.assert_array_equal(ou_facade.compute_cdf_limit(**kw)[1],
                               ou_facade.compute_cdf_limit(x0=99.0, **kw)[1])

    def test_compute_pdf_limit_default_sigma_is_one(self):
        # λ = 0.5 with the default σ = 1 gives var_limit = σ²/(2λ) = 1, so with μ = 0 the
        # limiting density is exactly the standard normal.
        x, p = ou_facade.compute_pdf_limit(μ=0.0, λ=0.5, xmin=-5.0, xmax=5.0, npts=11)
        npt.assert_allclose(x, numpy.linspace(-5.0, 5.0, 11))
        npt.assert_allclose(p, norm.pdf(x), rtol=1e-12)

    def test_compute_cdf_limit_default_sigma_is_one(self):
        x, c = ou_facade.compute_cdf_limit(μ=0.0, λ=0.5, xmin=-5.0, xmax=5.0, npts=11)
        npt.assert_allclose(c, norm.cdf(x), rtol=1e-12)

    @pytest.mark.parametrize("λ", [0.25, 1.0, 4.0])
    def test_compute_mean_half_life(self, λ):
        assert ou_facade.compute_mean_half_life(λ=λ) == pytest.approx(LN2 / λ)
        assert ou_facade.compute_mean_half_life(λ=λ) == ou.mean_halflife(λ)


class TestFacadeDocumentedKwargs:
    """Kwargs the docstrings advertise that the implementations never read.

    Each one is silently dropped: the call succeeds and returns the default-sized grid, so a
    notebook asking for 4000 samples gets 10 and never learns about it.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="lib/data/impl/ou.py:417 documents an `n` kwarg for create_xt_source but the "
               "body reads `npts` (line 427), so create_xt_source(t=1.0, n=4000) silently "
               "returns 10 samples instead of 4000.",
    )
    def test_create_xt_source_honours_the_documented_n(self):
        assert ou_facade.create_xt_source(t=1.0, n=4000).shape == (4000,)

    @pytest.mark.xfail(
        strict=True,
        reason="lib/data/impl/ou.py:448 documents an `n` kwarg for create_source but the body "
               "reads `npts` (line 462), so create_source(n=25) silently returns 10 points.",
    )
    def test_create_source_honours_the_documented_n(self):
        t, x = ou_facade.create_source(n=25)
        assert len(t) == 25 and len(x) == 25

    @pytest.mark.parametrize(
        "func, required",
        [
            (ou_facade.compute_pdf, {"t": 1.0}),
            (ou_facade.compute_cdf, {"t": 1.0}),
            (ou_facade.compute_pdf_limit, {}),
            (ou_facade.compute_cdf_limit, {}),
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="compute_pdf/compute_cdf/compute_pdf_limit/compute_cdf_limit all document a Δx "
               "kwarg (lib/data/impl/ou.py:209, 251, 293, 329) but each builds its grid with "
               "create_space(xmin=…, xmax=…, npts=…) and never reads Δx, so Δx = 0.5 over "
               "[-5, 5] still returns the default 10 points spaced 10/9 = 1.111 apart.",
    )
    def test_dx_kwarg_sets_the_grid_spacing(self, func, required):
        x, _ = func(xmin=-5.0, xmax=5.0, Δx=0.5, **required)
        npt.assert_allclose(numpy.diff(x), 0.5)


class TestFacadeSources:
    def test_create_source_grid_and_initial_value(self):
        t, x = ou_facade.create_source(μ=1.0, λ=0.5, Δt=0.25, σ=1.0, x0=3.0, npts=41)
        npt.assert_allclose(t, 0.25 * numpy.arange(41))
        assert x.shape == (41,) and x[0] == 3.0

    def test_create_source_agrees_with_model_under_same_seed(self):
        numpy.random.seed(99)
        _, x_facade = ou_facade.create_source(μ=1.0, λ=0.5, Δt=1.0, σ=1.3, x0=3.0, npts=50)
        numpy.random.seed(99)
        x_model = ou.ou(1.0, 0.5, 1.0, 50, 1.3, 3.0)
        npt.assert_array_equal(x_facade, x_model)

    def test_create_source_zero_noise_relaxation(self):
        μ, λ, Δt, x0 = 2.0, 0.8, 0.25, -1.0
        t, x = ou_facade.create_source(μ=μ, λ=λ, Δt=Δt, σ=0.0, x0=x0, npts=30)
        npt.assert_allclose(x, μ + (x0 - μ) * (1.0 - λ * Δt) ** numpy.arange(30), rtol=1e-12)

    def test_create_xt_source_routes_kwargs(self):
        x = ou_facade.create_xt_source(t=1.0, μ=1.0, λ=0.5, σ=0.0, x0=3.0, npts=6)
        assert x.shape == (6,)
        npt.assert_allclose(x, ou.mean(1.0, 0.5, numpy.array([1.0]), 3.0)[0])

    def test_create_xt_source_agrees_with_model_under_same_seed(self):
        numpy.random.seed(99)
        a = ou_facade.create_xt_source(t=1.0, μ=1.0, λ=0.5, σ=1.3, x0=3.0, npts=50)
        numpy.random.seed(99)
        b = ou.xt(1.0, 0.5, 1.0, 1.3, 3.0, 50)
        npt.assert_array_equal(a, b)

    def test_create_xt_source_moments(self):
        x = ou_facade.create_xt_source(t=1.0, μ=1.0, λ=0.5, σ=1.0, x0=3.0, npts=4000)
        mean_t = ou.mean(1.0, 0.5, numpy.array([1.0]), 3.0)[0]
        var_t = ou.var(0.5, numpy.array([1.0]), 1.0)[0]
        # same tolerances as TestXt: 4 SE of mean and of variance at n = 4000
        assert x.mean() == pytest.approx(mean_t, abs=0.05)
        assert x.var(ddof=1) == pytest.approx(var_t, rel=0.09)

    def test_ensemble_moments_track_compute_mean_and_var(self):
        # The notebook comparison: an ensemble of paths versus the closed-form mean and
        # variance curves. Δt = 1, λ = 0.1 keeps the Euler bias ≤ 2% in the mean and
        # ≤ 10% in the variance. With nsim = 1000, SE(mean) ≤ 0.07 and SE(var) ≈ 4.5%
        # per time point; the max over 31 correlated points stays within ~3 SE.
        kw = dict(μ=1.0, λ=0.1, Δt=1.0, σ=1.0, x0=4.0, npts=31)
        t, ensemble = create_ensemble(ou_facade.create_source, 1000, **kw)
        paths = numpy.array(ensemble)
        assert paths.shape == (1000, 31)
        t_mean, mean_curve = ou_facade.compute_mean(**kw)
        t_var, var_curve = ou_facade.compute_var(**kw)
        npt.assert_array_equal(t, t_mean)
        npt.assert_array_equal(t, t_var)
        npt.assert_allclose(paths.mean(axis=0), mean_curve, atol=0.35)
        npt.assert_allclose(paths.var(axis=0)[1:], var_curve[1:], rtol=0.3)
        assert paths.var(axis=0)[0] == 0.0

    def test_ensemble_mean_tracks_compute_mean_at_non_unit_dt(self):
        # The drift half of the Euler step is untouched by the σ·Δt noise bug, so at Δt ≠ 1
        # the ensemble mean must still follow compute_mean. λΔt = 0.05 keeps the Euler mean
        # bias (0.95^i vs e^{-0.05i}, scaled by |x0 - μ| = 3) under 0.03; path sd ≤ 2.3 at
        # nsim = 1000 gives SE ≤ 0.073, so atol = 0.35 is bias + ~4 SE.
        kw = dict(μ=1.0, λ=0.1, Δt=0.5, σ=1.0, x0=4.0, npts=41)
        t, ensemble = create_ensemble(ou_facade.create_source, 1000, **kw)
        paths = numpy.array(ensemble)
        t_mean, mean_curve = ou_facade.compute_mean(**kw)
        npt.assert_array_equal(t, t_mean)
        npt.assert_allclose(t, 0.5 * numpy.arange(41))
        npt.assert_allclose(paths.mean(axis=0), mean_curve, atol=0.35)

    @pytest.mark.xfail(
        strict=True,
        reason="façade-level consequence of ou.ou scaling its noise as σ·Δt·ε instead of "
               "σ·√Δt·ε: an ensemble from create_source has variance ≈ Δt × the closed-form "
               "compute_var curve. Measured at Δt = 0.5, λ = 0.2, nsim = 1000: the ensemble "
               "variance at t = 20 is 1.306 against the closed form's 2.499 (ratio 0.522).",
    )
    def test_ensemble_variance_tracks_compute_var_at_non_unit_dt(self):
        # λΔt = 0.1 keeps the Euler variance bias ≈ 5%; SE(var) ≈ 4.5% at nsim = 1000, so
        # rtol = 0.2 would be comfortable for a correctly scaled sampler.
        kw = dict(μ=0.0, λ=0.2, Δt=0.5, σ=1.0, x0=0.0, npts=41)
        _, ensemble = create_ensemble(ou_facade.create_source, 1000, **kw)
        paths = numpy.array(ensemble)
        _, var_curve = ou_facade.compute_var(**kw)
        npt.assert_allclose(paths.var(axis=0)[1:], var_curve[1:], rtol=0.2)


# ---------------------------------------------------------------------------
# Façade layer: half-life estimation round trip
# ---------------------------------------------------------------------------


class TestHalfLifeEstimate:
    μ, λ, σ = 5.0, 0.8, 1.0

    @pytest.fixture
    def fit(self):
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=1.0, σ=self.σ, x0=0.0, npts=4000)
        return ou_facade.compute_mean_half_life_estimate(x, dt=1.0)

    def test_return_types(self, fit):
        report, result = fit
        assert isinstance(report, RegressionResultsWrapper)
        assert isinstance(result, OLSResult)
        assert result.est_model == EstModel.OLS
        assert len(result.params) == 1

    def test_report_and_result_describe_the_same_regression(self, fit):
        # OLS.single_variable_estimate maps exog column 0 (the constant statsmodels adds)
        # onto result.const and column 1 onto result.params[0]. Without this, a façade that
        # returned a stale or unrelated RegressionResultsWrapper alongside a correct
        # OLSResult would satisfy every other test here.
        report, result = fit
        params, bse = numpy.asarray(report.params), numpy.asarray(report.bse)
        assert params.shape == (2,) and bse.shape == (2,)
        assert params[0] == pytest.approx(result.const.est)
        assert bse[0] == pytest.approx(result.const.err)
        assert params[1] == pytest.approx(result.params[0].est)
        assert bse[1] == pytest.approx(result.params[0].err)
        assert report.rsquared == pytest.approx(result.r2)
        assert report.nobs == 3999  # 4000 samples -> 3999 first differences

    def test_transforms_are_populated(self, fit):
        _, result = fit
        assert result.model is not None and r"\lambda" in result.model and r"\Delta t" in result.model
        assert result.param_transforms is not None and len(result.param_transforms) == 2
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert half_life.est_label == r"$t_{1/2}$" and lam.est_label == r"$\lambda$"
        assert half_life.param_type == OLSParamType.TRANS_PARAM.value
        assert lam.param_type == OLSParamType.TRANS_PARAM.value
        assert half_life.row == 0 and lam.row == 1
        assert result.const_transform is not None
        assert result.const_transform.param.est_label == r"$\mu$"
        assert result.const_transform.param.param_type == OLSParamType.TRANS_CONST.value
        assert {half_life.est_id, lam.est_id, result.const_transform.param.est_id} == {result.est_id}

    def test_recovers_half_life_and_lambda(self, fit):
        # Regressing ΔX on X_{t-1} gives slope -λΔt.
        #
        # DELIBERATE CONVENTION PIN, not a library defect: the transform's own model string
        # is ΔX_t = λ X_{t-1} Δt + μ Δt + √Δt ε_t, in which mean reversion means λ < 0,
        # whereas lib.models.ou uses dX = λ(μ - X)dt with λ > 0. So the ParamEst labelled
        # $\lambda$ legitimately holds -0.79 for a path simulated with λ = +0.8, and the
        # half-life t½ = -ln2/λ comes out positive. A reader of the assertion alone would
        # assume the library is broken; it is not. If navi ever unifies the two conventions
        # these are the assertions that must change.
        #
        # AR(1) coefficient 1 - λ = 0.2, so SE(slope) ≈ √((1 - 0.04)/4000) ≈ 0.016 →
        # 3 SE ≈ 6% of λ; rel = 0.1.
        _, result = fit
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert lam.est < 0  # the sign convention above, stated explicitly
        assert lam.est == pytest.approx(-self.λ, rel=0.1)
        assert half_life.est == pytest.approx(ou.mean_halflife(self.λ), rel=0.1)
        assert half_life.est > 0 and half_life.err > 0
        # the reported error should be consistent with the actual estimation error
        assert abs(half_life.est - ou.mean_halflife(self.λ)) < 4.0 * half_life.err

    def test_half_life_error_is_the_delta_method_of_the_slope_error(self, fit):
        # t½ = -ln2/λ with λ = slope/dt, so dt½/dλ = ln2/λ² and the delta method gives
        # σ_{t½} = (ln2/λ²)(σ_slope/dt).
        #
        # This restates the implementation's own arithmetic, so it is a CHANGE DETECTOR only:
        # it cannot tell a correct delta method from a mis-derived one, only from a source
        # edit. The magnitude of the reported error is validated independently in
        # test_reported_half_life_error_matches_the_spread_over_repeated_fits below.
        _, result = fit
        half_life, lam = (tr.param for tr in result.param_transforms)
        dt = 1.0
        assert lam.est == pytest.approx(result.params[0].est / dt)
        assert lam.err == pytest.approx(result.params[0].err / dt)
        assert half_life.err == pytest.approx((LN2 / lam.est**2) * (result.params[0].err / dt))

    def test_reported_half_life_error_matches_the_spread_over_repeated_fits(self):
        # The one assertion here that validates the delta method's MAGNITUDE instead of
        # restating its formula: fit 60 independent paths and compare the empirical spread of
        # the half-life estimates against the mean reported error. The sample sd of 60 values
        # has a relative SE of 1/√(2·59) ≈ 9%, so rel = 0.3 is ~3.3 SE; measured over 15
        # independent blocks of 60 fits the ratio spanned 0.85-1.17.
        ests, errs = [], []
        for _ in range(60):
            _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=1.0, σ=self.σ, x0=0.0, npts=4000)
            _, result = ou_facade.compute_mean_half_life_estimate(x, dt=1.0)
            half_life = result.param_transforms[0].param
            ests.append(half_life.est)
            errs.append(half_life.err)
        assert numpy.std(ests, ddof=1) == pytest.approx(numpy.mean(errs), rel=0.3)

    def test_recovers_long_run_mean_from_intercept(self, fit):
        # Intercept is λμΔt and slope is -λΔt, so -intercept/slope recovers μ for any Δt.
        # This is the working route to μ; the labelled const transform is not (below).
        _, result = fit
        assert result.const.est == pytest.approx(self.λ * self.μ, rel=0.1)
        assert -result.const.est / result.params[0].est == pytest.approx(self.μ, rel=0.1)
        # R² has a closed form here, so pin the value rather than a threshold: for the Euler
        # AR(1) with φ = 1 - λΔt, regressing ΔX on X_{t-1} gives
        #   R² = (λΔt)²Var(X) / ((λΔt)²Var(X) + σ_ε²) = λΔt/2,
        # exactly, independent of σ (because Var(X) = σ_ε²/(1 - φ²)). λΔt = 0.8 → 0.40;
        # measured 0.381-0.421 over 80 seeds, so abs = 0.03 is comfortable and — unlike
        # "> 0.3", which admits 25% more residual variance than the model allows — it would
        # catch a mis-specified design matrix or a wrongly differenced series.
        assert result.r2 == pytest.approx(self.λ * 1.0 / 2.0, abs=0.03)

    def test_r2_is_half_lambda_dt_at_non_unit_dt(self):
        # Same λΔt/2 identity, now with λΔt = 0.4 → R² = 0.20, which exercises the identity's
        # Δt dependence. Measured 0.190-0.221 over 60 seeds.
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=0.5, σ=self.σ, x0=0.0, npts=4000)
        _, result = ou_facade.compute_mean_half_life_estimate(x, dt=0.5)
        assert result.r2 == pytest.approx(self.λ * 0.5 / 2.0, abs=0.03)

    def test_const_transform_labels_the_drift_constant_not_the_long_run_mean(self):
        # The const transform is labelled $\\mu$, which invites reading it as the OU long-run
        # mean. It is not. Under the transform's own model string (lib/data/impl/ou.py:483)
        #     ΔX_t = λ X_{t-1} Δt + μ Δt + √Δt ε_t
        # the labelled μ is the DRIFT CONSTANT, whose OLS value for this data is λ·μ_OU·Δt =
        # 0.8·5.0·0.5 = 2.0 — the long-run mean in that parameterisation is -μ/λ, which
        # test_recovers_long_run_mean_from_intercept checks as -const.est/params[0].est.
        # The file already accepts this same model-string convention for λ (see the sign
        # discussion in test_recovers_half_life_and_lambda), so accept it here too and pin
        # what the value actually is.
        Δt = 0.5
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=Δt, σ=self.σ, x0=0.0, npts=4000)
        _, result = ou_facade.compute_mean_half_life_estimate(x, dt=Δt)
        const = result.const_transform.param
        assert const.est_label == r"$\mu$"
        assert const.est == pytest.approx(self.λ * self.μ * Δt, rel=0.15)
        assert const.est != pytest.approx(self.μ, rel=0.1)  # emphatically not the long-run mean

    @pytest.mark.xfail(
        strict=True,
        reason="the $\\mu$ transform stores the raw OLS intercept (lib/data/impl/ou.py:505-513). "
               "Under the transform's own model string the intercept is μ Δt, so the "
               "transformed value should be intercept/dt — the λ transform divides by dt at "
               "line 484, the const at 505-513 does not. Measured at Δt = 0.5: the intercept "
               "1.9855 is stored unchanged where intercept/dt = 3.971 is what the model string "
               "implies. Δt = 1 is not parametrized here because the two agree there, so it "
               "carries no signal.",
    )
    def test_const_transform_divides_the_intercept_by_dt(self):
        Δt = 0.5
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=Δt, σ=self.σ, x0=0.0, npts=4000)
        _, result = ou_facade.compute_mean_half_life_estimate(x, dt=Δt)
        assert result.const_transform.param.est == pytest.approx(result.const.est / Δt)

    def test_every_reported_quantity_scales_correctly_with_dt(self):
        # dt is a unit conversion applied after the fit, so refitting the SAME series with dt
        # halved must double every rate and halve every time, errors included: λ ∝ 1/dt and
        # t½ ∝ dt. Comparing two fits of one series (rather than restating the transform's
        # formula) is what makes this discriminating: dropping the /dt from half_life.err, for
        # instance, would make it scale as dt² instead of dt.
        #
        # It also covers the two quantities nothing else checks at dt ≠ 1 — half_life.err's
        # scaling, and const_transform.param.err at any dt.
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=1.0, σ=self.σ, x0=0.0, npts=4000)
        _, one = ou_facade.compute_mean_half_life_estimate(x, dt=1.0)
        _, half = ou_facade.compute_mean_half_life_estimate(x, dt=0.5)
        hl_one, lam_one = (tr.param for tr in one.param_transforms)
        hl_half, lam_half = (tr.param for tr in half.param_transforms)
        assert lam_half.est == pytest.approx(2.0 * lam_one.est)
        assert lam_half.err == pytest.approx(2.0 * lam_one.err)
        assert hl_half.est == pytest.approx(hl_one.est / 2.0)
        assert hl_half.err == pytest.approx(hl_one.err / 2.0)
        # The const transform is the one reported quantity that does not move with dt: it is
        # the raw OLS intercept and error, carrying no /dt at all — see
        # test_const_transform_divides_the_intercept_by_dt.
        assert half.const_transform.param.est == pytest.approx(one.const_transform.param.est)
        assert half.const_transform.param.err == pytest.approx(one.const_transform.param.err)
        assert half.const_transform.param.err == pytest.approx(half.const.err)
        # …and the underlying regression itself is untouched by dt.
        assert half.params[0].est == pytest.approx(one.params[0].est)
        assert half.params[0].err == pytest.approx(one.params[0].err)
        assert half.r2 == pytest.approx(one.r2)

    def test_dt_is_applied_to_rate_and_half_life(self):
        # The slope is -λΔt whatever the noise scaling, so with dt passed the half-life
        # in time units must come back as ln2/λ. Δt = 0.5 → AR(1) coefficient 0.6,
        # SE(slope) ≈ √(0.64/4000) ≈ 0.013 on 0.4 → 3 SE ≈ 10%.
        _, x = ou_facade.create_source(μ=0.0, λ=self.λ, Δt=0.5, σ=self.σ, x0=0.0, npts=4000)
        _, result = ou_facade.compute_mean_half_life_estimate(x, dt=0.5)
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert result.params[0].est == pytest.approx(-self.λ * 0.5, rel=0.1)
        assert lam.est == pytest.approx(result.params[0].est / 0.5)
        assert lam.err == pytest.approx(result.params[0].err / 0.5)
        assert half_life.est == pytest.approx(ou.mean_halflife(self.λ), rel=0.1)

    def test_dt_is_applied_with_non_zero_mu(self):
        # Δt ≠ 1 together with μ ≠ 0 — the combination that exercises the const transform's
        # missing /dt, and the one every other Δt test avoids by setting μ = 0. The intercept
        # is λμΔt = 2.0 and the slope -λΔt = -0.4, so -intercept/slope = μ = 5.0 and the
        # half-life comes back in time units. AR(1) coefficient here is 1 - λΔt = 0.6, so
        # SE(slope) ≈ √((1 - 0.36)/4000) ≈ 0.0126 on 0.4 → rel = 0.15 is ~4.7 SE (rel = 0.1
        # is only 3.2 SE and tripped 1 seed in 40). The intercept and the μ ratio are far
        # better determined, so they keep rel = 0.1.
        _, x = ou_facade.create_source(μ=self.μ, λ=self.λ, Δt=0.5, σ=self.σ, x0=0.0, npts=4000)
        _, result = ou_facade.compute_mean_half_life_estimate(x, dt=0.5)
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert result.params[0].est == pytest.approx(-self.λ * 0.5, rel=0.15)
        assert result.const.est == pytest.approx(self.λ * self.μ * 0.5, rel=0.15)
        assert -result.const.est / result.params[0].est == pytest.approx(self.μ, rel=0.1)
        assert lam.est == pytest.approx(-self.λ, rel=0.15)
        assert half_life.est == pytest.approx(ou.mean_halflife(self.λ), rel=0.15)

    def test_dt_defaults_to_one(self):
        _, x = ou_facade.create_source(μ=0.0, λ=self.λ, Δt=1.0, σ=self.σ, x0=0.0, npts=500)
        _, with_default = ou_facade.compute_mean_half_life_estimate(x)
        _, explicit = ou_facade.compute_mean_half_life_estimate(x, dt=1.0)
        assert with_default.param_transforms[0].param.est == explicit.param_transforms[0].param.est

    def test_result_serialises_with_transforms(self, fit):
        _, result = fit
        payload = json.loads(result.to_json())
        assert payload["est_model"] == "OLS"
        assert payload["model"] == result.model
        assert [tr["param"]["est_label"] for tr in payload["param_transforms"]] == [r"$t_{1/2}$", r"$\lambda$"]
        assert payload["param_transforms"][0]["param"]["est"] == pytest.approx(result.param_transforms[0].param.est)
        assert payload["const_transform"]["param"]["est"] == pytest.approx(result.const.est)
        assert "OLSEst(" in repr(result)


class TestHalfLifeEstimateDegenerateInput:
    """``compute_mean_half_life_estimate`` validates nothing about its input.

    Characterisation of what it currently does for the inputs a notebook can plausibly hand
    it — a flat series, a series too short to fit, and a series with no mean reversion at
    all. None of these produce a diagnosable error today.
    """

    def test_constant_series_raises_an_index_error(self):
        # A constant series differences to all-zeros against a constant regressor, so
        # sm.add_constant sees an already-constant column and skips it. The design matrix ends
        # up with one column, OLSResult.params is empty, and __half_life_transform's
        # result.params[0] raises IndexError('list index out of range') — an internal leak,
        # not a domain error naming the degenerate input.
        with pytest.raises(IndexError):
            ou_facade.compute_mean_half_life_estimate(numpy.full(50, 3.0), dt=1.0)

    def test_two_point_series_raises(self):
        # 2 samples -> 1 difference and a one-row regressor, which is trivially constant: the
        # same collapse. Anything shorter than 3 points cannot be fitted.
        with pytest.raises(IndexError):
            ou_facade.compute_mean_half_life_estimate(numpy.array([1.0, 2.0]), dt=1.0)

    def test_three_points_is_the_shortest_accepted_input_and_is_degenerate(self):
        # 3 samples -> 2 observations for 2 parameters: an exact fit with zero residual
        # degrees of freedom. R² is 1 and every reported error is infinite, yet a finite
        # half-life comes back — nothing marks the result as meaningless.
        _, result = ou_facade.compute_mean_half_life_estimate(numpy.array([1.0, 2.0, 0.5]), dt=1.0)
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert result.r2 == pytest.approx(1.0)
        assert math.isinf(result.params[0].err)
        assert math.isinf(half_life.err) and math.isinf(lam.err)
        assert numpy.isfinite(half_life.est)

    def test_random_walk_returns_an_unusable_half_life_with_no_guard(self):
        # A pure random walk has λ = 0 exactly, and __half_life_transform (lib/data/impl/ou.py:486)
        # divides by λ with no guard, so the half-life is whatever the sampling noise in the
        # slope happens to make it — including a sign. Over 20 seeds |λ̂| ≤ 0.007, |t½| ≥ 100
        # and the reported relative error ≥ 0.27; the thresholds below are well inside that.
        rw = numpy.cumsum(numpy.random.normal(0.0, 1.0, 4000))
        _, result = ou_facade.compute_mean_half_life_estimate(rw, dt=1.0)
        half_life, lam = (tr.param for tr in result.param_transforms)
        assert abs(lam.est) < 0.05           # no mean reversion to find
        assert result.r2 < 0.02              # …and the regression knows it
        assert abs(half_life.est) > 20.0     # a half-life ~30x the sample's own λ scale
        assert half_life.err / abs(half_life.est) > 0.15  # the only usable signal: it is noise
