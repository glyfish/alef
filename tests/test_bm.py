"""
Tests for the Brownian-motion module group.

  lib.models.bm     -- pure simulators: noise, bm, bm_with_drift, bm_geometric
  lib.data.impl.bm  -- **kwargs façade returning (t, values) tuples (what the
                       notebooks call), plus closed-form mean / sd curves

Model-layer tests check closed forms (increment distribution, self-similarity
in Δt, Cov(B_s, B_t) = min(s, t), the Itô drift correction of GBM) and
round-trip parameter estimation. Façade tests check the (t, values) contract,
kwargs handling, that the façade reproduces the model layer under the same RNG
seed, and — following notebooks/random_processes/brownian_motion/bm_ensembles —
that ensembles built through the façade track the façade's closed forms.

All noise goes through numpy's global RNG, which tests/conftest.py reseeds
before every test. Tolerances are stated as multiples of the estimator's
standard error so the tests are not tuned to the seed.

Three navi behaviours are deliberately pinned as-is rather than asserted to be
correct, so that changing them fails loudly:

  * the two layers use DIFFERENT time grids — create_* sources return the step
    indices 0..npts-1 regardless of Δt, while compute_* returns the Δt-scaled
    grid Δt..npts·Δt (test_source_time_grid_is_step_indexed_and_ignores_dt vs
    test_closed_form_time_grid_is_dt_scaled_from_dt);
  * lib.stats.from_noise discards dB[0], so from_noise(to_noise(x)) is one
    sample short of x and offset by the first increment
    (TestFromNoiseIntegrationLive);
  * every **kwargs entry point silently swallows misspelled optional
    parameters — create_bm_source(npts=n, dt=…) quietly uses Δt=1.0
    (TestUnknownKwargsAreSwallowed).
"""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lib import stats
from lib.models import bm as model
from lib.data.impl import bm as facade
from lib.utils import create_ensemble, create_space

SEED = 12345


def _reseed() -> None:
    numpy.random.seed(SEED)


def _lag1_autocorr(x: numpy.ndarray) -> float:
    x = x - x.mean()
    return float(numpy.dot(x[:-1], x[1:]) / numpy.dot(x, x))


# ---------------------------------------------------------------------------
# lib.models.bm.noise
# ---------------------------------------------------------------------------

class TestNoise:
    def test_shape_and_dtype(self):
        z = model.noise(7)
        assert isinstance(z, numpy.ndarray)
        assert z.shape == (7,)
        assert z.dtype == numpy.float64
        assert model.noise(0).shape == (0,)

    def test_draws_reproducibly_from_the_global_rng(self):
        # Contract: noise draws from numpy's GLOBAL legacy RandomState, so
        # numpy.random.seed() (what tests/conftest.py uses) fully determines the
        # stream and reseeding mid-stream restarts it.
        _reseed()
        first = model.noise(50)
        _reseed()
        assert_array_equal(model.noise(50), first)
        _reseed()
        model.noise(13)                       # advance, then restart
        _reseed()
        assert_array_equal(model.noise(50), first)

    def test_is_not_a_private_generator_stream(self):
        # The distinguishing property of the global RandomState: it is NOT the
        # modern Generator bit stream. If noise() ever moved to default_rng(),
        # numpy.random.seed() would stop controlling it and every seeded test in
        # this suite would silently lose determinism.
        _reseed()
        z = model.noise(50)
        assert not numpy.array_equal(z, numpy.random.default_rng(SEED).normal(0.0, 1.0, 50))
        # Draws are shared with the global stream, not independent of it:
        # consuming from numpy.random first changes what noise() returns.
        _reseed()
        numpy.random.normal()
        assert not numpy.array_equal(model.noise(50), z)

    def test_zero_mean_unit_variance_uncorrelated(self):
        n = 20_000
        z = model.noise(n)
        # SE(mean) = 1/sqrt(n); SE(var) ~ sqrt(2/n); SE(r1) ~ 1/sqrt(n). 5 SE each.
        assert abs(z.mean()) < 5 / numpy.sqrt(n)
        assert abs(z.var(ddof=1) - 1.0) < 5 * numpy.sqrt(2 / n)
        assert abs(_lag1_autocorr(z)) < 5 / numpy.sqrt(n)


# ---------------------------------------------------------------------------
# lib.models.bm.bm
# ---------------------------------------------------------------------------

class TestBm:
    def test_starts_at_zero_with_requested_length(self):
        b = model.bm(25, 0.5)
        assert b.shape == (25,)
        assert b.dtype == numpy.float64
        assert b[0] == 0.0
        assert numpy.any(b[1:] != 0.0)

    def test_increments_are_iid_normal_with_variance_dt(self):
        n, dt = 4001, 0.25
        inc = numpy.diff(model.bm(n, dt))
        m = n - 1
        # increments ~ N(0, Δt): SE(mean) = sqrt(Δt/m), SE(var) = Δt sqrt(2/m),
        # SE(lag-1 acf) = 1/sqrt(m). 5 SE each.
        assert abs(inc.mean()) < 5 * numpy.sqrt(dt / m)
        assert abs(inc.var(ddof=1) - dt) < 5 * dt * numpy.sqrt(2 / m)
        assert abs(_lag1_autocorr(inc)) < 5 / numpy.sqrt(m)

    def test_self_similar_scaling_in_dt(self):
        # B with step Δt is sqrt(Δt) times the unit-step path built from the
        # same gaussian stream.
        _reseed()
        unit = model.bm(200, 1.0)
        _reseed()
        scaled = model.bm(200, 4.0)
        assert_allclose(scaled, 2.0 * unit, rtol=1e-12)

    def test_is_integral_of_unit_noise(self):
        # B_k = sqrt(Δt) * sum_{i<=k} z_i with B_0 = 0, z ~ noise().
        n, dt = 300, 0.3
        _reseed()
        path = model.bm(n, dt)
        _reseed()
        z = model.noise(n - 1)
        expected = numpy.concatenate([[0.0], numpy.sqrt(dt) * numpy.cumsum(z)])
        assert_allclose(path, expected, rtol=1e-12, atol=1e-14)

    def test_ensemble_covariance_is_min_s_t(self):
        nsim, n, dt = 2000, 41, 0.5
        paths = numpy.array([model.bm(n, dt) for _ in range(nsim)])
        i, j = 20, 40                 # s = 10, t = 20
        s, t = i * dt, j * dt
        bs, bt = paths[:, i], paths[:, j]

        # E[B_t] = 0: SE = sqrt(t/nsim).
        assert abs(bt.mean()) < 4 * numpy.sqrt(t / nsim)
        # Var[B_t] = t: relative SE sqrt(2/nsim).
        assert abs(bt.var(ddof=1) - t) < 4 * t * numpy.sqrt(2 / nsim)
        # Cov(B_s, B_t) = min(s, t) = s: SE = sqrt((s t + s^2)/nsim).
        cov = numpy.cov(bs, bt)[0, 1]
        assert abs(cov - s) < 4 * numpy.sqrt((s * t + s * s) / nsim)
        # Corr(B_s, B_t) = sqrt(s/t): SE ~ (1 - rho^2)/sqrt(nsim).
        rho = numpy.sqrt(s / t)
        corr = numpy.corrcoef(bs, bt)[0, 1]
        assert abs(corr - rho) < 4 * (1 - rho**2) / numpy.sqrt(nsim)


# ---------------------------------------------------------------------------
# lib.models.bm.bm_with_drift
# ---------------------------------------------------------------------------

class TestBmWithDrift:
    def test_starts_at_zero_with_requested_length(self):
        x = model.bm_with_drift(0.1, 0.5, 30, 0.5)
        assert x.shape == (30,)
        assert x[0] == 0.0

    def test_zero_volatility_is_deterministic_linear_drift(self):
        mu, dt, n = 0.7, 0.5, 11
        x = model.bm_with_drift(mu, 0.0, n, dt)
        assert_allclose(x, mu * dt * numpy.arange(n), rtol=1e-12, atol=1e-14)

    def test_reduces_to_bm_without_drift(self):
        n, dt = 150, 0.4
        _reseed()
        drift = model.bm_with_drift(0.0, 1.0, n, dt)
        _reseed()
        plain = model.bm(n, dt)
        assert_allclose(drift, plain, rtol=1e-12, atol=1e-14)

    def test_round_trip_estimates_mu_and_sigma_from_increments(self):
        mu, sigma, dt, n = 0.3, 0.8, 0.5, 4001
        inc = numpy.diff(model.bm_with_drift(mu, sigma, n, dt))
        m = n - 1
        mu_hat = inc.mean() / dt
        sigma_hat = inc.std(ddof=1) / numpy.sqrt(dt)
        # SE(mu_hat) = sigma / sqrt(Δt m); SE(sigma_hat) ~ sigma / sqrt(2 m). 5 SE.
        assert abs(mu_hat - mu) < 5 * sigma / numpy.sqrt(dt * m)
        assert abs(sigma_hat - sigma) < 5 * sigma / numpy.sqrt(2 * m)
        assert abs(_lag1_autocorr(inc)) < 5 / numpy.sqrt(m)

    def test_ensemble_mean_mu_t_and_sd_sigma_sqrt_t(self):
        nsim, n, dt, mu, sigma = 1000, 51, 1.0, 0.2, 1.5
        final = numpy.array([model.bm_with_drift(mu, sigma, n, dt)[-1] for _ in range(nsim)])
        t = (n - 1) * dt
        sd = sigma * numpy.sqrt(t)
        # SE(mean) = sd/sqrt(nsim); relative SE(sd) ~ 1/sqrt(2 nsim). 4 SE.
        assert abs(final.mean() - mu * t) < 4 * sd / numpy.sqrt(nsim)
        assert abs(final.std(ddof=1) - sd) < 4 * sd / numpy.sqrt(2 * nsim)


# ---------------------------------------------------------------------------
# lib.models.bm.bm_geometric
# ---------------------------------------------------------------------------

class TestBmGeometric:
    def test_starts_at_s0_and_stays_positive(self):
        s = model.bm_geometric(0.1, 0.5, 3.5, 40, 0.5)
        assert s.shape == (40,)
        assert s[0] == 3.5
        assert numpy.all(s > 0.0)

    def test_zero_volatility_is_exponential_growth(self):
        mu, s0, dt, n = 0.3, 2.0, 0.5, 11
        s = model.bm_geometric(mu, 0.0, s0, n, dt)
        assert_allclose(s, s0 * numpy.exp(mu * dt * numpy.arange(n)), rtol=1e-12)

    def test_log_returns_carry_ito_correction_and_round_trip(self):
        mu, sigma, s0, dt, n = 0.1, 0.5, 2.0, 0.5, 4001
        s = model.bm_geometric(mu, sigma, s0, n, dt)
        logret = numpy.diff(numpy.log(s))
        m = n - 1
        # log S has drift (mu - sigma^2/2): SE(mean logret) = sigma sqrt(Δt/m).
        se_mean = sigma * numpy.sqrt(dt / m)
        assert abs(logret.mean() - (mu - 0.5 * sigma**2) * dt) < 5 * se_mean
        sigma_hat = logret.std(ddof=1) / numpy.sqrt(dt)
        assert abs(sigma_hat - sigma) < 5 * sigma / numpy.sqrt(2 * m)
        # Round trip: mu_hat = mean/Δt + sigma_hat^2/2. The Itô correction
        # (sigma^2/2 = 0.125) is far larger than 5 SE (~0.056), so a missing
        # correction would fail here.
        mu_hat = logret.mean() / dt + 0.5 * sigma_hat**2
        assert abs(mu_hat - mu) < 5 * se_mean / dt

    def test_ensemble_moments_match_lognormal_closed_form(self):
        nsim, n, dt, mu, sigma, s0 = 1000, 21, 0.1, 0.1, 0.3, 2.0
        final = numpy.array([model.bm_geometric(mu, sigma, s0, n, dt)[-1] for _ in range(nsim)])
        t = (n - 1) * dt
        mean = s0 * numpy.exp(mu * t)
        sd = numpy.sqrt(s0**2 * numpy.exp(2 * mu * t) * (numpy.exp(sigma**2 * t) - 1))
        # SE(mean) = sd/sqrt(nsim) ~ 0.034 -> 4 SE ~ 5.6% of the mean.
        assert abs(final.mean() - mean) < 4 * sd / numpy.sqrt(nsim)
        # Lognormal excess kurtosis ~1.6 here, so SE(sd)/sd ~ 3%; allow 15%.
        assert abs(final.std(ddof=1) - sd) < 0.15 * sd


# ---------------------------------------------------------------------------
# Model layer: degenerate lengths and degenerate Δt
# ---------------------------------------------------------------------------

class TestModelDegenerateLengths:
    """Every simulator loops `for i in range(1, n)`, so n = 0 and n = 1 short
    circuit: the returned array is the untouched numpy.zeros(n) seed (S0 for
    GBM) and no gaussian is drawn."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_returns_seed_array_without_simulating(self, n):
        assert_array_equal(model.bm(n, 0.5), numpy.zeros(n))
        assert_array_equal(model.bm_with_drift(3.0, 2.0, n, 0.5), numpy.zeros(n))
        assert_array_equal(model.bm_geometric(3.0, 2.0, 7.0, n, 0.5), numpy.full(n, 7.0))

    @pytest.mark.parametrize("n", [0, 1])
    @pytest.mark.parametrize(
        "call",
        [
            lambda n: model.bm(n, 0.5),
            lambda n: model.bm_with_drift(3.0, 2.0, n, 0.5),
            lambda n: model.bm_geometric(3.0, 2.0, 7.0, n, 0.5),
        ],
        ids=["bm", "bm_with_drift", "bm_geometric"],
    )
    def test_consumes_no_rng_draws(self, call, n):
        # The loop body never runs, so the global stream is left untouched: the
        # next draw is the one that would have come first anyway.
        _reseed()
        call(n)
        after = numpy.random.normal()
        _reseed()
        assert after == numpy.random.normal()

    def test_noise_of_zero_length_is_empty(self):
        assert_array_equal(model.noise(0), numpy.zeros(0))


class TestModelDegenerateTimeStep:
    """Δt = 0 and Δt < 0 are accepted silently — neither simulator validates it.
    These tests state the (undocumented) resulting behaviour rather than assert
    an error, which is what the library actually does today."""

    def test_zero_dt_collapses_every_path_to_a_constant(self):
        # σ sqrt(0) = 0 and μ·0 = 0, so the recurrence becomes x_i = x_{i-1}.
        assert_array_equal(model.bm(6, 0.0), numpy.zeros(6))
        assert_array_equal(model.bm_with_drift(3.0, 2.0, 6, 0.0), numpy.zeros(6))
        assert_array_equal(model.bm_geometric(3.0, 2.0, 7.0, 6, 0.0), numpy.full(6, 7.0))

    def test_zero_dt_still_consumes_the_rng(self):
        # The draws happen and are then multiplied by zero: unlike n <= 1 this
        # does advance the global stream, once per step.
        _reseed()
        model.bm(6, 0.0)
        after = numpy.random.normal()
        _reseed()
        assert after == numpy.random.normal(size=6)[-1]

    def test_negative_dt_poisons_the_path_with_nan(self):
        # numpy.sqrt(Δt) is NaN for Δt < 0 and NaN propagates through the
        # recurrence; only the seed value survives.
        with numpy.errstate(invalid="ignore"):
            b = model.bm(5, -1.0)
            d = model.bm_with_drift(0.5, 2.0, 5, -1.0)
            g = model.bm_geometric(0.5, 2.0, 7.0, 5, -1.0)
        assert b[0] == 0.0 and numpy.all(numpy.isnan(b[1:]))
        assert d[0] == 0.0 and numpy.all(numpy.isnan(d[1:]))
        assert g[0] == 7.0 and numpy.all(numpy.isnan(g[1:]))

    def test_facade_sources_inherit_the_same_dt_behaviour(self):
        _, v = facade.create_bm_source(npts=6, Δt=0.0)
        assert_array_equal(v, numpy.zeros(6))
        with numpy.errstate(invalid="ignore"):
            _, w = facade.create_bm_geometric_source(npts=5, S0=3.0, Δt=-2.0)
        assert w[0] == 3.0 and numpy.all(numpy.isnan(w[1:]))


# ---------------------------------------------------------------------------
# Façade: closed-form curves
# ---------------------------------------------------------------------------

def _check_pair(result, npts):
    assert isinstance(result, tuple) and len(result) == 2
    t, v = result
    assert isinstance(t, numpy.ndarray) and isinstance(v, numpy.ndarray)
    assert t.shape == (npts,) and v.shape == (npts,)
    return t, v


class TestComputeMean:
    def test_defaults(self):
        t, v = _check_pair(facade.compute_mean(), 10)
        assert_allclose(t, numpy.arange(1, 11, dtype=float))
        assert_array_equal(v, numpy.zeros(10))

    def test_kwargs(self):
        t, v = _check_pair(facade.compute_mean(npts=5, Δt=0.5, μ=2.0), 5)
        assert_allclose(t, [0.5, 1.0, 1.5, 2.0, 2.5])
        assert_array_equal(v, numpy.full(5, 2.0))


class TestComputeSd:
    def test_defaults_are_sqrt_t(self):
        t, v = _check_pair(facade.compute_sd(), 10)
        assert_allclose(t, numpy.arange(1, 11, dtype=float))
        assert_allclose(v, numpy.sqrt(numpy.arange(1, 11)))

    def test_hand_values(self):
        # σ sqrt(t) with t = 0.25 * [1, 2, 3, 4] and σ = 2.
        t, v = _check_pair(facade.compute_sd(npts=4, Δt=0.25, σ=2.0), 4)
        assert_allclose(t, [0.25, 0.5, 0.75, 1.0])
        assert_allclose(v, [1.0, numpy.sqrt(2.0), numpy.sqrt(3.0), 2.0])


class TestComputeBmDriftMean:
    def test_requires_mu(self):
        with pytest.raises(Exception, match="μ parameter is required"):
            facade.compute_bm_drift_mean(npts=4)

    def test_hand_values(self):
        t, v = _check_pair(facade.compute_bm_drift_mean(npts=4, Δt=2.0, μ=0.5), 4)
        assert_allclose(t, [2.0, 4.0, 6.0, 8.0])
        assert_allclose(v, [1.0, 2.0, 3.0, 4.0])

    def test_default_dt_and_npts(self):
        t, v = _check_pair(facade.compute_bm_drift_mean(μ=-0.25), 10)
        assert_allclose(t, numpy.arange(1, 11, dtype=float))
        assert_allclose(v, -0.25 * numpy.arange(1, 11))


class TestComputeBmGeometricMean:
    def test_defaults_are_flat_at_one(self):
        t, v = _check_pair(facade.compute_bm_geometric_mean(), 10)
        assert_allclose(t, numpy.arange(1, 11, dtype=float))
        assert_allclose(v, numpy.ones(10))

    def test_hand_values(self):
        t, v = _check_pair(facade.compute_bm_geometric_mean(npts=3, μ=0.1, S0=3.0), 3)
        assert_allclose(t, [1.0, 2.0, 3.0])
        assert_allclose(v, 3.0 * numpy.exp([0.1, 0.2, 0.3]))

    def test_dt_rescales_time(self):
        t, v = _check_pair(facade.compute_bm_geometric_mean(npts=2, μ=1.0, Δt=0.5), 2)
        assert_allclose(t, [0.5, 1.0])
        assert_allclose(v, [numpy.exp(0.5), numpy.e])


class TestComputeBmGeometricSd:
    def test_hand_values_with_defaults(self):
        # S0 = 1, μ = 0, σ = 1: sd(t) = sqrt(e^t - 1).
        t, v = _check_pair(facade.compute_bm_geometric_sd(npts=3), 3)
        assert_allclose(t, [1.0, 2.0, 3.0])
        assert_allclose(v, [1.310832, 2.527658, 4.368700], rtol=1e-6)

    def test_hand_values_at_non_default_sigma(self):
        # sd(t) = sqrt(S0² e^{2μt} (e^{σ²t} - 1)) = S0 e^{μt} sqrt(e^{σ²t} - 1).
        # With npts=3, Δt=0.4 the grid is t = [0.4, 0.8, 1.2]; at σ=0.5, S0=4,
        # μ=0.2 that is 4 e^{0.2t} sqrt(e^{0.25t} - 1), evaluated by hand below.
        # This is the only exact check at a σ that is neither the default 1.0
        # nor the σ→0 asymptote, so it is what actually pins the formula.
        t, v = _check_pair(facade.compute_bm_geometric_sd(npts=3, σ=0.5, Δt=0.4,
                                                          S0=4.0, μ=0.2), 3)
        assert_allclose(t, [0.4, 0.8, 1.2])
        assert_allclose(v, [1.405242844061, 2.208709615546, 3.007717708311], rtol=1e-9)

    def test_scales_with_s0_and_drift(self):
        # Structural companion to the exact check above: sd(S0, μ) = S0 e^{μt} sd(1, 0).
        # Self-referential by construction (it compares the function to itself at
        # two parameter settings), so it can only detect a break in the
        # multiplicative structure — the closed form itself is pinned by
        # test_hand_values_with_defaults and test_hand_values_at_non_default_sigma.
        t, base = facade.compute_bm_geometric_sd(npts=5, σ=0.7, Δt=0.3)
        _, v = facade.compute_bm_geometric_sd(npts=5, σ=0.7, Δt=0.3, S0=4.0, μ=0.2)
        assert_allclose(v, 4.0 * numpy.exp(0.2 * t) * base, rtol=1e-12)

    def test_small_sigma_limit_is_bm_sd(self):
        # sqrt(e^{σ² t} - 1) -> σ sqrt(t) as σ -> 0 (relative error ~ σ² t / 4).
        t, v = facade.compute_bm_geometric_sd(npts=10, σ=0.01, μ=0.2, S0=5.0)
        assert_allclose(v, 5.0 * numpy.exp(0.2 * t) * 0.01 * numpy.sqrt(t), rtol=1e-3)

    def test_zero_sigma_gives_zero_sd(self):
        _, v = facade.compute_bm_geometric_sd(npts=4, σ=0.0, μ=0.5, S0=2.0)
        assert_array_equal(v, numpy.zeros(4))


def test_closed_forms_share_time_grid():
    grids = [
        facade.compute_mean(npts=6, Δt=0.2)[0],
        facade.compute_sd(npts=6, Δt=0.2)[0],
        facade.compute_bm_drift_mean(npts=6, Δt=0.2, μ=1.0)[0],
        facade.compute_bm_geometric_mean(npts=6, Δt=0.2)[0],
        facade.compute_bm_geometric_sd(npts=6, Δt=0.2)[0],
    ]
    for g in grids:
        assert_allclose(g, 0.2 * numpy.arange(1, 7))


# ---------------------------------------------------------------------------
# Façade: kwargs that are documented-but-dead, or silently swallowed
# ---------------------------------------------------------------------------

class TestInertKwargs:
    def test_compute_sd_ignores_its_documented_mu(self):
        # compute_sd's docstring documents a μ parameter, but the body never
        # reads it — σ sqrt(t) carries no drift term. Pin that the documented
        # kwarg is inert so the docstring/behaviour gap cannot silently change.
        t, v = _check_pair(facade.compute_sd(npts=4, Δt=0.25, σ=2.0, μ=1000.0), 4)
        assert_allclose(t, [0.25, 0.5, 0.75, 1.0])
        assert_allclose(v, [1.0, numpy.sqrt(2.0), numpy.sqrt(3.0), 2.0])

    def test_compute_mean_ignores_sigma(self):
        # The BM mean is μ everywhere, independent of the volatility.
        _, v = facade.compute_mean(npts=4, Δt=0.5, μ=2.0, σ=99.0)
        assert_array_equal(v, numpy.full(4, 2.0))


class TestUnknownKwargsAreSwallowed:
    """Every façade entry point takes **kwargs and looks parameters up by name,
    so a misspelling is not an error — the default silently applies. The
    notebooks pass these by hand, so this failure mode is worth pinning."""

    def test_ascii_dt_is_not_the_real_parameter(self):
        # The real parameter is the non-ASCII 'Δt'. 'dt' is dropped on the floor.
        _reseed()
        _, mistyped = facade.create_bm_source(npts=8, dt=0.25)
        _reseed()
        _, defaulted = facade.create_bm_source(npts=8)
        _reseed()
        _, intended = facade.create_bm_source(npts=8, Δt=0.25)
        assert_array_equal(mistyped, defaulted)          # silently Δt = 1.0
        assert not numpy.allclose(mistyped, intended)    # and NOT what was asked for

    def test_ascii_sigma_and_mu_are_not_the_real_parameters(self):
        _, mistyped = facade.compute_sd(npts=4, sigma=2.0)
        _, defaulted = facade.compute_sd(npts=4)
        assert_array_equal(mistyped, defaulted)
        _, mistyped_mean = facade.compute_bm_geometric_mean(npts=4, mu=5.0, S0=2.0)
        assert_array_equal(mistyped_mean, numpy.full(4, 2.0))   # μ stayed 0.0

    def test_required_kwargs_do_catch_the_misspelling(self):
        # Contrast: a get_param_throw_if_missing parameter has no default to
        # fall back to, so the same typo raises instead of passing silently.
        with pytest.raises(Exception, match="μ parameter is required"):
            facade.compute_bm_drift_mean(npts=4, mu=1.0)
        with pytest.raises(Exception, match="npts parameter is required"):
            facade.create_bm_source(NPTS=4)


# ---------------------------------------------------------------------------
# Façade: degenerate npts
# ---------------------------------------------------------------------------

_CLOSED_FORMS = [
    (facade.compute_mean, {}),
    (facade.compute_sd, {}),
    (facade.compute_bm_drift_mean, {"μ": 1.0}),
    (facade.compute_bm_geometric_mean, {}),
    (facade.compute_bm_geometric_sd, {}),
]

_ALL_ENTRY_POINTS = _CLOSED_FORMS + [(s, {}) for s in [
    facade.create_bm_noise_source,
    facade.create_bm_source,
    facade.create_bm_drift_source,
    facade.create_bm_geometric_source,
]]


class TestDegenerateNpts:
    @pytest.mark.parametrize("f, extra", _ALL_ENTRY_POINTS, ids=lambda a: getattr(a, "__name__", ""))
    def test_npts_zero_yields_two_empty_arrays(self, f, extra):
        # create_space degenerates to linspace(xmin, xmin - 1, 0) -> empty; no
        # entry point rejects npts=0, they all return a well-formed empty pair.
        t, v = _check_pair(f(npts=0, **extra), 0)
        assert t.dtype == numpy.float64 and v.dtype == numpy.float64

    def test_npts_one_grids_diverge_between_layers(self):
        # The two grid conventions collide hardest at npts=1: sources are
        # step-indexed from 0, closed forms are Δt-scaled from Δt.
        assert_array_equal(facade.create_bm_source(npts=1)[0], [0.0])
        assert_array_equal(facade.compute_mean(npts=1)[0], [1.0])
        assert_array_equal(facade.compute_sd(npts=1, Δt=0.5)[0], [0.5])
        assert_array_equal(facade.create_bm_source(npts=1, Δt=0.5)[0], [0.0])

    def test_npts_one_source_values_are_the_unsimulated_seed(self):
        assert_array_equal(facade.create_bm_source(npts=1)[1], [0.0])
        assert_array_equal(facade.create_bm_drift_source(npts=1, μ=9.0, σ=9.0)[1], [0.0])
        assert_array_equal(facade.create_bm_geometric_source(npts=1, S0=3.0)[1], [3.0])
        # The noise source is the exception: it draws npts values, not npts-1.
        _reseed()
        z = facade.create_bm_noise_source(npts=1)[1]
        _reseed()
        assert_array_equal(z, model.noise(1))

    def test_npts_one_closed_forms_evaluate_at_t_equals_dt(self):
        assert_array_equal(facade.compute_mean(npts=1, μ=2.0)[1], [2.0])
        assert_allclose(facade.compute_sd(npts=1, Δt=0.25, σ=2.0)[1], [1.0])
        assert_allclose(facade.compute_bm_drift_mean(npts=1, Δt=0.5, μ=3.0)[1], [1.5])
        assert_allclose(facade.compute_bm_geometric_mean(npts=1, μ=1.0, S0=2.0)[1],
                        [2.0 * numpy.e])
        assert_allclose(facade.compute_bm_geometric_sd(npts=1)[1],
                        [numpy.sqrt(numpy.e - 1.0)])


# ---------------------------------------------------------------------------
# Façade: compute_bm_from_noise
# ---------------------------------------------------------------------------

class TestComputeBmFromNoise:
    def test_requires_db(self):
        with pytest.raises(Exception, match="dB parameter is required"):
            facade.compute_bm_from_noise()

    @pytest.mark.xfail(
        strict=True,
        reason="compute_bm_from_noise calls verify_type(dB, NDArray[numpy.floating[Any]]); "
               "isinstance() rejects a parameterized generic with TypeError, so the function "
               "raises for every input",
    )
    def test_integrates_noise_with_zero_start(self):
        # stats.from_noise: B_0 = 0, B_k = B_{k-1} + dB_k (dB_0 is not used).
        # NOTE: this body is dead while the xfail stands, so the expectations
        # below are ALSO asserted live against the two pieces the function
        # composes — see TestFromNoiseIntegrationLive.
        dB = numpy.array([5.0, 1.0, 2.0, 3.0])
        t, v = _check_pair(facade.compute_bm_from_noise(dB=dB), 4)
        assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
        assert_allclose(v, [0.0, 1.0, 3.0, 6.0])


class TestFromNoiseIntegrationLive:
    """compute_bm_from_noise is unreachable (see the xfail above), so its two
    halves — create_space(xmax=npts-1, npts=npts) for the grid and
    stats.from_noise for the integration — are pinned here directly. That keeps
    the contract under live assertion, and documents a SECOND navi defect:
    from_noise loops from i=1, so dB[0] is discarded."""

    def test_grid_is_zero_to_npts_minus_one(self):
        npts = 4
        assert_allclose(create_space(xmax=npts - 1, npts=npts), [0.0, 1.0, 2.0, 3.0])
        assert_allclose(create_space(xmax=0, npts=1), [0.0])

    def test_from_noise_discards_the_first_increment(self):
        # B_0 = 0 and B_k = B_{k-1} + dB_k for k >= 1: the leading 5.0 never
        # enters the sum, which is the integration bug this pins.
        dB = numpy.array([5.0, 1.0, 2.0, 3.0])
        B = stats.from_noise(dB)
        assert B.shape == dB.shape
        assert_allclose(B, [0.0, 1.0, 3.0, 6.0])
        # A faithful integration would be the running total of EVERY increment.
        assert not numpy.allclose(B, numpy.cumsum(dB))
        assert_allclose(B, numpy.cumsum(dB) - dB[0])   # exactly dB[0] is missing

    def test_from_noise_is_insensitive_to_its_first_element(self):
        base = numpy.array([5.0, 1.0, 2.0, 3.0])
        poisoned = base.copy()
        poisoned[0] = -1234.5
        assert_array_equal(stats.from_noise(base), stats.from_noise(poisoned))

    def test_round_trip_loses_a_sample_and_the_first_increment(self):
        # from_noise(to_noise(x)) is one element SHORT of x (to_noise differences
        # away a sample) and is offset: it equals x[1:] re-based to zero, i.e.
        # x[k+1] - x[1], not x[k] - x[0].
        x = numpy.array([0.0, -1.41, -2.43, -2.457, -4.846, -5.828])
        y = stats.from_noise(stats.to_noise(x))
        assert y.shape == (len(x) - 1,)
        assert_allclose(y, x[1:] - x[1], atol=1e-12)
        assert not numpy.allclose(y, x[: len(x) - 1])

    def test_notebook_flow_noise_then_integrate_is_offset_from_create_bm_source(self):
        # The bm notebooks build a path two ways: create_bm_noise_source ->
        # compute_bm_from_noise, and create_bm_source directly. Under one seed
        # those should agree; they do not, because from_noise drops dB[0]. The
        # exact relationship: rec[k] = path[k+1] - z[0].
        npts = 12
        _reseed()
        _, z = facade.create_bm_noise_source(npts=npts)
        _reseed()
        _, path = facade.create_bm_source(npts=npts, Δt=1.0)
        # create_bm_source integrates the FIRST npts-1 draws, keeping all of them.
        assert_allclose(path, numpy.concatenate([[0.0], numpy.cumsum(z[: npts - 1])]),
                        rtol=1e-12, atol=1e-14)
        rec = stats.from_noise(z)
        assert not numpy.allclose(rec, path)
        assert_allclose(rec[: npts - 1], path[1:] - z[0], rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Façade: simulation sources
# ---------------------------------------------------------------------------

SOURCES = [
    facade.create_bm_noise_source,
    facade.create_bm_source,
    facade.create_bm_drift_source,
    facade.create_bm_geometric_source,
]


@pytest.mark.parametrize("source", SOURCES, ids=lambda f: f.__name__)
def test_source_requires_npts(source):
    with pytest.raises(Exception, match="npts parameter is required"):
        source(Δt=1.0)


@pytest.mark.parametrize("Δt", [0.25, 1.0, 4.0], ids=lambda d: f"dt={d}")
@pytest.mark.parametrize("source", SOURCES, ids=lambda f: f.__name__)
def test_source_time_grid_is_step_indexed_and_ignores_dt(source, Δt):
    # Sources build their grid with create_space(xmax=npts-1, npts=npts), i.e.
    # the STEP INDICES 0, 1, ..., npts-1 — deliberately NOT scaled by Δt, even
    # though the values they return advance by Δt per step. This is pinned at
    # Δt != 1 so that "fixing" it to Δt*arange(npts) fails loudly rather than
    # silently rescaling every notebook plot.
    # (create_bm_noise_source takes no Δt; **kwargs swallows it harmlessly.)
    npts = 17
    t, v = _check_pair(source(npts=npts, Δt=Δt), npts)
    assert t.dtype == numpy.float64 and v.dtype == numpy.float64
    assert_allclose(t, numpy.arange(npts, dtype=float))
    assert t[0] == 0.0 and t[-1] == npts - 1


@pytest.mark.parametrize("Δt", [0.25, 1.0, 4.0], ids=lambda d: f"dt={d}")
@pytest.mark.parametrize("f, extra", _CLOSED_FORMS, ids=lambda a: getattr(a, "__name__", ""))
def test_closed_form_time_grid_is_dt_scaled_from_dt(f, extra, Δt):
    # The counterpart to the source grid above, and the design inconsistency in
    # one place: compute_* uses Δt*create_space(xmin=1, npts=npts), so it is
    # Δt-scaled and starts at Δt (not 0). The two layers therefore do NOT share
    # an x axis unless Δt == 1, and even then they are offset by one step.
    npts = 17
    t, _ = _check_pair(f(npts=npts, Δt=Δt, **extra), npts)
    assert_allclose(t, Δt * numpy.arange(1, npts + 1, dtype=float))
    assert t[0] == pytest.approx(Δt)


class TestCreateBmNoiseSource:
    def test_matches_model_layer(self):
        _reseed()
        _, v = facade.create_bm_noise_source(npts=64)
        _reseed()
        assert_array_equal(v, model.noise(64))

    def test_is_standard_normal(self):
        n = 20_000
        _, z = facade.create_bm_noise_source(npts=n)
        assert abs(z.mean()) < 5 / numpy.sqrt(n)
        assert abs(z.var(ddof=1) - 1.0) < 5 * numpy.sqrt(2 / n)


class TestCreateBmSource:
    def test_matches_model_layer_with_dt(self):
        _reseed()
        _, v = facade.create_bm_source(npts=80, Δt=0.3)
        _reseed()
        assert_array_equal(v, model.bm(80, 0.3))
        assert v[0] == 0.0

    def test_default_dt_is_one(self):
        _reseed()
        _, implicit = facade.create_bm_source(npts=80)
        _reseed()
        _, explicit = facade.create_bm_source(npts=80, Δt=1.0)
        _reseed()
        _, quadrupled = facade.create_bm_source(npts=80, Δt=4.0)
        assert_array_equal(implicit, explicit)
        assert_allclose(quadrupled, 2.0 * implicit, rtol=1e-12)


class TestCreateBmDriftSource:
    def test_matches_model_layer(self):
        _reseed()
        _, v = facade.create_bm_drift_source(npts=80, μ=0.3, σ=0.8, Δt=0.5)
        _reseed()
        assert_array_equal(v, model.bm_with_drift(0.3, 0.8, 80, 0.5))

    def test_defaults_reduce_to_plain_bm(self):
        # μ = 0, σ = 1, Δt = 1 by default.
        _reseed()
        _, v = facade.create_bm_drift_source(npts=80)
        _reseed()
        assert_allclose(v, model.bm(80, 1.0), rtol=1e-12, atol=1e-14)

    def test_zero_sigma_is_linear_drift(self):
        _, v = facade.create_bm_drift_source(npts=6, μ=2.0, σ=0.0, Δt=0.5)
        assert_allclose(v, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


class TestCreateBmGeometricSource:
    def test_matches_model_layer(self):
        _reseed()
        _, v = facade.create_bm_geometric_source(npts=80, μ=0.1, σ=0.3, S0=2.0, Δt=0.5)
        _reseed()
        assert_array_equal(v, model.bm_geometric(0.1, 0.3, 2.0, 80, 0.5))
        assert v[0] == 2.0

    def test_defaults(self):
        # μ = 0, σ = 1, S0 = 1, Δt = 1 by default.
        _reseed()
        _, v = facade.create_bm_geometric_source(npts=80)
        _reseed()
        assert_array_equal(v, model.bm_geometric(0.0, 1.0, 1.0, 80, 1.0))
        assert v[0] == 1.0

    def test_zero_sigma_is_exponential(self):
        _, v = facade.create_bm_geometric_source(npts=4, μ=0.5, σ=0.0, S0=3.0, Δt=2.0)
        assert_allclose(v, 3.0 * numpy.exp([0.0, 1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Cross-layer: ensembles built through the façade track the façade closed forms
# (the bm_ensembles notebook overlays compute_*(Δt=npts/10, npts=10) on a
#  step-indexed ensemble; the same alignment is used here).
# ---------------------------------------------------------------------------

class TestEnsemblesTrackClosedForms:
    def test_bm_ensemble_sd_tracks_compute_sd(self):
        nsim, npts = 1000, 41
        t, ensemble = create_ensemble(facade.create_bm_source, nsim, npts=npts, Δt=1.0)
        assert t.shape == (npts,) and len(ensemble) == nsim
        paths = numpy.array(ensemble)
        idx = numpy.array([10, 20, 30, 40])
        sd_t, sd = facade.compute_sd(npts=4, Δt=10.0)
        assert_allclose(sd_t, idx)
        # relative SE(sd) ~ 1/sqrt(2 nsim) = 2.2%; 4 SE < 10%.
        assert_allclose(paths[:, idx].std(axis=0, ddof=1), sd, rtol=0.10)
        # mean is zero: SE = sqrt(t/nsim) <= 0.2 at t = 40.
        assert numpy.all(numpy.abs(paths[:, idx].mean(axis=0)) < 4 * numpy.sqrt(idx / nsim))

    def test_drift_ensemble_tracks_drift_mean_and_sd(self):
        nsim, npts, mu, sigma = 1000, 41, 0.25, 0.5
        _, ensemble = create_ensemble(facade.create_bm_drift_source, nsim,
                                      npts=npts, μ=mu, σ=sigma, Δt=1.0)
        paths = numpy.array(ensemble)
        idx = numpy.array([10, 20, 30, 40])
        mean_t, mean = facade.compute_bm_drift_mean(npts=4, Δt=10.0, μ=mu)
        _, sd = facade.compute_sd(npts=4, Δt=10.0, σ=sigma)
        assert_allclose(mean_t, idx)
        assert_allclose(mean, [2.5, 5.0, 7.5, 10.0])
        # SE(mean) = σ sqrt(t)/sqrt(nsim) <= 0.1 -> 4 SE.
        assert numpy.all(numpy.abs(paths[:, idx].mean(axis=0) - mean)
                         < 4 * sigma * numpy.sqrt(idx / nsim))
        assert_allclose(paths[:, idx].std(axis=0, ddof=1), sd, rtol=0.10)

    def test_geometric_ensemble_tracks_lognormal_mean_and_sd(self):
        nsim, npts, mu, sigma, s0, dt = 1000, 21, 0.1, 0.3, 2.0, 0.1
        _, ensemble = create_ensemble(facade.create_bm_geometric_source, nsim,
                                      npts=npts, μ=mu, σ=sigma, S0=s0, Δt=dt)
        paths = numpy.array(ensemble)
        idx = numpy.array([5, 10, 15, 20])
        mean_t, mean = facade.compute_bm_geometric_mean(npts=4, Δt=0.5, μ=mu, S0=s0)
        _, sd = facade.compute_bm_geometric_sd(npts=4, Δt=0.5, μ=mu, σ=sigma, S0=s0)
        assert_allclose(mean_t, idx * dt)
        # SE(mean) = sd/sqrt(nsim); 4 SE.
        assert numpy.all(numpy.abs(paths[:, idx].mean(axis=0) - mean) < 4 * sd / numpy.sqrt(nsim))
        # sd: lognormal kurtosis inflates SE(sd)/sd to ~3%; allow 15%.
        assert_allclose(paths[:, idx].std(axis=0, ddof=1), sd, rtol=0.15)
