"""
Tests for navi's ARIMA family: the model layer ``lib.models.arima`` and its
kwargs facade ``lib.data.impl.arima`` (the layer notebooks call, returning
``(t, values)`` tuples).

Three kinds of test, following the architecture doc:

1. Closed forms   -- pure functions checked against hand-derived values.
2. Round trips    -- simulate with known parameters, estimate them back.
   Every tolerance is sized to ~4-5 standard errors of the relevant sampling
   distribution (stated inline) so the test holds for most seeds, not just
   the conftest seed.
3. Contracts      -- tuple/shape/dtype structure, required kwargs, type
   guards, enum tags, serialization.

Every simulator draws from numpy's global RNG; conftest reseeds before each
test. Tests that need to compare two draws of the *same* stream reseed
explicitly with ``SEED``.
"""
import json
import uuid

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels.tsa.arima.model import ARIMAResults, ARIMAResultsWrapper

from lib.data.impl import arima as facade
from lib.data.param_est import ARMAEst, ARMAEstType, ARMAParamType, ParamEst
from lib.models import arima as model
from lib.utils import create_ensemble

SEED = 7_261_993


def reseed():
    numpy.random.seed(SEED)


def sample_acf(x, nlags):
    """Biased sample autocorrelation, lags 0..nlags, computed directly in numpy."""
    x = numpy.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    c0 = x @ x / n
    return numpy.array([(x[: n - k] @ x[k:]) / n / c0 for k in range(nlags + 1)])


MAQ_COV_BUG = pytest.mark.xfail(
    strict=True,
    reason="maq_cov adds the lag-n cross-product sum sum_i θ_i θ_{i+n} to the lag-(n+1) "
           "direct term θ_{n+1}, so for q>=2 every lag 1..q is mis-indexed "
           "(θ=[0.5,0.3] gives [0.5,0.45]·σ² instead of the true γ=[0.65,0.3]·σ²)",
)


# ---------------------------------------------------------------------------
# Model layer: closed forms
# ---------------------------------------------------------------------------

class TestMAqClosedForms:
    def test_maq_sigma(self):
        # Var MA(q) = σ²(1 + Σθ²);  θ=[0.6, 0.8] → 1 + 0.36 + 0.64 = 2
        assert model.maq_sigma([0.6, 0.8], 3.0) == pytest.approx(3.0 * numpy.sqrt(2.0))
        assert model.maq_sigma([0.6]) == pytest.approx(numpy.sqrt(1.36))   # σ defaults to 1
        assert model.maq_sigma([]) == pytest.approx(1.0)                   # q=0 is white noise

    # γ(h) = σ² Σ_{j=0}^{q-h} θ_j θ_{j+h} with θ_0 = 1, hand-computed for lags 1..q.
    @pytest.mark.parametrize("θ, σ, expected", [
        ([0.6], 2.0, [2.4]),
        pytest.param([0.5, 0.3], 2.0, [2.6, 1.2], marks=MAQ_COV_BUG),
        pytest.param([0.4, 0.3, 0.2], 1.0, [0.58, 0.38, 0.2], marks=MAQ_COV_BUG),
    ])
    def test_maq_cov(self, θ, σ, expected):
        cov = model.maq_cov(θ, σ)
        assert cov.shape == (len(θ),)
        assert_allclose(cov, expected, rtol=1e-12)

    def test_maq_acf_ma1(self):
        # ρ(1) = θ/(1+θ²), ρ(k>1) = 0; padded to max_lag+1 entries with ρ(0)=1.
        acf = model.maq_acf([0.6], σ=1.0, max_lag=4)
        assert acf.shape == (5,)
        assert_allclose(acf, [1.0, 0.6 / 1.36, 0.0, 0.0, 0.0], rtol=1e-12)

    def test_sigma_scaling_of_cov_and_sd(self):
        # NOT a σ-invariance check on maq_acf: that ratio is maq_cov/maq_sigma² =
        # σ²(c+s)/(σ²(v+1)), so σ cancels *algebraically* and comparing two σ can
        # never fail for any implementation of the bracketed terms. Assert the σ
        # dependence where it is real instead — γ ∝ σ², sd ∝ σ — plus the defaults.
        θ = [0.6, -0.2]
        assert_allclose(model.maq_cov(θ, 2.0), 4.0 * model.maq_cov(θ, 1.0), rtol=1e-12)
        assert model.maq_sigma(θ, 2.0) == pytest.approx(2.0 * model.maq_sigma(θ, 1.0))
        assert_allclose(model.maq_cov([0.6]), [0.6], rtol=1e-12)   # σ defaults to 1
        assert model.maq_acf([0.6]).shape == (16,)                 # default max_lag=15

    def test_maq_acf_overruns_when_q_exceeds_max_lag(self):
        # maq_acf writes ac_eq[i+1] for every θ_i into an array of max_lag+1 entries,
        # so q > max_lag runs off the end. Pinning the current unguarded behaviour.
        with pytest.raises(IndexError):
            model.maq_acf([0.4, 0.3, 0.2], 1.0, 2)

    @pytest.mark.parametrize("θ, expected", [
        pytest.param([0.5, 0.3], [1.0, 0.65 / 1.34, 0.3 / 1.34, 0.0], marks=MAQ_COV_BUG),
    ])
    def test_maq_acf_ma2(self, θ, expected):
        assert_allclose(model.maq_acf(θ, max_lag=3), expected, rtol=1e-12)


class TestAR1ClosedForms:
    def test_ar1_sigma(self):
        # sqrt(σ²/(1-φ²)):  φ=0.6, σ=2 → sqrt(4/0.64) = 2.5
        assert model.ar1_sigma(0.6, 2.0) == pytest.approx(2.5)
        assert model.ar1_sigma(0.0) == pytest.approx(1.0)

    def test_ar1_acf(self):
        acf = model.ar1_acf(0.5, 4)
        assert isinstance(acf, list) and len(acf) == 4
        assert acf == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_ar1_offset_mean(self):
        # μ/(1-φ): φ=0.6, μ=2 → 5
        assert model.ar1_offset_mean(0.6, 2.0) == pytest.approx(5.0)
        assert model.ar1_offset_mean(0.0, 1.5) == pytest.approx(1.5)

    def test_ar1_offset_sigma(self):
        # σ/sqrt(1-φ²): φ=0.6, σ=2 → 2/0.8 = 2.5
        assert model.ar1_offset_sigma(0.6, 2.0) == pytest.approx(2.5)

    @pytest.mark.parametrize("φ", [0.0, 0.3, -0.7, 0.95])
    def test_offset_sigma_equals_ar1_sigma(self, φ):
        # The offset only shifts the mean; the stationary sd is that of AR(1).
        assert model.ar1_offset_sigma(φ, 1.7) == pytest.approx(model.ar1_sigma(φ, 1.7))


# ---------------------------------------------------------------------------
# Model layer: simulators written in navi (explicit recursions)
# ---------------------------------------------------------------------------

class TestNoise:
    def test_noise_is_standard_normal(self):
        n = 20_000
        ε = model.noise(n)
        assert ε.shape == (n,)
        # SE(mean) = 1/sqrt(n) = 0.0071 → 0.04 is ~5.6σ; SE(var) = sqrt(2/n) = 0.014 → 0.07 is ~5σ.
        assert ε.mean() == pytest.approx(0.0, abs=0.04)
        assert ε.var() == pytest.approx(1.0, abs=0.07)


class TestAR:
    def test_deterministic_recursion(self):
        # σ=0: x_t = 0.5 x_{t-1} + 0.3 x_{t-2} from x0=[1, 2]
        x = model.ar([0.5, 0.3], [1.0, 2.0], 5, 0.0)
        assert_allclose(x, [1.0, 2.0, 1.3, 1.25, 1.015], rtol=1e-12)

    def test_initial_values_and_noise_recovery(self):
        φ, x0, n, σ = [0.5, -0.3], [1.0, 2.0], 200, 1.5
        reseed()
        ε = σ * numpy.random.normal(0.0, 1.0, n)
        reseed()
        x = model.ar(φ, x0, n, σ)
        assert x.shape == (n,)
        assert_array_equal(x[:2], x0)
        # x_t - φ1 x_{t-1} - φ2 x_{t-2} must be exactly the scaled noise draw.
        resid = x[2:] - φ[0] * x[1:-1] - φ[1] * x[:-2]
        assert_allclose(resid, ε[2:], atol=1e-12)

    def test_round_trip_yule_walker(self):
        φ, n = 0.7, 4000
        x = model.ar([φ], [0.0], n, 1.0)
        # SE(φ̂) ≈ sqrt((1-φ²)/n) = 0.0113 → 0.05 is ~4.4σ
        assert model.yw(x, 1)[0] == pytest.approx(φ, abs=0.05)

    def test_too_few_initial_values_raise(self):
        # ar() copies x0[i] for i in range(p) with no length guard, so an x0 shorter
        # than p is an IndexError rather than a diagnosed argument error.
        with pytest.raises(IndexError):
            model.ar([0.5, 0.3], [1.0], 5, 0.0)


class TestOU:
    def test_deterministic_relaxation_to_mean(self):
        # σ=0, x_0=0: x_t = μ (1 - (1-λ)^t)
        λ, μ, n = 0.3, 5.0, 50
        t = numpy.arange(n)
        assert_allclose(model.ou(λ, μ, n, 0.0), μ * (1.0 - (1.0 - λ) ** t), rtol=1e-12)

    def test_stationary_mean_and_sd(self):
        λ, μ, σ, n = 0.5, 2.0, 1.0, 4000
        x = model.ou(λ, μ, n, σ)
        φ = 1.0 - λ
        sd = σ / numpy.sqrt(1.0 - φ**2)                       # 1.1547
        # SE(mean) = sd·sqrt((1+φ)/((1-φ)n)) = 0.032 → 0.15 is ~4.7σ
        assert x.mean() == pytest.approx(μ, abs=0.15)
        # SE(sd) ≈ sd·sqrt((1+φ²)/((1-φ²)·2n)) = 0.017 → 0.08 is ~4.8σ
        assert x.std() == pytest.approx(sd, abs=0.08)


class TestARpOffset:
    def test_deterministic_closed_form(self):
        # σ=0, x_0=0, AR(1): x_t = μ (1-φ^t)/(1-φ)
        φ, μ, n = 0.5, 2.0, 40
        t = numpy.arange(n)
        assert_allclose(model.arp_offset([φ], μ, n, 0.0), μ * (1 - φ**t) / (1 - φ), rtol=1e-12)

    def test_round_trip_offset_fit(self):
        φ, μ, n = 0.5, 2.0, 4000
        x = model.arp_offset([φ], μ, n, 1.0)
        result = model.ar_offset_fit(x, 1)
        # statsmodels' ARIMA 'const' is the process mean μ/(1-φ) = 4 (regression with
        # ARMA errors), not the intercept. bse(const) ≈ 0.031 → 0.15 is ~5σ;
        # bse(φ) ≈ 0.014 → 0.06 is ~4σ.
        assert result.params[0] == pytest.approx(model.ar1_offset_mean(φ, μ), abs=0.15)
        assert result.params[1] == pytest.approx(φ, abs=0.06)


class TestARpDrift:
    def test_pure_drift_when_phi_is_zero(self):
        # p=1, φ=0, σ=0: x_0 = 0 (initial slot), x_t = γ t + μ for t ≥ 1
        μ, γ, n = 1.0, 0.5, 10
        x = model.arp_drift([0.0], μ, γ, n, 0.0)
        assert x[0] == 0.0
        assert_allclose(x[1:], γ * numpy.arange(1, n) + μ, rtol=1e-12)

    def test_linear_asymptote(self):
        # Steady state of x_t = φ x_{t-1} + γ t + μ is a t + b with
        # a = γ/(1-φ), b = (μ - φ a)/(1-φ); transient decays like φ^t.
        φ, μ, γ, n = 0.5, 1.0, 0.25, 100
        x = model.arp_drift([φ], μ, γ, n, 0.0)
        a = γ / (1 - φ)
        b = (μ - φ * a) / (1 - φ)
        assert x[-1] == pytest.approx(a * (n - 1) + b, rel=1e-9)
        assert x[-2] == pytest.approx(a * (n - 2) + b, rel=1e-9)

    def test_noise_recovery(self):
        φ, μ, γ, n, σ = 0.6, 1.0, 0.1, 300, 2.0
        reseed()
        ε = σ * numpy.random.normal(0.0, 1.0, n)
        reseed()
        x = model.arp_drift([φ], μ, γ, n, σ)
        t = numpy.arange(1, n)
        resid = x[1:] - φ * x[:-1] - γ * t - μ
        assert x[0] == 0.0
        assert_allclose(resid, ε[1:], atol=1e-12)

    def test_p2_warm_up_leaves_the_first_p_samples_at_zero(self):
        # arp_drift takes no initial-value argument and its loop is `range(p, n)`,
        # so the first p slots stay at the numpy.zeros default. For p=2 that is two
        # hard zeros before the recursion starts, hand-computed here:
        #   x2 = μ = 1, x3 = 1 + 0.5·1 = 1.5, x4 = 1 + 0.5·1.5 + 0.2·1 = 1.95,
        #   x5 = 1 + 0.5·1.95 + 0.2·1.5 = 2.275   (γ = 0, σ = 0)
        x = model.arp_drift([0.5, 0.2], 1.0, 0.0, 6, 0.0)
        assert_allclose(x, [0.0, 0.0, 1.0, 1.5, 1.95, 2.275], rtol=1e-12)
        _, x_facade = facade.create_ar_drift_source(φ=[0.5, 0.2], μ=1.0, γ=0.0, npts=6, σ=0.0)
        assert_array_equal(x_facade, x)

    def test_p2_drift_term_is_indexed_by_absolute_time(self):
        # γ multiplies the absolute index i, not i - p. φ=[0.4,-0.2], μ=0.5, γ=0.25:
        #   x2 = 0.25·2 + 0.5                            = 1.0
        #   x3 = 0.25·3 + 0.5 + 0.4·1.0                  = 1.65
        #   x4 = 0.25·4 + 0.5 + 0.4·1.65 - 0.2·1.0       = 1.96
        #   x5 = 0.25·5 + 0.5 + 0.4·1.96 - 0.2·1.65      = 2.204
        x = model.arp_drift([0.4, -0.2], 0.5, 0.25, 6, 0.0)
        assert_allclose(x, [0.0, 0.0, 1.0, 1.65, 1.96, 2.204], rtol=1e-12)

    def test_p2_noise_recovery(self):
        φ, μ, γ, n, σ = [0.4, -0.2], 0.5, 0.1, 300, 2.0
        reseed()
        ε = σ * numpy.random.normal(0.0, 1.0, n)
        reseed()
        x = model.arp_drift(φ, μ, γ, n, σ)
        t = numpy.arange(2, n)
        resid = x[2:] - φ[0] * x[1:-1] - φ[1] * x[:-2] - γ * t - μ
        assert_array_equal(x[:2], [0.0, 0.0])
        assert_allclose(resid, ε[2:], atol=1e-12)


# ---------------------------------------------------------------------------
# Model layer: statsmodels-backed simulators
# ---------------------------------------------------------------------------

class TestARpStatsmodels:
    def test_ar1_variance_and_coefficient(self):
        φ, σ, n = 0.6, 1.5, 6000
        x = model.ar1(φ, n, σ)
        assert x.shape == (n,)
        # var(γ̂0) ≈ (2γ0²/n)(1+φ²)/(1-φ²) → SE 0.094 on γ0 = 3.516; 12% (0.42) is ~4.5σ
        assert x.var() == pytest.approx(model.ar1_sigma(φ, σ) ** 2, rel=0.12)
        # SE(φ̂) = sqrt((1-φ²)/n) = 0.0103 → 0.05 is ~4.8σ
        assert model.yw(x, 1)[0] == pytest.approx(φ, abs=0.05)

    def test_ar1_acf_matches_phi_powers(self):
        φ, n = 0.6, 6000
        acf = sample_acf(model.ar1(φ, n), 3)
        # SE(r_k) ≲ sqrt((1-φ²)/n)·k^½-ish ≈ 0.01-0.02 → 0.06 is ≥3σ at every lag
        assert_allclose(acf, model.ar1_acf(φ, 4), atol=0.06)

    def test_arp_sign_convention_via_residuals(self):
        # φ_sim = [1, -φ] must mean x_t = φ1 x_{t-1} + φ2 x_{t-2} + ε_t with ε ~ N(0, σ²).
        φ, σ, n = numpy.array([0.5, 0.3]), 1.5, 5000
        x = model.arp(φ, n, σ)
        resid = x[2:] - φ[0] * x[1:-1] - φ[1] * x[:-2]
        # SE(var) = σ²·sqrt(2/n) = 0.045 → 8% (0.18) is ~4σ
        assert resid.var() == pytest.approx(σ**2, rel=0.08)
        # White residuals: SE(r_1) = 1/sqrt(n) = 0.014 → 0.06 is ~4.2σ
        assert abs(sample_acf(resid, 1)[1]) < 0.06

    def test_arp_round_trip_yule_walker(self):
        φ, n = numpy.array([0.5, 0.3]), 4000
        x = model.arp(φ, n)
        # SE(φ̂_i) ≈ sqrt((1-φ2²)/n) = 0.015 → 0.07 is ~4.6σ
        assert_allclose(model.yw(x, 2), φ, atol=0.07)


class TestMAqStatsmodels:
    def test_variance_and_acf(self):
        θ, σ, n = 0.6, 1.5, 5000
        x = model.maq(numpy.array([θ]), n, σ)
        assert x.shape == (n,)
        # var(γ̂0) ≈ (2/n)(γ0² + 2γ1²) → SE 0.072 on γ0 = 3.06; 12% (0.37) is ~5σ
        assert x.var() == pytest.approx(model.maq_sigma([θ], σ) ** 2, rel=0.12)
        acf = sample_acf(x, 2)
        # θ_sim = [1, θ] → x_t = ε_t + θ ε_{t-1}, so ρ1 = θ/(1+θ²) = 0.441 (positive).
        # Bartlett: SE(r_1) = sqrt((1-3ρ²+4ρ⁴)/n) = 0.011 → 0.05 is ~4.7σ;
        # SE(r_2) = sqrt((1+2ρ²)/n) = 0.017 → 0.07 is ~4.2σ.
        assert acf[1] == pytest.approx(θ / (1 + θ**2), abs=0.05)
        assert acf[2] == pytest.approx(0.0, abs=0.07)

    # --- q=2: empirical anchor for the hand-derived γ the MAQ_COV xfails assert ---
    # θ=[0.5,0.3] → γ0 = 1+0.25+0.09 = 1.34, γ1 = θ1+θ1θ2 = 0.65, γ2 = θ2 = 0.3,
    # so ρ = [0.48507, 0.22388]. Monte Carlo over 300 independent seeds at n=20000
    # gives sd(r_1) = 0.0058 and sd(r_2) = 0.0079, so the tolerances below are ~5σ.
    MA2_θ = [0.5, 0.3]
    MA2_N = 20_000
    MA2_ρ = [0.65 / 1.34, 0.3 / 1.34]

    def ma2_sample_acf(self):
        _, x = facade.create_ma_source(θ=self.MA2_θ, npts=self.MA2_N)
        assert x.shape == (self.MA2_N,)
        return sample_acf(x, 2)

    def test_ma2_sample_acf_matches_hand_derived_autocovariance(self):
        # Pins the q=2 simulator (model.maq via create_ma_source) numerically and
        # anchors the MAQ_COV_BUG expectations to data rather than algebra alone.
        r = self.ma2_sample_acf()
        assert r[1] == pytest.approx(self.MA2_ρ[0], abs=0.03)
        assert r[2] == pytest.approx(self.MA2_ρ[1], abs=0.04)

    @MAQ_COV_BUG
    def test_ma2_navi_acf_matches_the_sample_acf(self):
        # The same comparison from the library's side: navi's maq_acf returns
        # [0.3731, 0.3358] where the simulated process shows [0.485, 0.224].
        r = self.ma2_sample_acf()
        assert_allclose(model.maq_acf(self.MA2_θ, max_lag=2)[1:], r[1:], atol=0.04)

    def test_round_trip_ma_fit(self):
        θ, σ, n = 0.6, 1.0, 4000
        result = model.ma_fit(model.maq(numpy.array([θ]), n, σ), 1)
        # SE(θ̂) ≈ sqrt((1-θ²)/n) = 0.0126 → 0.05 is ~4σ;  SE(σ̂²) = σ²sqrt(2/n) = 0.022 → 0.1 is ~4.5σ
        assert result.params[1] == pytest.approx(θ, abs=0.05)
        assert result.params[2] == pytest.approx(σ**2, abs=0.1)


class TestARMA:
    def test_arma11_closed_form_acf_and_variance(self):
        φ, θ, σ, n = 0.5, 0.4, 1.0, 5000
        x = model.arma(numpy.array([φ]), numpy.array([θ]), n, σ)
        assert x.shape == (n,)
        # ARMA(1,1): γ0 = σ²(1+2φθ+θ²)/(1-φ²), ρ1 = (1+φθ)(φ+θ)/(1+2φθ+θ²), ρk = φ^{k-1} ρ1
        γ0 = σ**2 * (1 + 2 * φ * θ + θ**2) / (1 - φ**2)          # 2.08
        ρ1 = (1 + φ * θ) * (φ + θ) / (1 + 2 * φ * θ + θ**2)       # 0.692
        # SE(γ̂0) ≈ γ0·sqrt(2(1+φ)/((1-φ)n)) = 0.07 → 15% (0.31) is ~4.4σ
        assert x.var() == pytest.approx(γ0, rel=0.15)
        acf = sample_acf(x, 3)
        # The sampling SE of r_k here is NOT the white-noise value sqrt((1-ρ1²)/n):
        # for a strongly autocorrelated process Bartlett's sum grows with lag. Monte
        # Carlo over 300 independent draws at n=5000 gives sd(r_k) = 0.0082 / 0.0159 /
        # 0.0195 at k = 1/2/3, so one flat atol=0.05 is 6σ at lag 1 but only ~2.5σ at
        # lag 3 (measured ~1% per-run failure rate). Per-lag tolerances ≈ 6σ/5.6σ/5.1σ:
        assert acf[1] == pytest.approx(ρ1, abs=0.05)
        assert acf[2] == pytest.approx(φ * ρ1, abs=0.09)
        assert acf[3] == pytest.approx(φ**2 * ρ1, abs=0.10)

    def test_empty_orders_give_white_noise(self):
        n = 5000
        x = model.arma(numpy.array([]), numpy.array([]), n, 2.0)
        assert x.var() == pytest.approx(4.0, rel=0.08)          # SE = 4·sqrt(2/n) = 0.08 → 4σ
        assert abs(sample_acf(x, 1)[1]) < 0.06                   # 1/sqrt(n) = 0.014 → 4.2σ


class TestARIMA:
    φ = numpy.array([0.5])
    θ = numpy.array([0.3])

    def test_d1_is_cumulative_sum_of_arma(self):
        n = 500
        reseed()
        x = model.arma(self.φ, self.θ, n, 1.0)
        reseed()
        y = model.arima(self.φ, self.θ, 1, n, 1.0)
        assert y.shape == (n,)
        assert y[0] == x[0]
        assert_allclose(numpy.diff(y), x[1:], atol=1e-10)

    def test_d2_second_difference_recovers_arma(self):
        n = 500
        reseed()
        x = model.arma(self.φ, self.θ, n, 1.0)
        reseed()
        y = model.arima(self.φ, self.θ, 2, n, 1.0)
        assert_array_equal(y[:2], x[:2])
        assert_allclose(numpy.diff(y, n=2), x[2:], atol=1e-7)

    @pytest.mark.parametrize("d", [0, 3, -1])
    def test_invalid_d_raises(self, d):
        with pytest.raises(AssertionError, match="d must equal 1 or 2"):
            model.arima(self.φ, self.θ, d, 10)


class TestARIMAFromARMA:
    def test_d1_hand_example(self):
        assert_array_equal(model.arima_from_arma(numpy.array([1.0, 2.0, 3.0, 4.0]), 1), [1, 3, 6, 10])

    def test_d2_hand_example_and_input_untouched(self):
        # y_t = x_t + 2 y_{t-1} - y_{t-2}, y_0 = x_0, y_1 = x_1
        x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = model.arima_from_arma(x, 2)
        assert_array_equal(y, [1, 2, 6, 14, 27])
        assert_array_equal(numpy.diff(y, n=2), x[2:])
        assert_array_equal(x, [1, 2, 3, 4, 5])          # not mutated in place

    @pytest.mark.parametrize("d", [1, 2])
    def test_agrees_with_arima_simulator(self, d):
        φ, θ, n = numpy.array([0.4]), numpy.array([-0.2]), 300
        reseed()
        y_from_samples = model.arima_from_arma(model.arma(φ, θ, n, 1.0), d)
        reseed()
        y_direct = model.arima(φ, θ, d, n, 1.0)
        assert_allclose(y_from_samples, y_direct, rtol=1e-12, atol=1e-9)

    @pytest.mark.parametrize("d", [0, 3, -1])
    def test_invalid_d_raises(self, d):
        with pytest.raises(AssertionError, match="d must equal 1 or 2"):
            model.arima_from_arma(numpy.ones(5), d)


# ---------------------------------------------------------------------------
# Model layer: estimators
# ---------------------------------------------------------------------------

class TestPACF:
    def test_shape_and_lag_zero(self):
        p = model.pacf(model.ar1(0.5, 500), 6)
        assert p.shape == (7,)
        assert p[0] == pytest.approx(1.0)

    def test_ar1_cuts_off_after_lag_one(self):
        φ, n = 0.6, 4000
        p = model.pacf(model.ar1(φ, n), 4)
        # SE(pacf_1) ≈ sqrt((1-φ²)/n) = 0.013 → 0.05 is ~4σ; SE(pacf_k>1) = 1/sqrt(n) = 0.016 → 0.07 is ~4.4σ
        assert p[1] == pytest.approx(φ, abs=0.05)
        assert_allclose(p[2:], 0.0, atol=0.07)

    def test_ar2_closed_form(self):
        # AR(2): pacf_1 = ρ1 = φ1/(1-φ2), pacf_2 = φ2, pacf_k>2 = 0
        φ1, φ2, n = 0.5, 0.3, 4000
        p = model.pacf(model.arp(numpy.array([φ1, φ2]), n), 4)
        assert p[1] == pytest.approx(φ1 / (1 - φ2), abs=0.06)
        assert p[2] == pytest.approx(φ2, abs=0.06)
        assert_allclose(p[3:], 0.0, atol=0.07)


class TestYuleWalker:
    def test_returns_order_coefficients(self):
        φ = 0.5
        coeffs = model.yw(model.ar1(φ, 500), 3)
        assert isinstance(coeffs, numpy.ndarray) and coeffs.shape == (3,)
        # MC sd ≈ 0.045 at n=500 → 0.2 is ~4.4σ; the extra terms are pure noise.
        assert coeffs[0] == pytest.approx(φ, abs=0.2)
        assert_allclose(coeffs[1:], 0.0, atol=0.2)

    def test_over_specified_order_has_vanishing_extra_terms(self):
        φ, n = 0.7, 4000
        coeffs = model.yw(model.ar1(φ, n), 3)
        assert coeffs[0] == pytest.approx(φ, abs=0.06)
        assert_allclose(coeffs[1:], 0.0, atol=0.07)        # 1/sqrt(n) = 0.016 → ~4.4σ


class TestFits:
    @pytest.mark.parametrize("fit, order, names", [
        (model.ar_fit, 2, ["const", "ar.L1", "ar.L2", "sigma2"]),
        (model.ar_offset_fit, 2, ["const", "ar.L1", "ar.L2", "sigma2"]),
        (model.ma_fit, 1, ["const", "ma.L1", "sigma2"]),
        (model.ma_offset_fit, 1, ["const", "ma.L1", "sigma2"]),
    ])
    def test_param_layout_the_facade_relies_on(self, fit, order, names):
        # The facade indexes params[0]=const, params[1:-1]=coefficients, params[-1]=sigma2.
        # Note ar_fit/ma_fit pass no trend, and statsmodels' d=0 default is 'c', so they
        # carry a constant exactly like the *_offset_fit variants.
        result = fit(model.ar1(0.4, 400), order)
        # statsmodels returns the results wrapper, which is not an ARIMAResults subclass
        # even though navi annotates the return as ARIMAResults.
        assert isinstance(result, (ARIMAResults, ARIMAResultsWrapper))
        assert list(result.param_names) == names
        assert result.params.shape == (len(names),)

    def test_offset_fits_are_exact_duplicates_of_the_plain_fits(self):
        # NAMING BUG pinned: ar_fit builds ARIMA(order=(p,0,0)) with no trend and
        # ar_offset_fit adds trend='c' — but statsmodels' default trend at d=0 is
        # already 'c', so the two models are the same model. Same for ma_fit /
        # ma_offset_fit. The *_offset_fit names promise a distinction that the code
        # does not make; the fitted parameters come back bit-identical.
        x = model.ar1(0.4, 400)
        assert_array_equal(model.ar_fit(x, 1).params, model.ar_offset_fit(x, 1).params)
        assert_array_equal(model.ma_fit(x, 1).params, model.ma_offset_fit(x, 1).params)

    def test_ar_fit_round_trip(self):
        φ, σ, n = 0.6, 1.5, 4000
        result = model.ar_fit(model.ar1(φ, n, σ), 1)
        assert result.params[1] == pytest.approx(φ, abs=0.05)           # SE 0.013 → ~4σ
        assert result.params[2] == pytest.approx(σ**2, rel=0.08)        # SE σ²sqrt(2/n)=0.05 → 0.18 is ~3.6σ
        # zero-mean process: SE(const) ≈ sqrt(σ²/((1-φ)² n)) = 0.06 → 0.3 is 5σ
        assert result.params[0] == pytest.approx(0.0, abs=0.3)

    def test_ma_offset_fit_recovers_mean(self):
        θ, c, n = 0.6, 3.0, 4000
        result = model.ma_offset_fit(model.maq(numpy.array([θ]), n) + c, 1)
        # SE(mean) = sqrt((1+θ)²/n) = 0.025 → 0.12 is ~4.7σ; SE(θ̂) ≈ 0.013 → 0.05 is ~4σ
        assert result.params[0] == pytest.approx(c, abs=0.12)
        assert result.params[1] == pytest.approx(θ, abs=0.05)


# ---------------------------------------------------------------------------
# Facade: closed-form compute_* functions
# ---------------------------------------------------------------------------

class TestFacadeClosedForms:
    def test_compute_ar1_acf(self):
        lags, acf = facade.compute_ar1_acf(φ=0.5, nlags=4)
        assert_array_equal(lags, [0.0, 1.0, 2.0, 3.0])
        assert_allclose(acf, [1.0, 0.5, 0.25, 0.125], rtol=1e-12)
        assert_allclose(acf, model.ar1_acf(0.5, 4), rtol=1e-12)

    def test_compute_maq_acf(self):
        lags, acf = facade.compute_maq_acf(θ=[0.6], nlags=5, σ=2.0)
        assert_array_equal(lags, numpy.arange(6.0))
        assert acf.shape == (6,)
        assert_allclose(acf, [1.0, 0.6 / 1.36, 0, 0, 0, 0], rtol=1e-12)

    @MAQ_COV_BUG
    def test_compute_maq_acf_ma2(self):
        # θ=[0.4,0.2] → γ0 = 1.2, γ1 = θ1+θ1θ2 = 0.48, γ2 = θ2 = 0.2, so the true
        # ACF is [1, 0.4, 1/6, 0]; navi returns [1, 1/3, 7/30, 0] via maq_cov.
        # NB the facade's σ default cannot be checked on this path at all: maq_acf
        # is exactly σ-invariant (σ² cancels), so compute_maq_acf(θ, nlags) is
        # bit-equal to model.maq_acf(θ, σ, nlags) for *every* σ. compute_maq_sd
        # below is the σ-sensitive path where that default is observable.
        _, acf = facade.compute_maq_acf(θ=[0.4, 0.2], nlags=3)
        assert_allclose(acf, [1.0, 0.4, 1.0 / 6.0, 0.0], rtol=1e-12)

    def test_compute_maq_acf_more_coefficients_than_lags_raises(self):
        # Reachable straight from the notebook-facing facade: maq_acf writes
        # ac_eq[i+1] for each θ_i into a max_lag+1 array. Bare IndexError today.
        with pytest.raises(IndexError):
            facade.compute_maq_acf(θ=[0.4, 0.3, 0.2], nlags=2)

    def test_compute_arma_mean(self):
        t, mean = facade.compute_arma_mean(npts=7)
        assert_array_equal(t, numpy.arange(7.0))
        assert_array_equal(mean, numpy.zeros(7))

    def test_compute_ar1_sd(self):
        t, sd = facade.compute_ar1_sd(npts=4, φ=0.6, σ=2.0)
        assert_array_equal(t, numpy.arange(4.0))
        assert_allclose(sd, numpy.full(4, 2.5), rtol=1e-12)
        _, sd_default = facade.compute_ar1_sd(npts=4, φ=0.6)       # σ defaults to 1
        assert_allclose(sd_default, numpy.full(4, 1.25), rtol=1e-12)

    def test_compute_maq_sd(self):
        t, sd = facade.compute_maq_sd(θ=[0.6, 0.8], σ=3.0, npts=3)
        assert_array_equal(t, [0.0, 1.0, 2.0])
        assert_allclose(sd, numpy.full(3, 3.0 * numpy.sqrt(2.0)), rtol=1e-12)
        _, sd_default = facade.compute_maq_sd(θ=[0.6, 0.8], npts=3)
        assert_allclose(sd_default, numpy.full(3, numpy.sqrt(2.0)), rtol=1e-12)

    def test_compute_ar1_offset_mean(self):
        t, mean = facade.compute_ar1_offset_mean(φ=0.6, μ=2.0, npts=3)
        assert_array_equal(t, [0.0, 1.0, 2.0])
        assert_allclose(mean, numpy.full(3, 5.0), rtol=1e-12)

    def test_compute_ar1_offset_sd(self):
        t, sd = facade.compute_ar1_offset_sd(φ=0.6, σ=2.0, npts=3)
        assert_array_equal(t, [0.0, 1.0, 2.0])
        assert_allclose(sd, numpy.full(3, 2.5), rtol=1e-12)
        _, sd_default = facade.compute_ar1_offset_sd(φ=0.6, npts=3)
        assert_allclose(sd_default, numpy.full(3, 1.25), rtol=1e-12)

    def test_compute_pacf_contract_and_round_trip(self):
        φ, n = 0.7, 4000
        x = model.ar1(φ, n)
        lags, coeffs = facade.compute_pacf(x, nlags=3)
        assert_array_equal(lags, [1.0, 2.0, 3.0])                 # lags start at 1
        assert coeffs.shape == (3,)
        assert_allclose(coeffs, model.yw(x, 3), rtol=1e-12)       # delegates to Yule-Walker
        assert coeffs[0] == pytest.approx(φ, abs=0.06)
        assert_allclose(coeffs[1:], 0.0, atol=0.07)
        # AR(1) is exactly the case where Yule-Walker and the PACF agree at lag 1,
        # so this test cannot distinguish them — see the AR(2) test below.

    def test_compute_pacf_returns_yule_walker_coefficients_not_the_pacf(self):
        # NAMING BUG pinned: compute_pacf delegates to arima.yw, which returns the
        # order-n Yule-Walker *coefficients* (φ̂1..φ̂n), not partial autocorrelations.
        # On AR(1) the two coincide at lag 1; on AR(2) they separate, because
        # PACF(1) = ρ1 = φ1/(1-φ2) = 0.714 while the YW lag-1 coefficient is φ1 = 0.5.
        φ1, φ2, n = 0.5, 0.3, 4000
        x = model.arp(numpy.array([φ1, φ2]), n)
        _, coeffs = facade.compute_pacf(x, nlags=2)
        true_pacf = model.pacf(x, 2)                    # statsmodels' actual PACF
        # SE(φ̂_i) ≈ sqrt((1-φ2²)/n) = 0.015 → 0.07 is ~4.6σ
        assert_allclose(coeffs, [φ1, φ2], atol=0.07)
        assert true_pacf[1] == pytest.approx(φ1 / (1 - φ2), abs=0.07)
        # The gap at lag 1 is 0.214 ≈ 14 SE, so this separation is not sampling noise.
        assert abs(coeffs[0] - true_pacf[1]) > 0.1
        # At the final lag the two definitions coincide (Durbin-Levinson), which is
        # why the mislabelling is invisible when only the last coefficient is read.
        assert coeffs[1] == pytest.approx(true_pacf[2], abs=0.02)


# ---------------------------------------------------------------------------
# Facade: simulation sources
# ---------------------------------------------------------------------------

SOURCE_CASES = [
    pytest.param(facade.create_ar_source, dict(φ=[0.5, 0.2], npts=64, σ=2.0),
                 lambda: model.arp(numpy.array([0.5, 0.2]), 64, 2.0), id="ar"),
    pytest.param(facade.create_ar_drift_source, dict(φ=[0.5], npts=64, μ=1.0, γ=0.1, σ=2.0),
                 lambda: model.arp_drift([0.5], 1.0, 0.1, 64, 2.0), id="ar_drift"),
    pytest.param(facade.create_ar_offset_source, dict(φ=[0.5], npts=64, μ=1.0, σ=2.0),
                 lambda: model.arp_offset([0.5], 1.0, 64, 2.0), id="ar_offset"),
    pytest.param(facade.create_ma_source, dict(θ=[0.6, -0.2], npts=64, σ=2.0),
                 lambda: model.maq(numpy.array([0.6, -0.2]), 64, 2.0), id="ma"),
    pytest.param(facade.create_arma_source, dict(φ=[0.5], θ=[0.3], npts=64, σ=2.0),
                 lambda: model.arma(numpy.array([0.5]), numpy.array([0.3]), 64, 2.0), id="arma"),
    pytest.param(facade.create_arima_source, dict(φ=[0.5], θ=[0.3], d=1, npts=64, σ=2.0),
                 lambda: model.arima(numpy.array([0.5]), numpy.array([0.3]), 1, 64, 2.0), id="arima_d1"),
    pytest.param(facade.create_arima_source, dict(φ=[0.5], θ=[0.3], d=2, npts=64, σ=2.0),
                 lambda: model.arima(numpy.array([0.5]), numpy.array([0.3]), 2, 64, 2.0), id="arima_d2"),
]


class TestFacadeSources:
    @pytest.mark.parametrize("source, kwargs, model_call", SOURCE_CASES)
    def test_kwargs_plumb_through_to_model_layer(self, source, kwargs, model_call):
        # This is a delegation mirror: each facade function calls exactly the model
        # function the lambda invokes, so it carries NO independent numeric content
        # and a wrong simulator would be invisible here. What it does pin is the
        # (t, values) contract and the kwargs plumbing — every case uses distinct
        # argument values, so a swapped argument or a dropped σ shows up.
        reseed()
        t, x = source(**kwargs)
        reseed()
        expected = model_call()
        npts = kwargs["npts"]
        assert isinstance(t, numpy.ndarray) and t.dtype.kind == "f"
        assert_array_equal(t, numpy.arange(npts, dtype=float))
        assert isinstance(x, numpy.ndarray) and x.shape == (npts,)
        assert_array_equal(x, expected)

    @pytest.mark.parametrize("source, kwargs", [
        (facade.create_ar_source, dict(φ=[0.5], npts=3000)),
        (facade.create_ma_source, dict(θ=[0.5], npts=3000)),
        (facade.create_arma_source, dict(φ=[0.5], θ=[0.5], npts=3000)),
    ])
    def test_sigma_defaults_to_one(self, source, kwargs):
        # Each process is driven by ε ~ N(0, σ²) and with σ=1 has var ≥ 1; with the
        # default applied the innovation variance recovered from the ARMA structure is 1.
        _, x = source(**kwargs)
        φ = kwargs.get("φ", [0.0])[0]
        # Innovation variance from the AR part only is exact for the pure-AR case;
        # for MA/ARMA bound it through the closed-form variance instead.
        if "θ" not in kwargs:
            resid = x[1:] - φ * x[:-1]
            assert resid.var() == pytest.approx(1.0, rel=0.1)       # SE sqrt(2/n)=0.026 → ~3.9σ
        else:
            θ = kwargs["θ"][0]
            γ0 = (1 + 2 * φ * θ + θ**2) / (1 - φ**2)
            assert x.var() == pytest.approx(γ0, rel=0.2)

    @pytest.mark.parametrize("source, kwargs, resid", [
        # Both simulators are explicit recursions over ε = σ·noise(n), so with the σ
        # default in force the residual must be the *unscaled* N(0,1) draw exactly.
        # A default of anything but 1.0 changes every residual by that factor.
        pytest.param(facade.create_ar_drift_source, dict(φ=[0.6], μ=1.0, γ=0.1, npts=300),
                     lambda x: x[1:] - 0.6 * x[:-1] - 0.1 * numpy.arange(1, 300) - 1.0,
                     id="ar_drift"),
        pytest.param(facade.create_ar_offset_source, dict(φ=[0.6], μ=1.0, npts=300),
                     lambda x: x[1:] - 0.6 * x[:-1] - 1.0,
                     id="ar_offset"),
    ])
    def test_recursive_sources_sigma_defaults_to_one(self, source, kwargs, resid):
        reseed()
        ε = numpy.random.normal(0.0, 1.0, kwargs["npts"])
        reseed()
        _, x = source(**kwargs)
        assert_allclose(resid(x), ε[1:], atol=1e-12)

    def test_create_arima_source_sigma_defaults_to_one(self):
        # ARIMA(0,1,0) is a random walk, so its first difference is the raw ε: with
        # the σ default applied the differenced series has unit variance.
        npts = 4000
        _, y = facade.create_arima_source(φ=[], θ=[], d=1, npts=npts)
        # SE(var) = sqrt(2/n) = 0.022 → 10% is ~4.5σ
        assert numpy.diff(y).var() == pytest.approx(1.0, rel=0.1)

    def test_create_arima_from_arma_source(self):
        samples = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t, y = facade.create_arima_from_arma_source(arma=samples, d=2)
        assert_array_equal(t, numpy.arange(5.0))
        assert_array_equal(y, [1, 2, 6, 14, 27])
        t1, y1 = facade.create_arima_from_arma_source(arma=samples, d=1)
        assert_array_equal(t1, numpy.arange(5.0))
        assert_array_equal(y1, [1, 3, 6, 10, 15])

    @pytest.mark.parametrize("d", [1, 2])
    def test_arima_source_agrees_with_integrating_an_arma_source(self, d):
        # The two facade entry points must describe the same process: simulating an
        # ARIMA directly, and integrating an ARMA drawn from the same noise stream.
        kwargs = dict(φ=[0.4], θ=[-0.2], npts=300, σ=1.5)
        reseed()
        _, y_direct = facade.create_arima_source(d=d, **kwargs)
        reseed()
        _, arma = facade.create_arma_source(**kwargs)
        t, y_integrated = facade.create_arima_from_arma_source(arma=arma, d=d)
        assert_array_equal(t, numpy.arange(300.0))
        assert_allclose(y_integrated, y_direct, rtol=1e-12, atol=1e-9)

    @pytest.mark.parametrize("d", [0, 3, -1])
    def test_create_arima_source_rejects_bad_d(self, d):
        with pytest.raises(AssertionError, match="d must equal 1 or 2"):
            facade.create_arima_source(φ=[0.5], θ=[0.3], d=d, npts=16)

    @pytest.mark.parametrize("d", [0, 3, -1])
    def test_create_arima_from_arma_source_rejects_bad_d(self, d):
        with pytest.raises(AssertionError, match="d must equal 1 or 2"):
            facade.create_arima_from_arma_source(arma=numpy.ones(5), d=d)

    def test_random_walk_variance_grows_linearly(self):
        # ARIMA(0,1,0) with σ: var(y_t) = σ² (t+1) for the cumulative sum of t+1 shocks.
        σ, npts, nsim = 2.0, 300, 600
        t, ensemble = create_ensemble(facade.create_arima_source, nsim, φ=[], θ=[], d=1, npts=npts, σ=σ)
        y = numpy.array(ensemble)
        assert t.shape == (npts,) and y.shape == (nsim, npts)
        # SE(var) = var·sqrt(2/nsim) = 5.8% → 25% is ~4.3σ
        assert y[:, -1].var() == pytest.approx(σ**2 * npts, rel=0.25)
        assert y[:, npts // 2 - 1].var() == pytest.approx(σ**2 * npts / 2, rel=0.25)
        # SE(mean) = sqrt(σ² npts / nsim) = 1.41 → 7 is ~5σ
        assert y[:, -1].mean() == pytest.approx(0.0, abs=7.0)


# ---------------------------------------------------------------------------
# Facade: kwargs contracts
# ---------------------------------------------------------------------------

REQUIRED_KWARGS = [
    (facade.compute_ar1_acf, dict(φ=0.5, nlags=4)),
    (facade.compute_maq_acf, dict(θ=[0.5], nlags=4)),
    (facade.compute_arma_mean, dict(npts=4)),
    (facade.compute_ar1_sd, dict(npts=4, φ=0.5)),
    (facade.compute_maq_sd, dict(θ=[0.5], npts=4)),
    (facade.compute_ar1_offset_mean, dict(φ=0.5, μ=1.0, npts=4)),
    (facade.compute_ar1_offset_sd, dict(φ=0.5, npts=4)),
    (facade.create_ar_source, dict(φ=[0.5], npts=16)),
    (facade.create_ar_drift_source, dict(φ=[0.5], npts=16, μ=1.0, γ=0.1)),
    (facade.create_ar_offset_source, dict(φ=[0.5], npts=16, μ=1.0)),
    (facade.create_ma_source, dict(θ=[0.5], npts=16)),
    (facade.create_arma_source, dict(φ=[0.5], θ=[0.3], npts=16)),
    (facade.create_arima_source, dict(φ=[0.5], θ=[0.3], d=1, npts=16)),
    (facade.create_arima_from_arma_source, dict(arma=numpy.ones(4), d=1)),
]

MISSING_CASES = [
    pytest.param(fn, kwargs, key, id=f"{fn.__name__}-missing-{key}")
    for fn, kwargs in REQUIRED_KWARGS
    for key in kwargs
]


class TestFacadeKwargs:
    @pytest.mark.parametrize("fn, kwargs, key", MISSING_CASES)
    def test_missing_required_kwarg_raises(self, fn, kwargs, key):
        incomplete = {k: v for k, v in kwargs.items() if k != key}
        with pytest.raises(Exception, match=f"{key} parameter is required"):
            fn(**incomplete)

    @pytest.mark.parametrize("fn, kwargs", REQUIRED_KWARGS, ids=lambda v: getattr(v, "__name__", ""))
    def test_complete_kwargs_return_tuple_pair(self, fn, kwargs):
        out = fn(**kwargs)
        assert isinstance(out, tuple) and len(out) == 2
        assert len(out[0]) == len(out[1])

    @pytest.mark.parametrize("fn, kwargs", [
        (facade.compute_pacf, dict(nlags=2)),
        (facade.compute_ar_estimate, dict(order=1)),
        (facade.compute_ar_offset_estimate, dict(order=1)),
        (facade.compute_ma_estimate, dict(order=1)),
        (facade.compute_ma_offset_estimate, dict(order=1)),
    ], ids=lambda v: getattr(v, "__name__", ""))
    def test_positional_data_functions_require_their_kwarg(self, fn, kwargs):
        (key,) = kwargs
        with pytest.raises(Exception, match=f"{key} parameter is required"):
            fn(numpy.ones(32))

    @pytest.mark.parametrize("fn, kwargs, list_param", [
        (facade.compute_maq_acf, dict(nlags=4), "θ"),
        (facade.compute_maq_sd, dict(npts=4), "θ"),
        (facade.create_ar_source, dict(npts=16), "φ"),
        (facade.create_ar_drift_source, dict(npts=16, μ=1.0, γ=0.1), "φ"),
        (facade.create_ar_offset_source, dict(npts=16, μ=1.0), "φ"),
        (facade.create_ma_source, dict(npts=16), "θ"),
        (facade.create_arma_source, dict(npts=16, φ=[0.5]), "θ"),
        (facade.create_arma_source, dict(npts=16, θ=[0.5]), "φ"),
        (facade.create_arima_source, dict(npts=16, φ=[0.5], d=1), "θ"),
        (facade.create_arima_source, dict(npts=16, θ=[0.5], d=1), "φ"),
    ], ids=lambda v: getattr(v, "__name__", v if isinstance(v, str) else ""))
    def test_coefficient_vectors_must_be_lists(self, fn, kwargs, list_param):
        with pytest.raises(Exception, match="Expected <class 'list'>"):
            fn(**kwargs, **{list_param: numpy.array([0.5])})


# ---------------------------------------------------------------------------
# Facade: estimates
# ---------------------------------------------------------------------------

ESTIMATE_CASES = [
    pytest.param(facade.compute_ar_estimate, ARMAEstType.AR, "phi", id="ar"),
    pytest.param(facade.compute_ar_offset_estimate, ARMAEstType.AR_OFFSET, "phi", id="ar_offset"),
    pytest.param(facade.compute_ma_estimate, ARMAEstType.MA, "theta", id="ma"),
    pytest.param(facade.compute_ma_offset_estimate, ARMAEstType.MA_OFFSET, "theta", id="ma_offset"),
]


class TestFacadeEstimates:
    @pytest.mark.parametrize("estimate, est_type, label_symbol", ESTIMATE_CASES)
    def test_estimate_contract(self, estimate, est_type, label_symbol):
        order = 2
        result, est = estimate(model.ar1(0.4, 600), order=order)
        assert isinstance(result, (ARIMAResults, ARIMAResultsWrapper))
        assert isinstance(est, ARMAEst)
        assert est.arma_est_type is est_type
        assert est.est_model == "ARMA"
        assert est.order == order and len(est.params) == order

        # Values mirror statsmodels' params/bse in the documented slots.
        assert est.const.est == result.params[0] and est.const.err == result.bse[0]
        for i, p in enumerate(est.params):
            assert p.est == result.params[i + 1] and p.err == result.bse[i + 1]
            assert p.param_type == ARMAParamType.ARMA_PARAM.value
            # The two indexes disagree: __arma_estimate_from_result numbers `order`
            # from statsmodels' 1-based params slot, while ARMAEst.__set_params_labels
            # subscripts the label with the 0-based list position. So params[0] is
            # order 1 but reads $\hat{\phi_{0}}$. Both halves asserted exactly.
            assert p.order == i + 1
            assert p.est_label == rf"$\hat{{\{label_symbol}_{{{i}}}}}$"
            assert p.err_label == rf"$\sigma_{{$\hat{{\{label_symbol}_{{{i}}}}}}}$"
        assert est.sigma2.est == result.params[-1] and est.sigma2.err == result.bse[-1]
        assert est.const.param_type == ARMAParamType.ARMA_CONST.value
        assert est.sigma2.param_type == ARMAParamType.ARMA_SIG2.value

        # One uuid4 tags the estimate and every parameter in it.
        assert uuid.UUID(est.est_id).version == 4
        assert {est.const.est_id, est.sigma2.est_id, *(p.est_id for p in est.params)} == {est.est_id}
        assert est.const.est_label == r"$\hat{\mu^*}$"
        assert est.sigma2.est_label == r"$\hat{\sigma^2}$"

    def test_estimate_ids_are_unique_per_call(self):
        x = model.ar1(0.4, 300)
        _, a = facade.compute_ar_estimate(x, order=1)
        _, b = facade.compute_ar_estimate(x, order=1)
        assert a.est_id != b.est_id

    def test_estimate_serialization_and_repr(self):
        _, est = facade.compute_ma_estimate(model.maq(numpy.array([0.5]), 400), order=1)
        payload = json.loads(est.to_json())
        assert payload["est_model"] == "ARMA"
        assert payload["arma_est_type"] == "MA"
        assert payload["est_id"] == est.est_id
        assert payload["order"] == 1 and len(payload["params"]) == 1
        assert payload["params"][0]["est"] == pytest.approx(est.params[0].est)
        assert payload["sigma2"]["param_type"] == "ARMA_SIG2"
        assert json.loads(est.to_json(pretty=True)) == payload
        assert repr(est).startswith("ARMAEst(") and est.est_id in repr(est)
        assert repr(est.const).startswith("ParamEst(")
        assert isinstance(est.const, ParamEst)

    def test_ar_estimate_round_trip(self):
        φ, n = [0.4, 0.2], 4000
        _, x = facade.create_ar_source(φ=φ, npts=n)
        _, est = facade.compute_ar_estimate(x, order=2)
        # SE(φ̂_i) ≈ sqrt((1-φ2²)/n) = 0.0155 → 0.07 is ~4.5σ
        assert est.params[0].est == pytest.approx(φ[0], abs=0.07)
        assert est.params[1].est == pytest.approx(φ[1], abs=0.07)
        # zero-mean: SE(const) = sqrt(1/((1-φ1-φ2)² n)) = 0.0395 → 0.2 is ~5σ
        assert est.const.est == pytest.approx(0.0, abs=0.2)
        assert est.sigma2.est == pytest.approx(1.0, rel=0.1)       # SE sqrt(2/n)=0.022 → ~4.5σ
        assert all(p.err > 0 for p in est.params)

    def test_ar_offset_estimate_round_trip(self):
        φ, μ, n = 0.5, 2.0, 4000
        _, x = facade.create_ar_offset_source(φ=[φ], μ=μ, npts=n)
        _, est = facade.compute_ar_offset_estimate(x, order=1)
        # const is the process mean μ/(1-φ) = 4: SE ≈ 0.031 → 0.15 is ~5σ
        assert est.const.est == pytest.approx(model.ar1_offset_mean(φ, μ), abs=0.15)
        assert est.params[0].est == pytest.approx(φ, abs=0.06)

    def test_ma_estimate_round_trip(self):
        θ, n = 0.6, 4000
        _, x = facade.create_ma_source(θ=[θ], npts=n)
        _, est = facade.compute_ma_estimate(x, order=1)
        assert est.params[0].est == pytest.approx(θ, abs=0.05)      # SE 0.0126 → ~4σ
        assert est.sigma2.est == pytest.approx(1.0, rel=0.1)

    def test_ma_offset_estimate_round_trip(self):
        θ, c, n = 0.6, 3.0, 4000
        _, x = facade.create_ma_source(θ=[θ], npts=n)
        _, est = facade.compute_ma_offset_estimate(x + c, order=1)
        assert est.const.est == pytest.approx(c, abs=0.12)          # SE (1+θ)/sqrt(n)=0.025 → ~4.7σ
        assert est.params[0].est == pytest.approx(θ, abs=0.05)
