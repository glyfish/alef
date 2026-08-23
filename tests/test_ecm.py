"""Tests for navi's Error Correction Model (ECM) module group.

Two layers are covered:

* ``lib.models.ecm`` — the pure model: closed-form variance/covariance of the
  ECM pair and the ``ecm`` simulator.
* ``lib.data.impl.ecm`` — the ``**kwargs`` façade notebooks call. It returns
  ``(t, values)`` tuples and wraps the OLS parameter estimation.

The model is

    x_t      = x_{t-1} + a_t,                   a_t ~ AR(1)(φ, σ)
    Δy_t     = δ + γ Δx_t + λ (y_{t-1} - α - β x_{t-1}) + ξ_t,   ξ_t ~ N(0, σ²)

so the cointegrating residual e_t = y_t - α - β x_t obeys

    e_t = (1 + λ) e_{t-1} + δ + (γ - β) Δx_t + ξ_t

which is stationary for -2 < λ < 0 and a pure AR(1)(1+λ, σ) when γ = β.

Note that ``arima.ar1`` generates the driving AR(1) with no burn-in, so the
simulated x path is the cumsum of a *zero-initialised* AR(1) rather than of a
stationary one. The closed forms in ``lib.models.ecm`` assume the stationary
case, so the two are compared against separately derived references here.

Every simulation draws from numpy's global RNG, which ``conftest`` reseeds
before each test.
"""

import ast
import inspect

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lib.data.impl import ecm as fecm
from lib.data.param_est import EstModel, OLSParamType, OLSResult, OLSTransform, ParamEst
from lib.models import arima, ecm

# ---------------------------------------------------------------------------
# Independent closed forms (derived by hand, not by calling the library)
# ---------------------------------------------------------------------------


def cumsum_ar1_var(φ: float, σ: float, n: int) -> float:
    """Var(Σ_{i=1}^{n} a_i) for stationary AR(1) a_i with acf φ^k.

    Σ_{k=1}^{n-1} (n-k) φ^k has the closed form φ[n(1-φ) - (1-φ^n)]/(1-φ)²,
    so the variance is γ0 [n + 2 φ (n(1-φ) - (1-φ^n)) / (1-φ)²] with
    γ0 = σ²/(1-φ²).
    """
    if n == 0:
        return 0.0
    γ0 = σ**2 / (1.0 - φ**2)
    geometric = φ * (n * (1.0 - φ) - (1.0 - φ**n)) / (1.0 - φ) ** 2
    return γ0 * (n + 2.0 * geometric)


def cumsum_ar1_var_matrix(φ: float, σ: float, n: int) -> float:
    """Same quantity as the full covariance matrix sum 1ᵀ Γ 1, Γ_ij = γ0 φ^|i-j|."""
    if n == 0:
        return 0.0
    idx = numpy.arange(n)
    Γ = σ**2 / (1.0 - φ**2) * φ ** numpy.abs(idx[:, None] - idx[None, :])
    return float(Γ.sum())


def zero_init_cumsum_ar1_var(φ: float, σ: float, k: int) -> float:
    """Exact Var(x_k) for the process ``ecm.ecm`` actually simulates.

    ``arima.ar1`` calls ``sm.tsa.arma_generate_sample`` with no burn-in, so the
    driving AR(1) starts from zero rather than from its stationary
    distribution: a_i = Σ_{j=0}^{i} φ^{i-j} e_j. Since x_k = Σ_{i=0}^{k} a_i,
    collecting the coefficient of each innovation gives

        x_k = Σ_{j=0}^{k} e_j (1 - φ^{k-j+1}) / (1 - φ)

    hence Var(x_k) = σ² Σ_{m=0}^{k} (1 - φ^{m+1})² / (1 - φ)².
    """
    m = numpy.arange(k + 1)
    return float(σ**2 * numpy.sum((1.0 - φ ** (m + 1)) ** 2) / (1.0 - φ) ** 2)


def lag1_autocorr(x: numpy.ndarray) -> float:
    z = x - x.mean()
    return float((z[1:] @ z[:-1]) / (z @ z))


def unused_imports(module) -> list[str]:
    """Names imported by ``module`` that no expression in it ever references."""
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {alias.asname or alias.name for alias in node.names}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return sorted(imported - used)


# Reference simulation parameters. λ < 0 so the ECM term mean-reverts,
# γ ≠ β so the γ and β estimates are distinguishable, and all values differ
# from the façade defaults so kwargs plumbing mistakes show up.
PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, SIGMA = 0.6, 0.0, 0.0, 1.5, 0.5, -0.4, 1.0
NPTS = 2000


@pytest.fixture
def simulated():
    xt, yt = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, NPTS, SIGMA)
    return xt, yt


# ===========================================================================
# Model layer: lib.models.ecm
# ===========================================================================


class TestXtVar:
    def test_random_walk_when_phi_zero(self):
        # φ = 0 makes a_t white noise, so x_t is a random walk: Var = t σ².
        t = numpy.arange(0, 11, dtype=float)
        assert_allclose(ecm.xt_var(0.0, 1.5, t), t * 1.5**2)

    @pytest.mark.parametrize("φ", [0.3, 0.7, -0.5])
    def test_small_t_hand_values(self, φ):
        σ = 1.3
        # n=0: empty sum. n=1: AR(1) stationary variance. n=2: 2γ0(1+φ) = 2σ²/(1-φ).
        out = ecm.xt_var(φ, σ, numpy.array([0.0, 1.0, 2.0]))
        assert_allclose(out, [0.0, σ**2 / (1 - φ**2), 2 * σ**2 / (1 - φ)])

    @pytest.mark.parametrize("φ", [0.3, 0.7, -0.5])
    @pytest.mark.parametrize("σ", [1.0, 2.0])
    def test_matches_geometric_closed_form(self, φ, σ):
        t = numpy.arange(0, 30, dtype=float)
        expected = [cumsum_ar1_var(φ, σ, int(n)) for n in t]
        assert_allclose(ecm.xt_var(φ, σ, t), expected, rtol=1e-12)

    @pytest.mark.parametrize("φ", [0.3, 0.7, -0.5])
    def test_matches_covariance_matrix_sum(self, φ):
        t = numpy.array([1.0, 5.0, 12.0, 25.0])
        expected = [cumsum_ar1_var_matrix(φ, 1.0, int(n)) for n in t]
        assert_allclose(ecm.xt_var(φ, 1.0, t), expected, rtol=1e-12)

    def test_large_t_long_run_variance(self):
        # Var(x_t)/t → σ²/(1-φ)² (the AR(1) long-run variance). The exact O(1/t)
        # correction is 2φ/(t(1-φ²)) = 5.49e-4 at φ=0.7, t=5000 (v/t = 11.10501
        # against 11.11111), so rtol=2e-3 leaves ~3.6x headroom.
        φ, σ, n = 0.7, 1.0, 5000.0
        (v,) = ecm.xt_var(φ, σ, numpy.array([n]))
        assert v / n == pytest.approx(σ**2 / (1 - φ) ** 2, rel=2e-3)
        # the correction has the sign and size the expansion predicts
        assert (v / n) / (σ**2 / (1 - φ) ** 2) - 1.0 == pytest.approx(-2 * φ / (n * (1 - φ**2)), rel=1e-3)

    def test_scales_with_sigma_squared(self):
        t = numpy.arange(1, 10, dtype=float)
        assert_allclose(ecm.xt_var(0.4, 2.0, t), 4.0 * ecm.xt_var(0.4, 1.0, t))

    def test_monotone_in_t_for_positive_phi(self):
        t = numpy.arange(0, 40, dtype=float)
        assert numpy.all(numpy.diff(ecm.xt_var(0.5, 1.0, t)) > 0)

    def test_truncates_non_integer_t(self):
        # Time values are cast with int(), so 2.9 evaluates as n=2.
        assert_allclose(ecm.xt_var(0.5, 1.0, numpy.array([2.9])), [cumsum_ar1_var(0.5, 1.0, 2)])

    def test_output_shape_and_dtype(self):
        t = numpy.linspace(0, 9, 10)
        out = ecm.xt_var(0.5, 1.0, t)
        assert isinstance(out, numpy.ndarray)
        assert out.shape == (10,)
        assert out.dtype.kind == "f"
        assert ecm.xt_var(0.5, 1.0, numpy.array([])).shape == (0,)

    @pytest.mark.xfail(
        strict=True,
        reason="xt_var never guards against t < 0: the inner `range(1, npts)` is empty for "
        "npts <= 1, so the result collapses to npts·σ²/(1-φ²), which is NEGATIVE for every "
        "negative time — ecm.xt_var(0.5, 1.0, [-3, -2.9, -1]) returns "
        "[-4.0, -2.667, -1.333]. A variance must never be negative; negative times should "
        "raise or clamp to 0",
    )
    def test_negative_t_never_yields_negative_variance(self):
        out = ecm.xt_var(0.5, 1.0, numpy.array([-3.0, -2.9, -1.0, 0.0]))
        assert numpy.all(out >= 0.0)


class TestYtVarAndCov:
    @pytest.mark.parametrize("β", [1.5, -0.8])
    def test_yt_var_is_beta_squared_times_closed_form(self, β):
        φ, σ = 0.6, 1.2
        t = numpy.arange(0, 20, dtype=float)
        expected = β**2 * numpy.array([cumsum_ar1_var(φ, σ, int(n)) for n in t])
        assert_allclose(ecm.yt_var(φ, σ, β, t), expected, rtol=1e-12)

    @pytest.mark.parametrize("β", [1.5, -0.8])
    def test_cov_is_beta_times_closed_form(self, β):
        φ, σ = 0.6, 1.2
        t = numpy.arange(0, 20, dtype=float)
        expected = β * numpy.array([cumsum_ar1_var(φ, σ, int(n)) for n in t])
        assert_allclose(ecm.cov(φ, σ, β, t), expected, rtol=1e-12)

    @pytest.mark.parametrize("β", [1.5, -0.8])
    def test_cov_saturates_cauchy_schwarz(self, β):
        # The three closed forms have to stay mutually consistent: the implied
        # correlation is exactly ±1, i.e. cov² = var_x var_y elementwise. This
        # is a relation *between* three public functions, so changing any one
        # of them in isolation breaks it — but note it carries no numeric
        # content on its own; the empirical claim is checked in the next test.
        t = numpy.arange(1, 20, dtype=float)
        c = ecm.cov(0.6, 1.0, β, t)
        assert_allclose(c**2, ecm.xt_var(0.6, 1.0, t) * ecm.yt_var(0.6, 1.0, β, t), rtol=1e-12)

    @pytest.mark.parametrize("β", [1.5, -0.8])
    def test_simulated_correlation_approaches_sign_of_beta(self, β):
        # The falsifiable form of "the implied correlation is ±1": x and y are
        # cointegrated, so the ensemble correlation of (x_T, y_T) tends to
        # sign(β) as the I(1) component swamps the stationary residual. At
        # T=50 the residual still contributes — observed ρ = 0.995 (β=1.5) and
        # -0.972 (β=-0.8), stable to ±0.005 across seeds — so tol 0.1.
        nsim, T = 250, 50
        X = numpy.empty((nsim, T))
        Y = numpy.empty((nsim, T))
        for i in range(nsim):
            X[i], Y[i] = ecm.ecm(PHI, DELTA, ALPHA, β, GAMMA, LAMBDA, T, SIGMA)
        ρ = numpy.corrcoef(X[:, -1], Y[:, -1])[0, 1]
        assert numpy.sign(ρ) == numpy.sign(β)
        assert ρ == pytest.approx(numpy.sign(β), abs=0.1)


class TestEcmSimulation:
    def test_shapes_and_initial_condition(self):
        xt, yt = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, 50, SIGMA)
        assert xt.shape == (50,) and yt.shape == (50,)
        assert xt.dtype.kind == "f" and yt.dtype.kind == "f"
        assert yt[0] == 0.0

    def test_deterministic_under_global_seed(self):
        numpy.random.seed(7)
        a = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, 100, SIGMA)
        numpy.random.seed(7)
        b = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, 100, SIGMA)
        assert_array_equal(a[0], b[0])
        assert_array_equal(a[1], b[1])
        # and a different stream gives a different path
        numpy.random.seed(8)
        c = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, 100, SIGMA)
        assert not numpy.array_equal(a[1], c[1])

    def test_x_is_the_ar1_driver_integrated_and_xi_is_drawn_after_it(self):
        # Pins the RNG contract of the simulator, not just its statistics:
        # ecm.ecm draws the AR(1) driver first, integrates it with
        # arima_from_arma(·, 1), and only then draws the ξ stream — and it
        # burns ξ[0], which the i-loop never reads. Replaying the same global
        # stream must reproduce both paths bit-for-bit; the NaN planted at ξ[0]
        # proves that first draw is discarded rather than used anywhere.
        n, δ, α = 60, 0.3, 1.0
        numpy.random.seed(31)
        xt, yt = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, n, SIGMA)

        numpy.random.seed(31)
        driver = arima.ar1(PHI, n, SIGMA)
        x_ref = arima.arima_from_arma(driver, 1)
        ξ = numpy.random.normal(0.0, SIGMA, n)
        ξ[0] = numpy.nan

        assert_array_equal(xt, x_ref)
        assert_array_equal(xt, numpy.cumsum(driver))

        y_ref = numpy.zeros(n)
        for i in range(1, n):
            Δx = x_ref[i] - x_ref[i - 1]
            y_ref[i] = δ + GAMMA * Δx + LAMBDA * (y_ref[i - 1] - α - BETA * x_ref[i - 1]) + ξ[i] + y_ref[i - 1]
        assert numpy.all(numpy.isfinite(y_ref))
        assert_array_equal(yt, y_ref)

    @pytest.mark.parametrize("δ,α,λ", [(0.0, 1.0, -0.5), (0.2, 1.0, -0.5), (0.3, 0.0, -0.25), (0.0, -2.0, -1.5)])
    def test_noise_free_relaxation_closed_form(self, δ, α, λ):
        # With σ = 0 there is no x process and y_t = (1+λ) y_{t-1} + δ - λ α,
        # y_0 = 0, which solves to y_t = (α - δ/λ)(1 - (1+λ)^t).
        n = 30
        xt, yt = ecm.ecm(0.5, δ, α, 1.0, 1.0, λ, n, σ=0.0)
        assert_array_equal(xt, numpy.zeros(n))
        t = numpy.arange(n)
        assert_allclose(yt, (α - δ / λ) * (1.0 - (1.0 + λ) ** t), atol=1e-12)

    def test_noise_free_fixed_point_is_alpha_minus_delta_over_lambda(self):
        _, yt = ecm.ecm(0.5, 0.4, 2.0, 1.0, 1.0, -0.5, 200, σ=0.0)
        assert yt[-1] == pytest.approx(2.0 - 0.4 / -0.5, abs=1e-10)

    def test_xt_increments_are_ar1(self, simulated):
        # Δx_t is the driving AR(1): lag-1 autocorrelation φ, variance σ²/(1-φ²).
        # n=2000: SE(acf1) ≈ sqrt((1-φ²)/n) ≈ 0.018, so abs=0.06 is ≈ 3.3 SE —
        # the largest deviation over 61 seeds was 0.052 and this seed gives
        # 0.6024. The sample variance has relative SE ≈ sqrt(2/n_eff) ≈ 0.06
        # with n_eff = n(1-φ)/(1+φ) = 500, so rel=0.15 is ≈ 2.5 SE (largest
        # deviation over the same 61 seeds: 0.104). Tight enough that a driver
        # simulated at φ=0.5 fails both.
        xt, _ = simulated
        dx = numpy.diff(xt)
        assert lag1_autocorr(dx) == pytest.approx(PHI, abs=0.06)
        assert dx.var() == pytest.approx(SIGMA**2 / (1 - PHI**2), rel=0.15)

    def test_recovered_innovations_are_white_with_variance_sigma2(self, simulated):
        # Inverting the documented recursion must give back iid N(0, σ²) noise:
        # ξ_t = Δy_t - δ - γ Δx_t - λ (y_{t-1} - α - β x_{t-1}).
        # SE(mean) = σ/sqrt(n) ≈ 0.022 (tol 0.15 ≈ 7 SE); SE(var) ≈ σ² sqrt(2/n)
        # ≈ 0.03 (tol 0.2 ≈ 6 SE); SE(acf1) ≈ 1/sqrt(n) ≈ 0.022 (tol 0.15).
        xt, yt = simulated
        ξ = numpy.diff(yt) - DELTA - GAMMA * numpy.diff(xt) - LAMBDA * (yt[:-1] - ALPHA - BETA * xt[:-1])
        assert ξ.mean() == pytest.approx(0.0, abs=0.15)
        assert ξ.var() == pytest.approx(SIGMA**2, abs=0.2)
        assert lag1_autocorr(ξ) == pytest.approx(0.0, abs=0.15)

    def test_cointegrating_residual_is_ar1_when_gamma_equals_beta(self):
        # With γ = β and δ = 0 the residual e_t = y_t - α - β x_t reduces to
        # e_t = (1+λ) e_{t-1} + ξ_t: acf1 = 1+λ, var = σ²/(1-(1+λ)²).
        # n=2000, 1+λ=0.6: SE(acf1) ≈ 0.018 (tol 0.15); var relative SE ≈ 0.06
        # with n_eff=500 (tol 0.4, observed max 0.16 over 40 seeds).
        β, λ, σ = 1.5, -0.4, 1.0
        xt, yt = ecm.ecm(0.6, 0.0, 0.0, β, β, λ, NPTS, σ)
        e = yt - β * xt
        assert lag1_autocorr(e) == pytest.approx(1 + λ, abs=0.15)
        assert e.var() == pytest.approx(σ**2 / (1 - (1 + λ) ** 2), rel=0.4)

    def test_ensemble_moments_match_closed_forms(self):
        # Ensemble of short paths. At T=60 the zero-initialised closed form
        # (359.77) and the stationary one (357.03) agree to 0.8%, so rtol=0.25
        # cannot tell them apart: this assertion only *bounds* the ensemble
        # variance (relative SE sqrt(2/nsim) ≈ 0.07 at nsim=400, so 0.25 ≈ 3.5
        # SE; observed max deviation 0.06 over 12 seeds). The burn-in question
        # is settled at small T by
        # test_xt_ensemble_variance_matches_zero_initialised_closed_form, where
        # the two derivations differ by 100% at k=0. The var_y/var_x and
        # cov/var_x ratios carry an O(1/T) bias from the stationary residual
        # and from y starting at 0 (≈ 6% and 3% at T=60), so tolerances
        # 0.15 / 0.1.
        nsim, T = 400, 60
        X = numpy.empty((nsim, T))
        Y = numpy.empty((nsim, T))
        for i in range(nsim):
            X[i], Y[i] = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, T, SIGMA)
        var_x = X[:, -1].var()
        var_y = Y[:, -1].var()
        cov_xy = numpy.cov(X[:, -1], Y[:, -1], bias=True)[0, 1]

        assert var_x == pytest.approx(zero_init_cumsum_ar1_var(PHI, SIGMA, T - 1), rel=0.25)
        assert var_y / var_x == pytest.approx(BETA**2, rel=0.15)
        assert cov_xy / var_x == pytest.approx(BETA, rel=0.1)

        # Mean of both processes is zero (δ = α = 0): within 4 SE of the ensemble mean.
        for Z in (X, Y):
            se = Z[:, -1].std() / numpy.sqrt(nsim)
            assert abs(Z[:, -1].mean()) < 4 * se

    def test_xt_ensemble_variance_matches_zero_initialised_closed_form(self):
        # The simulated x path is the cumsum of a burn-in-free AR(1), whose
        # variance profile is derived independently in
        # zero_init_cumsum_ar1_var. Checked over the whole grid, including the
        # transient at small t where Var(x_0) = σ² exactly.
        # rtol 0.25 ≈ 3.5 SE at nsim=500 (SE ≈ sqrt(2/500) = 0.063); observed
        # max deviation 0.11 over 12 seeds.
        nsim, T, φ, σ = 500, 25, 0.7, 1.3
        X = numpy.empty((nsim, T))
        for i in range(nsim):
            X[i], _ = ecm.ecm(φ, DELTA, ALPHA, BETA, GAMMA, LAMBDA, T, σ)
        expected = [zero_init_cumsum_ar1_var(φ, σ, k) for k in range(T)]
        assert_allclose(X.var(axis=0), expected, rtol=0.25)

    @pytest.mark.xfail(
        strict=True,
        reason="TWO independent defects keep the analytic curve off the simulated ensemble, and "
        "fixing either alone will not make this xpass. (1) Missing burn-in: xt_var(t) is the "
        "variance of a sum of t *stationary* AR(1) terms, but ecm.ecm builds x from "
        "arma_generate_sample with no burnin, so the driver starts at zero and x_0 = a_0 has "
        "variance σ², not 0. (2) Off-by-one between the grids: create_source returns t = k for "
        "k = 0..npts-1, while x[k] = Σ_{i=0}^{k} a_i is a sum of k+1 driver terms, so even a "
        "perfectly stationary driver would need xt_var evaluated at k+1, not k. Adopting the "
        "k+1 convention alone still leaves 96%/71%/53%/41% relative error at k=0..3 "
        "(φ=0.7, σ=1.3), and adding burn-in alone still leaves the index shift. The failure at "
        "index 0 is deterministic (σ² against 0) and seed-independent",
    )
    def test_xt_var_curve_matches_simulated_ensemble(self):
        nsim, T = 500, 6
        X = numpy.empty((nsim, T))
        for i in range(nsim):
            X[i], _ = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, T, SIGMA)
        t, expected = fecm.compute_xt_var(φ=PHI, σ=SIGMA, npts=T)
        assert_array_equal(t, numpy.arange(T, dtype=float))
        assert_allclose(X.var(axis=0), expected, rtol=0.25, atol=0.25)

    @pytest.mark.parametrize("δ,α,λ", [(0.3, 1.0, -0.5), (0.0, -2.0, -0.25), (-0.4, 0.7, -1.2)])
    def test_delta_and_alpha_enter_affinely_with_noise_present(self, δ, α, λ):
        # y is affine in (δ, α): on an identical RNG stream the difference
        # between a run with (δ, α) and one with (0, 0) obeys the noise-free
        # recursion d_t = (1+λ) d_{t-1} + δ - λα, d_0 = 0, whose solution is
        # (α - δ/λ)(1 - (1+λ)^t). Exact, so no statistical tolerance is needed.
        # The x path must be bit-identical too, since δ and α do not touch it.
        n = 40
        state = numpy.random.get_state()
        x_a, y_a = ecm.ecm(PHI, δ, α, BETA, GAMMA, λ, n, SIGMA)
        numpy.random.set_state(state)
        x_b, y_b = ecm.ecm(PHI, 0.0, 0.0, BETA, GAMMA, λ, n, SIGMA)

        assert_array_equal(x_a, x_b)
        t = numpy.arange(n)
        assert_allclose(y_a - y_b, (α - δ / λ) * (1.0 - (1.0 + λ) ** t), atol=1e-10)
        # The stationary offset the ECM term relaxes to is E[y - α - β x] = -δ/λ.
        # (1+λ)^39 ≤ 1.4e-5 for every λ above, so the residual transient is tiny.
        assert (y_a - y_b)[-1] == pytest.approx(α - δ / λ, abs=1e-3)


# ===========================================================================
# Façade layer: lib.data.impl.ecm
# ===========================================================================


class TestMeanFacades:
    @pytest.mark.parametrize("func", [fecm.compute_xt_mean, fecm.compute_yt_mean])
    def test_returns_unit_time_grid_and_zeros(self, func):
        t, μ = func(npts=7)
        assert isinstance(t, numpy.ndarray) and isinstance(μ, numpy.ndarray)
        assert_array_equal(t, numpy.arange(7, dtype=float))
        assert_array_equal(μ, numpy.zeros(7))
        assert μ.dtype.kind == "f"

    def test_zero_mean_is_backed_by_the_simulated_ensemble(self):
        # The zeros above are a restatement of numpy.full(npts, 0.0); this
        # establishes independently that zero really is the mean of the process
        # create_source builds at its δ = α = 0 defaults. nsim=400: the
        # ensemble mean of every coordinate lands within 4 SE of the façade
        # curve (largest observed |mean|/SE was 2.7 over 5 seeds). y_0 is
        # identically 0, so its SE is exactly 0 there and it is compared exactly.
        nsim, T = 400, 25
        X = numpy.empty((nsim, T))
        Y = numpy.empty((nsim, T))
        for i in range(nsim):
            _, values = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, npts=T)
            X[i], Y[i] = values
        for Z, func in ((X, fecm.compute_xt_mean), (Y, fecm.compute_yt_mean)):
            _, μ = func(npts=T)
            mean = Z.mean(axis=0)
            se = Z.std(axis=0) / numpy.sqrt(nsim)
            degenerate = se == 0.0
            assert_array_equal(mean[degenerate], μ[degenerate])
            assert numpy.all(numpy.abs(mean[~degenerate] - μ[~degenerate]) < 4.0 * se[~degenerate])

    @pytest.mark.parametrize("func", [fecm.compute_xt_mean, fecm.compute_yt_mean])
    def test_requires_npts(self, func):
        with pytest.raises(Exception, match="npts parameter is required"):
            func()

    @pytest.mark.parametrize("func", [fecm.compute_xt_mean, fecm.compute_yt_mean])
    @pytest.mark.xfail(
        strict=True,
        reason="compute_xt_mean/compute_yt_mean document Δt as the time-step width but pass "
        "both xmax=npts-1 and npts to create_space, which then ignores Δx; the grid is "
        "always unit-spaced and disagrees with compute_xt_var's grid for the same Δt",
    )
    def test_delta_t_sets_time_step(self, func):
        t, _ = func(npts=5, Δt=0.5)
        assert_allclose(numpy.diff(t), 0.5)
        t_var, _ = fecm.compute_xt_var(φ=0.5, npts=5, Δt=0.5)
        assert_allclose(t, t_var)

    @pytest.mark.xfail(
        strict=True,
        reason="compute_yt_mean returns numpy.full(npts, 0.0) unconditionally and takes no "
        "δ/α/λ kwargs, so it cannot express the ECM stationary mean E[y_t] = α - δ/λ. For "
        "δ=0.4, α=2.0, λ=-0.5 the simulated y relaxes to 2.8 (proved exactly by "
        "test_noise_free_fixed_point_is_alpha_minus_delta_over_lambda) while the façade still "
        "reports 0, so every notebook overlay of the mean on a source with δ≠0 or α≠0 is wrong",
    )
    def test_yt_mean_reflects_delta_and_alpha(self):
        _, μ = fecm.compute_yt_mean(npts=200, δ=0.4, α=2.0, λ=-0.5)
        assert μ[-1] == pytest.approx(2.0 - 0.4 / -0.5, abs=1e-3)


class TestVarianceFacades:
    def test_xt_var_contract_and_closed_form(self):
        φ, σ, npts = 0.6, 1.2, 8
        out = fecm.compute_xt_var(φ=φ, σ=σ, npts=npts)
        assert isinstance(out, tuple) and len(out) == 2
        t, v = out
        assert_array_equal(t, numpy.arange(npts, dtype=float))
        assert v.shape == (npts,)
        assert_allclose(v, [cumsum_ar1_var(φ, σ, n) for n in range(npts)], rtol=1e-12)
        assert v[0] == 0.0
        assert v[1] == pytest.approx(σ**2 / (1 - φ**2))

    def test_xt_var_sigma_defaults_to_one(self):
        t, v = fecm.compute_xt_var(φ=0.6, npts=6)
        assert_allclose(v, [cumsum_ar1_var(0.6, 1.0, n) for n in range(6)], rtol=1e-12)

    def test_xt_var_delta_t_spaces_grid(self):
        t, v = fecm.compute_xt_var(φ=0.6, npts=5, Δt=2.0)
        assert_array_equal(t, [0.0, 2.0, 4.0, 6.0, 8.0])
        assert_allclose(v, [cumsum_ar1_var(0.6, 1.0, n) for n in (0, 2, 4, 6, 8)], rtol=1e-12)

    def test_xt_var_tmax_sets_grid_end(self):
        t, v = fecm.compute_xt_var(φ=0.6, npts=4, tmax=9)
        assert_array_equal(t, [0.0, 3.0, 6.0, 9.0])
        assert_allclose(v, [cumsum_ar1_var(0.6, 1.0, n) for n in (0, 3, 6, 9)], rtol=1e-12)

    @pytest.mark.parametrize(
        "func,extra",
        [
            (fecm.compute_xt_var, {}),
            (fecm.compute_yt_var, {"β": 1.5}),
            (fecm.compute_cov, {"β": 1.5}),
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="when tmax and npts are both supplied create_space takes neither derivation "
        "branch (xmax is not None and npts is not None) and silently discards Δx, so the "
        "documented Δt kwarg is ignored: compute_xt_var(φ=0.6, npts=5, tmax=8, Δt=0.5) returns "
        "the grid [0, 2, 4, 6, 8] instead of honouring Δt. Same root cause as the "
        "compute_xt_mean grid bug. Over-determined input should either raise or honour Δt",
    )
    def test_tmax_and_delta_t_together_do_not_drop_delta_t(self, func, extra):
        try:
            t, _ = func(φ=0.6, npts=5, tmax=8, Δt=0.5, **extra)
        except Exception:
            # Rejecting the over-determined call outright is an acceptable
            # contract too, so an exception counts as fixed.
            return
        assert_allclose(numpy.diff(t), 0.5)

    @pytest.mark.parametrize("missing", ["φ", "npts"])
    def test_xt_var_required_kwargs(self, missing):
        kwargs = {"φ": 0.6, "npts": 5}
        del kwargs[missing]
        with pytest.raises(Exception, match=f"{missing} parameter is required"):
            fecm.compute_xt_var(**kwargs)

    def test_yt_var_contract_and_closed_form(self):
        φ, σ, β, npts = 0.6, 1.2, -0.8, 8
        t, v = fecm.compute_yt_var(φ=φ, σ=σ, β=β, npts=npts)
        assert_array_equal(t, numpy.arange(npts, dtype=float))
        assert_allclose(v, [β**2 * cumsum_ar1_var(φ, σ, n) for n in range(npts)], rtol=1e-12)

    def test_yt_var_delta_t_grid_matches_closed_form(self):
        # φ, σ and β are mutually distinct so an argument swap in the façade's
        # call to ecm.yt_var shows up, and the expected values come from the
        # hand-derived form rather than from the model layer the façade calls.
        φ, σ, β = 0.6, 1.2, -0.8
        t, v = fecm.compute_yt_var(φ=φ, σ=σ, β=β, npts=10, Δt=2.0)
        assert_array_equal(t, numpy.arange(10) * 2.0)
        assert_allclose(v, [β**2 * cumsum_ar1_var(φ, σ, 2 * n) for n in range(10)], rtol=1e-12)

    @pytest.mark.parametrize("missing", ["φ", "β", "npts"])
    def test_yt_var_required_kwargs(self, missing):
        kwargs = {"φ": 0.6, "β": 1.5, "npts": 5}
        del kwargs[missing]
        with pytest.raises(Exception, match=f"{missing} parameter is required"):
            fecm.compute_yt_var(**kwargs)

    def test_cov_contract_and_closed_form(self):
        φ, σ, β, npts = 0.6, 1.2, -0.8, 8
        t, c = fecm.compute_cov(φ=φ, σ=σ, β=β, npts=npts)
        assert_array_equal(t, numpy.arange(npts, dtype=float))
        assert_allclose(c, [β * cumsum_ar1_var(φ, σ, n) for n in range(npts)], rtol=1e-12)

    def test_cov_tmax_grid_matches_closed_form(self):
        φ, σ, β = 0.6, 1.2, -0.8
        t, c = fecm.compute_cov(φ=φ, σ=σ, β=β, npts=10, tmax=18)
        assert_array_equal(t, numpy.arange(10) * 2.0)
        assert_allclose(c, [β * cumsum_ar1_var(φ, σ, 2 * n) for n in range(10)], rtol=1e-12)

    @pytest.mark.parametrize("missing", ["φ", "β", "npts"])
    def test_cov_required_kwargs(self, missing):
        kwargs = {"φ": 0.6, "β": 1.5, "npts": 5}
        del kwargs[missing]
        with pytest.raises(Exception, match=f"{missing} parameter is required"):
            fecm.compute_cov(**kwargs)

    def test_facades_share_time_grid(self):
        kwargs = dict(φ=0.6, β=1.5, npts=7, Δt=0.5)
        t_x, _ = fecm.compute_xt_var(**kwargs)
        t_y, _ = fecm.compute_yt_var(**kwargs)
        t_c, _ = fecm.compute_cov(**kwargs)
        assert_array_equal(t_x, t_y)
        assert_array_equal(t_x, t_c)


class TestCreateSource:
    def test_contract(self):
        out = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, npts=50)
        assert isinstance(out, tuple) and len(out) == 2
        t, values = out
        assert_array_equal(t, numpy.arange(50, dtype=float))
        assert isinstance(values, numpy.ndarray)
        assert values.shape == (2, 50)
        assert values.dtype.kind == "f"
        assert values[1, 0] == 0.0  # y_0 = 0

    def test_default_npts_is_1000(self):
        t, values = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA)
        assert t.shape == (1000,)
        assert values.shape == (2, 1000)

    @pytest.mark.parametrize("missing", ["φ", "β", "γ", "λ"])
    def test_required_kwargs(self, missing):
        kwargs = {"φ": PHI, "β": BETA, "γ": GAMMA, "λ": LAMBDA, "npts": 20}
        del kwargs[missing]
        with pytest.raises(Exception, match=f"{missing} parameter is required"):
            fecm.create_source(**kwargs)

    def test_agrees_with_model_layer_using_defaults(self):
        # Same RNG stream → the façade must reproduce ecm.ecm with δ=0, α=0, σ=1.
        numpy.random.seed(11)
        _, values = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, npts=100)
        numpy.random.seed(11)
        xt, yt = ecm.ecm(PHI, 0.0, 0.0, BETA, GAMMA, LAMBDA, 100, 1.0)
        assert_array_equal(values[0], xt)
        assert_array_equal(values[1], yt)

    def test_agrees_with_model_layer_with_all_kwargs(self):
        # Exercises the positional plumbing of every optional parameter.
        numpy.random.seed(12)
        _, values = fecm.create_source(φ=0.3, δ=0.2, α=1.5, β=-0.7, γ=0.9, λ=-0.6, σ=2.0, npts=80)
        numpy.random.seed(12)
        xt, yt = ecm.ecm(0.3, 0.2, 1.5, -0.7, 0.9, -0.6, 80, 2.0)
        assert_array_equal(values[0], xt)
        assert_array_equal(values[1], yt)

    def test_sigma_scales_paths(self):
        # Noise enters linearly with δ = α = 0, so doubling σ doubles both paths.
        numpy.random.seed(13)
        _, v1 = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, σ=1.0, npts=60)
        numpy.random.seed(13)
        _, v2 = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, σ=2.0, npts=60)
        assert_allclose(v2, 2.0 * v1, rtol=1e-10)


class TestDegenerateSizes:
    def test_ecm_with_a_single_point(self):
        # npts=1: the i-loop body never executes, so y is exactly [0.0] while x
        # is the single (integrated) driver draw. Nothing may raise or return
        # an empty array.
        xt, yt = ecm.ecm(PHI, 0.5, 1.0, BETA, GAMMA, LAMBDA, 1, SIGMA)
        assert xt.shape == (1,) and yt.shape == (1,)
        assert_array_equal(yt, [0.0])
        assert numpy.isfinite(xt[0])

    def test_create_source_with_a_single_point(self):
        t, values = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, npts=1)
        assert_array_equal(t, [0.0])
        assert values.shape == (2, 1)
        assert values[1, 0] == 0.0

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (fecm.compute_xt_mean, {}),
            (fecm.compute_yt_mean, {}),
            (fecm.compute_xt_var, {"φ": PHI}),
            (fecm.compute_yt_var, {"φ": PHI, "β": BETA}),
            (fecm.compute_cov, {"φ": PHI, "β": BETA}),
        ],
    )
    def test_compute_facades_with_a_single_point(self, func, kwargs):
        # A one-point grid is [0.0], where every mean and every accumulated
        # variance/covariance is 0 — no empty-linspace or off-by-one crash.
        t, v = func(npts=1, **kwargs)
        assert_array_equal(t, [0.0])
        assert v.shape == (1,)
        assert v[0] == 0.0


class TestBetaEstimate:
    def test_recovers_beta(self, simulated):
        # OLS of y on an I(1) regressor is superconsistent (error O(1/n)):
        # observed |β̂ - β| ≤ 0.009 over 40 seeds at n=2000; tol 0.05.
        # The reported error must be the textbook OLS slope SE
        # s/sqrt(Σ(x-x̄)²), s² = SSR/(n-2), computed here from the normal
        # equations rather than read off statsmodels (value ≈ 0.00216, so the
        # band below is a sanity net around an exact check).
        xt, yt = simulated
        _, result = fecm.compute_beta_estimate(yt, xt)
        assert result.params[0].est == pytest.approx(BETA, abs=0.05)
        assert result.r2 > 0.99

        x̄, ȳ = xt.mean(), yt.mean()
        Sxx = ((xt - x̄) ** 2).sum()
        slope = ((xt - x̄) * (yt - ȳ)).sum() / Sxx
        resid = yt - (ȳ - slope * x̄) - slope * xt
        s2 = resid @ resid / (len(xt) - 2)
        assert result.params[0].err == pytest.approx(numpy.sqrt(s2 / Sxx), rel=1e-8)
        assert result.const.err == pytest.approx(numpy.sqrt(s2 * (1.0 / len(xt) + x̄**2 / Sxx)), rel=1e-8)
        assert 1e-3 < result.params[0].err < 5e-3

    def test_matches_independent_least_squares(self, simulated):
        xt, yt = simulated
        report, result = fecm.compute_beta_estimate(yt, xt)
        slope, intercept = numpy.polyfit(xt, yt, 1)
        assert result.params[0].est == pytest.approx(slope, rel=1e-8)
        assert result.const.est == pytest.approx(intercept, rel=1e-8, abs=1e-8)
        # statsmodels report and the result model describe the same fit
        params = numpy.asarray(report.params)
        bse = numpy.asarray(report.bse)
        assert result.const.est == params[0] and result.params[0].est == params[1]
        assert result.const.err == bse[0] and result.params[0].err == bse[1]
        assert result.r2 == report.rsquared

    def test_result_structure_and_transforms(self, simulated):
        xt, yt = simulated
        _, result = fecm.compute_beta_estimate(yt, xt)
        assert isinstance(result, OLSResult)
        assert result.est_model == EstModel.OLS
        assert len(result.params) == 1
        assert result.const.param_type == OLSParamType.OLS_CONST.value
        assert result.params[0].param_type == OLSParamType.OLS_PARAM.value

        assert result.model == r"$\hat{\alpha} + \hat{\beta} x_t$"
        assert result.param_transforms is not None and len(result.param_transforms) == 1
        β_tr = result.param_transforms[0]
        assert isinstance(β_tr, OLSTransform)
        assert β_tr.param.est == result.params[0].est
        assert β_tr.param.err == result.params[0].err
        assert β_tr.param.est_label == r"$\hat{\beta}$"
        assert β_tr.param.err_label == r"$\sigma_{\hat{\beta}}$"
        assert β_tr.param.param_type == OLSParamType.TRANS_PARAM.value
        assert β_tr.param.est_id == result.est_id

        assert result.const_transform is not None
        α_tr = result.const_transform
        assert α_tr.param.est == result.const.est
        assert α_tr.param.err == result.const.err
        assert α_tr.param.est_label == r"$\hat{\alpha}$"
        assert α_tr.param.err_label == r"$\sigma_{\hat{\alpha}}$"
        assert α_tr.param.param_type == OLSParamType.TRANS_CONST.value

    def test_result_serializes_to_json(self, simulated):
        import json

        xt, yt = simulated
        _, result = fecm.compute_beta_estimate(yt, xt)
        data = json.loads(result.to_json())
        assert data["est_id"] == result.est_id
        assert data["params"][0]["est"] == result.params[0].est
        assert data["param_transforms"][0]["param"]["est_label"] == r"$\hat{\beta}$"
        assert data["model"] == result.model


class TestGammaLambdaEstimate:
    def test_recovers_gamma_and_lambda(self, simulated):
        # Regression of Δy on (Δx, ε_{t-1}). At n=2000 the coefficient SEs are
        # ≈ 0.02 (γ) and ≈ 0.014 (λ); observed max errors over 40 seeds were
        # 0.04 and 0.02. Tolerances 0.15 / 0.1 are ≥ 5 SE.
        xt, yt = simulated
        _, β_result = fecm.compute_beta_estimate(yt, xt)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, β_result.params[0].est)
        assert len(result.params) == 2
        assert result.params[0].est == pytest.approx(GAMMA, abs=0.15)
        assert result.params[1].est == pytest.approx(LAMBDA, abs=0.1)

    def test_matches_independent_least_squares(self, simulated):
        xt, yt = simulated
        est_beta = 1.48
        report, result = fecm.compute_gamma_lambda_estimate(yt, xt, est_beta)
        ε = yt - est_beta * xt
        dx, dy = numpy.diff(xt), numpy.diff(yt)
        A = numpy.column_stack([numpy.ones(len(dx)), dx, ε[:-1]])
        coef, *_ = numpy.linalg.lstsq(A, dy, rcond=None)
        assert result.const.est == pytest.approx(coef[0], abs=1e-8)
        assert result.params[0].est == pytest.approx(coef[1], rel=1e-8)
        assert result.params[1].est == pytest.approx(coef[2], rel=1e-8)
        params = numpy.asarray(report.params)
        assert_allclose([result.const.est, result.params[0].est, result.params[1].est], params)

    def test_standard_errors_match_independent_covariance(self, simulated):
        # SE_i = sqrt(s² (AᵀA)⁻¹_ii) with s² = SSR/(n - 3), built from the
        # design matrix here rather than read off the statsmodels report — the
        # two-parameter path never had its errors checked, only its estimates.
        # Observed 0.0224 (const), 0.0199 (γ̂), 0.0097 (λ̂) at n=2000.
        xt, yt = simulated
        est_beta = 1.48
        report, result = fecm.compute_gamma_lambda_estimate(yt, xt, est_beta)
        ε = yt - est_beta * xt
        dx, dy = numpy.diff(xt), numpy.diff(yt)
        A = numpy.column_stack([numpy.ones(len(dx)), dx, ε[:-1]])
        coef, *_ = numpy.linalg.lstsq(A, dy, rcond=None)
        resid = dy - A @ coef
        s2 = resid @ resid / (len(dy) - A.shape[1])
        se = numpy.sqrt(numpy.diag(s2 * numpy.linalg.inv(A.T @ A)))
        assert_allclose([result.const.err, result.params[0].err, result.params[1].err], se, rtol=1e-8)
        assert_allclose(numpy.asarray(report.bse), se, rtol=1e-8)
        assert numpy.all(se > 0.0)

    def test_result_surface_and_json_round_trip(self, simulated):
        import json

        xt, yt = simulated
        report, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert isinstance(result, OLSResult)
        assert result.est_model == EstModel.OLS
        assert result.r2 == report.rsquared
        assert 0.0 < result.r2 < 1.0

        # Index bookkeeping: both regressors are OLS_PARAM and keep the column
        # they occupied in the design matrix; every parameter carries the
        # result's est_id, transforms included.
        assert result.const.param_type == OLSParamType.OLS_CONST.value
        assert [p.param_type for p in result.params] == [OLSParamType.OLS_PARAM.value] * 2
        assert [p.column for p in result.params] == [1, 2]
        assert all(p.row == 0 and p.order == 0 for p in result.params)
        assert result.const.row == 0 and result.const.column == 0
        transforms = [tr.param for tr in result.param_transforms] + [result.const_transform.param]
        for p in [result.const, *result.params, *transforms]:
            assert p.est_id == result.est_id
        assert all(p.order == 1 and p.row == 0 and p.column == 0 for p in transforms)

        data = json.loads(result.to_json())
        assert data["est_id"] == result.est_id
        assert data["est_model"] == EstModel.OLS.value
        assert data["r2"] == result.r2
        assert data["model"] == result.model
        assert [p["est"] for p in data["params"]] == [p.est for p in result.params]
        assert [p["err"] for p in data["params"]] == [p.err for p in result.params]
        assert data["const"]["est"] == result.const.est and data["const"]["err"] == result.const.err
        assert len(data["param_transforms"]) == 2
        assert data["param_transforms"][0]["param"]["est_label"] == r"$\hat{\gamma}$"
        assert data["const_transform"]["param"]["est_id"] == result.est_id
        # each serialized parameter reconstructs through ParamEst.from_dict
        for p, d in zip(result.params, data["params"], strict=True):
            rebuilt = ParamEst.from_dict(d)
            assert (rebuilt.est, rebuilt.err, rebuilt.est_id, rebuilt.param_type, rebuilt.column) == (
                p.est,
                p.err,
                p.est_id,
                p.param_type,
                p.column,
            )

    def test_transform_structure(self, simulated):
        xt, yt = simulated
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert result.param_transforms is not None and len(result.param_transforms) == 2
        γ_tr, λ_tr = result.param_transforms
        assert γ_tr.param.est == result.params[0].est
        assert γ_tr.param.err == result.params[0].err
        assert γ_tr.param.est_label == r"$\hat{\gamma}$"
        assert γ_tr.param.err_label == r"$\sigma_{\hat{\gamma}}$"
        assert γ_tr.param.param_type == OLSParamType.TRANS_PARAM.value
        assert λ_tr.param.est_label == r"$\hat{\lambda}$"
        assert λ_tr.param.err_label == r"$\sigma_{\hat{\lambda}}$"
        assert λ_tr.param.param_type == OLSParamType.TRANS_PARAM.value
        assert result.const_transform is not None
        assert result.const_transform.param.est == result.const.est
        assert result.const_transform.param.param_type == OLSParamType.TRANS_CONST.value

    @pytest.mark.xfail(
        strict=True,
        reason="__add_gamma_lambda_transform builds the λ transform from result.params[0] "
        "(the γ estimate) instead of result.params[1], so the reported λ̂ is the γ̂ value",
    )
    def test_lambda_transform_reports_lambda_estimate(self, simulated):
        xt, yt = simulated
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        λ_tr = result.param_transforms[1]
        assert λ_tr.param.est == result.params[1].est
        assert λ_tr.param.err == result.params[1].err

    @pytest.mark.xfail(
        strict=True,
        reason="__add_gamma_lambda_transform labels the constant transform with the λ̂ labels, "
        "duplicating the second parameter transform's. The constant of the Δy regression "
        "estimates δ - λα, so its label has to name δ — anything else (including another "
        "wrong label) is still a bug, which is why this asserts the δ form positively rather "
        "than merely differing from the λ̂ label",
    )
    def test_const_transform_is_labelled_for_delta(self, simulated):
        xt, yt = simulated
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        λ_tr = result.param_transforms[1]
        est_label = result.const_transform.param.est_label
        err_label = result.const_transform.param.err_label
        assert r"\delta" in est_label
        assert r"\delta" in err_label
        assert est_label != λ_tr.param.est_label

    @pytest.mark.xfail(
        strict=True,
        reason="__add_gamma_lambda_transform hard-codes model = r'$\\hat{\\alpha} + "
        "\\hat{\\beta} x_t$' — the β regression's formula copied verbatim (lib/data/impl/"
        "ecm.py:308) — for a regression of Δy on (Δx, ε_{t-1}). "
        "compute_gamma_lambda_estimate(...).model is byte-identical to "
        "compute_beta_estimate(...).model, so every report of the γ/λ fit renders the wrong "
        "formula. It should name γ̂, λ̂ and Δx",
    )
    def test_model_formula_describes_the_delta_y_regression(self, simulated):
        xt, yt = simulated
        _, β_result = fecm.compute_beta_estimate(yt, xt)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert result.model != β_result.model
        assert r"\gamma" in result.model and r"\lambda" in result.model


class TestEndToEnd:
    def test_facade_pipeline_recovers_parameters(self):
        # Notebook-style flow: create_source → β̂ → (γ̂, λ̂), with non-default σ.
        σ = 0.5
        _, values = fecm.create_source(φ=PHI, β=BETA, γ=GAMMA, λ=LAMBDA, σ=σ, npts=NPTS)
        xt, yt = values
        _, β_result = fecm.compute_beta_estimate(yt, xt)
        β_hat = β_result.params[0].est
        _, γλ_result = fecm.compute_gamma_lambda_estimate(yt, xt, β_hat)
        assert β_hat == pytest.approx(BETA, abs=0.05)
        assert γλ_result.params[0].est == pytest.approx(GAMMA, abs=0.15)
        assert γλ_result.params[1].est == pytest.approx(LAMBDA, abs=0.1)
        # residual variance of the Δy regression is σ² (tol ≈ 6 SE at n=2000)
        report, _ = fecm.compute_gamma_lambda_estimate(yt, xt, β_hat)
        assert numpy.asarray(report.resid).var() == pytest.approx(σ**2, rel=0.2)


class TestModuleHygiene:
    @pytest.mark.parametrize(
        "module,dead",
        [
            pytest.param(ecm, ["sm", "tsa"], id="lib.models.ecm"),
            pytest.param(fecm, ["tsa", "verify_type"], id="lib.data.impl.ecm"),
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="Both ECM modules carry imports nothing in them references: lib/models/ecm.py "
        "imports statsmodels.api as sm and statsmodels.tsa as tsa (lines 9-10) and uses "
        "neither — the AR(1) driver comes from lib.models.arima — and lib/data/impl/ecm.py "
        "imports statsmodels.tsa as tsa and lib.utils.verify_type without referencing either. "
        "Dead imports pull statsmodels into the import graph of the pure-model module and "
        "there is no lint gate to catch them",
    )
    def test_no_dead_imports(self, module, dead):
        # AST scan of the module's own source: every imported binding must be
        # referenced by some Name node (annotations included, since these
        # modules do not stringify them).
        found = unused_imports(module)
        assert found == [], f"unused imports {found} (expected dead set {dead})"
