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

Validation stance: none of the model functions validate their inputs, and the
docstrings state preconditions (|φ| < 1, integer t ≥ 0) rather than an error
contract. The suite therefore *documents* what the code does outside those
preconditions (``TestOutOfDomainInputs``, ``TestDegenerateSizes``) instead of
asserting a guard that was never promised; ``xfail`` is reserved for cases
where the code contradicts its own documentation.

Every simulation draws from numpy's global RNG, which ``conftest`` reseeds
before each test.
"""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lib.data.impl import ecm as fecm
from lib.data.param_est import EstModel, OLSParamType, OLSResult, OLSTransform, ParamEst
from lib.models import arima, ecm
from helpers import present

# ---------------------------------------------------------------------------
# Independent closed forms (derived by hand, not by calling the library)
# ---------------------------------------------------------------------------


def cumsum_ar1_var(φ: float, σ: float, n: int) -> float:
    """Var(Σ_{i=1}^{n} a_i) for stationary AR(1) a_i with acf φ^k.

    Σ_{k=1}^{n-1} (n-k) φ^k has the closed form φ[n(1-φ) - (1-φ^n)]/(1-φ)²,
    so the variance is γ0 [n + 2 φ (n(1-φ) - (1-φ^n)) / (1-φ)²] with
    γ0 = σ²/(1-φ²).

    The identity is algebraic, so it is also the analytic continuation of the
    library's summation to |φ| > 1 — where it is negative, and no longer a
    variance. ``TestOutOfDomainInputs`` uses it in exactly that role.
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


def ols_line(x: numpy.ndarray, y: numpy.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the OLS fit of y on x, from the normal equations."""
    x̄, ȳ = x.mean(), y.mean()
    slope = ((x - x̄) * (y - ȳ)).sum() / ((x - x̄) ** 2).sum()
    return float(slope), float(ȳ - slope * x̄)


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
        # Parametrised over σ, so this also pins the σ² scaling against the
        # hand-derived form rather than against another call to xt_var.
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

    def test_monotone_in_t_for_positive_phi(self):
        t = numpy.arange(0, 40, dtype=float)
        assert numpy.all(numpy.diff(ecm.xt_var(0.5, 1.0, t)) > 0)

    def test_truncates_non_integer_t(self):
        # Time values are cast with int(), so 2.9 evaluates as n=2. The model is
        # a discrete-time process, so only integer t is defined; the cast is a
        # floor, not an interpolation. Consequence: any grid finer than one time
        # unit is a staircase — see
        # TestVarianceFacades.test_sub_unit_delta_t_duplicates_values, which
        # pins what that does to the documented Δt kwarg.
        assert_allclose(ecm.xt_var(0.5, 1.0, numpy.array([2.9])), [cumsum_ar1_var(0.5, 1.0, 2)])
        assert_allclose(ecm.xt_var(0.5, 1.0, numpy.array([2.0, 2.5, 2.9])), [cumsum_ar1_var(0.5, 1.0, 2)] * 3)

    def test_output_shape_and_dtype(self):
        t = numpy.linspace(0, 9, 10)
        out = ecm.xt_var(0.5, 1.0, t)
        assert isinstance(out, numpy.ndarray)
        assert out.shape == (10,)
        assert out.dtype.kind == "f"
        assert ecm.xt_var(0.5, 1.0, numpy.array([])).shape == (0,)


class TestOutOfDomainInputs:
    """What the closed forms do outside their documented preconditions.

    ``xt_var``/``yt_var``/``cov`` document ``|φ| < 1`` and are only defined on
    integer times t ≥ 0, and they validate neither. These tests are the record
    of the current, unguarded behaviour, so that any future guard (raise, clamp)
    shows up as a red test rather than as a silent change of meaning. They are
    deliberately *not* xfails: no documented contract is being violated.
    """

    @pytest.mark.parametrize("φ", [1.0, -1.0])
    @pytest.mark.parametrize(
        "func,args",
        [(ecm.xt_var, ()), (ecm.yt_var, (1.5,)), (ecm.cov, (1.5,))],
    )
    def test_unit_root_phi_raises_zero_division(self, func, args, φ):
        # lib/models/ecm.py:38 divides by (1 - φ²) with Python floats, so the
        # boundary of the documented |φ| < 1 precondition raises rather than
        # returning inf — and it raises for *every* t, t = 0 included.
        with pytest.raises(ZeroDivisionError):
            func(φ, 1.0, *args, numpy.array([0.0, 1.0, 5.0]))

    def test_explosive_phi_returns_negative_variances(self):
        # |φ| > 1: the summation still evaluates, and equals the analytic
        # continuation of the stationary closed form — which is negative for
        # every t ≥ 1 because γ0 = σ²/(1-φ²) < 0. So the function silently
        # returns something that cannot be a variance.
        φ, σ = 1.5, 1.0
        t = numpy.arange(0, 6, dtype=float)
        out = ecm.xt_var(φ, σ, t)
        assert_allclose(out, [cumsum_ar1_var(φ, σ, int(n)) for n in t], rtol=1e-12)
        assert out[0] == 0.0
        assert numpy.all(out[1:] < 0.0)
        assert_allclose(out[1:3], [-0.8, -4.0], rtol=1e-12)

    @pytest.mark.parametrize("β", [1.5, -0.8])
    def test_explosive_phi_propagates_through_yt_var_and_cov(self, β):
        # yt_var and cov are β²- and β-scalings of xt_var, so they inherit the
        # defect: yt_var stays negative for either sign of β, while cov flips
        # sign with β — a "covariance" whose sign no longer tracks β.
        φ, σ = 1.5, 1.0
        t = numpy.arange(0, 6, dtype=float)
        ref = numpy.array([cumsum_ar1_var(φ, σ, int(n)) for n in t])
        assert_allclose(ecm.yt_var(φ, σ, β, t), β**2 * ref, rtol=1e-12)
        assert_allclose(ecm.cov(φ, σ, β, t), β * ref, rtol=1e-12)
        assert numpy.all(ecm.yt_var(φ, σ, β, t)[1:] < 0.0)
        assert numpy.all(numpy.sign(ecm.cov(φ, σ, β, t)[1:]) == -numpy.sign(β))

    def test_negative_t_returns_negative_values(self):
        # t < 0 is outside the domain and unreachable through any façade (every
        # compute_* builds its grid with create_space(xmin=0, ...)). The inner
        # `range(1, npts)` is empty for npts <= 0, so the result collapses to
        # npts·σ²/(1-φ²) — negative, and int()-truncated toward zero, so -2.9
        # evaluates as -2 rather than -3.
        out = ecm.xt_var(0.5, 1.0, numpy.array([-3.0, -2.9, -1.0, 0.0]))
        γ0 = 1.0 / (1.0 - 0.5**2)
        assert_allclose(out, [-3 * γ0, -2 * γ0, -1 * γ0, 0.0], rtol=1e-12)
        assert numpy.all(out[:3] < 0.0)


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
    def test_simulated_correlation_approaches_sign_of_beta(self, β):
        # The falsifiable form of "the implied correlation is ±1" (which the
        # three closed forms assert algebraically, cov² = var_x var_y, since
        # yt_var and cov are one-line scalings of xt_var): x and y are
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
        # ecm.ecm draws the AR(1) driver first, integrates it, and only then
        # draws the ξ stream — and it burns ξ[0], which the i-loop never reads.
        # Replaying the same global stream must reproduce both paths
        # bit-for-bit; the NaN planted at ξ[0] proves that first draw is
        # discarded rather than used anywhere.
        #
        # x_ref is numpy.cumsum(driver), NOT arima.arima_from_arma(driver, 1):
        # the latter is the very helper ecm.ecm calls, so comparing against it
        # would validate no value. The y recursion below is a transcription of
        # the model documented in this file's header, so it pins the code
        # against the documented model.
        n, δ, α = 60, 0.3, 1.0
        numpy.random.seed(31)
        xt, yt = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, n, SIGMA)

        numpy.random.seed(31)
        driver = arima.ar1(PHI, n, SIGMA)
        x_ref = numpy.cumsum(driver)
        ξ = numpy.random.normal(0.0, SIGMA, n)
        ξ[0] = numpy.nan

        assert_array_equal(xt, x_ref)

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
        # Both quantities are random variables with heavy-ish tails at fixed n,
        # so the tolerances are set outside the range observed over 1500 seeds
        # rather than at a nominal SE multiple. n=2000: SE(acf1) ≈
        # sqrt((1-φ²)/n) ≈ 0.018, largest |dev| over 1500 seeds was 0.065, so
        # abs=0.08 clears the observed range. The sample variance has relative
        # SE ≈ sqrt(2/n_eff) ≈ 0.06 with n_eff = n(1-φ)/(1+φ) = 500, largest
        # relative deviation over the same 1500 seeds was 0.159, so rel=0.25
        # clears it. Both still discriminate the stated alternative: a driver
        # simulated at φ=0.5 is 0.10 away in acf and 26% away in variance.
        xt, _ = simulated
        dx = numpy.diff(xt)
        assert lag1_autocorr(dx) == pytest.approx(PHI, abs=0.08)
        assert dx.var() == pytest.approx(SIGMA**2 / (1 - PHI**2), rel=0.25)

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
        "always unit-spaced and disagrees with compute_xt_var's grid for the same Δt. Unlike "
        "the over-determined create_space calls a *caller* can make, here the façade itself "
        "manufactures the conflicting xmax from npts and so silently voids its own documented "
        "kwarg (lib/data/impl/ecm.py:43,66)",
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
        # σ enters as σ², so a wrong default would rescale the whole curve.
        _, v2 = fecm.compute_xt_var(φ=0.6, σ=2.0, npts=6)
        assert_allclose(v2, 4.0 * v, rtol=1e-12)

    def test_yt_var_sigma_defaults_to_one(self):
        # The default was only ever pinned for compute_xt_var; yt_var's own
        # get_param_default_if_missing("σ", 1.0, ...) is checked here against
        # the hand-derived form, not against another façade call.
        φ, β = 0.6, -0.8
        t, v = fecm.compute_yt_var(φ=φ, β=β, npts=6)
        assert_allclose(v, [β**2 * cumsum_ar1_var(φ, 1.0, n) for n in range(6)], rtol=1e-12)
        _, v2 = fecm.compute_yt_var(φ=φ, β=β, σ=2.0, npts=6)
        assert_allclose(v2, 4.0 * v, rtol=1e-12)

    def test_cov_sigma_defaults_to_one(self):
        φ, β = 0.6, -0.8
        t, c = fecm.compute_cov(φ=φ, β=β, npts=6)
        assert_allclose(c, [β * cumsum_ar1_var(φ, 1.0, n) for n in range(6)], rtol=1e-12)
        _, c2 = fecm.compute_cov(φ=φ, β=β, σ=2.0, npts=6)
        assert_allclose(c2, 4.0 * c, rtol=1e-12)

    def test_xt_var_delta_t_spaces_grid(self):
        t, v = fecm.compute_xt_var(φ=0.6, npts=5, Δt=2.0)
        assert_array_equal(t, [0.0, 2.0, 4.0, 6.0, 8.0])
        assert_allclose(v, [cumsum_ar1_var(0.6, 1.0, n) for n in (0, 2, 4, 6, 8)], rtol=1e-12)

    @pytest.mark.parametrize(
        "func,extra,scale",
        [
            (fecm.compute_xt_var, {}, 1.0),
            (fecm.compute_yt_var, {"β": 1.5}, 1.5**2),
            (fecm.compute_cov, {"β": 1.5}, 1.5),
        ],
    )
    def test_sub_unit_delta_t_duplicates_values(self, func, extra, scale):
        # Documents the consequence of ecm.xt_var's int(t) cast (lib/models/
        # ecm.py:40) for the documented Δt kwarg: below one time unit the grid
        # is refined but the curve is not, so the result is a staircase and Δt
        # carries no information. compute_xt_var(φ=0.6, npts=6, Δt=0.5) returns
        # t = [0, 0.5, 1, 1.5, 2, 2.5] against v = [0, 0, 1.5625, 1.5625, 5, 5]
        # — in particular Var is reported as 0 at t=0.5, not as the (undefined)
        # value between 0 and σ²/(1-φ²). Δt ≥ 1 and integer-valued is the only
        # regime in which the kwarg means what its docstring says.
        t, v = func(φ=0.6, npts=6, Δt=0.5, **extra)
        assert_array_equal(t, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        expected = [scale * cumsum_ar1_var(0.6, 1.0, n) for n in (0, 0, 1, 1, 2, 2)]
        assert_allclose(v, expected, rtol=1e-12)
        assert v[0] == v[1] and v[2] == v[3] and v[4] == v[5]

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
    def test_tmax_and_npts_together_ignore_delta_t(self, func, extra):
        # Documentation test, not a defect report. Supplying tmax, npts and Δt
        # over-determines a two-parameter linspace; create_space
        # (lib/utils.py:150-163) documents only "xmax or npts is required" and
        # derives whichever of the two is missing, so with both present it
        # takes neither branch and Δx never enters — the same precedence
        # numpy.linspace itself has. The conflict here is the caller's, so the
        # current behaviour is pinned rather than xfailed. (Contrast
        # TestMeanFacades.test_delta_t_sets_time_step, where the façade
        # fabricates the conflicting xmax itself and voids its own kwarg.)
        t, _ = func(φ=0.6, npts=5, tmax=8, Δt=0.5, **extra)
        assert_array_equal(t, [0.0, 2.0, 4.0, 6.0, 8.0])

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


class TestDegenerateEstimatorInput:
    """The two estimator façades on samples too short (or mis-shaped) to fit.

    Neither validates its input, so these record what the underlying
    statsmodels call does. They are documentation of the current failure modes,
    not defect reports: the functions promise nothing about n ≤ 3.
    """

    @staticmethod
    def _short(n):
        xt, yt = ecm.ecm(PHI, DELTA, ALPHA, BETA, GAMMA, LAMBDA, n, SIGMA)
        return xt, yt

    def test_single_sample_raises_in_both_facades(self):
        xt, yt = self._short(1)
        # β fit: sm.add_constant sees a one-row design whose only column is
        # trivially constant and skips the intercept, so the fit returns a
        # single parameter, result.params ends up empty, and
        # __add_beta_transform's result.params[0] is an IndexError.
        with pytest.raises(IndexError):
            fecm.compute_beta_estimate(yt, xt)
        # γ/λ fit: diff() leaves zero rows, so statsmodels reduces over an
        # empty array.
        with pytest.raises(ValueError):
            fecm.compute_gamma_lambda_estimate(yt, xt, BETA)

    def test_mismatched_lengths_raise(self):
        xt, yt = self._short(50)
        with pytest.raises(ValueError):
            fecm.compute_beta_estimate(yt, xt[:40])
        with pytest.raises(ValueError):
            fecm.compute_gamma_lambda_estimate(yt, xt[:40], BETA)

    def test_beta_estimate_with_two_samples_is_an_exact_but_error_free_fit(self):
        # Two points, two parameters: zero residual degrees of freedom. The fit
        # succeeds and interpolates (r² = 1) but every standard error is inf,
        # so a caller that trusts result.params[0].err gets no signal at all.
        xt, yt = self._short(2)
        _, result = fecm.compute_beta_estimate(yt, xt)
        slope, intercept = ols_line(xt, yt)
        assert result.params[0].est == pytest.approx(slope, rel=1e-8)
        assert result.const.est == pytest.approx(intercept, rel=1e-8, abs=1e-8)
        assert result.r2 == pytest.approx(1.0)
        # zero residual degrees of freedom: s² = ssr/0 is inf when the residual
        # is a nonzero rounding artefact and nan when it is exactly 0, so the
        # invariant is that no standard error is a usable number.
        assert not numpy.isfinite(result.params[0].err)
        assert not numpy.isfinite(result.const.err)

    def test_gamma_lambda_estimate_with_two_samples_silently_loses_a_parameter(self):
        # n=2 leaves a single differenced observation, so every column of the
        # 1x2 design is trivially constant and sm.add_constant skips adding an
        # intercept. The result model then reads column 0 as the constant and
        # reports ONE parameter where the caller asked for two — the γ̂/λ̂
        # unpacking every caller does would silently shift by one column.
        xt, yt = self._short(2)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert len(result.params) == 1
        # a single observation makes the centred total sum of squares exactly 0,
        # so r² is 0/0 or ssr/0 — never a usable number.
        assert not numpy.isfinite(result.r2)

    def test_gamma_lambda_estimate_with_three_samples_returns_two_parameters(self):
        # n=3 → 2 differenced rows against 3 columns: still rank deficient, but
        # add_constant now does add the intercept, so the (const, γ̂, λ̂)
        # structure is intact and the pseudo-inverse gives an exact fit.
        xt, yt = self._short(3)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert len(result.params) == 2
        assert numpy.all(numpy.isfinite([p.est for p in result.params]))
        assert result.r2 == pytest.approx(1.0)


class TestBetaEstimate:
    def test_recovers_beta(self, simulated):
        # OLS of y on an I(1) regressor is superconsistent (error O(1/n)):
        # observed |β̂ - β| ≤ 0.018 over 300 seeds at n=2000; tol 0.05.
        # The reported error must be the textbook OLS slope SE
        # s/sqrt(Σ(x-x̄)²), s² = SSR/(n-2), computed here from the normal
        # equations rather than read off statsmodels.
        xt, yt = simulated
        _, result = fecm.compute_beta_estimate(yt, xt)
        assert result.params[0].est == pytest.approx(BETA, abs=0.05)

        x̄, ȳ = xt.mean(), yt.mean()
        Sxx = ((xt - x̄) ** 2).sum()
        slope = ((xt - x̄) * (yt - ȳ)).sum() / Sxx
        resid = yt - (ȳ - slope * x̄) - slope * xt
        s2 = resid @ resid / (len(xt) - 2)
        assert result.params[0].err == pytest.approx(numpy.sqrt(s2 / Sxx), rel=1e-8)
        assert result.const.err == pytest.approx(numpy.sqrt(s2 * (1.0 / len(xt) + x̄**2 / Sxx)), rel=1e-8)

        # r² is the centred identity, checked exactly rather than against a
        # threshold: at fixed n it is the ratio of the stationary residual
        # variance to the *realized* sample variance of a random walk, so it
        # does not concentrate near 1 and any fixed cutoff is seed-tuned.
        sst = ((yt - ȳ) ** 2).sum()
        assert result.r2 == pytest.approx(1.0 - resid @ resid / sst, rel=1e-8)
        # loose magnitude net: a cointegrating fit is always high-r², but the
        # lower tail crosses 0.99 regularly, so the bound is 0.95.
        assert result.r2 > 0.95

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

    def test_beta_recovery_survives_a_non_zero_intercept(self):
        # Every other estimator test runs at δ = α = 0. β̂ must be unaffected by
        # a non-zero level: over 300 seeds at (δ, α) = (0.3, 1.0) the largest
        # |β̂ - β| was 0.018, identical to the δ = α = 0 case, so tol 0.05.
        δ, α = 0.3, 1.0
        xt, yt = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, NPTS, SIGMA)
        _, result = fecm.compute_beta_estimate(yt, xt)
        assert result.params[0].est == pytest.approx(BETA, abs=0.05)

    def test_intercept_shift_from_delta_and_alpha_is_exact(self):
        # Exact (seed-free) proof that δ and α reach the β regression's
        # constant. On an identical RNG stream, y(δ, α) - y(0, 0) is the
        # deterministic offset d_t = (α - δ/λ)(1 - (1+λ)^t) (proved in
        # test_delta_and_alpha_enter_affinely_with_noise_present) and x is
        # bit-identical, so OLS linearity forces the two fits to differ by
        # exactly the OLS fit of d on x — no statistical tolerance involved.
        δ, α, n = 0.4, 2.0, 800
        state = numpy.random.get_state()
        xt, y_a = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, n, SIGMA)
        numpy.random.set_state(state)
        _, y_b = ecm.ecm(PHI, 0.0, 0.0, BETA, GAMMA, LAMBDA, n, SIGMA)

        _, res_a = fecm.compute_beta_estimate(y_a, xt)
        _, res_b = fecm.compute_beta_estimate(y_b, xt)

        d = (α - δ / LAMBDA) * (1.0 - (1.0 + LAMBDA) ** numpy.arange(n))
        slope_d, intercept_d = ols_line(xt, d)
        assert res_a.params[0].est - res_b.params[0].est == pytest.approx(slope_d, abs=1e-10)
        assert res_a.const.est - res_b.const.est == pytest.approx(intercept_d, abs=1e-10)
        # And the offset being absorbed really is the ECM level: d is entirely
        # deterministic, and d̄ = (α - δ/λ)(1 - mean((1+λ)^t)) = 3.0 x (1 -
        # 1/320) at these parameters. The *split* of d̄ between intercept_d and
        # slope_d x̄ is not deterministic — slope_d = Sxd/Sxx picks up the
        # random walk's scatter against d's short transient — which is exactly
        # why the level claim is made statistically in
        # test_intercept_estimates_alpha_minus_delta_over_lambda instead.
        assert d.mean() == pytest.approx(α - δ / LAMBDA, rel=1e-2)

    def test_intercept_estimates_alpha_minus_delta_over_lambda(self):
        # The statistical form of the same claim. The OLS intercept
        # ȳ - β̂ x̄ is NOT superconsistent — its error is dominated by
        # (β̂ - β) x̄ = O_p(n^{-1/2}) — so a single path is a poor estimator
        # (sd ≈ 0.23 at n=2000, with a tail out past 1.5). Averaging M
        # independent paths gives SE ≈ 0.05; deviations of the ensemble mean
        # from α - δ/λ ranged to 0.093 over 6 seeds, so abs=0.35 is ≈ 7 SE and
        # ~4x the largest observed miss. It still separates α - δ/λ = 3.0 from
        # α alone (2.0), from -δ/λ alone (1.0), and from 0.
        δ, α, M, n = 0.4, 2.0, 60, 600
        consts = numpy.empty(M)
        for i in range(M):
            xt, yt = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, n, SIGMA)
            _, result = fecm.compute_beta_estimate(yt, xt)
            consts[i] = result.const.est
        assert consts.mean() == pytest.approx(α - δ / LAMBDA, abs=0.35)

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
        # ≈ 0.02 (γ) and ≈ 0.014 (λ); observed max errors over 300 seeds were
        # 0.069 and 0.033. Tolerances 0.15 / 0.1 are ≥ 5 SE and clear the
        # observed range with room to spare.
        xt, yt = simulated
        _, β_result = fecm.compute_beta_estimate(yt, xt)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, β_result.params[0].est)
        assert len(result.params) == 2
        assert result.params[0].est == pytest.approx(GAMMA, abs=0.15)
        assert result.params[1].est == pytest.approx(LAMBDA, abs=0.1)

    def test_constant_estimates_delta_minus_lambda_alpha(self):
        # The identity the const-label xfail below is built on, asserted
        # numerically against the TRUE (δ, α, λ) rather than against lstsq.
        # Rewriting the recursion,
        #     Δy_t = (δ - λα) + γ Δx_t + λ (y_{t-1} - β x_{t-1}) + ξ_t,
        # so with ε_t = y_t - β x_t the regression constant estimates δ - λα.
        # δ - λα = 0.7 here, which is distinct from δ (0.3), from -λα (0.4)
        # and from 0, so the assertion discriminates all the plausible
        # alternatives. True β is passed rather than β̂ because the residual
        # (β̂ - β) x_{t-1} leaks an I(1) term into the constant: with true β the
        # constant's sd is 0.028 and its largest deviation over 300 seeds was
        # 0.113 (tol 0.2 ≈ 7 SD), with β̂ the same numbers are 0.078 and 0.485.
        δ, α = 0.3, 1.0
        xt, yt = ecm.ecm(PHI, δ, α, BETA, GAMMA, LAMBDA, NPTS, SIGMA)
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert result.const.est == pytest.approx(δ - LAMBDA * α, abs=0.2)
        # γ and λ are unaffected by the level parameters.
        assert result.params[0].est == pytest.approx(GAMMA, abs=0.15)
        assert result.params[1].est == pytest.approx(LAMBDA, abs=0.1)

    def test_constant_is_zero_when_delta_and_alpha_vanish(self, simulated):
        # The complementary half of the identity: δ = α = 0 ⇒ δ - λα = 0.
        # Largest |const| over 300 seeds at these settings was 0.073.
        xt, yt = simulated
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        assert result.const.est == pytest.approx(0.0, abs=0.2)

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
        transforms = [tr.param for tr in present(result.param_transforms)] + [present(result.const_transform).param]
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
        λ_tr = present(result.param_transforms)[1]
        assert λ_tr.param.est == result.params[1].est
        assert λ_tr.param.err == result.params[1].err

    @pytest.mark.xfail(
        strict=True,
        reason="__add_gamma_lambda_transform labels the constant transform with the λ̂ labels, "
        "duplicating the second parameter transform's. The constant of the Δy regression "
        "estimates δ - λα (asserted numerically against the true δ, α, λ in "
        "test_constant_estimates_delta_minus_lambda_alpha), so its label has to name δ — "
        "anything else (including another wrong label) is still a bug, which is why this "
        "asserts the δ form positively rather than merely differing from the λ̂ label",
    )
    def test_const_transform_is_labelled_for_delta(self, simulated):
        xt, yt = simulated
        _, result = fecm.compute_gamma_lambda_estimate(yt, xt, BETA)
        λ_tr = present(result.param_transforms)[1]
        est_label = present(present(result.const_transform).param.est_label)
        err_label = present(present(result.const_transform).param.err_label)
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
        assert r"\gamma" in present(result.model) and r"\lambda" in present(result.model)


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

    def test_facade_pipeline_with_non_zero_delta_and_alpha(self):
        # Same flow with the level parameters engaged, so every estimator sees
        # δ ≠ 0 and α ≠ 0 end to end: β̂ → β, (γ̂, λ̂) → (γ, λ), and the Δy
        # regression's constant → δ - λα. β̂ is used for ε here (the notebook
        # flow), which inflates the constant's spread — largest deviation over
        # 300 seeds was 0.485 — so its tolerance is 0.6, still well inside the
        # 0.7 separation from 0.
        δ, α, σ = 0.3, 1.0, 1.0
        _, values = fecm.create_source(φ=PHI, δ=δ, α=α, β=BETA, γ=GAMMA, λ=LAMBDA, σ=σ, npts=NPTS)
        xt, yt = values
        _, β_result = fecm.compute_beta_estimate(yt, xt)
        β_hat = β_result.params[0].est
        _, γλ_result = fecm.compute_gamma_lambda_estimate(yt, xt, β_hat)
        assert β_hat == pytest.approx(BETA, abs=0.05)
        assert γλ_result.params[0].est == pytest.approx(GAMMA, abs=0.15)
        assert γλ_result.params[1].est == pytest.approx(LAMBDA, abs=0.1)
        assert γλ_result.const.est == pytest.approx(δ - LAMBDA * α, abs=0.6)
