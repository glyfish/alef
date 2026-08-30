"""Tests for navi's VECM support: ``lib.models.vecm`` and its kwargs façade
``lib.data.impl.vecm``.

Process simulated by both ``vecm1`` and ``vecm``::

    Δx_t = λ β x_{t-1} + Σ_{j=1..m} a_j Δx_{t-j} + ε_t ,   ε_t ~ N(0, Ω)

with λ an (n × r) damping matrix, β an (r × n) matrix of cointegration vectors
(rows), and Π = λβ the (rank r) long-run impact matrix. Both simulators pin
x_0 … x_m to zero and draw the whole ε path up front with a single call to
``numpy.random.multivariate_normal``, so a test can reseed and redraw the very
same shocks and then check the path against a formula the library does not use.

Closed forms exercised below (n = 2, β = [1, -1], λ = [-κ, 0]ᵀ, so Π has rank 1
and the second series never error-corrects):

* levels form. Δx_t = Πx_{t-1} + Σ a_j Δx_{t-j} + ε_t is the VAR(m+1)
  x_t = (I + Π + a_1)x_{t-1} + Σ_{j=2..m}(a_j - a_{j-1})x_{t-j} - a_m x_{t-m-1} + ε_t.
  With a = 0 that collapses to x_t = A x_{t-1} + ε_t, A = I + Π, whose solution
  is x_t = Σ_{s=2}^{t} A^{t-s} ε_s.
* cointegrating combination. z_t = βx_t obeys Δz_t = (βλ)z_{t-1} + βε_t, i.e.
  AR(1) with φ = 1 + βλ = 1 - κ, so Var(z) = βΩβᵀ/(1 - φ²) and ρ(k) = φ^k.
  For r > 1, z is a VAR(1) with matrix I + βλ and Var(z) solves the discrete
  Lyapunov equation Σ = AΣAᵀ + βΩβᵀ.
* the non-adjusting series. Row 2 of Π is zero, so with a = 0 the second series
  is exactly the running sum of ε_2 — a random walk with Var(x_2,t) = (t-1)Ω₂₂
  ((t-1), not t, because x_0 and x_1 are both pinned to zero).
* Johansen. trace(r) = -T Σ_{i≥r} log(1 - λ_i) and λ_max(r) = -T log(1 - λ_r)
  for the eigenvalues λ_i reported by the test, T the effective sample size.

Statistical tolerances are set from the spread observed over seven to ten
unrelated seeds and stated as a multiple of it, so they are not tuned to the
fixed seed conftest installs.
"""
import io
import json
import re
from contextlib import redirect_stdout

import numpy
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm
from statsmodels.tsa.vector_ar.var_model import LagOrderResults
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult, VECMResults

from lib.data.hyp_test import (
    JohansenCointTestEigenVector,
    JohansenCointTestRank,
    JohansenCointTestReport,
    JohansenCointTestStatistic,
    VAROrderTestReport,
)
from lib.data.impl import vecm as facade
from lib.data.param_est import EstModel, ParamEst, VARParamType, VECMEst, VECMParamType
from lib.data.reports import JohansenTestReport
from lib.models import vecm as model
from lib.utils import create_ensemble

# numpy.matrix is what the simulators build internally (and what the notebooks
# feed them); coint_johansen drops the zero imaginary parts of its eigenvalues.
pytestmark = [
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
    pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning"),
]

# ############################################################################
# Model parameters shared by the tests
# ############################################################################

KAPPA = 0.3                      # error correction speed, φ = 1 - κ = 0.7
PHI = 0.4                        # lagged difference coefficient
BETA = numpy.matrix([[1.0, -1.0]])
LAMBDA = numpy.matrix([[-KAPPA], [0.0]])
A_ZERO = numpy.matrix(numpy.zeros((2, 2)))
A_ONE = numpy.matrix([[PHI, 0.0], [0.0, PHI]])
OMEGA_I = numpy.matrix(numpy.eye(2))
# deliberately neither identity nor diagonal
OMEGA = numpy.matrix([[2.0, 0.6], [0.6, 1.5]])

# rank 2 system: three series, one common stochastic trend
BETA2 = numpy.matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
LAMBDA2 = numpy.matrix([[-KAPPA, 0.0], [0.0, -KAPPA], [0.0, 0.0]])
A2_ZERO = numpy.matrix(numpy.zeros((3, 3)))


def redraw_noise(seed: int, Ω, nsamp: int) -> numpy.ndarray:
    """Reproduce the ε path a simulator drew when seeded with ``seed``.

    Both simulators consume the global RNG with exactly one
    ``multivariate_normal(0, Ω, nsamp)`` call before their recursion starts.
    """
    n = numpy.asarray(Ω).shape[0]
    numpy.random.seed(seed)
    return numpy.random.multivariate_normal(numpy.zeros(n), numpy.asarray(Ω), nsamp)


def lyapunov(A: numpy.ndarray, Q: numpy.ndarray) -> numpy.ndarray:
    """Stationary covariance of z_t = A z_{t-1} + η_t, Cov(η) = Q.

    Solves Σ = AΣAᵀ + Q as vec(Σ) = (I - A⊗A)⁻¹ vec(Q) (row major vec, so
    (AΣAᵀ).flatten() = (A⊗A) Σ.flatten()).
    """
    k = A.shape[0]
    return numpy.linalg.solve(numpy.eye(k * k) - numpy.kron(A, A), Q.flatten()).reshape(k, k)


def acf(x: numpy.ndarray, k: int) -> float:
    """Sample autocorrelation at lag k."""
    xc = x - x.mean()
    return float((xc[:-k] * xc[k:]).sum() / (xc * xc).sum())


def find_param(params: list[ParamEst], row: int, column: int, order: int = 0) -> ParamEst:
    """The single ParamEst at the given matrix position and lag order."""
    hits = [p for p in params if p.row == row and p.column == column and p.order == order]
    assert len(hits) == 1, f"expected one estimate at ({row},{column}) order {order}, got {len(hits)}"
    return hits[0]


# Module scoped fixtures run before conftest's function scoped autouse seeding,
# so they seed the global RNG themselves.

@pytest.fixture(scope="module")
def sim_vecm1():
    """VECM(1): κ = 0.3 error correction, a = 0.4·I, Ω = I, 2000 points."""
    numpy.random.seed(6041)
    return facade.create_vecm1_source(LAMBDA, BETA, A_ONE, npts=2000)


@pytest.fixture(scope="module")
def fit_vecm1(sim_vecm1):
    """Estimate of ``sim_vecm1`` with the true rank and lag order."""
    _, xt = sim_vecm1
    return facade.compute_estimate(xt, maxlags=1, rank=1, trend="co")


@pytest.fixture(scope="module")
def sim_rank_two():
    """Three series with a single common stochastic trend: Π = λ₂β₂ has rank 2,
    so two independent combinations error correct and one random walk remains."""
    numpy.random.seed(4477)
    return facade.create_vecm1_source(LAMBDA2, BETA2, A2_ZERO, npts=4000)


@pytest.fixture(scope="module")
def sim_random_walks():
    """Two independent random walks: Π = 0, so nothing is cointegrated."""
    zeros_λ = numpy.matrix(numpy.zeros((2, 1)))
    zeros_β = numpy.matrix(numpy.zeros((1, 2)))
    numpy.random.seed(8123)
    return facade.create_vecm1_source(zeros_λ, zeros_β, A_ZERO, npts=1000)


# ############################################################################
# Model layer: simulators
# ############################################################################

class TestVecm1Simulator:

    def test_path_solves_the_var1_closed_form(self):
        """With a = 0 the model is x_t = A x_{t-1} + ε_t, A = I + λβ, whose
        solution x_t = Σ_{s=2}^t A^{t-s} ε_s is computed here from matrix powers
        rather than by iterating the difference equation."""
        seed, npts = 4242, 250
        numpy.random.seed(seed)
        xt = numpy.array(model.vecm1(LAMBDA, BETA, A_ZERO, OMEGA, npts))
        εt = redraw_noise(seed, OMEGA, npts)

        A = numpy.eye(2) + numpy.array(LAMBDA) @ numpy.array(BETA)
        for t in (2, 3, 5, 17, 120, 249):
            expected = sum(numpy.linalg.matrix_power(A, t - s) @ εt[s] for s in range(2, t + 1))
            assert_allclose(xt[:, t], expected, rtol=1e-10, atol=1e-10)

    def test_first_two_samples_are_zero_and_third_is_the_shock(self):
        """The recursion starts at i = 2, so x_0 = x_1 = 0 and x_2 = ε_2."""
        seed, npts = 71, 40
        numpy.random.seed(seed)
        xt = numpy.array(model.vecm1(LAMBDA, BETA, A_ONE, OMEGA_I, npts))
        εt = redraw_noise(seed, OMEGA_I, npts)

        assert_allclose(xt[:, 0], numpy.zeros(2), atol=0.0)
        assert_allclose(xt[:, 1], numpy.zeros(2), atol=0.0)
        assert_allclose(xt[:, 2], εt[2], rtol=1e-12)

    def test_returns_matrix_of_shape_series_by_samples(self):
        numpy.random.seed(11)
        xt = model.vecm1(LAMBDA, BETA, A_ONE, OMEGA_I, 30)
        assert isinstance(xt, numpy.matrix)
        assert xt.shape == (2, 30)
        assert numpy.asarray(xt).dtype == numpy.float64

    def test_non_adjusting_series_is_the_running_sum_of_its_shocks(self):
        """Row 2 of λβ is zero and a = 0, so Δx₂,t = ε₂,t exactly."""
        seed, npts = 909, 2000
        numpy.random.seed(seed)
        xt = numpy.array(model.vecm1(LAMBDA, BETA, A_ZERO, OMEGA, npts))
        εt = redraw_noise(seed, OMEGA, npts)

        walk = numpy.concatenate([[0.0, 0.0], numpy.cumsum(εt[2:, 1])])
        assert_allclose(xt[1], walk, rtol=1e-10, atol=1e-10)

    def test_increment_covariance_reproduces_omega(self):
        """Δx₂ = ε₂ and Δx₁ = -κ z_{t-1} + ε₁ with z_{t-1} ⟂ ε_t, so
        Var(Δx₂) = Ω₂₂, Cov(Δx₁, Δx₂) = Ω₁₂ and
        Var(Δx₁) = κ²Var(z) + Ω₁₁.  n = 4000 gives the variance of an iid
        estimate a relative spread of √(2/n) ≈ 2.2%; 10% is ~4.5σ.  The off
        diagonal is the noisier of the three: over 150 unrelated seeds its
        relative error averaged 3.7% with p95 8.7% and a maximum of 12.4%, so
        15% clears the observed maximum while still rejecting a simulator whose
        cross covariance is a quarter off — which the old 25% did not."""
        numpy.random.seed(2718)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=4000, Ω=OMEGA)
        Δ = numpy.diff(xt, axis=1)[:, 2:]
        cov = numpy.cov(Δ)

        βΩβ = (numpy.array(BETA) @ numpy.array(OMEGA) @ numpy.array(BETA).T).item()
        var_z = βΩβ / (1.0 - (1.0 - KAPPA) ** 2)

        assert cov[1, 1] == pytest.approx(OMEGA[1, 1], rel=0.10)
        assert cov[0, 1] == pytest.approx(OMEGA[0, 1], rel=0.15)
        assert cov[0, 0] == pytest.approx(KAPPA**2 * var_z + OMEGA[0, 0], rel=0.10)

    def test_cointegrating_combination_is_ar1(self):
        """z = βx is AR(1) with φ = 1 - κ: Var(z) = βΩβᵀ/(1 - φ²), ρ(k) = φ^k.

        n = 4000 of an AR(1) with φ = 0.7 carries n(1-φ)/(1+φ) ≈ 700 independent
        observations, so the variance estimate has a ~5% relative spread and 20%
        is ~4σ. Bartlett's formula puts the spread of ρ̂(k) between 0.011 (k=1)
        and 0.025 (k=5), which the 0.03 + 0.012k band covers at ~3.5σ throughout
        while staying well inside φ^5 = 0.17.
        """
        numpy.random.seed(9315)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=4000, Ω=OMEGA)
        z = (numpy.array(BETA) @ xt)[0, 2:]

        βΩβ = (numpy.array(BETA) @ numpy.array(OMEGA) @ numpy.array(BETA).T).item()
        assert z.var() == pytest.approx(βΩβ / (1.0 - (1.0 - KAPPA) ** 2), rel=0.20)
        for k in range(1, 6):
            assert acf(z, k) == pytest.approx((1.0 - KAPPA) ** k, abs=0.03 + 0.012 * k)

    def test_lagged_difference_term_matches_var2_representation(self):
        """a ≠ 0 makes the levels process a VAR(2):
        x_t = (I + Π + a)x_{t-1} - a x_{t-2} + ε_t."""
        seed, npts = 5150, 300
        numpy.random.seed(seed)
        xt = numpy.array(model.vecm1(LAMBDA, BETA, A_ONE, OMEGA, npts))
        εt = redraw_noise(seed, OMEGA, npts)

        Π = numpy.array(LAMBDA) @ numpy.array(BETA)
        a = numpy.array(A_ONE)
        expected = numpy.zeros((npts, 2))
        for i in range(2, npts):
            expected[i] = (numpy.eye(2) + Π + a) @ expected[i - 1] - a @ expected[i - 2] + εt[i]
        assert_allclose(xt.T, expected, rtol=1e-9, atol=1e-9)

    def test_rank_two_cointegration_has_lyapunov_covariance(self, sim_rank_two):
        """Three series, rank 2: z = βx is a stationary VAR(1) with matrix
        I + βλ, and Cov(z) solves the discrete Lyapunov equation.  The off
        diagonal, expected at -0.346, is compared on an absolute scale because
        its sampling spread (0.105 over 150 unrelated seeds) is a third of its
        size; atol = 0.32 is ~3σ and the acceptance window stays clear of zero,
        so two uncorrelated combinations would be rejected."""
        _, xt = sim_rank_two
        z = numpy.array(BETA2) @ xt

        A = numpy.eye(2) + numpy.array(BETA2) @ numpy.array(LAMBDA2)
        Q = numpy.array(BETA2) @ numpy.array(BETA2).T
        expected = lyapunov(A, Q)
        sample = numpy.cov(z[:, 2:])

        assert_allclose(numpy.diag(sample), numpy.diag(expected), rtol=0.20)
        assert_allclose(sample[0, 1], expected[0, 1], atol=0.32)

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="vecm1 annotates λ and β as NDArray but evaluates λ*β, which is "
               "elementwise for plain ndarrays; only numpy.matrix inputs give the "
               "matrix product. Rank 1 survives by accident (column*row broadcasts "
               "to the outer product), rank>1 raises on the broadcast.",
    )
    def test_accepts_plain_ndarray_parameters(self):
        numpy.random.seed(31)
        as_matrix = numpy.array(model.vecm1(LAMBDA2, BETA2, A2_ZERO, numpy.matrix(numpy.eye(3)), 50))
        numpy.random.seed(31)
        as_array = numpy.array(model.vecm1(numpy.array(LAMBDA2), numpy.array(BETA2),
                                           numpy.array(A2_ZERO), numpy.eye(3), 50))
        assert_allclose(as_array, as_matrix)


class TestVecmSimulator:

    def test_single_lag_agrees_with_vecm1(self):
        """vecm with a of shape (1, n, n) is vecm1 with a[0]: same recursion
        start (i = 2) and the same single draw from the global RNG."""
        numpy.random.seed(313)
        first_order = numpy.array(model.vecm1(LAMBDA, BETA, A_ONE, OMEGA, 400))
        numpy.random.seed(313)
        general = numpy.array(model.vecm(LAMBDA, BETA, numpy.array([numpy.array(A_ONE)]), OMEGA, 400))
        assert_allclose(general, first_order, rtol=1e-12, atol=1e-12)

    def test_two_lags_match_var3_representation(self):
        """With m = 2 the levels process is the VAR(3)
        x_t = (I+Π+a₁)x_{t-1} + (a₂-a₁)x_{t-2} - a₂x_{t-3} + ε_t."""
        seed, npts = 6161, 300
        a = numpy.array([[[0.4, 0.0], [0.0, 0.3]], [[-0.2, 0.0], [0.0, 0.1]]])
        numpy.random.seed(seed)
        xt = numpy.array(model.vecm(LAMBDA, BETA, a, OMEGA, npts))
        εt = redraw_noise(seed, OMEGA, npts)

        Π = numpy.array(LAMBDA) @ numpy.array(BETA)
        a1, a2 = a[0], a[1]
        expected = numpy.zeros((npts, 2))
        for i in range(3, npts):
            expected[i] = ((numpy.eye(2) + Π + a1) @ expected[i - 1]
                           + (a2 - a1) @ expected[i - 2]
                           - a2 @ expected[i - 3]
                           + εt[i])
        assert_allclose(xt.T, expected, rtol=1e-9, atol=1e-9)

    def test_first_m_plus_one_samples_are_zero(self):
        """The m lag recursion cannot start before i = m + 1."""
        a = numpy.zeros((3, 2, 2))
        numpy.random.seed(17)
        xt = numpy.array(model.vecm(LAMBDA, BETA, a, OMEGA_I, 40))
        assert_allclose(xt[:, :4], numpy.zeros((2, 4)), atol=0.0)
        # a = 0 and every lag pinned to zero leave Δx₄ = ε₄, so the first
        # computed sample is exactly the shock drawn for it
        assert_allclose(xt[:, 4], redraw_noise(17, OMEGA_I, 40)[4], rtol=1e-12)


# ############################################################################
# Model layer: estimation
# ############################################################################

class TestFit:

    def test_recovers_known_parameters(self, sim_vecm1):
        """Round trip on 2000 points. Over 200 unrelated seeds the largest
        deviations seen were λ 0.021, β 0.004, a 0.093, Ω 0.135 and const 0.052;
        the tolerances below clear those maxima rather than the p95, because a
        blanket atol over a whole matrix takes the worst entry of four."""
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=1, rank=1, trend="co")

        assert result.k_ar == 2                 # k_ar_diff + 1
        assert result.coint_rank == 1
        assert result.neqs == 2
        # statsmodels normalises β so its first `rank` rows are the identity
        assert_allclose(result.beta[:, 0], [1.0, -1.0], atol=0.02)
        assert_allclose(result.alpha[:, 0], [-KAPPA, 0.0], atol=0.06)
        assert_allclose(result.gamma, numpy.array(A_ONE), atol=0.14)
        assert_allclose(result.sigma_u, numpy.eye(2), atol=0.18)
        assert_allclose(result.det_coef[:, 0], [0.0, 0.0], atol=0.15)

    def test_long_run_impact_matrix_is_recovered(self, sim_vecm1):
        """λ and β are identified only up to an r×r rotation; Π = λβᵀ is not."""
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=1, rank=1, trend="co")
        Π = numpy.array(LAMBDA) @ numpy.array(BETA)
        assert_allclose(result.alpha @ result.beta.T, Π, atol=0.06)

    def test_rank_two_recovers_the_cointegration_space(self, sim_rank_two):
        """β normalised on its leading 2×2 block spans the same space as
        [1,-1,0] and [0,1,-1], i.e. it is [[1,0],[0,1],[-1,-1]]."""
        _, xt = sim_rank_two
        result = model.fit(xt.T, maxlags=1, rank=2, trend="co")

        assert result.coint_rank == 2
        assert_allclose(result.beta, [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], atol=0.03)
        assert_allclose(result.alpha @ result.beta.T,
                        numpy.array(LAMBDA2) @ numpy.array(BETA2), atol=0.06)

    @pytest.mark.parametrize(
        "trend, outside_columns, inside_rows",
        [("n", 0, 0), ("co", 1, 0), ("ci", 0, 1), ("lo", 1, 0), ("li", 0, 1)],
    )
    def test_trend_string_selects_deterministic_terms(self, sim_vecm1, trend, outside_columns, inside_rows):
        """Each documented trend code places its term outside (det_coef) or
        inside (det_coef_coint) the cointegration relation."""
        _, xt = sim_vecm1
        result = model.fit(xt[:, :600].T, maxlags=1, rank=1, trend=trend)
        assert result.det_coef.shape == (2, outside_columns)
        assert result.det_coef_coint.shape == (inside_rows, 1)

    def test_default_trend_is_a_constant_outside_the_relation(self, sim_vecm1):
        """fit's trend defaults to "co": one deterministic column outside the
        cointegration relation, none inside, and the same fit as passing it."""
        _, xt = sim_vecm1
        default = model.fit(xt.T, maxlags=1, rank=1)
        explicit = model.fit(xt.T, maxlags=1, rank=1, trend="co")
        assert default.det_coef.shape == (2, 1)
        assert default.det_coef_coint.shape == (0, 1)
        assert_allclose(default.alpha, explicit.alpha, rtol=1e-12)
        assert_allclose(default.det_coef, explicit.det_coef, rtol=1e-12)
        assert not numpy.allclose(default.alpha, model.fit(xt.T, maxlags=1, rank=1, trend="n").alpha)

    def test_maxlags_sets_the_number_of_lagged_differences(self, sim_vecm1):
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=3, rank=1, trend="co")
        assert result.k_ar == 4
        assert result.gamma.shape == (2, 6)     # neqs × neqs·k_ar_diff


class TestOrderEstimate:

    def test_selects_the_true_number_of_lagged_differences(self, sim_vecm1):
        """The simulated process has one lagged difference. BIC picked 1 on all
        200 seeds tried; HQIC missed on 0.5% of them and AIC over-selects far
        more often, which is its documented small sample behaviour, so only BIC
        is pinned exactly and HQIC is held to a bound."""
        _, xt = sim_vecm1
        result = model.lag_order_estimate(xt.T, 6, "co")
        assert isinstance(result, LagOrderResults)
        assert result.bic == 1
        assert result.hqic <= 1

    def test_selects_zero_lags_for_a_pure_var1(self):
        """a = 0 leaves no lagged differences to fit. BIC selected 0 on all 200
        seeds tried; HQIC over-selected on 1% of them, so it is held to a bound
        the same way AIC is elsewhere."""
        numpy.random.seed(5309)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=2000)
        result = model.lag_order_estimate(xt.T, 6, "co")
        assert result.bic == 0
        assert result.hqic <= 1

    def test_criteria_are_reported_for_every_candidate_order(self, sim_vecm1):
        """select_order scores VAR orders 1…maxlags+1, reported as the
        equivalent lagged difference counts 0…maxlags, and the selection is the
        argmin of each criterion's list."""
        _, xt = sim_vecm1
        result = model.lag_order_estimate(xt.T, 6, "co")
        for name in ("aic", "bic", "fpe", "hqic"):
            values = numpy.array(result.ics[name])
            assert len(values) == 7
            assert int(values.argmin()) == getattr(result, name)

    def test_default_maxlags_is_twelve(self, sim_vecm1):
        _, xt = sim_vecm1
        assert len(model.lag_order_estimate(xt.T).ics["aic"]) == 13


# ############################################################################
# Model layer: Johansen cointegration test
# ############################################################################

class TestJohansenTest:

    def test_statistics_follow_the_eigenvalue_closed_form(self, sim_vecm1):
        """trace(r) = -T Σ_{i≥r} log(1-λ_i), λ_max(r) = -T log(1-λ_r), with
        T = nobs - k_ar_diff - 1 the effective sample."""
        _, xt = sim_vecm1
        lags = 2
        result = model.johansen_test_coint(xt.T, lags)

        eig = numpy.real(result.eig)
        T = xt.shape[1] - lags - 1
        expected_trace = [-T * numpy.log(1.0 - eig[i:]).sum() for i in range(2)]
        expected_max = [-T * numpy.log(1.0 - eig[i]) for i in range(2)]
        assert_allclose(result.lr1, expected_trace, rtol=1e-10)
        assert_allclose(result.lr2, expected_max, rtol=1e-10)

    def test_detects_a_single_cointegrating_relation(self, sim_vecm1):
        """Π has rank 1, so exactly one canonical correlation is large and the
        r = 0 null is rejected by an order of magnitude at 99%.

        The r ≤ 1 statistic is deliberately not asserted: statsmodels tabulates
        χ²(1) values for that last row, which this design exceeds on roughly a
        third of seeds even though the second component really is a unit root,
        so any claim about it would be a seed-tuned claim.
        """
        _, xt = sim_vecm1
        result = model.johansen_test_coint(xt.T, 2)
        eig = numpy.real(result.eig)
        assert result.lr1[0] > 5.0 * result.cvt[0, 2]
        assert eig[0] > 0.12
        assert eig[1] < 0.02

    def test_leading_eigenvector_is_the_cointegration_vector(self, sim_vecm1):
        """The eigenvectors are the columns of evec; the leading one, scaled to
        lead with 1, is the simulated β = [1, -1]."""
        _, xt = sim_vecm1
        result = model.johansen_test_coint(xt.T, 2)
        leading = numpy.real(result.evec[:, 0])
        assert_allclose(leading / leading[0], [1.0, -1.0], atol=0.05)

    def test_finds_no_cointegration_between_independent_walks(self, sim_random_walks):
        """Π = 0, so the r = 0 null holds. The 99% critical value on its own is
        not a safe bound: the trace test's finite sample over rejection puts
        lr1[0] above it on ~4% of seeds (measured over 400 draws, and the same
        4% shows up for unpinned random walks), so the bound carries a factor of
        two, which 400 draws never violated."""
        _, xt = sim_random_walks
        result = model.johansen_test_coint(xt.T, 1)
        assert result.lr1[0] < 2.0 * result.cvt[0, 2]

    def test_trend_argument_switches_critical_value_table(self, sim_vecm1):
        """det_order -1/0/1 (no trend / constant / linear trend) selects three
        different tabulated critical value sets and three different eigenvalue
        problems."""
        _, xt = sim_vecm1
        none, const, linear = (model.johansen_test_coint(xt.T, 2, trend) for trend in (-1, 0, 1))
        assert not numpy.allclose(none.cvt, const.cvt)
        assert not numpy.allclose(const.cvt, linear.cvt)
        assert not numpy.allclose(none.eig, const.eig)
        # the constant case is the tabulated χ²(1) value for the last statistic
        assert const.cvt[1, 1] == pytest.approx(3.8415, abs=1e-4)


# ############################################################################
# Model layer: prediction
# ############################################################################

class TestPredict:

    def test_returns_forecast_and_interval_of_requested_shape(self, fit_vecm1):
        result, _ = fit_vecm1
        out = model.predict(result, 5)
        assert isinstance(out, tuple) and len(out) == 3
        for part in out:
            assert part.shape == (5, 2)
        forecast, lower, upper = out
        assert numpy.all(lower < forecast)
        assert numpy.all(forecast < upper)

    def test_forecast_follows_the_fitted_recursion(self, sim_vecm1, fit_vecm1):
        """Re-run the VECM recursion by hand from the reported Π̂, Γ̂ and
        constant, with the noise set to its mean."""
        _, xt = sim_vecm1
        result, _ = fit_vecm1
        forecast, _, _ = model.predict(result, 3)

        levels = xt.T
        Π = result.alpha @ result.beta.T
        const = result.det_coef[:, 0]
        previous, current = levels[-2], levels[-1]
        for step in range(3):
            nxt = current + Π @ current + result.gamma @ (current - previous) + const
            assert_allclose(forecast[step], nxt, rtol=1e-10)
            previous, current = current, nxt

    def test_interval_width_scales_with_the_normal_quantile(self, fit_vecm1):
        """The band is ±z_{1-α/2}·se, so its width ratio across two α values is
        exactly the ratio of the quantiles, and at one step se = √diag(Ω̂)."""
        result, _ = fit_vecm1
        forecast_05, _, upper_05 = model.predict(result, 4, alpha=0.05)
        forecast_01, _, upper_01 = model.predict(result, 4, alpha=0.01)

        assert_allclose(forecast_05, forecast_01, rtol=1e-12)
        ratio = (upper_01 - forecast_01) / (upper_05 - forecast_05)
        assert_allclose(ratio, numpy.full((4, 2), norm.ppf(0.995) / norm.ppf(0.975)), rtol=1e-10)
        assert_allclose((upper_05 - forecast_05)[0],
                        norm.ppf(0.975) * numpy.sqrt(numpy.diag(result.sigma_u)), rtol=1e-10)

    def test_multi_step_interval_follows_the_vma_expansion(self, fit_vecm1):
        """The h step forecast error is Σ_{i<h} Φ_i ε_{h-i}, for the VMA weights
        of the fitted levels VAR(2): Φ₀ = I and Φ_i = A₁Φ_{i-1} + A₂Φ_{i-2} with
        A₁ = I + Π̂ + Γ̂ and A₂ = -Γ̂.  So the half width at step h is
        z_{1-α/2}·√diag(Σ_{i<h} Φ_iΩ̂Φ_iᵀ), which is checked here at every step
        rather than only at h = 1, where it collapses to √diag(Ω̂)."""
        result, _ = fit_vecm1
        steps = 5
        forecast, lower, upper = model.predict(result, steps)

        A1 = numpy.eye(2) + result.alpha @ result.beta.T + result.gamma
        A2 = -result.gamma
        phi = [numpy.eye(2), A1]
        for i in range(2, steps):
            phi.append(A1 @ phi[i - 1] + A2 @ phi[i - 2])

        covariance = numpy.zeros((2, 2))
        for step in range(steps):
            covariance = covariance + phi[step] @ result.sigma_u @ phi[step].T
            half_width = norm.ppf(0.975) * numpy.sqrt(numpy.diag(covariance))
            assert_allclose(upper[step] - forecast[step], half_width, rtol=1e-10)
            assert_allclose(forecast[step] - lower[step], half_width, rtol=1e-10)
        # uncertainty accumulates: every step is wider than the one before it
        assert numpy.all(numpy.diff(upper - forecast, axis=0) > 0.0)

    def test_default_alpha_is_five_percent(self, fit_vecm1):
        result, _ = fit_vecm1
        assert_allclose(model.predict(result, 3), model.predict(result, 3, alpha=0.05), rtol=1e-12)


# ############################################################################
# Façade: simulation sources
# ############################################################################

class TestSourceFacade:

    def test_vecm1_source_returns_time_and_values(self):
        numpy.random.seed(101)
        t, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ONE, npts=64)
        assert isinstance(t, numpy.ndarray) and isinstance(xt, numpy.ndarray)
        assert not isinstance(xt, numpy.matrix)      # unwrapped for the notebooks
        assert t.shape == (64,)
        assert xt.shape == (2, 64)
        assert xt.dtype == numpy.float64
        assert_allclose(t, numpy.arange(64.0))

    def test_vecm1_source_agrees_with_the_model_layer(self):
        numpy.random.seed(555)
        _, from_facade = facade.create_vecm1_source(LAMBDA, BETA, A_ONE, npts=120, Ω=OMEGA)
        numpy.random.seed(555)
        from_model = numpy.array(model.vecm1(LAMBDA, BETA, A_ONE, OMEGA, 120))
        assert_allclose(from_facade, from_model, rtol=1e-12, atol=1e-12)

    def test_vecm1_source_defaults_to_1000_points_and_unit_noise(self):
        """Ω defaults to the identity: Δx₂ = ε₂ has unit variance."""
        numpy.random.seed(202)
        t, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO)
        assert t.shape == (1000,)
        assert xt.shape == (2, 1000)
        assert numpy.diff(xt, axis=1)[1, 2:].var() == pytest.approx(1.0, rel=0.18)

    def test_vecm1_source_honours_the_noise_covariance(self):
        """Scaling Ω by 4 scales the random walk's increment variance by 4;
        n = 3000 gives ~2.6% relative spread, so 15% is comfortably ~5σ."""
        numpy.random.seed(303)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=3000, Ω=numpy.matrix(4.0 * numpy.eye(2)))
        assert numpy.diff(xt, axis=1)[1, 2:].var() == pytest.approx(4.0, rel=0.15)

    def test_vecm_source_returns_time_and_values(self):
        a = numpy.zeros((2, 2, 2))
        numpy.random.seed(404)
        t, xt = facade.create_vecm_source(LAMBDA, BETA, a, npts=80)
        assert t.shape == (80,)
        assert xt.shape == (2, 80)
        assert_allclose(t, numpy.arange(80.0))

    def test_vecm_source_defaults_to_1000_points_and_unit_noise(self):
        """create_vecm_source sizes its identity Ω default from ``_, n, _ =
        a.shape`` — the (m, n, n) unpack, a different code path from
        create_vecm1_source's (n, n) — so m ≠ n pins which axis is the series
        count. Ω = I leaves Δx₂ = ε₂ with unit variance."""
        a = numpy.zeros((3, 2, 2))       # three lags, two series
        numpy.random.seed(909)
        t, xt = facade.create_vecm_source(LAMBDA, BETA, a)
        assert isinstance(xt, numpy.ndarray)
        assert not isinstance(xt, numpy.matrix)      # unwrapped for the notebooks
        assert t.shape == (1000,)
        assert xt.shape == (2, 1000)
        assert xt.dtype == numpy.float64
        assert_allclose(t, numpy.arange(1000.0))
        # the recursion starts at i = m + 1 = 4; 996 increments give the variance
        # a √(2/996) ≈ 4.5% spread, so 18% is ~4σ
        assert numpy.diff(xt, axis=1)[1, 4:].var() == pytest.approx(1.0, rel=0.18)

    def test_vecm_source_agrees_with_the_model_layer(self):
        a = numpy.array([[[0.4, 0.0], [0.0, 0.3]], [[-0.2, 0.0], [0.0, 0.1]]])
        numpy.random.seed(606)
        _, from_facade = facade.create_vecm_source(LAMBDA, BETA, a, npts=150, Ω=OMEGA)
        numpy.random.seed(606)
        from_model = numpy.array(model.vecm(LAMBDA, BETA, a, OMEGA, 150))
        assert_allclose(from_facade, from_model, rtol=1e-12, atol=1e-12)

    def test_ensemble_variance_of_the_random_walk_component(self):
        """Var(x₂,t) = (t-1)Ω₂₂ and E[x₂,t] = 0 over an ensemble of independent
        paths. 300 paths give the variance a 8.2% relative spread, so 30% is
        ~3.7σ; the mean is checked at 4 standard errors."""
        npts, nsim = 150, 300
        numpy.random.seed(707)
        t, ensemble = create_ensemble(
            lambda **kwargs: facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, **kwargs),
            nsim, npts=npts,
        )
        assert t.shape == (npts,)
        assert len(ensemble) == nsim

        paths = numpy.array(ensemble)
        assert paths.shape == (nsim, 2, npts)
        for index in (75, 149):
            walk = paths[:, 1, index]
            assert walk.var() == pytest.approx(index - 1, rel=0.30)
            assert abs(walk.mean()) < 4.0 * numpy.sqrt((index - 1) / nsim)


# ############################################################################
# Façade: estimation
# ############################################################################

class TestEstimateFacade:

    def test_transposes_samples_and_agrees_with_the_model_layer(self, sim_vecm1, fit_vecm1):
        """The façade takes (nseries, npts) — what the sources emit — and hands
        statsmodels the transpose."""
        _, xt = sim_vecm1
        result, est = fit_vecm1
        direct = model.fit(xt.T, maxlags=1, rank=1, trend="co")

        assert isinstance(result, VECMResults)
        assert isinstance(est, VECMEst)
        assert result.neqs == 2 and result.nobs == 1998
        assert_allclose(result.alpha, direct.alpha, rtol=1e-12)
        assert_allclose(result.beta, direct.beta, rtol=1e-12)
        assert_allclose(result.gamma, direct.gamma, rtol=1e-12)

    def test_defaults_are_twelve_lags_rank_one_and_a_constant(self, sim_vecm1):
        _, xt = sim_vecm1
        result, est = facade.compute_estimate(xt)
        assert result.k_ar == 13            # maxlags 12 lagged differences
        assert result.coint_rank == 1
        assert result.det_coef.shape == (2, 1)   # trend "co"
        assert est.rank == 1
        assert est.order == 12

    def test_estimate_report_mirrors_the_statsmodels_result(self, fit_vecm1):
        result, est = fit_vecm1

        assert est.est_model == EstModel.VECM
        assert est.rank == 1
        assert est.order == 1
        assert len(est.lambda_est) == 2      # neqs × rank
        assert len(est.beta_est) == 2
        assert len(est.const) == 2
        assert len(est.omega) == 4           # neqs × neqs
        assert len(est.a_est) == 4           # neqs × neqs × order

        for i in range(2):
            λ = find_param(est.lambda_est, i, 0)
            assert λ.est == result.alpha[i, 0] and λ.err == result.stderr_alpha[i, 0]
            assert λ.param_type == VECMParamType.VECM_LAMBDA.value

            β = find_param(est.beta_est, i, 0)
            assert β.est == result.beta[i, 0] and β.err == result.stderr_beta[i, 0]
            assert β.param_type == VECMParamType.VECM_BETA.value

            c = find_param(est.const, i, 0)
            assert c.est == result.det_coef[i, 0] and c.err == result.stderr_det_coef[i, 0]
            assert c.param_type == VECMParamType.VECM_CONST.value

            for j in range(2):
                a = find_param(est.a_est, i, j, order=1)
                assert a.est == result.gamma[i, j] and a.err == result.stderr_gamma[i, j]
                assert a.param_type == VECMParamType.VECM_ALPHA.value

                ω = find_param(est.omega, i, j)
                # Ω has no reported standard error
                assert ω.est == result.sigma_u[i, j] and ω.err == 0.0
                assert ω.param_type == VECMParamType.VECM_OMEGA.value

    def test_rank_two_reports_every_column_of_lambda_and_beta(self, sim_rank_two):
        """Three series at rank 2: the λ/β loop has to walk 3×2 entries rather
        than the single column every other estimation test exercises, and Γ̂,
        Ω̂ and the constant grow with the series count."""
        _, xt = sim_rank_two
        result, est = facade.compute_estimate(xt, maxlags=1, rank=2, trend="co")

        assert est.rank == 2 and est.order == 1
        assert len(est.lambda_est) == 6 and len(est.beta_est) == 6
        assert len(est.a_est) == 9 and len(est.omega) == 9
        assert len(est.const) == 3
        assert {(p.row, p.column) for p in est.lambda_est} == {(i, j) for i in range(3) for j in range(2)}

        # cache_readonly descriptors since statsmodels 0.15, so read them as arrays
        alpha, s_alpha = numpy.asarray(result.alpha), numpy.asarray(result.stderr_alpha)
        beta, s_beta = numpy.asarray(result.beta), numpy.asarray(result.stderr_beta)
        sigma_u, gamma = numpy.asarray(result.sigma_u), numpy.asarray(result.gamma)
        for i in range(3):
            for j in range(2):
                λ = find_param(est.lambda_est, i, j)
                assert λ.est == alpha[i, j] and λ.err == s_alpha[i, j]
                β = find_param(est.beta_est, i, j)
                assert β.est == beta[i, j] and β.err == s_beta[i, j]
            for j in range(3):
                ω = find_param(est.omega, i, j)
                assert ω.est == sigma_u[i, j]
                a = find_param(est.a_est, i, j, order=1)
                assert a.est == gamma[i, j]

        # the second cointegration vector is not a copy of the first, so the
        # column index really is being read
        assert [p.est for p in est.beta_est if p.column == 1] != [p.est for p in est.beta_est if p.column == 0]

    def test_supports_a_linear_trend_outside_the_relation(self, sim_vecm1):
        """"lo" is the one documented trend besides "co" that compute_estimate
        survives: det_coef keeps shape (neqs, 1), but it now holds the linear
        trend slope rather than an intercept. The slope multiplies t, so its
        standard error falls like n^{-3/2} against the intercept's n^{-1/2} —
        three orders of magnitude apart on 2000 points."""
        _, xt = sim_vecm1
        result, est = facade.compute_estimate(xt, maxlags=1, rank=1, trend="lo")
        constant, _ = facade.compute_estimate(xt, maxlags=1, rank=1, trend="co")

        assert result.det_coef.shape == (2, 1)
        assert result.det_coef_coint.shape == (0, 1)
        assert est.rank == 1 and est.order == 1 and len(est.const) == 2
        det_coef = numpy.asarray(result.det_coef)
        s_det = numpy.asarray(result.stderr_det_coef)
        for i in range(2):
            c = find_param(est.const, i, 0)
            assert c.est == det_coef[i, 0]
            assert c.err == s_det[i, 0]
        assert numpy.abs(s_det).max() < 0.01 * numpy.abs(numpy.asarray(constant.stderr_det_coef)).max()

    def test_linear_trend_coefficient_is_not_labelled_as_the_constant(self, sim_vecm1):
        _, xt = sim_vecm1
        _, trend_est = facade.compute_estimate(xt, maxlags=1, rank=1, trend="lo")
        _, const_est = facade.compute_estimate(xt, maxlags=1, rank=1, trend="co")
        assert ({(p.param_type, p.est_label) for p in trend_est.const}
                != {(p.param_type, p.est_label) for p in const_est.const})

    def test_estimate_report_labels_are_latex(self, fit_vecm1):
        _, est = fit_vecm1
        labels = {(p.est_label, p.err_label) for p in est.lambda_est}
        assert labels == {("$\\hat{\\lambda}$", "$\\sigma_{\\lambda}$")}
        assert {(p.est_label, p.err_label) for p in est.beta_est} == {("$\\hat{\\beta}$", "$\\sigma_{\\beta}$")}
        assert {p.est_label for p in est.a_est} == {"$\\hat{A}$"}
        assert {p.est_label for p in est.omega} == {"$\\hat{\\Omega}$"}
        assert {p.est_label for p in est.const} == {"$\\hat{M}$"}

    def test_all_estimates_share_one_estimate_id(self, fit_vecm1):
        _, est = fit_vecm1
        ids = {p.est_id for p in est.lambda_est + est.beta_est + est.a_est + est.omega + est.const}
        assert len(ids) == 1

    def test_estimate_serialises_and_a_param_round_trips(self, fit_vecm1):
        _, est = fit_vecm1
        payload = json.loads(est.to_json())
        assert set(payload) == {"est_model", "rank", "order", "const", "lambda_est", "beta_est", "a_est", "omega"}
        assert payload["est_model"] == EstModel.VECM.value
        assert payload["rank"] == est.rank and payload["order"] == est.order
        assert len(payload["lambda_est"]) == 2

        restored = ParamEst.from_dict(payload["lambda_est"][0])
        original = est.lambda_est[0]
        assert restored.est == original.est
        assert restored.err == original.err
        assert restored.row == original.row and restored.column == original.column
        assert restored.order == original.order
        assert restored.est_id == original.est_id
        assert restored.param_type == original.param_type
        assert json.loads(est.to_json(pretty=True)) == payload

    def test_omega_estimates_serialise_with_their_own_literal(self, fit_vecm1):
        """A persisted VECM noise covariance carries a VECM discriminator, so it
        is distinguishable from a VAR result's omega rows."""
        _, est = fit_vecm1
        payload = json.loads(est.to_json())
        assert {p["param_type"] for p in payload["omega"]} == {"VECM_OMEGA"}
        assert {p["param_type"] for p in payload["lambda_est"]} == {"VECM_LAMBDA"}

    def test_omega_param_type_is_distinct_from_the_var_one(self):
        assert VECMParamType.VECM_OMEGA.value != VARParamType.VAR_OMEGA.value

    def test_estimate_repr_names_the_model(self, fit_vecm1):
        _, est = fit_vecm1
        text = repr(est)
        assert text.startswith("VECMEst(")
        assert "est_model=(EstModel.VECM)" in text
        assert str(est) in text

    @pytest.mark.parametrize("trend", ["n", "ci", "li"])
    def test_supports_trends_without_a_constant_outside_the_relation(self, sim_vecm1, trend):
        _, xt = sim_vecm1
        _, est = facade.compute_estimate(xt[:, :600], maxlags=1, rank=1, trend=trend)
        assert est.rank == 1

    def test_reports_a_distinct_coefficient_block_per_lag(self, sim_vecm1):
        _, xt = sim_vecm1
        result, est = facade.compute_estimate(xt, maxlags=2, rank=1, trend="co")
        assert est.order == 2 and len(est.a_est) == 8
        for i in range(2):
            for j in range(2):
                assert find_param(est.a_est, i, j, order=2).est == result.gamma[i, j + 2]


# ############################################################################
# Façade: order selection
# ############################################################################

class TestOrderFacade:

    def test_returns_result_and_report(self, sim_vecm1):
        """BIC recovers the one lagged difference the process was simulated
        with; HQIC is held to a bound for the reason given on
        ``TestOrderEstimate.test_selects_the_true_number_of_lagged_differences``."""
        _, xt = sim_vecm1
        result, report = facade.compute_lag_order(xt, maxlags=6)
        assert isinstance(result, LagOrderResults)
        assert isinstance(report, VAROrderTestReport)
        assert result.bic == 1 and result.hqic <= 1

    def test_agrees_with_the_model_layer_on_transposed_samples(self, sim_vecm1):
        _, xt = sim_vecm1
        result, _ = facade.compute_lag_order(xt, maxlags=6, trend="co")
        direct = model.lag_order_estimate(xt.T, 6, "co")
        assert (result.aic, result.bic, result.fpe, result.hqic) == (direct.aic, direct.bic, direct.fpe, direct.hqic)
        assert_allclose(result.ics["aic"], direct.ics["aic"], rtol=1e-12)

    def test_default_trend_is_a_constant_outside_the_relation(self, sim_vecm1):
        """compute_lag_order defaults to trend "co" where the model layer's
        lag_order_estimate defaults to "n"; the two score every candidate order
        differently, so the façade default is not a pass through of the model
        default."""
        _, xt = sim_vecm1
        result, _ = facade.compute_lag_order(xt, maxlags=6)
        assert_allclose(result.ics["aic"], model.lag_order_estimate(xt.T, 6, "co").ics["aic"], rtol=1e-12)
        assert_allclose(result.ics["bic"], model.lag_order_estimate(xt.T, 6, "co").ics["bic"], rtol=1e-12)
        assert not numpy.allclose(result.ics["aic"], model.lag_order_estimate(xt.T, 6, "n").ics["aic"])

    def test_default_maxlags_scores_thirteen_orders(self, sim_vecm1):
        _, xt = sim_vecm1
        result, _ = facade.compute_lag_order(xt)
        assert len(result.ics["aic"]) == 13

    def test_report_carries_each_criterion_and_its_minimum(self, sim_vecm1):
        """The report's per-criterion order is the criterion's selection and its
        value is that criterion evaluated there, i.e. its minimum. The headline
        order is the modal selection of AIC, BIC and HQIC — here 1, the true
        number of lagged differences."""
        _, xt = sim_vecm1
        result, report = facade.compute_lag_order(xt, maxlags=6)
        payload = json.loads(report.to_json())
        prefix = "_VAROrderTestReport__"

        assert payload[prefix + "order"]["value"] == 1
        for name, metric in (("aic", "AIC"), ("bic", "BIC"), ("fpe", "FPE"), ("hqic", "HQIC")):
            entry = payload[prefix + name]
            selected = getattr(result, name)
            assert entry["error_metric"] == metric
            assert entry["order"]["value"] == selected
            assert entry["value"]["value"] == pytest.approx(min(result.ics[name]))
            assert entry["value"]["value"] == pytest.approx(result.ics[name][selected])

        test_ids = {payload[prefix + "test_id"]} | {payload[prefix + n]["order"]["test_id"] for n in ("aic", "bic")}
        assert len(test_ids) == 1
        assert repr(report).startswith("VAROrderTestReport(")


# ############################################################################
# Façade: Johansen cointegration test
# ############################################################################

@pytest.fixture(scope="module")
def coint_test(sim_vecm1):
    """Johansen test of the cointegrated pair, through the façade."""
    _, xt = sim_vecm1
    return facade.compute_johansen_coint_test(xt, 2)


class TestJohansenFacade:

    def test_returns_text_report_data_report_and_raw_result(self, coint_test):
        text_report, report, result = coint_test
        assert isinstance(text_report, JohansenTestReport)
        assert isinstance(report, JohansenCointTestReport)
        assert isinstance(result, JohansenTestResult)
        assert all(isinstance(s, JohansenCointTestStatistic) for s in report.trace_test)
        assert all(isinstance(s, JohansenCointTestStatistic) for s in report.eigen_test)
        assert all(isinstance(v, JohansenCointTestEigenVector) for v in report.eigen_vectors)
        assert isinstance(report.ranks, JohansenCointTestRank)

    def test_agrees_with_the_model_layer_on_transposed_samples(self, sim_vecm1, coint_test):
        _, xt = sim_vecm1
        _, _, result = coint_test
        direct = model.johansen_test_coint(xt.T, 2, 0)   # trend defaults to 0
        assert_allclose(result.lr1, direct.lr1, rtol=1e-12)
        assert_allclose(result.evec, direct.evec, rtol=1e-12)

    def test_trend_kwarg_reaches_the_model_layer(self, sim_vecm1):
        _, xt = sim_vecm1
        _, _, no_trend = facade.compute_johansen_coint_test(xt, 2, trend=-1)
        direct = model.johansen_test_coint(xt.T, 2, -1)
        assert_allclose(no_trend.cvt, direct.cvt, rtol=1e-12)
        assert_allclose(numpy.real(no_trend.eig), numpy.real(direct.eig), rtol=1e-12)
        assert not numpy.allclose(no_trend.cvt, model.johansen_test_coint(xt.T, 2, 0).cvt)

    def test_statistic_flags_every_critical_value_it_exceeds(self):
        """JohansenCointTestStatistic on hand picked numbers, so the comparison
        rule is pinned without re-deriving it from a test result: the flag is a
        strict inequality, so a statistic equal to its critical value does not
        reject."""
        stat = JohansenCointTestStatistic("id", 3, 10.0, numpy.array([5.0, 10.0, 15.0]))
        assert stat.test_rank == 3
        assert stat.null_hypothesis == "r<=3"
        assert stat.test_stat == 10.0
        assert stat.critical_values == [5.0, 10.0, 15.0]
        assert stat.test_result == [True, False, False]
        assert stat.significance_levels == ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
        assert repr(stat).startswith("JohansenCointTestStatistic(")

    def test_report_rank_is_the_most_conservative_per_level_rank(self):
        """JohansenCointTestReport reduces the per level ranks to one headline
        rank, the one the strictest level still supports."""
        ranks = JohansenCointTestRank("id", [3, 2, 1])
        report = JohansenCointTestReport("id", [], [], ranks, [])
        assert report.rank == 1
        assert report.ranks.significance_levels == ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]

    def test_statistics_report_mirrors_the_raw_result(self, coint_test):
        """Index wiring: entry i carries row i of lr1/cvt for the trace test and
        of lr2/cvm for the maximum eigenvalue test. The critical values are
        pinned against Osterwald-Lenum's published table rather than against the
        result object, and the flags against the structural invariant that a
        stricter level can only be harder to reject."""
        _, report, result = coint_test
        assert len(report.trace_test) == 2
        assert len(report.eigen_test) == 2
        # two series with a constant; the r <= 1 row is the χ²(1) quantile set
        assert report.trace_test[0].critical_values == pytest.approx([13.4294, 15.4943, 19.9349])
        assert report.trace_test[1].critical_values == pytest.approx([2.7055, 3.8415, 6.6349])
        assert report.eigen_test[0].critical_values == pytest.approx([12.2971, 14.2639, 18.52])
        assert report.eigen_test[1].critical_values == pytest.approx([2.7055, 3.8415, 6.6349])

        for i, stat in enumerate(report.trace_test):
            assert stat.test_rank == i
            assert stat.null_hypothesis == f"r<={i}"
            assert stat.test_stat == result.lr1[i]
            assert stat.critical_values == result.cvt[i].tolist()
            assert stat.significance_levels == ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
            assert stat.critical_values == sorted(stat.critical_values)
            assert stat.test_result == sorted(stat.test_result, reverse=True)
        for i, stat in enumerate(report.eigen_test):
            assert stat.test_rank == i
            assert stat.test_stat == result.lr2[i]
            assert stat.critical_values == result.cvm[i].tolist()
            assert stat.significance_levels == ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
            assert stat.test_result == sorted(stat.test_result, reverse=True)

    def test_eigen_test_null_hypothesis_is_not_the_trace_null(self, coint_test):
        _, report, _ = coint_test
        assert ([s.null_hypothesis for s in report.eigen_test]
                != [s.null_hypothesis for s in report.trace_test])

    def test_rank_report_counts_the_rejected_nulls(self, coint_test):
        """The rank at a significance level is how many trace nulls that level
        rejects.

        The counter in the façade and the per statistic ``test_result`` flags
        compare statistic to critical value independently of each other, so they
        have to agree. Only the r = 0 rejection is asserted outright — it holds
        by an order of magnitude — because the χ²(1) values statsmodels
        tabulates for the r ≤ 1 row are rejected on a sizeable share of seeds.
        The headline rank's own reduction rule is pinned separately in
        ``test_report_rank_is_the_most_conservative_per_level_rank``.
        """
        _, report, _ = coint_test
        ranks = report.ranks.test_ranks

        assert report.trace_test[0].test_result == [True, True, True]
        counted = [sum(stat.test_result[level] for stat in report.trace_test)
                   for level in range(len(ranks))]
        assert list(ranks) == counted
        # every level rejects the r = 0 null, so no level can report rank 0
        assert report.rank >= 1

    def test_text_report_ranks_every_significance_level(self, coint_test):
        """The text report's rank is the index of the last rejected null plus
        one, computed here from the raw statistics for all three levels."""
        text_report, _, result = coint_test
        rejected = [numpy.nonzero(result.lr1 > result.cvt[:, level])[0] for level in range(3)]
        expected = [int(hits.max()) + 1 if len(hits) > 0 else 0 for hits in rejected]
        assert [int(r) for r in text_report.compute_rank()] == expected

    def test_eigenvalues_are_reported_as_reals(self, coint_test):
        _, report, result = coint_test
        assert [v.eigen_value for v in report.eigen_vectors] == pytest.approx(numpy.real(result.eig))
        assert all(isinstance(v.eigen_value, float) for v in report.eigen_vectors)
        assert all(len(v.eigen_vector) == 2 for v in report.eigen_vectors)

    def test_report_serialises(self, coint_test):
        _, report, _ = coint_test
        payload = json.loads(report.to_json())
        assert set(payload) == {"test_id", "trace_test", "eigen_test", "ranks", "eigen_vectors", "rank"}
        assert payload["rank"] == report.rank
        assert payload["ranks"]["test_ranks"] == list(report.ranks.test_ranks)
        assert payload["ranks"]["significance_levels"] == ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
        assert payload["trace_test"][0]["test_result"] == [True, True, True]
        assert len(payload["trace_test"]) == 2
        assert payload["trace_test"][0]["null_hypothesis"] == "r<=0"
        assert payload["eigen_vectors"][0]["eigen_value"] == pytest.approx(report.eigen_vectors[0].eigen_value)
        assert json.loads(report.to_json(pretty=True)) == payload

    def test_text_report_summary_tabulates_every_section(self, coint_test):
        text_report, _, result = coint_test
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            text_report.summary()
        printed = buffer.getvalue()
        for heading in ("Trace Statistic", "Rank", "Eigenvalue Statistic", "Eigenvalues and Eigenvectors"):
            assert heading in printed
        assert "r <= 0" in printed and "r <= 1" in printed
        assert f"{result.lr1[0]:.3f}" in printed

    def test_text_report_summary_prints_the_eigenvector_columns(self, coint_test):
        """The eigenvectors are the *columns* of evec, and reports.py prints
        ``evec[:, i]`` — the code path the data report gets wrong (xfailed
        below). The printed leading vector is parsed back out of the table and
        checked against the simulated β = [1, -1]."""
        text_report, _, _ = coint_test
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            text_report.summary()
        section = buffer.getvalue().split("Eigenvalues and Eigenvectors")[1]
        vectors = re.findall(r"\[([^\]]*)\]", section)
        assert len(vectors) == 2
        leading = [complex(token).real for token in vectors[0].split()]
        assert len(leading) == 2
        assert_allclose([v / leading[0] for v in leading], [1.0, -1.0], atol=0.05)

    def test_rank_is_reported_for_every_significance_level(self, coint_test):
        _, report, _ = coint_test
        assert len(report.ranks.test_ranks) == len(report.ranks.significance_levels)

    def test_handles_systems_with_more_series_than_significance_levels(self):
        zeros_λ = numpy.matrix(numpy.zeros((4, 1)))
        zeros_β = numpy.matrix(numpy.zeros((1, 4)))
        numpy.random.seed(2024)
        _, xt = facade.create_vecm1_source(zeros_λ, zeros_β, numpy.matrix(numpy.zeros((4, 4))), npts=400)
        _, report, _ = facade.compute_johansen_coint_test(xt, 1)
        assert report.rank == 0

    def test_reported_eigenvector_is_the_cointegration_vector(self, coint_test):
        _, report, _ = coint_test
        leading = numpy.array(report.eigen_vectors[0].eigen_vector)
        assert_allclose(leading / leading[0], [1.0, -1.0], atol=0.05)


# ############################################################################
# Façade: prediction
# ############################################################################

class TestPredictionFacade:

    def test_matches_the_model_layer(self, fit_vecm1):
        result, _ = fit_vecm1
        from_facade = facade.compute_prediction(result, 6)
        from_model = model.predict(result, 6, alpha=0.05)
        assert len(from_facade) == 3
        for actual, expected in zip(from_facade, from_model):
            assert actual.shape == (6, 2)
            assert_allclose(actual, expected, rtol=1e-12)

    def test_alpha_kwarg_widens_the_interval(self, fit_vecm1):
        """Default α = 0.05; a smaller α must give a strictly wider band in the
        exact ratio of the normal quantiles."""
        result, _ = fit_vecm1
        forecast, lower, upper = facade.compute_prediction(result, 4)
        tight_forecast, tight_lower, tight_upper = facade.compute_prediction(result, 4, alpha=0.01)

        assert_allclose(forecast, tight_forecast, rtol=1e-12)
        assert numpy.all(tight_upper > upper)
        assert numpy.all(tight_lower < lower)
        assert_allclose((tight_upper - tight_forecast) / (upper - forecast),
                        norm.ppf(0.995) / norm.ppf(0.975), rtol=1e-10)
