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
from lib.data.param_est import EstModel, ParamEst, VECMEst, VECMParamType
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
        estimate a relative spread of √(2/n) ≈ 2.2%; 10% is ~4.5σ."""
        numpy.random.seed(2718)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=4000, Ω=OMEGA)
        Δ = numpy.diff(xt, axis=1)[:, 2:]
        cov = numpy.cov(Δ)

        βΩβ = (numpy.array(BETA) @ numpy.array(OMEGA) @ numpy.array(BETA).T).item()
        var_z = βΩβ / (1.0 - (1.0 - KAPPA) ** 2)

        assert cov[1, 1] == pytest.approx(OMEGA[1, 1], rel=0.10)
        assert cov[0, 1] == pytest.approx(OMEGA[0, 1], rel=0.25)
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

    def test_rank_two_cointegration_has_lyapunov_covariance(self):
        """Three series, rank 2: z = βx is a stationary VAR(1) with matrix
        I + βλ, and Cov(z) solves the discrete Lyapunov equation.  Off diagonal
        entries are compared on an absolute scale — their sampling spread,
        √((σ₁₁σ₂₂ + σ₁₂²)/n_eff) ≈ 0.16, dwarfs their size."""
        numpy.random.seed(4477)
        _, xt = facade.create_vecm1_source(LAMBDA2, BETA2, A2_ZERO, npts=4000)
        z = numpy.array(BETA2) @ xt

        A = numpy.eye(2) + numpy.array(BETA2) @ numpy.array(LAMBDA2)
        Q = numpy.array(BETA2) @ numpy.array(BETA2).T
        expected = lyapunov(A, Q)
        sample = numpy.cov(z[:, 2:])

        assert_allclose(numpy.diag(sample), numpy.diag(expected), rtol=0.20)
        assert_allclose(sample[0, 1], expected[0, 1], atol=0.6)

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
        assert numpy.abs(xt[:, 4]).min() > 0.0


# ############################################################################
# Model layer: estimation
# ############################################################################

class TestFit:

    def test_recovers_known_parameters(self, sim_vecm1):
        """Round trip on 2000 points. Over seven unrelated seeds the largest
        deviations were λ 0.021, β 0.004, a 0.037, Ω 0.062, const 0.052, so the
        tolerances below sit at roughly three times the observed spread."""
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=1, rank=1, trend="co")

        assert result.k_ar == 2                 # k_ar_diff + 1
        assert result.coint_rank == 1
        assert result.neqs == 2
        # statsmodels normalises β so its first `rank` rows are the identity
        assert_allclose(result.beta[:, 0], [1.0, -1.0], atol=0.02)
        assert_allclose(result.alpha[:, 0], [-KAPPA, 0.0], atol=0.06)
        assert_allclose(result.gamma, numpy.array(A_ONE), atol=0.09)
        assert_allclose(result.sigma_u, numpy.eye(2), atol=0.12)
        assert_allclose(result.det_coef[:, 0], [0.0, 0.0], atol=0.15)

    def test_long_run_impact_matrix_is_recovered(self, sim_vecm1):
        """λ and β are identified only up to an r×r rotation; Π = λβᵀ is not."""
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=1, rank=1, trend="co")
        Π = numpy.array(LAMBDA) @ numpy.array(BETA)
        assert_allclose(result.alpha @ result.beta.T, Π, atol=0.06)

    def test_rank_two_recovers_the_cointegration_space(self):
        """β normalised on its leading 2×2 block spans the same space as
        [1,-1,0] and [0,1,-1], i.e. it is [[1,0],[0,1],[-1,-1]]."""
        numpy.random.seed(4477)
        _, xt = facade.create_vecm1_source(LAMBDA2, BETA2, A2_ZERO, npts=4000)
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

    def test_maxlags_sets_the_number_of_lagged_differences(self, sim_vecm1):
        _, xt = sim_vecm1
        result = model.fit(xt.T, maxlags=3, rank=1, trend="co")
        assert result.k_ar == 4
        assert result.gamma.shape == (2, 6)     # neqs × neqs·k_ar_diff


class TestOrderEstimate:

    def test_selects_the_true_number_of_lagged_differences(self, sim_vecm1):
        """The simulated process has one lagged difference. BIC and HQIC picked
        1 on all seven seeds tried; AIC over-selected on one of them, which is
        its documented small sample behaviour, so it is not asserted."""
        _, xt = sim_vecm1
        result = model.order_estimate(xt.T, 6, "co")
        assert isinstance(result, LagOrderResults)
        assert result.bic == 1
        assert result.hqic == 1

    def test_selects_zero_lags_for_a_pure_var1(self):
        """a = 0 leaves no lagged differences to fit."""
        numpy.random.seed(5309)
        _, xt = facade.create_vecm1_source(LAMBDA, BETA, A_ZERO, npts=2000)
        result = model.order_estimate(xt.T, 6, "co")
        assert result.bic == 0
        assert result.hqic == 0

    def test_criteria_are_reported_for_every_candidate_order(self, sim_vecm1):
        """select_order scores VAR orders 1…maxlags+1, reported as the
        equivalent lagged difference counts 0…maxlags, and the selection is the
        argmin of each criterion's list."""
        _, xt = sim_vecm1
        result = model.order_estimate(xt.T, 6, "co")
        for name in ("aic", "bic", "fpe", "hqic"):
            values = numpy.array(result.ics[name])
            assert len(values) == 7
            assert int(values.argmin()) == getattr(result, name)

    def test_default_maxlags_is_twelve(self, sim_vecm1):
        _, xt = sim_vecm1
        assert len(model.order_estimate(xt.T).ics["aic"]) == 13


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
        """Π = 0, so the r = 0 null holds; judged at 99% to keep the test's own
        false rejection rate at 1% across seeds."""
        _, xt = sim_random_walks
        result = model.johansen_test_coint(xt.T, 1)
        assert result.lr1[0] < result.cvt[0, 2]

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

    def test_estimate_repr_names_the_model(self, fit_vecm1):
        _, est = fit_vecm1
        text = repr(est)
        assert text.startswith("VECMEst(")
        assert "est_model=(EstModel.VECM)" in text
        assert str(est) in text

    @pytest.mark.parametrize("trend", ["n", "ci", "li"])
    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="compute_estimate documents trends 'n', 'ci' and 'li', but "
               "__vecm_estimate_from_result reads det_coef[i, 0] unconditionally; "
               "those trends leave det_coef with shape (neqs, 0) because the "
               "deterministic term is absent or lives in det_coef_coint",
    )
    def test_supports_trends_without_a_constant_outside_the_relation(self, sim_vecm1, trend):
        _, xt = sim_vecm1
        _, est = facade.compute_estimate(xt[:, :600], maxlags=1, rank=1, trend=trend)
        assert est.rank == 1

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="__vecm_estimate_from_result reads a_est[i, j] inside the lag loop, "
               "ignoring k, so every lag order repeats the first neqs columns of "
               "result.gamma instead of gamma[:, (k-1)*neqs:k*neqs]",
    )
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
        _, xt = sim_vecm1
        result, report = facade.compute_order(xt, maxlags=6)
        assert isinstance(result, LagOrderResults)
        assert isinstance(report, VAROrderTestReport)
        assert result.bic == 1 and result.hqic == 1

    def test_agrees_with_the_model_layer_on_transposed_samples(self, sim_vecm1):
        _, xt = sim_vecm1
        result, _ = facade.compute_order(xt, maxlags=6, trend="co")
        direct = model.order_estimate(xt.T, 6, "co")
        assert (result.aic, result.bic, result.fpe, result.hqic) == (direct.aic, direct.bic, direct.fpe, direct.hqic)
        assert_allclose(result.ics["aic"], direct.ics["aic"], rtol=1e-12)

    def test_default_maxlags_scores_thirteen_orders(self, sim_vecm1):
        _, xt = sim_vecm1
        result, _ = facade.compute_order(xt)
        assert len(result.ics["aic"]) == 13

    def test_report_carries_each_criterion_and_its_minimum(self, sim_vecm1):
        """The report's per-criterion order is the criterion's selection and its
        value is that criterion evaluated there, i.e. its minimum. The headline
        order is the modal selection of AIC, BIC and HQIC — here 1, the true
        number of lagged differences."""
        _, xt = sim_vecm1
        result, report = facade.compute_order(xt, maxlags=6)
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

    def test_statistics_report_mirrors_the_raw_result(self, coint_test):
        _, report, result = coint_test
        assert len(report.trace_test) == 2
        assert len(report.eigen_test) == 2
        for i, stat in enumerate(report.trace_test):
            assert stat.test_rank == i
            assert stat.null_hypothesis == f"r<={i}"
            assert stat.test_stat == result.lr1[i]
            assert stat.critical_values == result.cvt[i].tolist()
            assert stat.test_result == [bool(result.lr1[i] > cv) for cv in result.cvt[i]]
            assert len(stat.significance_levels) == 3
        for i, stat in enumerate(report.eigen_test):
            assert stat.test_stat == result.lr2[i]
            assert stat.critical_values == result.cvm[i].tolist()

    def test_rank_report_counts_the_rejected_nulls(self, coint_test):
        """The rank at a significance level is how many trace nulls it rejects,
        and the headline rank is the most conservative of those.

        The counter in the façade and the per statistic ``test_result`` flags
        compare statistic to critical value independently of each other, so they
        have to agree. Only the r = 0 rejection is asserted outright — it holds
        by an order of magnitude — because the χ²(1) values statsmodels
        tabulates for the r ≤ 1 row are rejected on a sizeable share of seeds.
        """
        text_report, report, _ = coint_test
        ranks = report.ranks.test_ranks

        assert report.trace_test[0].test_result == [True, True, True]
        assert report.rank == min(ranks)
        assert report.rank >= 1
        for level, rank in enumerate(ranks):
            assert rank == sum(stat.test_result[level] for stat in report.trace_test)
        # the text report counts the same rejections, taking the index of the
        # last rejected null rather than the total
        assert [int(r) for r in text_report.compute_rank()] == list(ranks)

    def test_eigenvalues_are_reported_as_reals(self, coint_test):
        _, report, result = coint_test
        assert [v.eigen_value for v in report.eigen_vectors] == pytest.approx(numpy.real(result.eig))
        assert all(isinstance(v.eigen_value, float) for v in report.eigen_vectors)
        assert all(len(v.eigen_vector) == 2 for v in report.eigen_vectors)

    def test_report_serialises(self, coint_test):
        _, report, _ = coint_test
        payload = json.loads(report.to_json())
        assert set(payload) == {"test_id", "trace_test", "eigen_test", "ranks", "eigen_vectors", "rank"}
        assert payload["rank"] == min(payload["ranks"]["test_ranks"]) == report.rank
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

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="the rank counter in __vecm_johansen_coint_test_report_from_result "
               "iterates its output over range(len(trace_statistic)) — the number "
               "of series — while the columns it indexes are the three significance "
               "levels, so a two series test reports only two of the three ranks",
    )
    def test_rank_is_reported_for_every_significance_level(self, coint_test):
        _, report, _ = coint_test
        assert len(report.ranks.test_ranks) == len(report.ranks.significance_levels)

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="the same rank counter indexes significance level columns with the "
               "series index, so any system with more than three series walks off "
               "the end of the 3 column critical value table",
    )
    def test_handles_systems_with_more_series_than_significance_levels(self):
        zeros_λ = numpy.matrix(numpy.zeros((4, 1)))
        zeros_β = numpy.matrix(numpy.zeros((1, 4)))
        numpy.random.seed(2024)
        _, xt = facade.create_vecm1_source(zeros_λ, zeros_β, numpy.matrix(numpy.zeros((4, 4))), npts=400)
        _, report, _ = facade.compute_johansen_coint_test(xt, 1)
        assert report.rank == 0

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="__vecm_johansen_coint_test_report_from_result stores evec[i], a row "
               "of the eigenvector matrix, as eigenvector i; statsmodels' "
               "eigenvectors are the columns evec[:, i], which is what "
               "JohansenTestReport.summary displays",
    )
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
