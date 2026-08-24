"""Tests for navi's VAR(n) model layer (``lib.models.var``) and its kwargs
facade (``lib.data.impl.var``), which is what the notebooks call.

Three kinds of test, in priority order:

1. Closed-form checks of the pure model functions against hand-derived values
   (AR(1)/AR(2) Yule-Walker moments, companion-form block layout, eigenvalues
   as roots of the characteristic polynomial) or an independent solver
   (``scipy.linalg.solve_discrete_lyapunov``).
2. Round trips: simulate with known parameters, estimate, compare. Tolerances
   are stated in terms of the estimator's standard error so the tests are not
   tuned to the conftest seed.
3. Contract checks on the facade: ``(t, values)`` tuples, kwargs defaults,
   input validation, the ``VAREst`` / ``VAROrderTestReport`` structure and
   their JSON serialization.
"""
import json
from types import SimpleNamespace

import numpy
import numpy.polynomial.polynomial as npoly
import pandas
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve_discrete_lyapunov

from lib.models import var as model
from lib.data.impl import var as facade
from lib.data.param_est import EstModel, ParamEst, VAREst, VARParamType
from lib.data.hyp_test import VAROrderTestReport
# The report builder is module private but is imported the same way by
# lib.data.impl.var; it is the only way to reach the majority-vote branches with
# criteria that disagree.
from lib.data.hyp_test import __var_order_test_report_from_result as build_order_report

SEED = 20260820

# The library returns numpy.matrix from the companion-form helpers and vec/unvec
# only accept numpy.matrix input, so numpy's matrix deprecation notice fires on
# nearly every call; it is not something these tests are about.
pytestmark = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")

# VAR(1): two independent AR(1) processes with coefficients 0.5 and 0.2.
PHI_DIAG1 = numpy.array([numpy.diag([0.5, 0.2])])
# VAR(1): coupled processes (the var_parameter_estimation notebook case).
PHI_COUPLED1 = numpy.array([[[0.25, 0.05],
                             [0.50, 0.15]]])
# VAR(2): two independent AR(2) processes, phi1 = 0.25, phi2 = 0.5.
PHI_DIAG2 = numpy.array([numpy.diag([0.25, 0.25]), numpy.diag([0.5, 0.5])])
# VAR(3): coupled (the var_properties notebook case).
PHI_3 = numpy.array([[[0.25, 0.50], [0.50, 0.25]],
                     [[0.15, 0.35], [0.25, 0.15]],
                     [[0.05, 0.50], [0.75, 0.10]]])
OMEGA_COUPLED = numpy.array([[2.0, 0.2],
                             [0.2, 2.0]])
MU = numpy.array([1.0, -1.0])
# VAR(2) in three endogenous variables. Both lag blocks are asymmetric and no two
# rows or columns are equal, so the (lag, row, column) -> stderr row arithmetic
# 1 + lag*m + k cannot pass under a transposed or m-hardcoded index the way the
# m = 2 diagonal cases can. Spectral radius 0.83.
PHI_3D2 = numpy.array([[[0.40, 0.10, 0.00],
                        [0.00, 0.30, 0.20],
                        [0.10, 0.00, 0.35]],
                       [[0.15, 0.00, 0.10],
                        [0.10, 0.20, 0.00],
                        [0.00, 0.15, 0.15]]])
OMEGA_3D = numpy.array([[1.0, 0.2, 0.1],
                        [0.2, 1.0, 0.3],
                        [0.1, 0.3, 1.0]])
MU_3D = numpy.array([0.5, -0.5, 1.0])


def ar2_gamma(phi1: float, phi2: float, sigma2: float = 1.0, nlags: int = 3) -> list[float]:
    """Yule-Walker autocovariances gamma_0..gamma_nlags of a scalar AR(2)."""
    rho1 = phi1 / (1.0 - phi2)
    rho = [1.0, rho1]
    for _ in range(2, nlags + 1):
        rho.append(phi1 * rho[-1] + phi2 * rho[-2])
    gamma0 = sigma2 / (1.0 - phi1 * rho[1] - phi2 * rho[2])
    return [gamma0 * r for r in rho]


def char_poly_roots(phi):
    """Roots of ``det(z^n I - z^{n-1} phi_1 - ... - phi_n)``, the reverse
    characteristic polynomial of a VAR(n).

    Built with polynomial arithmetic on the entries of the matrix polynomial, so
    it shares no code with the library's companion-form construction or with
    ``numpy.linalg.eig``. Stationarity requires every root inside the unit circle.
    Supports the 1x1 and 2x2 coefficient blocks used by these tests.
    """
    n, m, _ = phi.shape
    assert m in (1, 2)
    # c[i, j] holds the coefficients of entry (i, j) in ascending powers of z.
    c = numpy.zeros((m, m, n + 1))
    c[:, :, n] = numpy.eye(m)
    for k in range(n):
        c[:, :, n - 1 - k] -= phi[k]
    if m == 1:
        det = c[0, 0]
    else:
        det = npoly.polysub(npoly.polymul(c[0, 0], c[1, 1]),
                            npoly.polymul(c[0, 1], c[1, 0]))
    return npoly.polyroots(det)


def ma_acov(phi, omega, nlag: int, terms: int = 250):
    """Autocovariances gamma_0 .. gamma_{nlag-1} from the MA(infinity) form.

    For the companion VAR(1) ``z_t = Phi z_{t-1} + e_t`` the moving-average
    representation gives ``gamma_k = E[z_{t-k} z_t'] = sum_j Phi^j Omega
    (Phi^{j+k})'``. This shares no code with ``lib.models.var.cov`` (a Kronecker
    solve) or with ``acov``'s recursion, so it is an independent expectation for
    both gamma_0 and every lag. Only ``phi_comp`` / ``omega_comp`` are reused, and
    those are pinned separately against hard-coded block layouts. The series is
    truncated at ``terms``; for a spectral radius below 0.4 the tail is ~1e-100.
    """
    Phi = numpy.asarray(model.phi_comp(phi))
    Omega = numpy.asarray(model.omega_comp(omega, len(phi)))
    powers = [numpy.eye(Phi.shape[0])]
    for _ in range(terms + nlag):
        powers.append(powers[-1] @ Phi)
    return numpy.array([sum(powers[j] @ Omega @ powers[j + k].T for j in range(terms))
                        for k in range(nlag)])


def ols_var_stderr(xt, nlags: int):
    """Standard errors of a VAR(nlags) OLS fit with a constant, straight from the
    definition ``se(regressor k, equation j) = sqrt((Z'Z)^-1_kk sigma_jj)`` with
    ``sigma = resid' resid / (nobs - 1 - nlags m)``.

    Written from the OLS formula rather than obtained from statsmodels, so it is an
    independent check on the magnitude of the errors the facade reports (it agrees
    with ``VARResults.stderr`` to 3e-18 on these fixtures). Rows are ``[const,
    L1.y1 ... L1.ym, L2.y1 ...]`` and columns are equations, matching the
    ``result.stderr`` layout the facade indexes into.
    """
    m, npts = xt.shape
    Y = xt[:, nlags:].T
    nobs = Y.shape[0]
    Z = numpy.hstack([numpy.ones((nobs, 1))] +
                     [xt[:, nlags - lag:npts - lag].T for lag in range(1, nlags + 1)])
    ZtZ_inv = numpy.linalg.inv(Z.T @ Z)
    resid = Y - Z @ (ZtZ_inv @ Z.T @ Y)
    sigma = resid.T @ resid / (nobs - Z.shape[1])
    return numpy.sqrt(numpy.outer(numpy.diag(ZtZ_inv), numpy.diag(sigma)))


def simulate(phi, seed: int = SEED, **kwargs):
    """Seed explicitly so module-scoped fixtures don't depend on fixture order."""
    numpy.random.seed(seed)
    return facade.create_source(phi, **kwargs)


@pytest.fixture(scope="module")
def var1_sim():
    return simulate(PHI_COUPLED1, Ω=OMEGA_COUPLED, μ=MU, npts=3000)


@pytest.fixture(scope="module")
def var1_estimate(var1_sim):
    _, xt = var1_sim
    return facade.compute_estimate(xt, maxlags=1)


@pytest.fixture(scope="module")
def var2_sim():
    return simulate(PHI_DIAG2, npts=6000)


@pytest.fixture(scope="module")
def var3d_sim():
    """VAR(2) with three endogenous variables: the only fixture with m != 2."""
    return simulate(PHI_3D2, Ω=OMEGA_3D, μ=MU_3D, npts=5000)


@pytest.fixture(scope="module")
def var1_large_offset_sim():
    """VAR(1) with a stationary mean of ~[13, -4], large enough that dropping the
    constant regressor moves every information criterion by a wide margin."""
    return simulate(PHI_COUPLED1, Ω=OMEGA_COUPLED, μ=10.0 * MU, npts=1500)


# ---------------------------------------------------------------------------
# Companion forms and vec / unvec (model layer)
# ---------------------------------------------------------------------------

class TestCompanionForms:

    def test_phi_comp_var1_is_the_coefficient_matrix(self):
        Φ = model.phi_comp(PHI_COUPLED1)
        assert Φ.shape == (2, 2)
        assert_array_equal(numpy.asarray(Φ), PHI_COUPLED1[0])

    def test_phi_comp_var2_block_layout(self):
        φ1, φ2 = PHI_DIAG2
        expected = numpy.block([[φ1, φ2],
                                [numpy.eye(2), numpy.zeros((2, 2))]])
        assert_array_equal(numpy.asarray(model.phi_comp(PHI_DIAG2)), expected)

    def test_phi_comp_var3_block_layout(self):
        φ1, φ2, φ3 = PHI_3
        I, Z = numpy.eye(2), numpy.zeros((2, 2))
        expected = numpy.block([[φ1, φ2, φ3],
                                [I, Z, Z],
                                [Z, I, Z]])
        Φ = model.phi_comp(PHI_3)
        assert Φ.shape == (6, 6)
        assert_array_equal(numpy.asarray(Φ), expected)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mean_comp_pads_offset_with_zeros(self, n):
        μ = numpy.array([1.0, 2.0, 3.0])
        Μ = model.mean_comp(μ, n)
        assert Μ.shape == (3 * n, 1)
        assert_array_equal(numpy.asarray(Μ)[:, 0], numpy.concatenate([μ, numpy.zeros(3 * (n - 1))]))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_omega_comp_places_noise_cov_in_top_left_block(self, n):
        ω = numpy.array([[1.0, 0.5], [0.5, 1.0]])
        Ω = numpy.asarray(model.omega_comp(ω, n))
        assert Ω.shape == (2 * n, 2 * n)
        assert_array_equal(Ω[:2, :2], ω)
        assert numpy.count_nonzero(Ω) == 4  # nothing outside the top-left block

    def test_vec_stacks_columns(self):
        A = numpy.matrix([[1.0, 4.0, 7.0],
                          [2.0, 5.0, 8.0],
                          [3.0, 6.0, 9.0]])
        v = model.vec(A)
        assert v.shape == (9, 1)
        assert_array_equal(numpy.asarray(v)[:, 0], numpy.arange(1.0, 10.0))
        assert_array_equal(numpy.asarray(v), numpy.asarray(A).reshape(-1, 1, order="F"))

    def test_unvec_inverts_vec(self):
        A = numpy.matrix(numpy.arange(16.0).reshape(4, 4) ** 2)
        assert_array_equal(numpy.asarray(model.unvec(model.vec(A))), numpy.asarray(A))
        col = numpy.matrix(numpy.array([1.0, 2.0, 3.0, 4.0])).T
        assert_array_equal(numpy.asarray(model.unvec(col)), [[1.0, 3.0], [2.0, 4.0]])


# ---------------------------------------------------------------------------
# Closed-form stationary moments and eigenvalues (model layer)
# ---------------------------------------------------------------------------

class TestClosedForms:

    def test_mean_var1_independent(self):
        μ = numpy.array([1.0, 2.0])
        # x_i = mu_i / (1 - phi_i)
        assert_allclose(numpy.asarray(model.mean(PHI_DIAG1, μ))[:, 0], [1.0 / 0.5, 2.0 / 0.8])

    def test_mean_var1_coupled_solves_linear_system(self):
        expected = numpy.linalg.solve(numpy.eye(2) - PHI_COUPLED1[0], MU)
        assert_allclose(numpy.asarray(model.mean(PHI_COUPLED1, MU))[:, 0], expected)

    def test_mean_var2_repeats_over_lagged_state(self):
        μ = numpy.array([1.0, 2.0])
        # AR(2): x = mu / (1 - phi1 - phi2) = mu / 0.25; companion state is [x_t, x_{t-1}].
        assert_allclose(numpy.asarray(model.mean(PHI_DIAG2, μ))[:, 0], [4.0, 8.0, 4.0, 8.0])

    def test_mean_zero_offset_is_zero(self):
        assert_allclose(numpy.asarray(model.mean(PHI_3, numpy.zeros(2))), numpy.zeros((6, 1)), atol=1e-15)

    def test_cov_var1_independent(self):
        Σ = numpy.asarray(model.cov(PHI_DIAG1, numpy.eye(2)))
        # AR(1): gamma_0 = sigma^2 / (1 - phi^2)
        assert_allclose(Σ, numpy.diag([1.0 / (1 - 0.25), 1.0 / (1 - 0.04)]))

    def test_cov_scales_with_noise_variance(self):
        Σ1 = numpy.asarray(model.cov(PHI_DIAG1, numpy.eye(2)))
        Σ4 = numpy.asarray(model.cov(PHI_DIAG1, 4.0 * numpy.eye(2)))
        assert_allclose(Σ4, 4.0 * Σ1)

    @pytest.mark.parametrize("phi, omega", [
        (PHI_COUPLED1, OMEGA_COUPLED),
        (PHI_DIAG2, numpy.eye(2)),
        (PHI_3, numpy.array([[1.0, 0.5], [0.5, 1.0]])),
    ])
    def test_cov_solves_discrete_lyapunov_equation(self, phi, omega):
        n = len(phi)
        Φ = numpy.asarray(model.phi_comp(phi))
        Ω = numpy.asarray(model.omega_comp(omega, n))
        Σ = numpy.asarray(model.cov(phi, omega))
        assert Σ.shape == (2 * n, 2 * n)
        assert_allclose(Σ, solve_discrete_lyapunov(Φ, Ω), atol=1e-10)
        assert_allclose(Σ, Σ.T, atol=1e-12)

    def test_cov_var2_matches_ar2_yule_walker(self):
        γ = ar2_gamma(0.25, 0.5)
        Σ = numpy.asarray(model.cov(PHI_DIAG2, numpy.eye(2)))
        # Companion covariance is [[G0, G1], [G1^T, G0]] with G_k = gamma_k * I.
        assert_allclose(numpy.diag(Σ), [γ[0]] * 4)
        assert_allclose(Σ[0, 2], γ[1])
        assert_allclose(Σ[1, 3], γ[1])
        assert Σ[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert γ[0] == pytest.approx(16.0 / 9.0)

    def test_acov_var1_independent(self):
        γ = model.acov(PHI_DIAG1, numpy.eye(2), 5)
        assert γ.shape == (5, 2, 2)
        for k in range(5):
            expected = numpy.diag([0.5 ** k / 0.75, 0.2 ** k / 0.96])
            assert_allclose(γ[k], expected)

    def test_acov_matches_ma_infinity_representation(self):
        """Independent expectation for every lag: gamma[k] = sum_j Phi^j Omega
        (Phi^{j+k})', the MA(infinity) form of the companion VAR(1).

        This does not reconstruct the closed form the library evaluates (acov
        iterates a matrix power and then multiplies by cov's Kronecker solve), so
        it re-derives gamma[0] as well as the decay, and it fixes the
        Sigma Phi' / Phi Sigma orientation on its own rather than leaning on the
        sample-moment test in TestSimulation.
        """
        γ = model.acov(PHI_COUPLED1, OMEGA_COUPLED, 6)
        assert γ.shape == (6, 2, 2)
        assert_allclose(γ, ma_acov(PHI_COUPLED1, OMEGA_COUPLED, 6), atol=1e-12)
        # gamma[0] is the stationary covariance, and the lags satisfy the
        # Yule-Walker recursion gamma[k] = gamma[k-1] Phi' built from the companion
        # matrix and the k = 0 block only.
        Φ = numpy.asarray(model.phi_comp(PHI_COUPLED1))
        assert_allclose(γ[0], numpy.asarray(model.cov(PHI_COUPLED1, OMEGA_COUPLED)))
        for k in range(1, 6):
            assert_allclose(γ[k], γ[k - 1] @ Φ.T, atol=1e-12)

    def test_acov_var2_matches_ar2_autocovariances(self):
        γ_true = ar2_gamma(0.25, 0.5, nlags=4)
        γ = model.acov(PHI_DIAG2, numpy.eye(2), 5)
        assert_allclose([γ[k][0, 0] for k in range(5)], γ_true)
        assert_allclose([γ[k][1, 1] for k in range(5)], γ_true)

    def test_eig_var1_roots_of_characteristic_polynomial(self):
        φ = PHI_COUPLED1[0]
        λ = model.eig(PHI_COUPLED1)
        assert λ.shape == (2,)
        assert_allclose(λ.imag, 0.0, atol=1e-12)
        expected = numpy.roots([1.0, -numpy.trace(φ), numpy.linalg.det(φ)])
        assert_allclose(numpy.sort(λ.real), numpy.sort(expected))

    def test_eig_var2_roots_with_multiplicity(self):
        λ = model.eig(PHI_DIAG2)
        assert λ.shape == (4,)
        assert_allclose(λ.imag, 0.0, atol=1e-12)
        # Each AR(2) contributes the roots of z^2 - 0.25 z - 0.5.
        roots = numpy.sort(numpy.roots([1.0, -0.25, -0.5]))
        assert_allclose(numpy.sort(λ.real), numpy.repeat(roots, 2))

    # radius is the hand-derived spectral radius: the largest |root| of the
    # reverse characteristic polynomial. None means "no closed form" (PHI_3),
    # where the only claim is that the radius exceeds 1.
    @pytest.mark.parametrize("phi, expected, radius", [
        (PHI_DIAG1, True, 0.5),                                     # max(0.5, 0.2)
        (PHI_COUPLED1, True, 0.5 * (0.4 + numpy.sqrt(0.11))),       # z^2 - 0.4z + 0.0125
        (PHI_DIAG2, True, 0.5 * (0.25 + numpy.sqrt(2.0625))),       # z^2 - 0.25z - 0.5
        (numpy.array([numpy.eye(2)]), False, 1.0),                  # random walk
        (numpy.array([numpy.diag([0.5, 0.5]), numpy.diag([0.5, 0.5])]), False, 1.0),  # unit root
        (numpy.array([[[1.2]]]), False, 1.2),                       # explosive
        (PHI_3, False, None),                                       # notebook case
    ])
    def test_is_stationary(self, phi, expected, radius):
        result = model.is_stationary(phi)
        assert isinstance(result, bool)
        assert result is expected
        # Independent anchor for the decision: the spectral radius from the
        # characteristic polynomial, which does not go through phi_comp/eig.
        roots = char_poly_roots(phi)
        assert len(roots) == phi.shape[0] * phi.shape[1]
        if radius is None:
            assert numpy.abs(roots).max() > 1.0
        else:
            assert numpy.abs(roots).max() == pytest.approx(radius, abs=1e-6)
        # The whole spectrum, not just its radius: model.eig (phi_comp followed by
        # numpy.linalg.eig) must reproduce every root of the reverse characteristic
        # polynomial, including the complex pair and the 6x6 companion matrix of
        # PHI_3 and the repeated roots of the random-walk / unit-root cases. atol
        # 1e-6 is set by the repeated roots, where both routines lose half their
        # digits (worst observed deviation 1.2e-8; PHI_3 agrees to 1.3e-15).
        assert_allclose(numpy.sort_complex(model.eig(phi).astype(complex)),
                        numpy.sort_complex(roots.astype(complex)), atol=1e-6)


# ---------------------------------------------------------------------------
# Simulation (model layer)
# ---------------------------------------------------------------------------

class TestSimulation:

    def test_output_shape_and_initial_condition(self):
        x0 = numpy.array([[1.0, -2.0], [3.0, 4.0]])
        xt = model.var(x0, numpy.zeros(2), PHI_DIAG2, numpy.eye(2), 50)
        assert xt.shape == (2, 50)
        assert_array_equal(xt[:, :2], x0.T)

    def test_noise_free_var1_recursion(self):
        # With zero noise covariance x_t = mu + phi x_{t-1} exactly.
        φ = numpy.array([[[0.5, 0.0], [0.0, 0.2]]])
        μ = numpy.array([1.0, 0.0])
        xt = model.var(numpy.array([[2.0, 1.0]]), μ, φ, numpy.zeros((2, 2)), 8)
        k = numpy.arange(8)
        # x1: fixed point 2, starts at 2 -> constant; x2: 0.2^k
        assert_allclose(xt[0], numpy.full(8, 2.0))
        assert_allclose(xt[1], 0.2 ** k)

    def test_noise_free_var2_recursion_uses_correct_lag_order(self):
        φ = numpy.array([[[0.5]], [[0.25]]])
        xt = model.var(numpy.array([[1.0], [2.0]]), numpy.zeros(1), φ, numpy.zeros((1, 1)), 7)
        expected = [1.0, 2.0]
        for _ in range(5):
            expected.append(0.5 * expected[-1] + 0.25 * expected[-2])
        assert_allclose(xt[0], expected)

    def test_var1_sample_moments_match_stationary_values(self, var1_sim):
        _, xt = var1_sim
        n = xt.shape[1]
        mean = numpy.asarray(model.mean(PHI_COUPLED1, MU))[:, 0]
        Σ = numpy.asarray(model.cov(PHI_COUPLED1, OMEGA_COUPLED))
        # Std error of the sample mean ~ sqrt(gamma_0 (1+rho)/(1-rho) / n) ~ 0.04 with
        # n = 3000, rho ~ 0.3; atol 0.2 is ~5 sigma.
        assert_allclose(xt.mean(axis=1), mean, atol=0.2)
        # Std error of each covariance entry ~ gamma_0 sqrt(2/n_eff) ~ 0.07 (the
        # off-diagonal entry is small, so use an absolute tolerance); atol 0.4 is ~5.5 sigma.
        assert_allclose(numpy.cov(xt), Σ, atol=0.4)
        # Lag-1 autocovariance E[x_{t-1} x_t^T] = Sigma Phi^T = acov[1]; std error ~ 0.06.
        γ1 = model.acov(PHI_COUPLED1, OMEGA_COUPLED, 2)[1]
        dx = xt - mean[:, None]
        sample_γ1 = dx[:, :-1] @ dx[:, 1:].T / (n - 1)
        assert_allclose(sample_γ1, γ1, atol=0.3)
        # Orientation: acov[1] is E[x_{t-1} x_t'] = Sigma Phi', not Phi Sigma. For this
        # coupled Phi the two differ by ~0.9 (min over 30 seeds 0.81), well outside the
        # 0.3 sampling tolerance, so the sample moment is what pins the convention.
        Φ = numpy.asarray(model.phi_comp(PHI_COUPLED1))
        assert not numpy.allclose(sample_γ1, Φ @ Σ, atol=0.3)

    def test_var1_independent_acf_decays_geometrically(self):
        numpy.random.seed(SEED)
        xt = model.var(numpy.zeros((1, 2)), numpy.zeros(2), PHI_DIAG1, numpy.eye(2), 4000)
        # ACF estimate std error ~ 1/sqrt(n) = 0.016; atol 0.1 is ~6 sigma.
        for k in (1, 2, 3):
            assert numpy.corrcoef(xt[0, :-k], xt[0, k:])[0, 1] == pytest.approx(0.5 ** k, abs=0.1)
            assert numpy.corrcoef(xt[1, :-k], xt[1, k:])[0, 1] == pytest.approx(0.2 ** k, abs=0.1)
        # Independent components: cross-correlation ~ N(0, 1/n).
        assert numpy.corrcoef(xt[0], xt[1])[0, 1] == pytest.approx(0.0, abs=0.1)

    def test_var2_sample_moments_match_ar2_closed_form(self, var2_sim):
        _, xt = var2_sim
        γ = ar2_gamma(0.25, 0.5)
        # AR(2) with roots 0.84 / -0.59 has a slowly decaying ACF, so the variance
        # estimate has std error ~ gamma_0 sqrt(2 sum rho_k^2 / n) ~ 3% at n = 6000;
        # rtol 0.2 is ~6 sigma.
        assert_allclose(xt.var(axis=1), [γ[0], γ[0]], rtol=0.2)
        for k in (1, 2):
            ρk = γ[k] / γ[0]
            for i in (0, 1):
                assert numpy.corrcoef(xt[i, :-k], xt[i, k:])[0, 1] == pytest.approx(ρk, abs=0.1)


# ---------------------------------------------------------------------------
# Estimation (model layer)
# ---------------------------------------------------------------------------

class TestEstimation:

    def test_fit_recovers_var1_coefficients(self, var1_sim):
        _, xt = var1_sim
        result = model.fit(xt.T, maxlags=1)
        assert result.k_ar == 1
        assert result.coefs.shape == (1, 2, 2)
        # OLS std error per coefficient is ~0.018 at n = 3000 (max error over 20
        # seeds was 0.044); atol 0.08 is ~4.5 sigma.
        assert_allclose(result.coefs[0], PHI_COUPLED1[0], atol=0.08)
        # Constant std error ~0.036, residual covariance std error ~0.05: atol 0.2 / 0.3 is ~5.5 sigma.
        assert_allclose(result.params[0], MU, atol=0.2)
        assert_allclose(result.sigma_u, OMEGA_COUPLED, atol=0.3)

    def test_fit_recovers_var2_coefficients(self, var2_sim):
        _, xt = var2_sim
        result = model.fit(xt.T, maxlags=2)
        assert result.k_ar == 2
        assert_allclose(result.coefs, PHI_DIAG2, atol=0.08)

    def test_fit_default_maxlags_sets_lag_order(self, var1_sim):
        _, xt = var1_sim
        result = model.fit(xt.T)
        assert result.k_ar == 12
        assert result.coefs.shape == (12, 2, 2)

    def test_fit_trend_n_drops_the_constant_row(self, var1_sim):
        """Pin statsmodels' params/stderr layout for trend='n' (the layout the
        facade's trend xfails assume): no deterministic rows at all, so row i is
        regressor i and coefs[0] is params transposed."""
        _, xt = var1_sim
        result = model.fit(xt.T, maxlags=1, trend="n")
        assert result.exog_names == ["L1.y1", "L1.y2"]
        assert result.params.shape == (2, 2)
        assert result.stderr.shape == (2, 2)
        assert_array_equal(result.coefs[0], numpy.asarray(result.params).T)
        # The process mean is ~[1.3, -0.4]; with no constant to absorb it the fitted
        # coefficients are pulled far away from the true Phi (min error over 30 seeds
        # was 0.32, against the 0.08 tolerance the trend='c' fit is held to).
        assert not numpy.allclose(result.coefs[0], PHI_COUPLED1[0], atol=0.08)

    def test_fit_trend_ct_prepends_const_and_trend_rows(self, var1_sim):
        """Pin the deterministic-term row ordering for trend='ct': ['const',
        'trend', L1..., L2...], i.e. lag l occupies rows 2 + 2l .. 3 + 2l."""
        _, xt = var1_sim
        result = model.fit(xt.T, maxlags=2, trend="ct")
        assert result.exog_names == ["const", "trend", "L1.y1", "L1.y2", "L2.y1", "L2.y2"]
        assert result.params.shape == (6, 2)
        params = numpy.asarray(result.params)
        for lag in (0, 1):
            assert_array_equal(result.coefs[lag], params[2 + 2 * lag:4 + 2 * lag].T)
        # A stationary series has no linear trend, so the spurious trend coefficient
        # is ~0 (max 6.4e-5 over 30 seeds; atol 0.01 is two orders of magnitude of
        # headroom) and the lag-1 coefficients still recover Phi (max error 0.05).
        assert_allclose(params[1], [0.0, 0.0], atol=0.01)
        assert_allclose(result.coefs[0], PHI_COUPLED1[0], atol=0.08)

    def test_fit_accepts_dataframe_endog(self, var1_sim):
        """fit documents its endog as a DataFrame (lib/models/var.py:328). A
        DataFrame carries its column names into the equation and regressor labels,
        and must produce the identical fit to the equivalent ndarray."""
        _, xt = var1_sim
        result = model.fit(pandas.DataFrame(xt.T, columns=["y_a", "y_b"]), maxlags=1)
        assert result.names == ["y_a", "y_b"]
        assert result.exog_names == ["const", "L1.y_a", "L1.y_b"]
        assert isinstance(result.params, pandas.DataFrame)
        assert_array_equal(result.coefs, model.fit(xt.T, maxlags=1).coefs)
        assert_allclose(result.coefs[0], PHI_COUPLED1[0], atol=0.08)

    @pytest.mark.parametrize("fixture, order", [("var1_sim", 1), ("var2_sim", 2)])
    def test_order_estimate_selects_true_order(self, fixture, order, request):
        _, xt = request.getfixturevalue(fixture)
        result = model.order_estimate(xt.T, maxlags=4)
        # BIC and HQIC are consistent selectors; AIC/FPE overfit by a lag with ~10%
        # probability, so they are only required not to under-fit.
        assert result.bic == order
        assert result.hqic == order
        assert result.aic >= order
        assert result.fpe >= order
        assert len(result.ics["aic"]) == 5
        assert int(numpy.argmin(result.ics["bic"])) == order
        assert int(numpy.argmin(result.ics["hqic"])) == order

    @pytest.mark.xfail(strict=True, reason=(
        "lib.models.var.order_estimate accepts a trend argument and drops it: line 364 "
        "calls select_order(maxlags=maxlags) with no trend=trend, so order_estimate(..., "
        "trend='n') silently returns the trend='c' selection. statsmodels' own "
        "select_order(trend='n') returns a different, one-shorter set of criteria for "
        "this data (lag 0 has no regressors without a constant)"))
    def test_order_estimate_honours_trend(self, var1_large_offset_sim):
        _, xt = var1_large_offset_sim
        with_const = model.order_estimate(xt.T, maxlags=4, trend="c")
        no_const = model.order_estimate(xt.T, maxlags=4, trend="n")
        # The stationary mean is ~13, so a fit without a constant is far worse and
        # the criteria cannot coincide.
        assert not numpy.array_equal(numpy.asarray(with_const.ics["aic"]),
                                     numpy.asarray(no_const.ics["aic"]))


# ---------------------------------------------------------------------------
# Facade contract
# ---------------------------------------------------------------------------

class TestFacadeContract:

    def test_create_source_returns_t_and_samples(self):
        t, xt = simulate(PHI_COUPLED1, npts=300)
        assert isinstance(t, numpy.ndarray) and isinstance(xt, numpy.ndarray)
        assert t.shape == (300,)
        assert xt.shape == (2, 300)
        assert_array_equal(t, numpy.arange(300.0))
        assert_array_equal(xt[:, 0], [0.0, 0.0])  # x0 defaults to zero

    def test_create_source_default_npts(self):
        t, xt = simulate(PHI_DIAG1)
        assert len(t) == 1000
        assert xt.shape == (2, 1000)

    def test_create_source_x0_and_mu_kwargs(self):
        x0 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        t, xt = simulate(PHI_DIAG2, x0=x0, μ=numpy.array([5.0, 5.0]), Ω=numpy.zeros((2, 2)), npts=4)
        assert_array_equal(xt[:, :2], x0.T)
        # noise-free step: x_2 = mu + phi1 x_1 + phi2 x_0
        assert_allclose(xt[:, 2], 5.0 + 0.25 * x0[1] + 0.5 * x0[0])

    def test_create_source_agrees_with_model_layer(self):
        x0 = numpy.array([[0.5, -0.5]])
        numpy.random.seed(SEED)
        _, from_facade = facade.create_source(PHI_COUPLED1, Ω=OMEGA_COUPLED, μ=MU, x0=x0, npts=200)
        numpy.random.seed(SEED)
        from_model = model.var(x0, MU, PHI_COUPLED1, OMEGA_COUPLED, 200)
        assert_array_equal(from_facade, from_model)

    @pytest.mark.parametrize("bad_kwargs", [
        {"Ω": numpy.eye(3)},
        {"x0": numpy.zeros((2, 2))},   # VAR(1) wants (1, 2)
        {"x0": numpy.zeros((1, 3))},
        {"μ": numpy.zeros(3)},
    ])
    def test_create_source_rejects_misshapen_parameters(self, bad_kwargs):
        with pytest.raises(Exception, match="should have"):
            facade.create_source(PHI_COUPLED1, npts=10, **bad_kwargs)

    def test_create_source_rejects_non_array_offset(self):
        # The one verify_type branch create_source can actually reach: len() works
        # on a list, so the shape check passes and the type check fires.
        with pytest.raises(Exception, match="Expected"):
            facade.create_source(PHI_COUPLED1, npts=10, μ=[0.0, 0.0])

    @pytest.mark.parametrize("bad_kwargs", [
        {"Ω": [[1.0, 0.0], [0.0, 1.0]]},
        {"x0": [[0.0, 0.0]]},
    ])
    @pytest.mark.xfail(strict=True, reason=(
        "lib.data.impl.var.create_source runs its .shape checks before the verify_type "
        "calls, so a list-valued Ω or x0 raises AttributeError('list' object has no "
        "attribute 'shape') and the type checks that should report the expected "
        "numpy.ndarray type are unreachable for them"))
    def test_create_source_reports_non_array_parameters(self, bad_kwargs):
        with pytest.raises(Exception, match="Expected"):
            facade.create_source(PHI_COUPLED1, npts=10, **bad_kwargs)

    @pytest.mark.xfail(strict=True, reason=(
        "lib.data.impl.var.create_source does not guard npts against the process order: "
        "npts=1 for a VAR(2) dies with a raw IndexError from lib/models/var.py:314 "
        "(xt[i] = x0[i] with i past the end of xt) instead of a validation message"))
    def test_create_source_rejects_npts_below_order(self):
        with pytest.raises(Exception) as exc_info:
            facade.create_source(PHI_DIAG2, npts=1)
        assert not isinstance(exc_info.value, IndexError)

    def test_compute_acov_returns_lags_and_values(self):
        t, γ = facade.compute_acov(PHI_COUPLED1, OMEGA_COUPLED, nlag=10)
        assert_array_equal(t, numpy.arange(11.0))
        assert γ.shape == (11, 2, 2)
        assert_allclose(γ, model.acov(PHI_COUPLED1, OMEGA_COUPLED, 11))

    def test_compute_acov_default_nlag_and_omega(self):
        t, γ = facade.compute_acov(PHI_DIAG1)
        assert len(t) == 26 and t[-1] == 25.0
        assert γ.shape == (26, 2, 2)
        assert_allclose(γ[3], numpy.diag([0.5 ** 3 / 0.75, 0.2 ** 3 / 0.96]))

    def test_compute_acov_var2_returns_companion_sized_blocks(self):
        """For n > 1 the facade returns companion-sized (nlag+1, m*n, m*n) blocks,
        not (nlag+1, m, m): gamma[k] = E[z_{t-k} z_t'] for the stacked companion
        state z_t = [x_t, x_{t-1}]. Anchored on the scalar AR(2) Yule-Walker
        autocovariances, which fixes each block independently of the library."""
        t, γ = facade.compute_acov(PHI_DIAG2, nlag=4)
        assert_array_equal(t, numpy.arange(5.0))
        assert γ.shape == (5, 4, 4)
        γ_true = ar2_gamma(0.25, 0.5, nlags=5)
        for k in range(5):
            # Top-left block E[x_{t-k} x_t'] = gamma_x[k], for both components.
            assert_allclose(γ[k][0, 0], γ_true[k])
            assert_allclose(γ[k][1, 1], γ_true[k])
            # Lower-left block E[x_{t-k-1} x_t'] = gamma_x[k+1].
            assert_allclose(γ[k][2, 0], γ_true[k + 1])
            # Upper-right block E[x_{t-k} x_{t-1}'] = gamma_x[|k-1|].
            assert_allclose(γ[k][0, 2], γ_true[abs(k - 1)])
            # The two AR(2)s are independent, so every cross-component entry is 0.
            assert γ[k][0, 1] == pytest.approx(0.0, abs=1e-12)
            assert γ[k][0, 3] == pytest.approx(0.0, abs=1e-12)

    def test_compute_mean_default_offset_is_zero(self):
        assert_allclose(numpy.asarray(facade.compute_mean(PHI_COUPLED1)), numpy.zeros((2, 1)))
        assert_allclose(numpy.asarray(facade.compute_mean(PHI_COUPLED1, MU)),
                        numpy.asarray(model.mean(PHI_COUPLED1, MU)))

    def test_compute_cov_default_noise_is_identity(self):
        assert_allclose(numpy.asarray(facade.compute_cov(PHI_DIAG1)), numpy.diag([1 / 0.75, 1 / 0.96]))
        assert_allclose(numpy.asarray(facade.compute_cov(PHI_COUPLED1, OMEGA_COUPLED)),
                        numpy.asarray(model.cov(PHI_COUPLED1, OMEGA_COUPLED)))

    def test_compute_eig_values_of_diagonal_var1(self):
        # A diagonal VAR(1) is its own companion matrix, so its eigenvalues are the
        # two AR(1) coefficients. Independent of model.eig.
        assert_allclose(numpy.sort(facade.compute_eig_values(PHI_DIAG1)), [0.2, 0.5])

    def test_compute_is_stationary_hard_coded_cases(self):
        assert facade.compute_is_stationary(numpy.array([numpy.eye(2)])) is False  # random walk
        assert facade.compute_is_stationary(PHI_DIAG1) is True

    def test_compute_phi_companion_form_block_layout(self):
        φ1, φ2 = PHI_DIAG2
        expected = numpy.block([[φ1, φ2],
                                [numpy.eye(2), numpy.zeros((2, 2))]])
        assert_array_equal(numpy.asarray(facade.compute_phi_companion_form(PHI_DIAG2)), expected)

    def test_companion_form_facade_layout(self):
        μ = numpy.array([1.0, 2.0])
        ω = numpy.array([[1.0, 0.5], [0.5, 1.0]])
        Μ = numpy.asarray(facade.compute_mean_companion_form(μ, 3))
        assert Μ.shape == (6, 1)
        assert_array_equal(Μ[:, 0], [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        Ω = numpy.asarray(facade.compute_omega_companion_form(ω, 3))
        assert Ω.shape == (6, 6)
        assert_array_equal(Ω[:2, :2], ω)
        assert numpy.count_nonzero(Ω) == 4

    # Cheap extra on top of the hard-coded anchors above: these five facades are
    # one-line delegations, so this can only catch argument mangling.
    @pytest.mark.parametrize("facade_fn, model_fn", [
        (facade.compute_phi_companion_form, model.phi_comp),
        (facade.compute_eig_values, model.eig),
        (facade.compute_is_stationary, model.is_stationary),
    ])
    @pytest.mark.parametrize("phi", [PHI_COUPLED1, PHI_DIAG2, PHI_3])
    def test_phi_facades_agree_with_model(self, facade_fn, model_fn, phi):
        assert_array_equal(numpy.asarray(facade_fn(phi)), numpy.asarray(model_fn(phi)))

    def test_companion_form_facades_agree_with_model(self):
        μ = numpy.array([1.0, 2.0])
        ω = numpy.array([[1.0, 0.5], [0.5, 1.0]])
        assert_array_equal(numpy.asarray(facade.compute_mean_companion_form(μ, 3)),
                           numpy.asarray(model.mean_comp(μ, 3)))
        assert_array_equal(numpy.asarray(facade.compute_omega_companion_form(ω, 3)),
                           numpy.asarray(model.omega_comp(ω, 3)))

    def test_vec_facades_round_trip(self):
        A = numpy.matrix([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        v = facade.compute_vec(A)
        assert_array_equal(numpy.asarray(v)[:, 0], numpy.arange(1.0, 10.0))
        assert_array_equal(numpy.asarray(facade.compute_unvec(v)), numpy.asarray(A))

    def test_compute_unvec_rejects_non_column(self):
        with pytest.raises(Exception, match="column vector"):
            facade.compute_unvec(numpy.matrix([[1.0, 2.0, 3.0, 4.0]]))

    @pytest.mark.xfail(strict=True, reason=(
        "lib.models.var.unvec takes n = int(sqrt(len)) and silently truncates: a 6x1 "
        "column comes back as the 2x2 built from its first four entries and the last two "
        "are dropped with no error. lib.data.impl.var.compute_unvec validates only that "
        "the input has a single column, so the data loss is silent"))
    def test_compute_unvec_rejects_length_that_is_not_a_perfect_square(self):
        col = numpy.matrix(numpy.arange(1.0, 7.0)).T
        with pytest.raises(Exception):
            facade.compute_unvec(col)

    def test_compute_vec_non_square_input_is_unvalidated(self):
        """Behaviour, not a wish: non-square input is outside vec's documented use,
        so this pins what actually happens rather than demanding validation.
        compute_vec is a bare delegation (unlike its compute_unvec sibling, which
        checks for a column vector) and lib.models.var.vec sizes its output from the
        column count alone, so the failure is a raw numpy ValueError and not one of
        the library's own Exception("... should satisfy ...") messages.
        """
        with pytest.raises(Exception) as exc_info:
            facade.compute_vec(numpy.matrix(numpy.zeros((3, 2))))
        assert type(exc_info.value) is ValueError

    @pytest.mark.parametrize("fn", [
        facade.compute_eig_values,
        facade.compute_is_stationary,
        facade.compute_phi_companion_form,
        facade.compute_cov,
        facade.compute_acov,
        facade.create_source,
        pytest.param(facade.compute_mean, marks=pytest.mark.xfail(strict=True, reason=(
            "lib.data.impl.var.compute_mean is the only phi-taking facade that does not "
            "call __verify_phi: lines 34-35 do their own weaker len(φ) > 0 + "
            "verify_type(φ[0]) check and never verify that the blocks are square or "
            "equally sized. A (1, 2, 3) φ therefore reaches lib.models.var.mean and dies "
            "with a raw ValueError('operands could not be broadcast together with shapes "
            "(2,2) (2,3)'), and [eye(2), eye(3)] with an AttributeError, where every "
            "sibling reports 'should be square' / 'should have size (2, 2)'"))),
    ])
    def test_phi_validation(self, fn):
        with pytest.raises(Exception, match="square"):
            fn(numpy.zeros((1, 2, 3)))
        with pytest.raises(Exception, match="should have size"):
            fn([numpy.eye(2), numpy.eye(3)])
        with pytest.raises(Exception, match="len"):
            fn(numpy.zeros((0, 2, 2)))

    @pytest.mark.parametrize("fn", [facade.compute_cov, facade.compute_acov])
    @pytest.mark.xfail(strict=True, reason=(
        "lib.data.impl.var.compute_cov and compute_acov type-check Ω but never check its "
        "shape against φ (lib/data/impl/var.py:65-66 and 94-95 call verify_type only), "
        "although create_source validates exactly this mistake at line 304. A 3x3 Ω "
        "against a 2x2 Φ reaches lib.models.var.cov and dies with a raw ValueError("
        "'shapes (4,4) and (9,1) not aligned: 4 (dim 1) != 9 (dim 0)') from the Kronecker "
        "solve instead of the library's own 'Ω should satisfy should have shape(2,2)'"))
    def test_omega_shape_validation(self, fn):
        with pytest.raises(Exception, match="should have shape"):
            fn(PHI_COUPLED1, numpy.eye(3))

    def test_compute_mean_rejects_empty_and_non_array(self):
        with pytest.raises(Exception, match="len"):
            facade.compute_mean(numpy.zeros((0, 2, 2)))
        with pytest.raises(Exception, match="Expected"):
            facade.compute_mean(PHI_COUPLED1, [0.0, 0.0])

    def test_mean_companion_form_validation(self):
        with pytest.raises(Exception, match="Expected"):
            facade.compute_mean_companion_form([1.0, 2.0], 1)
        with pytest.raises(Exception, match="1-D"):
            facade.compute_mean_companion_form(numpy.ones((2, 1)), 1)
        with pytest.raises(Exception, match="positive"):
            facade.compute_mean_companion_form(numpy.ones(2), 0)

    def test_omega_companion_form_validation(self):
        with pytest.raises(Exception, match="Expected"):
            facade.compute_omega_companion_form([[1.0]], 1)
        with pytest.raises(Exception, match="square"):
            facade.compute_omega_companion_form(numpy.ones((2, 3)), 1)

    @pytest.mark.xfail(strict=True, reason=(
        "lib.data.impl.var.compute_omega_companion_form never checks n > 0, although its "
        "compute_mean_companion_form sibling does; n=0 dies inside numpy with 'could not "
        "broadcast input array from shape (2,2) into shape (0,0)' instead of the "
        "library's 'n should satisfy should be positive'"))
    def test_omega_companion_form_rejects_non_positive_order(self):
        with pytest.raises(Exception, match="positive"):
            facade.compute_omega_companion_form(numpy.eye(2), 0)

    @pytest.mark.xfail(strict=True, reason=(
        "lib.models.var.vec assigns m[:, i] into a (n, 1) column slice, which only works "
        "when m is a numpy.matrix; a plain 2-D ndarray (the annotated NDArray type) "
        "raises ValueError: could not broadcast (n,) into (n, 1)"))
    def test_compute_vec_accepts_plain_ndarray(self):
        A = numpy.array([[1.0, 3.0], [2.0, 4.0]])
        assert_array_equal(numpy.asarray(facade.compute_vec(A))[:, 0], [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Facade estimation: compute_estimate / compute_order and their reports
# ---------------------------------------------------------------------------

class TestFacadeEstimation:

    def test_compute_estimate_returns_results_and_varest(self, var1_estimate):
        result, est = var1_estimate
        assert result.k_ar == 1
        assert isinstance(est, VAREst)
        assert est.est_model == EstModel.VAR
        assert est.order == 1
        assert len(est.const) == 2
        assert len(est.params) == 4
        assert len(est.omega) == 4
        ids = {p.est_id for p in est.const + est.params + est.omega}
        assert len(ids) == 1

    def test_compute_estimate_param_types_and_indices(self, var1_estimate):
        _, est = var1_estimate
        assert {p.param_type for p in est.const} == {VARParamType.VAR_CONST.value}
        assert {p.param_type for p in est.params} == {VARParamType.VAR_PARAM.value}
        assert {p.param_type for p in est.omega} == {VARParamType.VAR_OMEGA.value}
        # est.const's (order, row, column) encoding is deliberately not asserted here;
        # it diverges from the VECM facade and is covered by
        # test_compute_estimate_const_index_matches_vecm_convention below.
        assert [(p.order, p.row, p.column) for p in est.params] == [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        assert [(p.order, p.row, p.column) for p in est.omega] == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
        assert all(p.err == 0.0 for p in est.omega)

    @pytest.mark.xfail(strict=True, reason=(
        "__var_estimate_from_result encodes the equation index of each constant in "
        "ParamEst.order and hard-codes row=column=0 (lib/data/impl/var.py:390-392), while "
        "the sibling VECM facade encodes the same $\\hat{M}$ constant as order=0, row=i, "
        "column=0 (lib/data/impl/vecm.py:285-293). ParamEst documents order as the "
        "parameter order index and row as the row index, so the VAR encoding is the "
        "divergent one; no navi consumer reads VAREst.const back, so nothing else pins it"))
    def test_compute_estimate_const_index_matches_vecm_convention(self, var1_estimate):
        _, est = var1_estimate
        assert [(p.order, p.row, p.column) for p in est.const] == [(0, 0, 0), (0, 1, 0)]

    def test_compute_estimate_display_labels(self, var1_estimate):
        """The est_label/err_label pairs are the notebook-facing display contract."""
        _, est = var1_estimate
        assert {(p.est_label, p.err_label) for p in est.const} == {(r"$\hat{M}$", r"$\sigma^M$")}
        assert {(p.est_label, p.err_label) for p in est.params} == {(r"$\hat{\Phi}$", r"$\sigma^\Phi$")}
        # Note the VAR omega error label is "$\sigma{\Omega}$" (no subscript), where the
        # VECM facade writes "$\sigma_{\Omega}$"; est.omega errors are always 0.0.
        assert {(p.est_label, p.err_label) for p in est.omega} == {(r"$\hat{\Omega}$", r"$\sigma{\Omega}$")}

    def test_compute_estimate_params_match_statsmodels_layout(self, var1_sim, var1_estimate):
        _, xt = var1_sim
        result, est = var1_estimate
        m = 2
        # statsmodels: params/stderr rows are [const, L1.y1, L1.y2, ...], columns are equations.
        assert_array_equal(result.coefs[0], result.params[1:1 + m].T)
        # est.const is emitted one entry per equation, in equation order.
        for i, p in enumerate(est.const):
            assert p.est == result.params[0, i]
            assert p.err == result.stderr[0, i]
        for p in est.params:
            lag, j, k = p.order - 1, p.row, p.column
            assert p.est == result.coefs[lag, j, k]
            assert p.err == result.stderr[1 + lag * m + k, j]
        assert_allclose([p.est for p in est.params], PHI_COUPLED1[0].ravel(), atol=0.08)
        # Magnitude as well as placement: the reported errors must be the OLS standard
        # errors sqrt((Z'Z)^-1_kk sigma_jj), recomputed here from the samples alone.
        # This is an exact identity rather than a statistical one (the two agree to
        # 3e-18 absolute), so rtol 1e-9; unlike a wide 0 < err < 0.05 band it fails for
        # an error inflated or deflated by any factor.
        se = ols_var_stderr(xt, 1)
        assert_allclose([p.err for p in est.params],
                        [se[1 + (p.order - 1) * m + p.column, p.row] for p in est.params],
                        rtol=1e-9)
        # Scale anchor for the same numbers: ~0.018 here, range 0.0156-0.0186 over 50 seeds.
        assert all(0.010 < p.err < 0.030 for p in est.params)

    def test_compute_estimate_var2_lag_indexing(self, var2_sim):
        _, xt = var2_sim
        result, est = facade.compute_estimate(xt, maxlags=2)
        assert est.order == 2
        assert len(est.params) == 8
        assert [p.order for p in est.params] == [1] * 4 + [2] * 4
        for p in est.params:
            assert p.est == result.coefs[p.order - 1, p.row, p.column]
            assert p.err == result.stderr[1 + (p.order - 1) * 2 + p.column, p.row]
        assert_allclose([p.est for p in est.params], PHI_DIAG2.ravel(), atol=0.08)

    def test_compute_estimate_three_variables_stderr_indexing(self, var3d_sim):
        """m = 3 is the only case that exercises the stderr row arithmetic
        1 + lag*m + k with m != 2, and PHI_3D2 is asymmetric in both lag blocks, so
        a transposed or m-hardcoded index cannot pass by accident the way it can on
        the diagonal m = 2 fixtures."""
        _, xt = var3d_sim
        m = 3
        result, est = facade.compute_estimate(xt, maxlags=2)
        assert result.exog_names == ["const"] + [f"L{lag}.y{i}"
                                                 for lag in (1, 2) for i in (1, 2, 3)]
        assert est.order == 2
        assert len(est.const) == m
        assert len(est.omega) == m * m
        assert [(p.order, p.row, p.column) for p in est.params] == [
            (lag + 1, j, k) for lag in range(2) for j in range(m) for k in range(m)]
        se = ols_var_stderr(xt, 2)
        for p in est.params:
            lag = p.order - 1
            assert p.est == result.coefs[lag, p.row, p.column]
            assert p.err == result.stderr[1 + lag * m + p.column, p.row]
            assert p.err == pytest.approx(se[1 + lag * m + p.column, p.row], rel=1e-9)
        # OLS std error per coefficient is ~0.013 at n = 5000; the largest error over
        # the 18 coefficients was 0.046 across 30 seeds, so atol 0.1 is ~7 sigma.
        assert_allclose(numpy.array([p.est for p in est.params]).reshape(2, m, m),
                        PHI_3D2, atol=0.1)
        # Constant std error ~0.06; atol 0.3 is ~5 sigma.
        assert_allclose([p.est for p in est.const], MU_3D, atol=0.3)

    def test_compute_order_three_variables(self, var3d_sim):
        """compute_order with m > 2: the true order is 2 and the consistent criteria
        must find it (AIC is allowed to over-fit, as elsewhere in this file)."""
        _, xt = var3d_sim
        result, report = facade.compute_order(xt, maxlags=4)
        assert (result.bic, result.hqic) == (2, 2)
        assert result.aic >= 2
        assert len(result.ics["bic"]) == 5
        assert json.loads(report.to_json())["_VAROrderTestReport__order"]["value"] == 2

    def test_compute_estimate_default_maxlags(self, var1_sim):
        _, xt = var1_sim
        result, est = facade.compute_estimate(xt)
        assert result.k_ar == 12
        assert est.order == 12
        assert len(est.params) == 12 * 4

    def test_varest_json_round_trip(self, var1_estimate):
        _, est = var1_estimate
        data = json.loads(est.to_json())
        assert data["est_model"] == "VAR"
        assert data["order"] == 1
        assert len(data["params"]) == 4
        restored = [ParamEst.from_dict(d) for d in data["params"]]
        for orig, back in zip(est.params, restored):
            assert back.est == orig.est
            assert back.err == orig.err
            assert back.est_id == orig.est_id
            assert (back.order, back.row, back.column) == (orig.order, orig.row, orig.column)
            assert back.param_type == VARParamType.VAR_PARAM.value
            assert back.est_label == orig.est_label == r"$\hat{\Phi}$"
            assert back.err_label == orig.err_label == r"$\sigma^\Phi$"
        assert json.loads(est.to_json(pretty=True)) == data
        assert repr(est).startswith("VAREst(")

    @pytest.mark.xfail(strict=True, reason=(
        "__var_estimate_from_result stores result.resid_corr (residual correlation: unit "
        "diagonal, off-diagonal 0.104 here) under the Omega label, but Omega is the noise "
        "covariance passed to create_source; it should track result.sigma_u, whose diagonal "
        "is ~2.0 and whose off-diagonal is ~0.200"))
    def test_compute_estimate_omega_is_noise_covariance(self, var1_estimate):
        _, est = var1_estimate
        omega = numpy.zeros((2, 2))
        for p in est.omega:
            omega[p.row, p.column] = p.est
        # Std error of a residual (co)variance estimate ~ sigma^2 sqrt(2/n) ~ 0.05, so
        # atol 0.3 is ~6 sigma. Asserting the whole matrix also pins the off-diagonal
        # scale (0.2, not the 0.104 correlation), which a diagonal-only check would miss.
        assert_allclose(omega, OMEGA_COUPLED, atol=0.3)

    # All four trend values are documented as supported by compute_estimate
    # (lib/data/impl/var.py:355-356), but __var_estimate_from_result hard-codes the
    # trend='c' layout in two independent ways: it drops exactly one leading row
    # (result.stderr[1:], the constant) whatever the number of deterministic terms,
    # and it feeds the remainder to numpy.array_split(..., n), which only divides
    # evenly when that remainder is exactly n*m rows. The three tests below pin the
    # correct row for each trend at maxlags = 1 and 2, so both faults are covered and
    # neither test can pass on a fix that repairs only the leading offset.
    @pytest.mark.parametrize("maxlags", [1, 2])
    @pytest.mark.xfail(strict=True, reason=(
        "__var_estimate_from_result assumes trend='c': with trend='n' result.stderr has no "
        "constant row, so stderr[1:] drops a real lag row. At maxlags=1 that leaves a "
        "single row and indexing est_stderr raises IndexError; at maxlags=2 the 3 "
        "remaining rows split unevenly into 2 chunks and numpy.array over the ragged list "
        "raises ValueError('... inhomogeneous shape after 2 dimensions')"))
    def test_compute_estimate_trend_n_stderr_alignment(self, var1_sim, maxlags):
        _, xt = var1_sim
        result, est = facade.compute_estimate(xt, maxlags=maxlags, trend="n")
        assert result.exog_names == [f"L{lag}.y{i}"
                                     for lag in range(1, maxlags + 1) for i in (1, 2)]
        assert len(est.params) == 4 * maxlags
        for p in est.params:
            assert p.est == result.coefs[p.order - 1, p.row, p.column]
            assert p.err == result.stderr[(p.order - 1) * 2 + p.column, p.row]

    @pytest.mark.parametrize("maxlags", [1, 2])
    @pytest.mark.xfail(strict=True, reason=(
        "__var_estimate_from_result assumes trend='c': with trend='ct' stderr[1:] starts at "
        "the linear-trend row, so at maxlags=1 every coefficient error is shifted by one "
        "regressor (the lag-1 errors come back as the trend stderr, 2.9e-05, instead of "
        "~0.018). At maxlags=2 the 5 rows left after the drop split unevenly into 2 chunks "
        "and numpy.array over the ragged list raises ValueError('... inhomogeneous shape "
        "after 2 dimensions'), so the row offset and the array_split are separate faults"))
    def test_compute_estimate_trend_ct_stderr_alignment(self, var1_sim, maxlags):
        _, xt = var1_sim
        result, est = facade.compute_estimate(xt, maxlags=maxlags, trend="ct")
        assert result.exog_names[:2] == ["const", "trend"]
        assert result.params.shape == (2 + 2 * maxlags, 2)
        for p in est.params:
            assert p.err == result.stderr[2 + (p.order - 1) * 2 + p.column, p.row]

    @pytest.mark.parametrize("maxlags", [1, 2])
    @pytest.mark.xfail(strict=True, reason=(
        "trend='ctt' is the silent variant of the same bug and the dangerous one: it never "
        "raises at any maxlags, so no crash-shaped test covers it. stderr[1:] starts at the "
        "'trend' row and the two extra deterministic rows keep the split even, so the "
        "reported Phi errors are the deterministic-term stderrs. At maxlags=1 the lag-1 "
        "errors come back as the trend and trend**2 stderrs, 0.000117 and literally 0.0 -- "
        "a notebook renders a 0.0 standard error for a Phi coefficient with no signal that "
        "anything is wrong; at maxlags=2 the lag-2 errors are a mix of the L1.y2 and L2 rows"))
    def test_compute_estimate_trend_ctt_stderr_alignment(self, var1_sim, maxlags):
        _, xt = var1_sim
        result, est = facade.compute_estimate(xt, maxlags=maxlags, trend="ctt")
        assert result.exog_names[:3] == ["const", "trend", "trend**2"]
        assert len(est.params) == 4 * maxlags
        for p in est.params:
            assert p.est == result.coefs[p.order - 1, p.row, p.column]
            assert p.err == result.stderr[3 + (p.order - 1) * 2 + p.column, p.row]

    def test_compute_order_returns_results_and_report(self, var1_sim):
        _, xt = var1_sim
        result, report = facade.compute_order(xt, maxlags=4)
        assert isinstance(report, VAROrderTestReport)
        assert result.bic == 1
        assert len(result.ics["aic"]) == 5
        data = json.loads(report.to_json())
        order = data["_VAROrderTestReport__order"]
        assert order["value"] == 1
        assert order["label"] == r"$\tau_{min}$"
        for name in ("aic", "bic", "fpe", "hqic"):
            entry = data[f"_VAROrderTestReport__{name}"]
            assert entry["error_metric"] == name.upper()
            assert entry["order"]["value"] == getattr(result, name)
            assert entry["order"]["label"] == rf"$\tau_{{{name.upper()}}}$"
            assert entry["value"]["value"] == pytest.approx(min(result.ics[name]))
            assert entry["value"]["label"] == rf"$\varepsilon_{{{name.upper()}}}$"
            assert entry["test_id"] == order["test_id"]
        assert json.loads(report.to_json(pretty=True)) == data
        assert repr(report).startswith("VAROrderTestReport(")

    def test_compute_order_var2(self, var2_sim):
        _, xt = var2_sim
        result, report = facade.compute_order(xt, maxlags=4)
        assert (result.bic, result.hqic) == (2, 2)
        # Majority vote over (aic, bic, hqic) is 2 whenever bic and hqic agree.
        assert json.loads(report.to_json())["_VAROrderTestReport__order"]["value"] == 2

    def test_compute_order_default_maxlags(self, var1_sim):
        _, xt = var1_sim
        result, _ = facade.compute_order(xt)
        assert len(result.ics["bic"]) == 13  # lags 0..12

    @pytest.mark.xfail(strict=True, reason=(
        "compute_order forwards trend to lib.models.var.order_estimate, which drops it: "
        "line 364 calls select_order(maxlags=maxlags) with no trend=trend, so "
        "compute_order(..., trend='n') silently returns the trend='c' result. The two "
        "calls are byte-identical, while statsmodels' own select_order(trend='n') returns "
        "a different, one-shorter set of criteria for this data"))
    def test_compute_order_honours_trend(self, var1_large_offset_sim):
        _, xt = var1_large_offset_sim
        with_const, _ = facade.compute_order(xt, maxlags=4, trend="c")
        no_const, _ = facade.compute_order(xt, maxlags=4, trend="n")
        # The stationary mean is ~13, so a no-constant fit is much worse: the ICs
        # cannot coincide once the argument is honoured.
        assert not numpy.array_equal(numpy.asarray(with_const.ics["bic"]),
                                     numpy.asarray(no_const.ics["bic"]))


# ---------------------------------------------------------------------------
# VAROrderTestReport majority vote
# ---------------------------------------------------------------------------

class TestOrderReportMajorityVote:
    """The report's reported order is a majority vote over (aic, bic, hqic).

    Real data of the size these tests can afford has all three criteria agreeing,
    so the disagreement and tie branches are reached with a stand-in result: the
    builder only reads .aic/.bic/.fpe/.hqic and .ics.
    """

    @staticmethod
    def fake_result(aic, bic, fpe, hqic, maxlags=4):
        ics = {name: list(numpy.linspace(2.0, 1.0, maxlags + 1))
               for name in ("aic", "bic", "fpe", "hqic")}
        return SimpleNamespace(aic=aic, bic=bic, fpe=fpe, hqic=hqic, ics=ics)

    @pytest.mark.parametrize("aic, bic, fpe, hqic, expected", [
        (1, 1, 1, 1, 1),   # unanimous
        (2, 1, 3, 1, 1),   # bic and hqic outvote aic
        (3, 3, 1, 2, 3),   # aic and bic outvote hqic
        (1, 2, 1, 2, 2),   # fpe takes no part in the vote: (1, 2, 2) -> 2
        (2, 1, 3, 4, 1),   # three-way tie -> numpy.argmax picks the smallest order
        (4, 0, 4, 0, 0),   # two-way tie -> the smaller of the two majorities
    ])
    def test_majority_vote(self, aic, bic, fpe, hqic, expected):
        report = build_order_report(self.fake_result(aic, bic, fpe, hqic))
        data = json.loads(report.to_json())
        assert data["_VAROrderTestReport__order"]["value"] == expected
        assert data["_VAROrderTestReport__order"]["label"] == r"$\tau_{min}$"

    def test_per_criterion_entries_index_their_own_selection(self):
        result = self.fake_result(aic=4, bic=1, fpe=3, hqic=2)
        data = json.loads(build_order_report(result).to_json())
        test_id = data["_VAROrderTestReport__test_id"]
        for name in ("aic", "bic", "fpe", "hqic"):
            entry = data[f"_VAROrderTestReport__{name}"]
            selected = getattr(result, name)
            assert entry["error_metric"] == name.upper()
            assert entry["order"]["value"] == selected
            assert entry["order"]["label"] == rf"$\tau_{{{name.upper()}}}$"
            # The metric value is the criterion at its own selected order, which is
            # not the minimum of this synthetic (monotone) IC curve.
            assert entry["value"]["value"] == pytest.approx(result.ics[name][selected])
            assert entry["value"]["label"] == rf"$\varepsilon_{{{name.upper()}}}$"
            assert entry["test_id"] == test_id
