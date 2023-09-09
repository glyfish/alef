from enum import Enum
import numpy
import uuid

from lib.models import var
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VARResults

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_condition, create_space, create_logspace)
from lib.data.param_est import (ParamEst, VAREst, VARParamType)


def compute_mean(φ: list[numpy.matrix[float]], μ: numpy.ndarray[float] = None) -> numpy.matrix[float]:
    """
    Compute the stationary mean matrix for a VAR(n) process with the given parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    μ: numpy.matrix[float]
        VAR(n) process offset matrix. (default column of zeros)

    Returns
    -------
    numpy.matrix[float]
        Stationary mean matrix.
    """

    verify_condition(φ, len(φ) > 0, "len(φ) > 0")
    verify_type(φ[0], numpy.ndarray)
    m, _ = φ[0].shape

    μ = numpy.zeros(m) if μ is None else μ
    verify_type(μ, numpy.ndarray)

    return var.mean(φ, μ)


def compute_cov(φ: list[numpy.matrix[float]], ω: numpy.matrix[float] = None) -> numpy.matrix[float]:
    """
    Compute the stationary covariance matrix for the given VAR(n) process
    parameters.
    
    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    ω: list[numpy.matrix[float]]
        VAR(n) process gaussian noise autocovariance function. (identity matrix)

    Returns
    -------
    numpy.matrix[float]
        Simulation results.
    """

    __verify_phi(φ)
    m, _ = φ[0].shape

    ω = numpy.matrix(numpy.eye(m)) if ω is None else ω
    verify_type(ω, numpy.ndarray)

    return var.cov(φ, ω)


def compute_acov(φ: list[numpy.matrix[float]], ω: numpy.matrix[float] = None, **kwargs) -> numpy.matrix[float]:
    """
    Compute the stationary auto covariance matrix for the given VAR(n)
    parameters.

    Parameters
    ----------
    φ: numpy.matrix[float]
        VAR(n) process coefficient matrix.
    Ω: numpy.matrix[float]
        VAR(n) process gaussian noise autocovariance matrix.
    nlag: int
        Number of lags. (default 25)

    Returns
    -------
    numpy.matrix[float]
        Stationary mean matrix.
    """

    __verify_phi(φ)
    m, _ = φ[0].shape

    ω = numpy.matrix(numpy.eye(m)) if ω is None else ω
    verify_type(ω, numpy.ndarray)
    nlag = get_param_default_if_missing("nlag", 25, **kwargs)

    return  create_space(npts=nlag + 1), var.acov(φ, ω, nlag + 1)

def compute_eig_values(Φ: list[numpy.matrix[float]]) -> numpy.ndarray[float]:
    """
    Compute eigen values of VAR(n) parameter matrix transformed to VAR(1) companion form. 
    Stationarity requires that |λ| < 1.

    Parameters
    ----------
    Φ: numpy.matrix[float]
       VAR(n) coefficient matrix in companion form.

    Returns
    -------
    numpy.ndarray[float]
        Array of eigen values.
    """

    __verify_phi(Φ)

    return var.eig(Φ)


def compute_is_stationary(Φ: list[numpy.matrix[float]]) -> bool:
    """
    Return True if the VAR(n) parameter matrix is stationary.

    Parameters
    ----------
    Φ: numpy.matrix[float]
        VAR(n) covariance matrix.

    Returns
    -------
    bool
        True if VAR(n) process is stationary.
    """

    __verify_phi(Φ)

    return var.is_stationary(Φ)


def compute_phi_companion_form(Φ: list[numpy.matrix[float]]) -> numpy.matrix[float]:
    """
    Convert the VAR(n) coefficient matrix to the VAR(1) companion form used for calculations. 

    Parameters
    ----------
    Φ: numpy.matrix[float]
        VAR(n) covariance matrix.

    Returns
    -------
    numpy.matrix[float]
        Companion form of noise covariance matrix.
    """

    __verify_phi(Φ)

    return var.phi_comp(Φ)


def compute_mean_companion_form(Μ: numpy.matrix[float], n: int) -> numpy.matrix[float]:
    """
    Convert the VAR(n) offset matrix to 

    Parameters
    ----------
    Μ: numpy.matrix[float]
        VAR(n) offset matrix.
    n: int
        Order of VAR process.

    Returns
    -------
    numpy.matrix[float]
        Companion form of VAR(n) offset matrix.
    """

    verify_type(Μ, numpy.ndarray)
    verify_condition("Μ", len(Μ.shape) == 1, f"should be 1-D vector")
    verify_condition("n", n > 0, f"should be positive")

    return var.mean_comp(Μ, n)
          

def compute_omega_companion_form(Ω: numpy.matrix[float], n: int) -> numpy.matrix[float]:
    """
    Convert VAR(n) gaussian noise covariance matrix to companion form.

    Parameters
    ----------
    Ω: list[numpy.matrix[float]]
        VAR(n) noise covariance matrix.
    n: int
        Order of VAR process.

    Returns
    -------
    numpy.matrix[float]
        Companion form of noise covariance matrix.
    """

    verify_type(Ω, numpy.ndarray)
    m, l = Ω.shape
    verify_condition("Φ", m == l, "Ω should be square")

    return var.omega_comp(Ω, n)


def compute_vec(m: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Apply the vec operator to the given matrix. The vec operation 
    applied to the matrix,

    A = [[a11, a12],
         [a21, a22]]

    gives,

    vec(A) = [[a11],
              [a21],
              [a12],
              [a22]]

    Parameters
    ----------
    m: numpy.matrix[float]
        Matrix to be converted to vec form.

    Returns
    -------
    numpy.matrix[float]
        Input vector converted to vec form.
    """

    return var.vec(m)

def compute_unvec(m: numpy.matrix[float]) -> numpy.matrix[float]:
    """
    Apply the inverse of the vec operation to the given matrix. For the following
    matrix in vec form,

    A = [[a11],
         [a21],
         [a12],
         [a22]]

    apply unvec gives,

    unvec(A) = [[a11, a12],
                [a21, a22]]

    Parameters
    ----------
    m: numpy.matrix[float]
        Matrix to be converted to unvec form.

    Returns
    -------
    numpy.matrix[float]
        Input vector in unvec form.
    """

    _, n = m.shape
    verify_condition("Input", n == 1, f"should be a column vector")

    return var.unvec(m)

def create_source(Φ: list[numpy.matrix[float]], **kwargs) -> numpy.matrix[float]:
    """
    Simulate a VAR(n) process using the provided parameters.
    
    Parameters
    ----------
    Φ: list[numpy.matrix[float]]
        VAR(n) process coefficient matrix.
    x0: numpy.matrix[float]
        VAR(n) process initial value matrix. (default zero column matrix)
    μ: numpy.ndarray[float]
        VAR(n) process offset matrix.(default zero column matrix)
    Ω: numpy.matrix[float]
        VAR(n) process gaussian noise autocovariance function. (identity matrix)
    npts: int
        Number of steps simulated. (default 1000)

    Returns
    -------
    numpy.matrix[float]
        Simulation results.
    """

    __verify_phi(Φ)

    n, m, _ = Φ.shape

    Ω_default = numpy.matrix(numpy.eye(m))
    μ_default = numpy.zeros(m)
    x0_default = numpy.matrix(numpy.zeros((n, m)))

    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    μ = get_param_default_if_missing("μ", μ_default, **kwargs)
    x0 = get_param_default_if_missing("x0", x0_default, **kwargs)

    m0, n0 = Ω.shape
    verify_condition("Ω", m0 == m and n0 == m, f"should have shape({m},{m})")
    n0, m0 = x0.shape
    verify_condition("x0", m0 == m and n0 == n, f"should have shape ({n},{m})")
    verify_condition("μ", len(μ) == m, f"should have length ({m},)")

    verify_type(x0, numpy.ndarray)
    verify_type(Ω, numpy.ndarray)
    verify_type(μ, numpy.ndarray)

    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    return create_space(npts=npts), var.var(x0, μ, Φ, Ω, npts)

def compute_estimate(samples: list[numpy.ndarray[float]], **kwargs):
    """
    Estimate the parameters for and assumed VAR(n) model.

    Parameters
    ----------
    samples: list[numpy.ndarray[float]]
        VAR(n) process endogenous variable samples.
    names: list[str]
        Names of sample variables. (default None)
    maxlags: int
        Maximum number of time lags tried. (default is 12)
    trend: str
        Assumed trend (default 'c'). 
        Values 'n'=no trend, 'c'=constant offset, 'ct'=linear trend, 'ctt'=quadratic and linear trend.

    Returns
    -------
    VARResult
        Analysis results.
    """
    
    nvar = len(samples)
    default_names = [f"S{i}" for i in range(nvar)]

    maxlags = get_param_default_if_missing("maxlags", 12, **kwargs)
    trend = get_param_default_if_missing("trend", 'c', **kwargs)
    names = get_param_default_if_missing("names", default_names, **kwargs)
    verify_condition("samples", nvar == len(names), f"and names should have the same length of {nvar}")

    endog = {}
    for i in range(nvar):
        endog[names[i]] = samples[i]
    endog=DataFrame(endog)

    result = var.fit(endog, maxlags=maxlags, trend=trend)
    return result, __var_estimate_from_result(result)

def __var_estimate_from_result(result: VARResults) -> VAREst:
    est_params = result.coefs
    n, m, _ = est_params.shape

    est_stderr = result.stderr.iloc[1:].to_numpy()
    est_stderr = numpy.array([numpy.transpose(a) for a in numpy.array_split(est_stderr, n)])

    est_const = result.params.loc['const'].to_numpy()
    est_const_stderr = result.stderr.loc['const'].to_numpy()

    est_omega = result.resid_corr

    est_id = str(uuid.uuid4())
    const = []
    params = []
    omega = []
 
    for i in range(m):
        const.append(ParamEst(est=est_const[i], 
                              err=est_const_stderr[i], 
                              est_label=f"$\hat{{M}}_{{i}}$", 
                              err_label=f"$\sigma^M_{{i}}$", 
                              column=i, 
                              est_id=est_id, 
                              param_type=VARParamType.VAR_CONST.value))

    for i in range(n):
        for j in range(m):
            for k in range(m):
                params.append(ParamEst(est=est_params[i,j,k], 
                                       err=est_stderr[i,j,k],
                                       est_label=f"$\\hat{{Phi}}_{{i}}$", 
                                       err_label=f"$\sigma^\Phi_{{i}}$", 
                                       order=i + 1,
                                       row=j,
                                       column=k, 
                                       est_id=est_id,
                                       param_type=VARParamType.VAR_PARAM.value))
    for i in range(m):
        for j in range(m):
            omega.append(ParamEst(est=est_omega[i,j], 
                                  est_label=f"$\\hat{{Omega}}_{{i}}$", 
                                  row=i,
                                  column=j,
                                  est_id=est_id,
                                  param_type=VARParamType.VAR_OMEGA.value))

    return VAREst(const=const, params=params, omega=omega)


def __verify_phi(φ):
    """
    Verify that Φ satisfies the required shape.
    """

    verify_condition(φ, len(φ) > 0, "len(φ) > 0")
    m0, n0 = φ[0].shape
    for i in range(len(φ)):
        verify_type(φ[i], numpy.ndarray)
        m, n = φ[i].shape
        verify_condition(f"φ[{i}]", m == n, f"should be square")
        verify_condition(f"φ[{i}]", m0 == m and n0 == n, f"should have size ({m0}, {n0})")

    
