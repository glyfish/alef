"""
Tests for lib.data.param_est -- the parameter-estimate result containers.

The module carries no numerics. It holds the enums and plain result objects
(ParamEst, OLSResult/OLSTransform, ARMAEst, VAREst, VECMEst) that the
estimators in lib.data.impl.{arima, var, vecm, ou, stats, ...} build and that
the reports/database layers serialise. So the tests here are contract checks:

  * enum membership, str-mixin behaviour and value lookup, plus the
    ARMAEstType.formula() / set_param_labels() dispatch branches;
  * JSON serialisation (to_json, compact and pretty) and the
    ParamEst.from_dict round trip, including the numpy.float64 scalars the
    estimators actually pass in;
  * the labels ARMAEst stamps on its parameters, and the (order, row, column)
    addressing plus mathtext labels VAREst/VECMEst rely on, built exactly the
    way the estimators build them;
  * __repr__ / __str__ structure;
  * module import hygiene -- what a consumer pays to `import lib.data.param_est`.

Tests that expose defects in the library are kept and marked
xfail(strict=True) so they flip to FAIL as soon as the defect is fixed.  Each
defect is owned by exactly ONE xfail: where a passing test would otherwise
hard-pin the same buggy string, it asserts the surrounding structure instead, so
a real fix produces one red test and not five.  Purely cosmetic divergences
(pretty-print indent widths, `est_id=x` vs `est_id=(x)` in a repr) are recorded
as comments next to a green assertion rather than as strict xfails.
"""

import inspect
import json
import subprocess
import sys

import numpy
import pytest
from numpy.testing import assert_array_equal

from lib.data import param_est as param_est_module
from lib.data.param_est import (
    ARMAEst,
    ARMAEstType,
    ARMAParamType,
    EstModel,
    OLSParamType,
    OLSResult,
    OLSTransform,
    ParamEst,
    VAREst,
    VARParamType,
    VECMEst,
    VECMParamType,
)
from helpers import present
from enum import Enum

EST_ID = "7f3c-est-id"

# The attribute set ParamEst carries; to_json must emit exactly these keys.
PARAM_KEYS = {"est", "err", "est_label", "err_label", "row", "column", "order", "est_id", "param_type"}


# ---------------------------------------------------------------------------
# Builders that mirror how lib.data.impl.* construct these objects
# ---------------------------------------------------------------------------

def _param(est=1.5, err=0.25, *, est_label=None, err_label=None, order=0, row=0, column=0,
           param_type=OLSParamType.OLS_PARAM.value, est_id=EST_ID) -> ParamEst:
    return ParamEst(est_id, est, err, est_label, err_label, order, row, column, param_type)


def _arma_est(arma_est_type=ARMAEstType.AR, nparams=2) -> ARMAEst:
    """Built like lib.data.impl.arima: unlabeled ParamEsts via from_dict, numpy
    scalars from the statsmodels result, 1-based ``order``."""
    params = [ParamEst.from_dict({"est": numpy.float64(0.5 / (i + 1)),
                                  "err": numpy.float64(0.01 * (i + 1)),
                                  "order": i + 1,
                                  "est_id": EST_ID,
                                  "param_type": ARMAParamType.ARMA_PARAM.value})
              for i in range(nparams)]
    const = ParamEst.from_dict({"est": numpy.float64(0.1), "err": numpy.float64(0.02),
                                "est_id": EST_ID, "param_type": ARMAParamType.ARMA_CONST.value})
    sigma2 = ParamEst.from_dict({"est": numpy.float64(2.0), "err": numpy.float64(0.1),
                                 "est_id": EST_ID, "param_type": ARMAParamType.ARMA_SIG2.value})
    return ARMAEst(EST_ID, const, params, sigma2, arma_est_type)


def _ols_result(nparams=2, r2=0.9) -> OLSResult:
    """Built like lib.data.impl.stats.ols: labelled const, params with column=i."""
    const = _param(0.3, 0.05, est_label=r"$\beta$", err_label=r"$\sigma_{\beta}$",
                   param_type=OLSParamType.OLS_CONST.value)
    params = [_param(float(i), 0.1 * i, est_label=rf"$\alpha_{i}$", err_label=rf"$\sigma_{{\alpha_{i}}}$",
                     column=i)
              for i in range(1, nparams + 1)]
    return OLSResult(EST_ID, const, params, r2)


# The mathtext labels lib.data.impl.var / lib.data.impl.vecm stamp on every row
# they build.  Reproduced here verbatim so the builders below carry the same
# backslash-laden strings the real estimators do -- escaping them through
# to_json is the interesting half of serialising a VAR/VECM result.
VAR_LABELS = {
    "const": (r"$\hat{M}$", r"$\sigma^M$"),
    "params": (r"$\hat{\Phi}$", r"$\sigma^\Phi$"),
    "omega": (r"$\hat{\Omega}$", r"$\sigma{\Omega}$"),
}
VECM_LABELS = {
    "const": (r"$\hat{M}$", r"$\sigma_{M}$"),
    "lambda_est": (r"$\hat{\lambda}$", r"$\sigma_{\lambda}$"),
    "beta_est": (r"$\hat{\beta}$", r"$\sigma_{\beta}$"),
    "a_est": (r"$\hat{A}$", r"$\sigma_A$"),
    "omega": (r"$\hat{\Omega}$", r"$\sigma_{\Omega}$"),
}


def _var_est(n=2, m=2) -> tuple[VAREst, numpy.ndarray, numpy.ndarray]:
    """Built like lib.data.impl.var: m equations, n lags. Φ[i, j, k] is the
    coefficient at lag i+1, row j, column k; Ω[i, j] the noise covariance.
    Labels are the ones lib.data.impl.var actually stamps (VAR_LABELS)."""
    Φ = numpy.arange(n * m * m, dtype=float).reshape(n, m, m) / 10.0
    Ω = numpy.arange(m * m, dtype=float).reshape(m, m) + 100.0
    const = [_param(float(i), 0.1, order=i + 1, param_type=VARParamType.VAR_CONST.value,
                    est_label=VAR_LABELS["const"][0], err_label=VAR_LABELS["const"][1]) for i in range(m)]
    params = [_param(Φ[i, j, k], 0.01, order=i + 1, row=j, column=k, param_type=VARParamType.VAR_PARAM.value,
                     est_label=VAR_LABELS["params"][0], err_label=VAR_LABELS["params"][1])
              for i in range(n) for j in range(m) for k in range(m)]
    omega = [_param(Ω[i, j], 0.0, row=i, column=j, param_type=VARParamType.VAR_OMEGA.value,
                    est_label=VAR_LABELS["omega"][0], err_label=VAR_LABELS["omega"][1])
             for i in range(m) for j in range(m)]
    return VAREst(order=n, const=const, params=params, omega=omega), Φ, Ω


def _vecm_est(neq=2, rank=1, order=2) -> VECMEst:
    """Built like lib.data.impl.vecm: λ and β are neq x rank, A has one
    neq x neq block per lag (order 1..p), Ω is neq x neq.  Labels are the ones
    lib.data.impl.vecm actually stamps (VECM_LABELS)."""
    lambda_est = [_param(0.1 * (i + 1), 0.01, row=i, column=j, param_type=VECMParamType.VECM_LAMBDA.value,
                         est_label=VECM_LABELS["lambda_est"][0], err_label=VECM_LABELS["lambda_est"][1])
                  for j in range(rank) for i in range(neq)]
    beta_est = [_param(1.0 if i == 0 else -0.5, 0.01, row=i, column=j, param_type=VECMParamType.VECM_BETA.value,
                       est_label=VECM_LABELS["beta_est"][0], err_label=VECM_LABELS["beta_est"][1])
                for j in range(rank) for i in range(neq)]
    const = [_param(0.01 * i, 0.001, row=i, param_type=VECMParamType.VECM_CONST.value,
                    est_label=VECM_LABELS["const"][0], err_label=VECM_LABELS["const"][1]) for i in range(neq)]
    omega = [_param(1.0 if i == j else 0.2, 0.0, row=i, column=j, param_type=VECMParamType.VECM_OMEGA.value,
                    est_label=VECM_LABELS["omega"][0], err_label=VECM_LABELS["omega"][1])
             for i in range(neq) for j in range(neq)]
    a_est = [_param(0.05 * (k + 1), 0.01, order=k + 1, row=i, column=j, param_type=VECMParamType.VECM_ALPHA.value,
                    est_label=VECM_LABELS["a_est"][0], err_label=VECM_LABELS["a_est"][1])
             for k in range(order) for j in range(neq) for i in range(neq)]
    return VECMEst(rank=rank, order=order, const=const, lambda_est=lambda_est, beta_est=beta_est,
                   a_est=a_est, omega=omega)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEstModel:
    def test_members_and_values(self):
        assert {m.name for m in EstModel} == {"ARMA", "OLS", "VAR", "VECM"}
        for member in EstModel:
            assert member.value == member.name

    def test_is_a_str_mixin(self):
        # The str mixin is what lets json.dumps emit the enum as its bare value
        # and lets the database layer compare against plain strings.  It does
        # NOT carry over to str()/format(): since the 3.11 Enum change a
        # (str, Enum) member renders as its dunder form, so every __repr__ in
        # this module leaks "EstModel.OLS" where to_json writes "OLS".
        # See test_repr_renders_enum_dunder_form_while_json_renders_value.
        assert isinstance(EstModel.OLS, str)
        assert EstModel.OLS == "OLS"
        assert json.dumps(EstModel.ARMA) == '"ARMA"'
        assert str(EstModel.OLS) == "EstModel.OLS"
        assert f"{EstModel.OLS}" == "EstModel.OLS"

    def test_lookup_by_value(self):
        assert EstModel("VAR") is EstModel.VAR
        with pytest.raises(ValueError):
            EstModel("ARIMA")


@pytest.mark.parametrize(
    "enum_cls, expected_names",
    [
        (OLSParamType, {"OLS_CONST", "OLS_R2", "OLS_PARAM", "TRANS_CONST", "TRANS_PARAM"}),
        (ARMAParamType, {"ARMA_CONST", "ARMA_PARAM", "ARMA_SIG2", "ARMA_OFFSET"}),
        (VARParamType, {"VAR_CONST", "VAR_PARAM", "VAR_OMEGA"}),
        (ARMAEstType, {"AR", "AR_OFFSET", "MA", "MA_OFFSET"}),
    ],
    ids=lambda x: getattr(x, "__name__", None),
)
def test_param_type_enums_value_equals_name(enum_cls: type[Enum], expected_names):
    assert {m.name for m in enum_cls} == expected_names
    for member in enum_cls:
        assert isinstance(member, str)
        assert member.value == member.name
        assert enum_cls(member.name) is member


class TestVECMParamType:
    # Held apart from test_param_type_enums_value_equals_name only because
    # VECM_OMEGA's value does not equal its name (see the xfail below); every
    # other property asserted of the sibling enums is asserted here too.
    def test_members(self):
        assert {m.name for m in VECMParamType} == {"VECM_CONST", "VECM_ALPHA", "VECM_LAMBDA", "VECM_BETA", "VECM_OMEGA"}
        for member in VECMParamType:
            # The str mixin is what lets the database layer compare a stored
            # param_type against a plain string and json.dumps emit the value.
            assert isinstance(member, str)
            assert VECMParamType(member.value) is member
            if member is not VECMParamType.VECM_OMEGA:
                assert member.value == member.name
                assert VECMParamType(member.name) is member
                assert member == member.name

    def test_vecm_omega_value_matches_name(self):
        assert VECMParamType.VECM_OMEGA.value == "VECM_OMEGA"
        assert VECMParamType("VECM_OMEGA") is VECMParamType.VECM_OMEGA


# ---------------------------------------------------------------------------
# ARMAEstType dispatch: formula() and set_param_labels()
# ---------------------------------------------------------------------------

class TestARMAEstTypeFormula:
    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_is_single_mathtext_expression(self, est_type):
        formula = est_type.formula()
        assert formula.startswith("$") and formula.endswith("$")
        assert formula.count("$") == 2
        assert "X_t =" in formula
        assert r"\varepsilon_{t}" in formula

    def test_ar_formula_is_autoregression_without_offset(self):
        formula = ARMAEstType.AR.formula()
        assert r"\sum_{i=1}^p \varphi_i X_{t-i}" in formula
        assert r"\mu^*" not in formula

    def test_ar_offset_adds_constant_term_to_ar_formula(self):
        # Pinned against the literal, not against AR.formula() re-derived through
        # the same dispatch (that would only fix AR_OFFSET relative to AR).
        assert ARMAEstType.AR_OFFSET.formula() == \
            r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        # Secondary: AR_OFFSET is the AR model plus a μ* term and nothing else.
        ar = ARMAEstType.AR.formula()
        expected = ar.replace(r"+ \varepsilon_{t}", r"+ \mu^* + \varepsilon_{t}")
        assert expected != ar
        assert ARMAEstType.AR_OFFSET.formula() == expected

    @pytest.mark.parametrize(
        "ma_type, ar_type",
        [(ARMAEstType.MA, ARMAEstType.AR), (ARMAEstType.MA_OFFSET, ARMAEstType.AR_OFFSET)],
        ids=["MA", "MA_OFFSET"],
    )
    def test_ma_formula_is_moving_average(self, ma_type, ar_type):
        formula = ma_type.formula()
        assert formula != ar_type.formula()
        assert r"\theta" in formula
        assert r"\varepsilon_{t-i}" in formula


class TestARMAEstTypeSetParamLabels:
    @pytest.mark.parametrize(
        "est_type, expected_est_label",
        [
            (ARMAEstType.AR, r"$\hat{\varphi_{3}}$"),
            (ARMAEstType.AR_OFFSET, r"$\hat{\varphi_{3}}$"),
            (ARMAEstType.MA, r"$\hat{\theta_{3}}$"),
            (ARMAEstType.MA_OFFSET, r"$\hat{\theta_{3}}$"),
        ],
        ids=lambda x: x.name if isinstance(x, ARMAEstType) else None,
    )
    def test_est_label_uses_model_symbol_and_index(self, est_type, expected_est_label):
        p = _param()
        est_type.set_param_labels(p, 3)
        assert p.est_label == expected_est_label

    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_err_label_is_sigma_of_est_symbol(self, est_type):
        p = _param()
        est_type.set_param_labels(p, 2)
        symbol = r"\varphi" if est_type in (ARMAEstType.AR, ARMAEstType.AR_OFFSET) else r"\theta"
        assert present(p.err_label).startswith(r"$\sigma_{")
        assert rf"\hat{{{symbol}_{{2}}}}" in present(p.err_label)
        assert p.err_label != p.est_label

    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_err_label_is_well_formed_mathtext(self, est_type):
        p = _param()
        est_type.set_param_labels(p, 0)
        assert present(p.err_label).count("$") == 2

    def test_overwrites_existing_labels(self):
        p = _param(est_label="old-est", err_label="old-err")
        ARMAEstType.AR.set_param_labels(p, 0)
        # Pin the concrete post-state, not merely "changed".  est_label is
        # pinned exactly; err_label is pinned structurally (symbol + index +
        # sigma prefix) because its exact spelling is the malformed
        # three-delimiter form owned by test_err_label_is_well_formed_mathtext
        # -- fixing that defect must produce exactly one red test, not two.
        assert p.est_label == r"$\hat{\varphi_{0}}$"
        assert present(p.err_label).startswith(r"$\sigma_{")
        assert present(p.err_label).endswith("$")
        assert r"\hat{\varphi_{0}}" in present(p.err_label)


@pytest.mark.parametrize(
    "est_type", [ARMAEstType.AR, ARMAEstType.AR_OFFSET], ids=lambda t: t.name
)
def test_formula_and_param_labels_agree_on_the_ar_symbol(est_type):
    # Direction-agnostic: the symbol is read out of formula() and the label
    # dispatcher is required to agree with it, so either spelling fixes this.
    formula = est_type.formula()
    formula_symbol = r"\varphi" if r"\varphi" in formula else r"\phi"
    p = _param()
    est_type.set_param_labels(p, 1)
    assert formula_symbol in present(p.est_label)
    assert formula_symbol in present(p.err_label)


# ---------------------------------------------------------------------------
# ParamEst
# ---------------------------------------------------------------------------

class TestParamEst:
    def test_constructor_stores_every_field(self):
        p = ParamEst("id-9", 2.5, 0.5, "L", "E", 3, 1, 2, "OLS_R2")
        assert (p.est_id, p.est, p.err) == ("id-9", 2.5, 0.5)
        assert (p.est_label, p.err_label) == ("L", "E")
        assert (p.order, p.row, p.column) == (3, 1, 2)
        assert p.param_type == "OLS_R2"

    def test_set_labels(self):
        p = _param()
        p.set_labels(r"$\lambda$", r"$\sigma_\lambda$")
        assert p.est_label == r"$\lambda$"
        assert p.err_label == r"$\sigma_\lambda$"

    def test_to_json_emits_exactly_the_fields(self):
        p = _param(1.5, 0.25, est_label=r"$\hat{\mu}$", order=2, row=3, column=4, param_type="TRANS_PARAM")
        loaded = json.loads(p.to_json())
        assert set(loaded) == PARAM_KEYS
        assert loaded["est"] == 1.5 and loaded["err"] == 0.25
        assert loaded["est_label"] == r"$\hat{\mu}$"   # backslashes survive the escape
        assert loaded["err_label"] is None
        assert (loaded["order"], loaded["row"], loaded["column"]) == (2, 3, 4)
        assert loaded["est_id"] == EST_ID
        assert loaded["param_type"] == "TRANS_PARAM"

    def test_to_json_accepts_numpy_float64_values(self):
        # Estimators hand over numpy scalars straight from statsmodels results.
        # numpy.float64 only survives because it subclasses float, so json's own
        # float encoder handles it before the ``default`` hook is consulted --
        # assert against the emitted text, not against json.loads' return type.
        p = _param(numpy.float64(0.75), numpy.float64(0.0625))
        text = p.to_json()
        assert '"est": 0.75' in text
        assert '"err": 0.0625' in text
        loaded = json.loads(text)
        assert loaded["est"] == 0.75 and loaded["err"] == 0.0625

    @pytest.mark.parametrize(
        "field, value, expected",
        [
            ("order", numpy.int64(2), 2),
            ("row", numpy.int32(1), 1),
            ("est", numpy.float32(0.5), 0.5),
        ],
        ids=["order-int64", "row-int32", "est-float32"],
    )
    def test_to_json_serialises_non_float_numpy_scalars(self, field, value, expected):
        p = _param(**{field: value})
        assert json.loads(p.to_json())[field] == expected

    def test_to_json_pretty_parses_to_same_document(self):
        p = _param()
        compact, pretty = p.to_json(), p.to_json(pretty=True)
        assert "\n" not in compact
        assert "\n" in pretty
        assert json.loads(pretty) == json.loads(compact)

    def test_from_dict_to_json_round_trip(self):
        original = _param(numpy.float64(1.25), 0.125, est_label=r"$\alpha_1$", err_label=r"$\sigma_{\alpha_1}$",
                          order=1, row=2, column=3, param_type=ARMAParamType.ARMA_PARAM.value)
        restored = ParamEst.from_dict(json.loads(original.to_json()))
        assert isinstance(restored, ParamEst)
        assert restored.__dict__ == original.__dict__

    def test_from_dict_defaults_for_optional_fields(self):
        p = ParamEst.from_dict({"est": 0.5, "err": 0.1, "est_id": "x", "param_type": "OLS_CONST"})
        assert (p.est, p.err, p.est_id, p.param_type) == (0.5, 0.1, "x", "OLS_CONST")
        assert p.est_label is None and p.err_label is None
        assert (p.order, p.row, p.column) == (0, 0, 0)

    def test_from_dict_distinguishes_an_absent_key_from_an_explicit_none(self):
        # from_dict tests membership (``"order" in data``) rather than taking a
        # default (``data.get("order", 0)``), so a key present with the value
        # None is NOT defaulted: it is passed straight through.  That is exactly
        # the shape to_json produces for an unlabelled row, so the distinction
        # decides what a JSON round trip of order/row/column yields.
        absent = ParamEst.from_dict({"est": 0.5, "err": 0.1, "est_id": "x", "param_type": "t"})
        explicit_none = ParamEst.from_dict({"est": 0.5, "err": 0.1, "est_id": "x", "param_type": "t",
                                            "est_label": None, "err_label": None,
                                            "order": None, "row": None, "column": None})
        assert (absent.order, absent.row, absent.column) == (0, 0, 0)
        assert (explicit_none.order, explicit_none.row, explicit_none.column) == (None, None, None)
        # est_label/err_label default to None either way, so those two agree.
        assert (absent.est_label, absent.err_label) == (explicit_none.est_label, explicit_none.err_label) == (None, None)

    def test_from_dict_reads_each_field_once(self):
        body = inspect.getsource(ParamEst.from_dict)
        assert body.count("err_label = data") == 1

    def test_from_dict_ignores_unknown_keys(self):
        p = ParamEst.from_dict({"est": 0.5, "err": 0.1, "est_id": "x", "param_type": "t", "pvalue": 0.03})
        assert not hasattr(p, "pvalue")
        assert set(p.__dict__) == PARAM_KEYS

    @pytest.mark.parametrize("missing", ["est", "err", "est_id", "param_type"])
    def test_from_dict_requires_core_fields(self, missing):
        data = {"est": 0.5, "err": 0.1, "est_id": "x", "param_type": "t"}
        del data[missing]
        # ``match`` is a regex *search*, so an unanchored "est" would also be
        # satisfied by KeyError('est_id'); anchor it and check the args so the
        # test proves *which* key from_dict complained about.
        with pytest.raises(KeyError, match=rf"^'{missing}'$") as excinfo:
            ParamEst.from_dict(data)
        assert excinfo.value.args == (missing,)

    def test_repr_and_str_structure(self):
        p = _param(1.5, 0.25, order=2, row=3, column=4, param_type="OLS_PARAM", est_id="abc")
        r, s = repr(p), str(p)
        assert r.startswith("ParamEst(") and r.endswith(")")
        assert not s.startswith("ParamEst(")
        for fragment in ("est=(1.5)", "order=(2)", "row=(3)", "column=(4)", "est_id=(abc)", "param_type=(OLS_PARAM)"):
            assert fragment in r
            assert fragment in s

    def test_repr_closes_err_parenthesis(self):
        p = _param(1.5, 0.25)
        assert "err=(0.25), est_label=(" in repr(p)


# ---------------------------------------------------------------------------
# OLSTransform
# ---------------------------------------------------------------------------

class TestOLSTransform:
    def test_wraps_param(self):
        p = _param(param_type=OLSParamType.TRANS_PARAM.value)
        t = OLSTransform(p)
        assert t.param is p

    def test_to_json_nests_param(self):
        p = _param(-0.0231, 0.004, est_label=r"$\lambda$", err_label=r"$\sigma_\lambda$",
                   order=1, row=1, param_type=OLSParamType.TRANS_PARAM.value)
        loaded = json.loads(OLSTransform(p).to_json())
        assert set(loaded) == {"param"}
        assert set(loaded["param"]) == PARAM_KEYS
        assert ParamEst.from_dict(loaded["param"]).__dict__ == p.__dict__

    def test_str_shows_param(self):
        p = _param(2.0, 0.5)
        s = str(OLSTransform(p))
        assert s.startswith("param=(")
        assert "est=(2.0)" in s

    def test_repr_embeds_the_wrapped_param(self):
        # The only green content assertion on OLSTransform.__repr__: its class
        # prefix is owned by test_ols_repr_prefix_is_class_name (xfail), so
        # without this nothing pins that the wrapped param reaches the repr.
        r = repr(OLSTransform(_param(2.0, 0.5, est_label=r"$\lambda$")))
        assert "param=(" in r
        assert "est=(2.0)" in r
        assert r"est_label=($\lambda$)" in r
        assert r.endswith(")")


# ---------------------------------------------------------------------------
# OLSResult
# ---------------------------------------------------------------------------

class TestOLSResult:
    def test_constructor_defaults(self):
        r = _ols_result(nparams=2, r2=0.9)
        assert r.est_model is EstModel.OLS
        assert r.est_id == EST_ID
        # r2 is stored as the raw float, not as the ParamEst its docstring and
        # the OLS_R2 enum member promise -- see
        # test_r2_is_the_param_est_row_its_type_is_declared_for.
        assert r.r2.est == 0.9
        assert len(r.params) == 2
        assert r.const.param_type == "OLS_CONST"
        assert r.param_transforms is None
        assert r.const_transform is None
        assert r.model is None

    def test_set_transforms(self):
        r = _ols_result(nparams=1)
        half_life = OLSTransform(_param(30.1, 2.0, param_type=OLSParamType.TRANS_PARAM.value))
        lam = OLSTransform(_param(-0.023, 0.001, row=1, param_type=OLSParamType.TRANS_PARAM.value))
        mu = OLSTransform(_param(0.5, 0.02, param_type=OLSParamType.TRANS_CONST.value))
        r.set_transforms("ou-model", [half_life, lam], mu)
        assert r.model == "ou-model"
        assert r.param_transforms == [half_life, lam]
        assert r.const_transform is mu

    def test_set_transforms_replaces_rather_than_accumulates(self):
        # set_transforms is a plain three-field assignment: a second call
        # silently discards the first set instead of appending or raising.
        r = _ols_result(nparams=1)
        first = OLSTransform(_param(1.0, 0.1, param_type=OLSParamType.TRANS_PARAM.value))
        second = OLSTransform(_param(2.0, 0.2, param_type=OLSParamType.TRANS_PARAM.value))
        r.set_transforms("first-model", [first], OLSTransform(_param(0.1)))
        r.set_transforms("second-model", [second], OLSTransform(_param(0.2)))
        assert r.model == "second-model"
        assert r.param_transforms == [second]
        assert first not in present(r.param_transforms)
        assert present(r.const_transform).param.est == 0.2

    def test_set_transforms_accepts_no_const_transform(self):
        # A model with a slope transform but no constant transform (the OU
        # half-life case with the offset left alone) passes const_transform=None;
        # the field must go back to its constructor state and serialise as null.
        r = _ols_result(nparams=1)
        r.set_transforms("no-const", [OLSTransform(_param(30.1, 2.0))], None)  # pyright: ignore[reportArgumentType]
        assert r.const_transform is None
        assert r.model == "no-const"
        assert len(present(r.param_transforms)) == 1
        loaded = json.loads(r.to_json())
        assert loaded["const_transform"] is None
        assert loaded["param_transforms"][0]["param"]["est"] == 30.1

    def test_to_json_before_transforms(self):
        r = _ols_result(nparams=2, r2=numpy.float64(0.87))   # statsmodels rsquared is a numpy scalar
        loaded = json.loads(r.to_json())
        assert set(loaded) == {"est_model", "const", "params", "r2", "param_transforms", "const_transform",
                               "est_id", "model"}
        assert loaded["est_model"] == "OLS"
        assert loaded["r2"]["est"] == 0.87
        assert loaded["r2"]["param_type"] == OLSParamType.OLS_R2.value
        assert loaded["est_id"] == EST_ID
        assert loaded["param_transforms"] is None
        assert loaded["const_transform"] is None
        assert loaded["model"] is None
        assert ParamEst.from_dict(loaded["const"]).__dict__ == r.const.__dict__
        assert [ParamEst.from_dict(d).__dict__ for d in loaded["params"]] == [p.__dict__ for p in r.params]
        assert [d["column"] for d in loaded["params"]] == [1, 2]

    def test_to_json_after_transforms(self):
        r = _ols_result(nparams=1)
        tp = _param(30.1, 2.0, est_label=r"$t_{1/2}$", param_type=OLSParamType.TRANS_PARAM.value)
        tc = _param(0.5, 0.02, est_label=r"$\mu$", param_type=OLSParamType.TRANS_CONST.value)
        r.set_transforms(r"$\Delta X_t = \lambda X_{t-1}$", [OLSTransform(tp)], OLSTransform(tc))
        loaded = json.loads(r.to_json())
        assert loaded["model"] == r"$\Delta X_t = \lambda X_{t-1}$"
        assert len(loaded["param_transforms"]) == 1
        assert ParamEst.from_dict(loaded["param_transforms"][0]["param"]).__dict__ == tp.__dict__
        assert ParamEst.from_dict(loaded["const_transform"]["param"]).__dict__ == tc.__dict__

    def test_repr_reflects_transform_state(self):
        r = _ols_result(nparams=1, r2=0.9)
        before = repr(r)
        assert "r2=(est=(0.9)" in before
        assert "model=(None)" in before
        assert f"est_id={EST_ID}" in before
        # est_model must be asserted explicitly: the "OLS" in the repr prefix
        # would otherwise satisfy any substring check even if the field vanished.
        assert "est_model=(EstModel.OLS)" in before
        r.set_transforms("my-model", [OLSTransform(_param())], OLSTransform(_param()))
        after = repr(r)
        assert "model=(my-model)" in after
        assert "model=(None)" not in after

    def test_str_is_the_props_without_the_class_prefix(self):
        # OLSResult.__str__ is otherwise the one statement in the module no test
        # executes; every sibling class gets a str() assertion.
        r = _ols_result(nparams=1, r2=0.9)
        s = str(r)
        assert not s.startswith("OLSEst(") and not s.startswith("OLSResult(")
        # Pin the field order and rendered values literally.  Comparing against
        # f"OLSEst({s})" would only re-derive repr from the same private
        # __props() that produced s, and so could not fail for any defect in it.
        assert s.startswith(f"est_id={EST_ID}, est_model=(EstModel.OLS), const=(")
        assert s.endswith("model=(None), const_transform=(None), param_transforms=(None)")
        assert "r2=(est=(0.9)" in s
        assert s.index("const=(") < s.index("params=(") < s.index("r2=(") < s.index("model=(None)")

    def test_repr_closes_params_parenthesis(self):
        assert "), r2=(" in repr(_ols_result())

    def test_r2_is_the_param_est_row_its_type_is_declared_for(self):
        r = _ols_result(r2=0.87)
        assert isinstance(r.r2, ParamEst)
        assert r.r2.param_type == OLSParamType.OLS_R2.value
        assert r.r2.est == 0.87


@pytest.mark.parametrize(
    "factory, cls_name",
    [(_ols_result, "OLSResult"), (lambda: OLSTransform(_param()), "OLSTransform")],
    ids=["OLSResult", "OLSTransform"],
)
def test_ols_repr_prefix_is_class_name(factory, cls_name):
    assert repr(factory()).startswith(f"{cls_name}(")


# ---------------------------------------------------------------------------
# ARMAEst
# ---------------------------------------------------------------------------

class TestARMAEst:
    def test_constructor_defaults(self):
        est = _arma_est(nparams=3)
        assert est.est_model is EstModel.ARMA
        assert est.arma_est_type is ARMAEstType.AR
        assert est.order == 3
        assert est.est_id == EST_ID
        assert est.const.param_type == "ARMA_CONST"
        assert est.sigma2.param_type == "ARMA_SIG2"
        assert [p.order for p in est.params] == [1, 2, 3]

    def test_order_zero_with_no_params(self):
        est = _arma_est(nparams=0)
        assert est.order == 0
        assert est.params == []

    def test_const_and_sigma2_labels(self):
        # AR_OFFSET is the type whose formula actually carries a μ* term, so it
        # The const row holds the process mean for every estimate type; μ* is a
        # separate row, present only when the model declares an offset, and is
        # pinned by test_offset_is_derived_from_the_mean_per_model_family.
        est = _arma_est(ARMAEstType.AR_OFFSET)
        assert est.const.est_label == r"$\hat{\mu}$"
        assert est.const.err_label == r"$\sigma_{\hat{\mu}}$"
        assert est.sigma2.est_label == r"$\hat{\sigma^2}$"
        assert est.sigma2.err_label == r"$\sigma_{\hat{\sigma^2}}$"

    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_sigma2_labels_are_est_type_independent(self, est_type):
        est = _arma_est(est_type)
        assert est.sigma2.est_label == r"$\hat{\sigma^2}$"
        assert est.sigma2.err_label == r"$\sigma_{\hat{\sigma^2}}$"

    @pytest.mark.parametrize("est_type", [ARMAEstType.AR, ARMAEstType.MA], ids=lambda t: t.name)
    def test_const_label_is_not_claimed_for_offset_free_models(self, est_type):
        est = _arma_est(est_type)
        assert r"\mu^*" not in est_type.formula()      # premise: no offset in the model
        assert r"\mu^*" not in (est.const.est_label or "")
        assert r"\mu^*" not in (est.const.err_label or "")

    @pytest.mark.parametrize(
        "est_type, symbol",
        [
            (ARMAEstType.AR, r"\varphi"),
            (ARMAEstType.AR_OFFSET, r"\varphi"),
            (ARMAEstType.MA, r"\theta"),
            (ARMAEstType.MA_OFFSET, r"\theta"),
        ],
        ids=lambda x: x.name if isinstance(x, ARMAEstType) else None,
    )
    def test_param_labels_follow_est_type(self, est_type, symbol):
        # What this owns: the est_type -> symbol dispatch (φ for the AR family,
        # θ for the MA family), one label per parameter, all distinct and in
        # parameter order.  The *index* inside each label is owned by
        # test_param_label_index_matches_the_stored_order below, so fixing that
        # off-by-one produces exactly one red test rather than five.
        est = _arma_est(est_type, nparams=3)
        labels = [present(p.est_label) for p in est.params]
        assert len(set(labels)) == len(labels) == 3
        assert all(label.startswith(rf"$\hat{{{symbol}_{{") and label.endswith("}}$") for label in labels)
        other = r"\theta" if symbol == r"\varphi" else r"\varphi"
        assert all(other not in label for label in labels)
        assert all(present(p.err_label).startswith(r"$\sigma_{") and symbol in present(p.err_label) for p in est.params)

    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_param_label_index_matches_the_stored_order(self, est_type):
        symbol = r"\varphi" if est_type in (ARMAEstType.AR, ARMAEstType.AR_OFFSET) else r"\theta"
        est = _arma_est(est_type, nparams=2)
        assert [p.order for p in est.params] == [1, 2]          # as the estimators store them
        assert [p.est_label for p in est.params] == [rf"$\hat{{{symbol}_{{{p.order}}}}}$" for p in est.params]

    def test_param_labels_are_one_based_like_the_stored_order(self):
        # The subscript is the stored order, so the first parameter -- the lag-1
        # coefficient -- is labelled 1, matching formula()'s sum from i = 1.
        est = _arma_est(ARMAEstType.AR, nparams=2)
        assert est.params[0].order == 1
        assert est.params[0].est_label == r"$\hat{\varphi_{1}}$"

    def test_offset_is_derived_from_the_mean_per_model_family(self):
        # statsmodels reports the process MEAN as its constant. For an AR the mean
        # is mu*/(1 - sum phi), so mu* = mean(1 - sum phi); for an MA the mean IS
        # mu*, because the offset does not pass through the moving average.
        # Applying the AR conversion to an MA would corrupt a correct value, so the
        # branch is on the model family, not on whether a constant was fitted.
        ar = _arma_est(ARMAEstType.AR_OFFSET, nparams=2)     # const 30.1, params 0.5 and 0.25
        φ_sum = sum(p.est for p in ar.params)
        assert present(ar.offset).est == pytest.approx(ar.const.est*(1.0 - φ_sum))
        assert present(ar.offset).est != pytest.approx(ar.const.est)

        ma = _arma_est(ARMAEstType.MA_OFFSET, nparams=2)
        assert present(ma.offset).est == pytest.approx(ma.const.est)
        assert present(ma.offset).err == pytest.approx(ma.const.err)

    @pytest.mark.parametrize("est_type", [ARMAEstType.AR, ARMAEstType.MA])
    def test_no_offset_row_for_models_that_declare_none(self, est_type):
        # AR and MA carry no mu* term in their own formula(), so no offset row is
        # emitted even though the estimator still fits a constant.
        est = _arma_est(est_type)
        assert est.offset is None
        assert r"\mu^*" not in est_type.formula()

    def test_offset_row_is_labelled_and_typed_for_mu_star(self):
        est = _arma_est(ARMAEstType.AR_OFFSET)
        offset = present(est.offset)
        assert offset.est_label == r"$\hat{\mu^*}$"
        assert offset.err_label == r"$\sigma_{\hat{\mu^*}}$"
        assert offset.param_type == ARMAParamType.ARMA_OFFSET.value
        assert offset.est_id == est.est_id

    def test_trend_records_what_the_fit_actually_used(self):
        # The estimate type says what was asked for; trend says what was fitted.
        # They are not the same: statsmodels defaults to 'c' at d = 0, so the
        # offset-free types are fitted with a constant too.
        assert _arma_est(ARMAEstType.AR).trend is None      # not supplied by the caller
        est = ARMAEst(EST_ID, _param(1.0), [_param(0.5)], _param(2.0),
                      ARMAEstType.AR_OFFSET, "c")
        assert est.trend == "c"

    def test_constructor_overwrites_labels_on_passed_objects(self):
        const = _param(est_label="c", err_label="c-err")
        params = [_param(est_label="p", err_label="p-err")]
        sigma2 = _param(est_label="s", err_label="s-err")
        ARMAEst(EST_ID, const, params, sigma2, ARMAEstType.MA)
        assert const.est_label == r"$\hat{\mu}$"
        assert const.err_label != "c-err"
        # Structural: the subscript is owned by
        # test_param_label_index_matches_the_stored_order.
        assert present(params[0].est_label).startswith(r"$\hat{\theta_{")
        assert present(params[0].err_label).startswith(r"$\sigma_{")
        assert sigma2.est_label == r"$\hat{\sigma^2}$"

    def test_to_json(self):
        est = _arma_est(ARMAEstType.MA_OFFSET, nparams=2)
        loaded = json.loads(est.to_json())
        assert set(loaded) == {"est_model", "arma_est_type", "trend", "const", "offset",
                               "order", "params", "sigma2", "est_id"}
        assert loaded["est_model"] == "ARMA"
        assert loaded["arma_est_type"] == "MA_OFFSET"
        assert ARMAEstType(loaded["arma_est_type"]) is ARMAEstType.MA_OFFSET
        assert loaded["order"] == 2
        assert loaded["est_id"] == EST_ID
        assert ParamEst.from_dict(loaded["const"]).__dict__ == est.const.__dict__
        assert ParamEst.from_dict(loaded["sigma2"]).__dict__ == est.sigma2.__dict__
        assert [ParamEst.from_dict(d).__dict__ for d in loaded["params"]] == [p.__dict__ for p in est.params]
        assert [d["est"] for d in loaded["params"]] == [0.5, 0.25]
        assert [d["order"] for d in loaded["params"]] == [1, 2]

    def test_repr_and_str(self):
        est = _arma_est(ARMAEstType.MA, nparams=2)
        r, s = repr(est), str(est)
        assert r.startswith("ARMAEst(") and r.endswith(")")
        assert not s.startswith("ARMAEst(")
        # Pin the rendered fields, not a substring of the class-name prefix
        # ("ARMAEst(" already contains both "ARMA" and "MA").  Note the repr
        # leaks the Enum dunder form while to_json emits the bare value "MA".
        assert "est_model=(EstModel.ARMA)" in r
        assert "arma_est_type=(ARMAEstType.MA)" in r
        assert "est_model=(EstModel.ARMA)" in s
        assert "arma_est_type=(ARMAEstType.MA)" in s
        assert "order=(2)" in r
        assert f"est_id={EST_ID}" in r
        assert r"$\hat{\sigma^2}$" in r


# ---------------------------------------------------------------------------
# VAREst
# ---------------------------------------------------------------------------

class TestVAREst:
    def test_constructor(self):
        est, _, _ = _var_est(n=3, m=2)
        assert est.est_model is EstModel.VAR
        assert est.order == 3
        assert len(est.const) == 2
        assert len(est.params) == 3 * 2 * 2
        assert len(est.omega) == 2 * 2
        assert {p.param_type for p in est.params} == {"VAR_PARAM"}
        assert {p.param_type for p in est.omega} == {"VAR_OMEGA"}

    def test_to_json_preserves_lag_row_column_addressing(self):
        n, m = 2, 2
        est, Φ, Ω = _var_est(n=n, m=m)
        loaded = json.loads(est.to_json())
        # NB: no top-level "est_id" -- unlike ParamEst/OLSResult/ARMAEst.  The
        # join key still travels on every nested row; see
        # test_var_and_vecm_rows_share_one_est_id_join_key.
        assert set(loaded) == {"est_model", "const", "order", "params", "omega"}
        assert loaded["est_model"] == "VAR"
        assert all(p["param_type"] == "VAR_OMEGA" for p in loaded["omega"])
        assert loaded["order"] == n
        assert len(loaded["params"]) == n * m * m

        # Rebuild Φ and Ω from the serialised records alone.
        Φ_rec = numpy.full((n, m, m), numpy.nan)
        for entry in loaded["params"]:
            Φ_rec[entry["order"] - 1, entry["row"], entry["column"]] = entry["est"]
        assert_array_equal(Φ_rec, Φ)

        Ω_rec = numpy.full((m, m), numpy.nan)
        for entry in loaded["omega"]:
            Ω_rec[entry["row"], entry["column"]] = entry["est"]
        assert_array_equal(Ω_rec, Ω)

        assert [c["order"] for c in loaded["const"]] == [1, 2]
        assert all(c["param_type"] == "VAR_CONST" for c in loaded["const"])

    def test_repr_and_str(self):
        est, _, _ = _var_est(n=2, m=2)
        r, s = repr(est), str(est)
        assert r.startswith("VAREst(") and r.endswith(")")
        assert not s.startswith("VAREst(")
        # "VAR" alone is implied by the "VAREst(" prefix; assert the field.
        assert "est_model=(EstModel.VAR)" in r
        assert "est_model=(EstModel.VAR)" in s
        assert "order=(2)" in r
        for section in ("const=(", "params=(", "omega=("):
            assert section in r


# ---------------------------------------------------------------------------
# VECMEst
# ---------------------------------------------------------------------------

class TestVECMEst:
    def test_constructor(self):
        est = _vecm_est(neq=2, rank=1, order=2)
        assert est.est_model is EstModel.VECM
        assert est.rank == 1
        assert est.order == 2
        assert len(est.const) == 2
        assert len(est.lambda_est) == 2 * 1
        assert len(est.beta_est) == 2 * 1
        assert len(est.a_est) == 2 * 2 * 2
        assert len(est.omega) == 2 * 2
        assert {p.param_type for p in est.lambda_est} == {"VECM_LAMBDA"}
        assert {p.param_type for p in est.beta_est} == {"VECM_BETA"}
        assert {p.param_type for p in est.a_est} == {"VECM_ALPHA"}
        assert {p.param_type for p in est.const} == {"VECM_CONST"}
        # The omega rows carry whatever VECMParamType.VECM_OMEGA's value is --
        # today the wrong string "VAR_OMEGA", which is owned by
        # TestVECMParamType.test_vecm_omega_value_matches_name.  Asserting it
        # through the enum keeps that defect to one red test while still
        # proving the type reaches the container unaltered.
        # (Its blast radius while unfixed: persisted VECM covariance rows are
        # stamped with the VAR type string and so are indistinguishable from the
        # VAR_OMEGA rows of TestVAREst.test_constructor.)
        assert {p.param_type for p in est.omega} == {VECMParamType.VECM_OMEGA.value}

    def test_to_json(self):
        neq, rank, order = 2, 1, 2
        est = _vecm_est(neq=neq, rank=rank, order=order)
        loaded = json.loads(est.to_json())
        # NB: no top-level "est_id" -- unlike ParamEst/OLSResult/ARMAEst.  The
        # join key still travels on every nested row; see
        # test_var_and_vecm_rows_share_one_est_id_join_key.
        assert set(loaded) == {"est_model", "rank", "const", "order", "lambda_est", "beta_est", "a_est", "omega"}
        assert loaded["est_model"] == "VECM"
        # Serialised through the enum, not the literal, so the VECM_OMEGA value
        # defect stays owned by test_vecm_omega_value_matches_name alone.
        assert {p["param_type"] for p in loaded["omega"]} == {VECMParamType.VECM_OMEGA.value}
        assert (loaded["rank"], loaded["order"]) == (rank, order)
        assert len(loaded["lambda_est"]) == neq * rank
        assert len(loaded["beta_est"]) == neq * rank
        assert len(loaded["omega"]) == neq * neq
        assert len(loaded["a_est"]) == order * neq * neq

        # Each lag block of A is addressed by a 1-based order and a full (row, column) grid.
        grid = {(i, j) for i in range(neq) for j in range(neq)}
        for k in range(1, order + 1):
            block = [d for d in loaded["a_est"] if d["order"] == k]
            assert {(d["row"], d["column"]) for d in block} == grid
            assert {d["est"] for d in block} == {0.05 * k}

        # β is the cointegrating vector: normalised first component, then -0.5.
        assert [d["est"] for d in sorted(loaded["beta_est"], key=lambda d: d["row"])] == [1.0, -0.5]
        assert [ParamEst.from_dict(d).__dict__ for d in loaded["lambda_est"]] == [p.__dict__ for p in est.lambda_est]

    def test_repr_and_str(self):
        est = _vecm_est(neq=2, rank=1, order=2)
        r, s = repr(est), str(est)
        assert r.startswith("VECMEst(") and r.endswith(")")
        assert not s.startswith("VECMEst(")
        # "VECM" alone is implied by the "VECMEst(" prefix; assert the field.
        assert "est_model=(EstModel.VECM)" in r
        assert "est_model=(EstModel.VECM)" in s
        assert "order=(2)" in r
        for section in ("const=(", "lambda=(", "beta=(", "A=(", "omega=("):
            assert section in r


# ---------------------------------------------------------------------------
# Cross-cutting serialisation contracts
# ---------------------------------------------------------------------------

ALL_SERIALISABLE = [
    pytest.param(lambda: _param(est_label=r"$\alpha$"), id="ParamEst"),
    pytest.param(lambda: OLSTransform(_param()), id="OLSTransform"),
    pytest.param(_ols_result, id="OLSResult"),
    pytest.param(_arma_est, id="ARMAEst"),
    pytest.param(lambda: _var_est()[0], id="VAREst"),
    pytest.param(_vecm_est, id="VECMEst"),
]


@pytest.mark.parametrize("factory", ALL_SERIALISABLE)
def test_pretty_json_is_multiline_and_equivalent(factory):
    obj = factory()
    compact, pretty = obj.to_json(), obj.to_json(pretty=True)
    assert "\n" not in compact
    assert "\n" in pretty
    assert json.loads(pretty) == json.loads(compact)


@pytest.mark.parametrize(
    "factory, est_model",
    [
        pytest.param(_ols_result, EstModel.OLS, id="OLSResult"),
        pytest.param(_arma_est, EstModel.ARMA, id="ARMAEst"),
        pytest.param(lambda: _var_est()[0], EstModel.VAR, id="VAREst"),
        pytest.param(_vecm_est, EstModel.VECM, id="VECMEst"),
    ],
)
def test_result_objects_serialise_est_model_as_value(factory, est_model):
    obj = factory()
    assert obj.est_model is est_model
    loaded = json.loads(obj.to_json())
    assert loaded["est_model"] == est_model.value
    assert EstModel(loaded["est_model"]) is est_model


@pytest.mark.parametrize("factory", ALL_SERIALISABLE)
def test_repr_is_single_line(factory):
    assert "\n" not in repr(factory())


def _pretty_indent(obj) -> int:
    """Leading-space count of the first nested line of ``to_json(pretty=True)``."""
    first_nested = obj.to_json(pretty=True).split("\n")[1]
    return len(first_nested) - len(first_nested.lstrip(" "))


@pytest.mark.parametrize(
    "factory, indent",
    [
        pytest.param(lambda: _param(), 4, id="ParamEst"),
        pytest.param(lambda: OLSTransform(_param()), 4, id="OLSTransform"),
        pytest.param(_ols_result, 3, id="OLSResult"),
        pytest.param(_arma_est, 3, id="ARMAEst"),
        pytest.param(lambda: _var_est()[0], 3, id="VAREst"),
        pytest.param(_vecm_est, 3, id="VECMEst"),
    ],
)
def test_pretty_json_indent_width(factory, indent):
    # Pinned because the widths disagree across the module: ParamEst and
    # OLSTransform pass indent=4 (param_est.py:74,123) while the four composite
    # results pass indent=3 (:207,:312,:403,:480).  Merely asserting "a newline
    # exists" hides that entirely.  It is a cosmetic inconsistency and nothing
    # more: json.dumps applies one indent to a whole tree and to_json is never
    # called on a nested object, so a ParamEst inside an OLSResult renders at
    # the container's width (3), not at its own -- there is no defect to xfail
    # here, only two top-level styles.
    assert _pretty_indent(factory()) == indent


def test_nested_objects_use_their_containers_indent():
    # The corollary that makes the width difference harmless: within one
    # document the indent is uniform, so the nested ParamEst rows of an
    # OLSResult step by the container's 3, never by ParamEst's own 4.
    pretty = _ols_result(nparams=1).to_json(pretty=True).split("\n")
    nested = [line for line in pretty if line.lstrip().startswith('"est":')]
    assert nested, "expected the const/params rows to appear in the pretty output"
    assert {len(line) - len(line.lstrip(" ")) for line in nested} == {6, 9}  # 2 x 3 and 3 x 3


# The paren imbalance every repr in the module inherits has exactly two root
# causes -- the missing ')' after ParamEst's err and after OLSResult's params.
# Both are owned by targeted strict xfails (TestParamEst.test_repr_closes_err_
# parenthesis, TestOLSResult.test_repr_closes_params_parenthesis); a generalised
# "count('(') == count(')')" sweep over all six classes would only restate those
# two defects six more times and turn five extra tests red on the one-line fix.


@pytest.mark.parametrize(
    "factory, est_model",
    [
        pytest.param(_ols_result, EstModel.OLS, id="OLSResult"),
        pytest.param(_arma_est, EstModel.ARMA, id="ARMAEst"),
        pytest.param(lambda: _var_est()[0], EstModel.VAR, id="VAREst"),
        pytest.param(_vecm_est, EstModel.VECM, id="VECMEst"),
    ],
)
def test_repr_renders_enum_dunder_form_while_json_renders_value(factory, est_model):
    # The (str, Enum) mixin does not reach f-string interpolation (3.11 Enum
    # change), so the two serialisations of est_model disagree: repr says
    # "EstModel.VAR", to_json says "VAR".  Pinned so the divergence is visible.
    obj = factory()
    r = repr(obj)
    assert f"est_model=({est_model.__class__.__name__}.{est_model.name})" in r
    assert f"est_model=({est_model.value})" not in r
    assert json.loads(obj.to_json())["est_model"] == est_model.value


def test_arma_est_type_repr_renders_enum_dunder_form_while_json_renders_value():
    est = _arma_est(ARMAEstType.MA_OFFSET)
    assert "arma_est_type=(ARMAEstType.MA_OFFSET)" in repr(est)
    assert json.loads(est.to_json())["arma_est_type"] == "MA_OFFSET"


@pytest.mark.parametrize(
    "factory", [pytest.param(_ols_result, id="OLSResult"), pytest.param(_arma_est, id="ARMAEst")]
)
def test_container_repr_carries_the_est_id(factory):
    # OLSResult/ARMAEst render est_id bare ("est_id=<id>") where ParamEst wraps
    # it ("est_id=(<id>)").  That difference is cosmetic -- nothing parses these
    # reprs -- so it is recorded here rather than as a strict xfail that would
    # fire on any incidental repr tidy-up.  What matters, and what is asserted,
    # is that the id reaches the repr at all.
    assert EST_ID in repr(factory())


@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: _var_est()[0], id="VAREst"), pytest.param(_vecm_est, id="VECMEst")]
)
def test_var_and_vecm_rows_share_one_est_id_join_key(factory):
    # VAREst/VECMEst carry no est_id attribute of their own, unlike
    # ParamEst/OLSResult/ARMAEst -- but that does NOT orphan the serialised
    # container: lib.data.impl.var and .vecm stamp the same uuid on every row
    # they build, so the join key is present on each record.  A top-level copy
    # would be redundant, which is why this is a green contract test and not an
    # xfail: what has to hold is that all the rows agree on one id.
    obj = factory()
    loaded = json.loads(obj.to_json())
    rows = [row for value in loaded.values() if isinstance(value, list) for row in value]
    assert len(rows) >= 8
    assert {row["est_id"] for row in rows} == {EST_ID}
    # The container itself has no est_id field to disagree with them.
    assert "est_id" not in loaded


def test_var_mathtext_labels_survive_serialisation():
    # The labels lib.data.impl.var stamps are backslash-heavy mathtext, and
    # JSON has to escape every backslash: the interesting failure is a label
    # arriving at the plot layer as "$\\hat{\\Phi}$" or with the backslashes
    # eaten.  Assert on the raw text as well as on the parsed document.
    est, _, _ = _var_est(n=2, m=2)
    text = est.to_json()
    assert r'"est_label": "$\\hat{\\Phi}$"' in text        # escaped on the wire
    loaded = json.loads(text)
    for field, (est_label, err_label) in VAR_LABELS.items():
        assert {row["est_label"] for row in loaded[field]} == {est_label}
        assert {row["err_label"] for row in loaded[field]} == {err_label}
        assert "\\\\" not in est_label                     # ... but not once parsed
    assert loaded["params"][0]["est_label"] == r"$\hat{\Phi}$"


def test_vecm_mathtext_labels_survive_serialisation():
    est = _vecm_est(neq=2, rank=1, order=2)
    loaded = json.loads(est.to_json())
    for field, (est_label, err_label) in VECM_LABELS.items():
        assert {row["est_label"] for row in loaded[field]} == {est_label}
        assert {row["err_label"] for row in loaded[field]} == {err_label}
    # λ and β must stay distinguishable after the round trip -- they share
    # (row, column) addressing and differ only by label and param_type.
    assert loaded["lambda_est"][0]["est_label"] != loaded["beta_est"][0]["est_label"]
    assert ParamEst.from_dict(loaded["lambda_est"][0]).est_label == r"$\hat{\lambda}$"


@pytest.mark.parametrize("factory", ALL_SERIALISABLE)
def test_every_param_est_row_rehydrates_through_param_est_from_dict(factory):
    # ParamEst.from_dict is the module's ONLY rehydrator -- the composite
    # results serialise one way by design (lib.data.impl.* build them from
    # statsmodels results and only ever call ParamEst.from_dict, e.g.
    # arima.py:507,513,517 and stats.py:867,877).  So the round trip that has to
    # hold is per row: every ParamEst-shaped record in any container's JSON
    # rebuilds into an identical ParamEst, labels and all.
    obj = factory()
    loaded = json.loads(obj.to_json())

    def rows(node):
        if isinstance(node, dict):
            if set(node) == PARAM_KEYS:
                yield node
            else:
                for value in node.values():
                    yield from rows(value)
        elif isinstance(node, list):
            for value in node:
                yield from rows(value)

    found = list(rows(loaded))
    assert found, "expected at least one ParamEst row in the serialised form"
    for row in found:
        restored = ParamEst.from_dict(row)
        assert restored.__dict__ == row
        assert json.loads(restored.to_json()) == row


# ---------------------------------------------------------------------------
# Dispatch error branches
# ---------------------------------------------------------------------------

class _UnhandledEstType:
    """Stands in for an estimate type none of the ARMAEstType if-chains handle.

    ``formula`` and ``set_param_labels`` dispatch on ``self.value``, so calling
    them unbound with this stub is the only way to reach their final ``raise``.
    """

    value = "GARCH"


def test_formula_raises_for_unhandled_est_type():
    with pytest.raises(Exception, match="Estimate type is invalid"):
        ARMAEstType.formula(_UnhandledEstType())  # pyright: ignore[reportArgumentType]


def test_set_param_labels_raises_for_unhandled_est_type_and_leaves_param_untouched():
    p = _param(est_label="untouched", err_label="also-untouched")
    with pytest.raises(Exception, match="Estimate type is invalid"):
        ARMAEstType.set_param_labels(_UnhandledEstType(), p, 0)  # pyright: ignore[reportArgumentType]
    assert (p.est_label, p.err_label) == ("untouched", "also-untouched")


def test_arma_est_defaults_to_ar():
    # arma_est_type is omitted the way a caller taking the default would omit it;
    # the AR default is what decides the φ (rather than θ) parameter labels.
    est = ARMAEst(EST_ID, _param(), [_param(), _param()], _param())
    assert est.arma_est_type is ARMAEstType.AR
    # φ (not θ) is the point; the subscripts are owned by
    # test_param_label_index_matches_the_stored_order.
    assert all(present(p.est_label).startswith(r"$\hat{\varphi_{") for p in est.params)
    assert est.params[0].est_label != est.params[1].est_label
    assert json.loads(est.to_json())["arma_est_type"] == "AR"


# ---------------------------------------------------------------------------
# Module import hygiene
#
# param_est is a pure data-container module: enums, five plain result objects
# and json.dumps.  Everything it imports at module scope is paid for by every
# consumer -- the plot layer, the database layer and the notebooks all import it
# long before any estimator runs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["numpy", "uuid"])
def test_module_level_imports_are_used(name):
    source = inspect.getsource(param_est_module)
    # An import that is present must also be used. Asserting `count > 1` alone
    # is unsatisfiable for a genuinely unused import: deleting the import takes
    # the count to 0, so the assertion fails whether or not the code is fixed.
    if f"import {name}" in source:
        assert source.count(name) > 1, f"{name} is imported but never used"


def test_importing_param_est_does_not_pull_in_statsmodels():
    code = (
        "import sys; import lib.data.param_est; "
        "print('statsmodels' if 'statsmodels' in sys.modules else 'clean')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean"
