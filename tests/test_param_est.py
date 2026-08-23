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
    addressing VAREst/VECMEst rely on, built exactly the way the estimators
    build them;
  * __repr__ / __str__ structure.

Tests that expose defects in the library are kept and marked
xfail(strict=True) so they flip to FAIL as soon as the defect is fixed.
"""

import json

import numpy
import pytest
from numpy.testing import assert_array_equal

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


def _var_est(n=2, m=2) -> tuple[VAREst, numpy.ndarray, numpy.ndarray]:
    """Built like lib.data.impl.var: m equations, n lags. Φ[i, j, k] is the
    coefficient at lag i+1, row j, column k; Ω[i, j] the noise covariance."""
    Φ = numpy.arange(n * m * m, dtype=float).reshape(n, m, m) / 10.0
    Ω = numpy.arange(m * m, dtype=float).reshape(m, m) + 100.0
    const = [_param(float(i), 0.1, order=i + 1, param_type=VARParamType.VAR_CONST.value) for i in range(m)]
    params = [_param(Φ[i, j, k], 0.01, order=i + 1, row=j, column=k, param_type=VARParamType.VAR_PARAM.value)
              for i in range(n) for j in range(m) for k in range(m)]
    omega = [_param(Ω[i, j], 0.0, row=i, column=j, param_type=VARParamType.VAR_OMEGA.value)
             for i in range(m) for j in range(m)]
    return VAREst(order=n, const=const, params=params, omega=omega), Φ, Ω


def _vecm_est(neq=2, rank=1, order=2) -> VECMEst:
    """Built like lib.data.impl.vecm: λ and β are neq x rank, A has one
    neq x neq block per lag (order 1..p), Ω is neq x neq."""
    lambda_est = [_param(0.1 * (i + 1), 0.01, row=i, column=j, param_type=VECMParamType.VECM_LAMBDA.value)
                  for j in range(rank) for i in range(neq)]
    beta_est = [_param(1.0 if i == 0 else -0.5, 0.01, row=i, column=j, param_type=VECMParamType.VECM_BETA.value)
                for j in range(rank) for i in range(neq)]
    const = [_param(0.01 * i, 0.001, row=i, param_type=VECMParamType.VECM_CONST.value) for i in range(neq)]
    omega = [_param(1.0 if i == j else 0.2, 0.0, row=i, column=j, param_type=VECMParamType.VECM_OMEGA.value)
             for i in range(neq) for j in range(neq)]
    a_est = [_param(0.05 * (k + 1), 0.01, order=k + 1, row=i, column=j, param_type=VECMParamType.VECM_ALPHA.value)
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
        (ARMAParamType, {"ARMA_CONST", "ARMA_PARAM", "ARMA_SIG2"}),
        (VARParamType, {"VAR_CONST", "VAR_PARAM", "VAR_OMEGA"}),
        (ARMAEstType, {"AR", "AR_OFFSET", "MA", "MA_OFFSET"}),
    ],
    ids=lambda x: getattr(x, "__name__", None),
)
def test_param_type_enums_value_equals_name(enum_cls, expected_names):
    assert {m.name for m in enum_cls} == expected_names
    for member in enum_cls:
        assert isinstance(member, str)
        assert member.value == member.name
        assert enum_cls(member.name) is member


class TestVECMParamType:
    def test_members(self):
        assert {m.name for m in VECMParamType} == {"VECM_CONST", "VECM_ALPHA", "VECM_LAMBDA", "VECM_BETA", "VECM_OMEGA"}
        for member in VECMParamType:
            if member is not VECMParamType.VECM_OMEGA:
                assert member.value == member.name

    @pytest.mark.xfail(
        strict=True,
        reason="VECMParamType.VECM_OMEGA's value is 'VAR_OMEGA' (copied from VARParamType); every other "
               "member's value equals its name, so VECM omega rows are stored under the VAR type string "
               "and VECMParamType('VECM_OMEGA') raises ValueError",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="ARMAEstType.formula() returns the AR formula (sum of φ_i X_{t-i}) for MA and MA_OFFSET; "
               "an MA(q) model is a sum of θ_i ε_{t-i}",
    )
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
            (ARMAEstType.AR, r"$\hat{\phi_{3}}$"),
            (ARMAEstType.AR_OFFSET, r"$\hat{\phi_{3}}$"),
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
        symbol = r"\phi" if est_type in (ARMAEstType.AR, ARMAEstType.AR_OFFSET) else r"\theta"
        assert p.err_label.startswith(r"$\sigma_{")
        assert rf"\hat{{{symbol}_{{2}}}}" in p.err_label
        assert p.err_label != p.est_label

    @pytest.mark.xfail(
        strict=True,
        reason="set_param_labels builds err_label as '$\\sigma_{$\\hat{...}}$' -- a stray '$' inside the "
               "subscript leaves three math delimiters, which mathtext cannot render",
    )
    @pytest.mark.parametrize("est_type", list(ARMAEstType), ids=lambda t: t.name)
    def test_err_label_is_well_formed_mathtext(self, est_type):
        p = _param()
        est_type.set_param_labels(p, 0)
        assert p.err_label.count("$") == 2

    def test_overwrites_existing_labels(self):
        p = _param(est_label="old-est", err_label="old-err")
        ARMAEstType.AR.set_param_labels(p, 0)
        # Pin the concrete post-state, not merely "changed": err_label is the
        # malformed three-delimiter form xfailed by
        # test_err_label_is_well_formed_mathtext.
        assert p.est_label == r"$\hat{\phi_{0}}$"
        assert p.err_label == r"$\sigma_{$\hat{\phi_{0}}}$"


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
        "field, value",
        [
            ("order", numpy.int64(2)),
            ("row", numpy.int32(1)),
            ("est", numpy.float32(0.5)),
        ],
        ids=["order-int64", "row-int32", "est-float32"],
    )
    def test_to_json_breaks_on_non_float_numpy_scalars(self, field, value):
        # The ``default=lambda o: o.__dict__`` hook is the whole numpy/JSON
        # boundary: any numpy scalar that is NOT a float subclass reaches it and
        # dies on the missing ``__dict__`` -- and it raises AttributeError, not
        # the TypeError a caller would expect from a serialiser.
        p = _param(**{field: value})
        with pytest.raises(AttributeError, match=r"has no attribute '__dict__'"):
            p.to_json()

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

    @pytest.mark.xfail(
        strict=True,
        reason="ParamEst.__props omits the closing ')' after err, so repr reads 'err=(0.25, est_label=(...'",
    )
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


# ---------------------------------------------------------------------------
# OLSResult
# ---------------------------------------------------------------------------

class TestOLSResult:
    def test_constructor_defaults(self):
        r = _ols_result(nparams=2, r2=0.9)
        assert r.est_model is EstModel.OLS
        assert r.est_id == EST_ID
        assert r.r2 == 0.9
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

    def test_to_json_before_transforms(self):
        r = _ols_result(nparams=2, r2=numpy.float64(0.87))   # statsmodels rsquared is a numpy scalar
        loaded = json.loads(r.to_json())
        assert set(loaded) == {"est_model", "const", "params", "r2", "param_transforms", "const_transform",
                               "est_id", "model"}
        assert loaded["est_model"] == "OLS"
        assert loaded["r2"] == 0.87
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
        assert "r2=(0.9)" in before
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
        assert repr(r) == f"OLSEst({s})"
        assert s.startswith(f"est_id={EST_ID}, ")
        assert "est_model=(EstModel.OLS)" in s
        assert "r2=(0.9)" in s

    @pytest.mark.xfail(
        strict=True,
        reason="OLSResult.__props omits the closing ')' after params, so repr reads 'params=([...], r2=('",
    )
    def test_repr_closes_params_parenthesis(self):
        assert "), r2=(" in repr(_ols_result())


@pytest.mark.xfail(
    strict=True,
    reason="OLSResult.__repr__ and OLSTransform.__repr__ both emit the prefix 'OLSEst(' rather than their "
           "class name, so the two are indistinguishable in output (every other class reprs as itself)",
)
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
        est = _arma_est()
        assert est.const.est_label == r"$\hat{\mu^*}$"
        assert est.const.err_label == r"$\sigma_{\hat{\mu^*}}$"
        assert est.sigma2.est_label == r"$\hat{\sigma^2}$"
        assert est.sigma2.err_label == r"$\sigma_{\hat{\sigma^2}}$"

    @pytest.mark.parametrize(
        "est_type, expected_labels",
        [
            (ARMAEstType.AR, [r"$\hat{\phi_{0}}$", r"$\hat{\phi_{1}}$"]),
            (ARMAEstType.AR_OFFSET, [r"$\hat{\phi_{0}}$", r"$\hat{\phi_{1}}$"]),
            (ARMAEstType.MA, [r"$\hat{\theta_{0}}$", r"$\hat{\theta_{1}}$"]),
            (ARMAEstType.MA_OFFSET, [r"$\hat{\theta_{0}}$", r"$\hat{\theta_{1}}$"]),
        ],
        ids=lambda x: x.name if isinstance(x, ARMAEstType) else None,
    )
    def test_param_labels_follow_est_type(self, est_type, expected_labels):
        # Labels are indexed by position in ``params`` (0-based), independent of
        # the 1-based ``order`` the estimators assign.
        est = _arma_est(est_type, nparams=2)
        assert [p.est_label for p in est.params] == expected_labels
        symbol = r"\phi" if est_type in (ARMAEstType.AR, ARMAEstType.AR_OFFSET) else r"\theta"
        assert all(p.err_label.startswith(r"$\sigma_{") and symbol in p.err_label for p in est.params)

    def test_constructor_overwrites_labels_on_passed_objects(self):
        const = _param(est_label="c", err_label="c-err")
        params = [_param(est_label="p", err_label="p-err")]
        sigma2 = _param(est_label="s", err_label="s-err")
        ARMAEst(EST_ID, const, params, sigma2, ARMAEstType.MA)
        assert const.est_label == r"$\hat{\mu^*}$"
        assert params[0].est_label == r"$\hat{\theta_{0}}$"
        assert sigma2.est_label == r"$\hat{\sigma^2}$"

    def test_to_json(self):
        est = _arma_est(ARMAEstType.MA_OFFSET, nparams=2)
        loaded = json.loads(est.to_json())
        assert set(loaded) == {"est_model", "arma_est_type", "const", "order", "params", "sigma2", "est_id"}
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
        # NB: no "est_id" -- unlike ParamEst/OLSResult/ARMAEst.  See
        # test_container_carries_est_id for that gap.
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
        # BUG (see TestVECMParamType.test_vecm_omega_value_matches_name): the
        # blast radius of VECM_OMEGA == "VAR_OMEGA" is right here -- VECM
        # covariance rows carry the VAR type string, so once persisted they are
        # indistinguishable from the VAR_OMEGA rows of TestVAREst.test_constructor.
        assert {p.param_type for p in est.omega} == {"VAR_OMEGA"}

    def test_to_json(self):
        neq, rank, order = 2, 1, 2
        est = _vecm_est(neq=neq, rank=rank, order=order)
        loaded = json.loads(est.to_json())
        # NB: no "est_id" -- unlike ParamEst/OLSResult/ARMAEst.  See
        # test_container_carries_est_id for that gap.
        assert set(loaded) == {"est_model", "rank", "const", "order", "lambda_est", "beta_est", "a_est", "omega"}
        assert loaded["est_model"] == "VECM"
        # Serialised VECM covariance rows are stamped "VAR_OMEGA" -- the
        # persisted form of the VECM_OMEGA defect.
        assert {p["param_type"] for p in loaded["omega"]} == {"VAR_OMEGA"}
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
    # OLSTransform pass indent=4 while the four composite results pass an odd
    # indent=3.  Merely asserting "a newline exists" hides that entirely.
    assert _pretty_indent(factory()) == indent


@pytest.mark.xfail(
    strict=True,
    reason="to_json(pretty=True) indent width is inconsistent: ParamEst/OLSTransform use indent=4 "
           "while OLSResult/ARMAEst/VAREst/VECMEst use indent=3, so a nested ParamEst re-indents "
           "at a different width than the container that holds it",
)
def test_pretty_json_indent_is_consistent_across_the_module():
    widths = {_pretty_indent(factory.values[0]()) for factory in ALL_SERIALISABLE}
    assert len(widths) == 1


@pytest.mark.xfail(
    strict=True,
    reason="ParamEst.__props omits the closing ')' after err, so every repr that embeds a ParamEst "
           "(all of them) is paren-unbalanced; OLSResult.__props adds a second imbalance after params",
)
@pytest.mark.parametrize("factory", ALL_SERIALISABLE)
def test_repr_parentheses_are_balanced(factory):
    # Generalises the two targeted xfails (ParamEst err, OLSResult params): any
    # newly introduced "field=(" without its closing paren fails here too.
    r = repr(factory())
    assert r.count("(") == r.count(")")


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
    "factory",
    [
        pytest.param(lambda: _param(), id="ParamEst"),
        pytest.param(
            _ols_result, id="OLSResult",
            marks=pytest.mark.xfail(
                strict=True,
                reason="OLSResult.__props emits a bare 'est_id={...}' while ParamEst.__props emits "
                       "'est_id=({...})'; the est_id field is the only one in the repr without parens",
            ),
        ),
        pytest.param(
            _arma_est, id="ARMAEst",
            marks=pytest.mark.xfail(
                strict=True,
                reason="ARMAEst.__props emits a bare 'est_id={...}' while ParamEst.__props emits "
                       "'est_id=({...})'; the est_id field is the only one in the repr without parens",
            ),
        ),
    ],
)
def test_est_id_repr_formatting_matches_every_other_field(factory):
    r = repr(factory())
    assert f"est_id=({EST_ID})" in r
    # The bare form can only come from the container's own __props -- a nested
    # ParamEst always renders "est_id=(...)".
    assert f"est_id={EST_ID}" not in r


@pytest.mark.xfail(
    strict=True,
    reason="VAREst and VECMEst carry no est_id attribute (ParamEst/OLSResult/ARMAEst all do), so the "
           "serialised container cannot be joined back to its estimate even though every ParamEst row "
           "it holds carries est_id",
)
@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: _var_est()[0], id="VAREst"), pytest.param(_vecm_est, id="VECMEst")]
)
def test_container_carries_est_id(factory):
    obj = factory()
    assert hasattr(obj, "est_id")
    assert "est_id" in json.loads(obj.to_json())


@pytest.mark.xfail(
    strict=True,
    reason="only ParamEst defines from_dict; OLSResult/ARMAEst/VAREst/VECMEst serialise one way, so a "
           "persisted composite estimate result cannot be rehydrated from its own JSON",
)
@pytest.mark.parametrize("cls", [OLSResult, ARMAEst, VAREst, VECMEst], ids=lambda c: c.__name__)
def test_composite_results_round_trip_through_from_dict(cls):
    assert hasattr(cls, "from_dict")


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
        ARMAEstType.formula(_UnhandledEstType())


def test_set_param_labels_raises_for_unhandled_est_type_and_leaves_param_untouched():
    p = _param(est_label="untouched", err_label="also-untouched")
    with pytest.raises(Exception, match="Estimate type is invalid"):
        ARMAEstType.set_param_labels(_UnhandledEstType(), p, 0)
    assert (p.est_label, p.err_label) == ("untouched", "also-untouched")


def test_arma_est_defaults_to_ar():
    # arma_est_type is omitted the way a caller taking the default would omit it;
    # the AR default is what decides the φ (rather than θ) parameter labels.
    est = ARMAEst(EST_ID, _param(), [_param(), _param()], _param())
    assert est.arma_est_type is ARMAEstType.AR
    assert [p.est_label for p in est.params] == [r"$\hat{\phi_{0}}$", r"$\hat{\phi_{1}}$"]
