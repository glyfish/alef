"""Tests for ``lib.data.hyp_test``.

``hyp_test`` is the data-model layer of navi's hypothesis testing: ``str``-backed
enums whose ``status()``/``desc()`` if-chains decide whether a test passed, the
``StatisticalTest*`` containers that the ADF and variance-ratio pipelines fill
in, the VAR lag-order report plus its private builder, and the Granger/Johansen
result containers.

Nothing here simulates on its own, so most tests are contract checks driven by
hand-built inputs whose expected outcome is derived by hand. Where the model is
only ever produced by an estimator (the VAR order report, the Johansen report,
the ADF/variance-ratio reports) a short simulation with known parameters feeds
the real pipeline and the report is checked against what the known parameters
imply.

Simulated assertions are pinned only where a sweep of 150 to 400 seeds showed
zero failures; the comment on each names the sweep and the configuration that
made it safe. Where an outcome is a nominal-size rejection under a null -- the
Johansen trace test on independent random walks -- nothing boolean is pinned on
a single draw at all: the structural contract is asserted on one draw and the
rejection *rate* is asserted over an ensemble.
"""
import json
import types
import uuid
import warnings

import numpy
import pytest
from statsmodels.tsa.vector_ar.var_model import LagOrderResults

import lib.data.impl.adf as adf_data
import lib.data.impl.fbm as fbm_data
import lib.data.impl.stats as stats_data
import lib.data.impl.var as var_data
import lib.data.impl.vecm as vecm_data
import lib.models.fbm as fbm_model
from lib.data import hyp_test
from lib.data.hyp_test import (
    ErrorMetric,
    GrangerCausalityTestReport,
    GrangerCausalityTestResult,
    HypothesisTestStatus,
    HypothesisTestType,
    HypothesisType,
    JohansenCointTestEigenVector,
    JohansenCointTestRank,
    JohansenCointTestReport,
    JohansenCointTestStatistic,
    LagOrderTestResult,
    StatisticalTestData,
    StatisticalTestParam,
    StatisticalTestReport,
    VAROrderTestReport,
)

# ``__var_order_test_report_from_result`` is a module level function, so its
# leading double underscore is not mangled -- but a reference to it written
# inside a class body would be. Bind it once here under a mangle-proof alias.
var_order_report_from_result = getattr(hyp_test, "__var_order_test_report_from_result")

ADF_TYPES = [
    HypothesisTestType.STATIONARITY,
    HypothesisTestType.STATIONARITY_OFFSET,
    HypothesisTestType.STATIONARITY_DRIFT,
]

FBM_TYPES = [
    HypothesisTestType.FBM_AUTO_CORR,
    HypothesisTestType.FBM_NEG_AUTO_CORR,
]

S_VALS = [4, 6, 10, 16, 24]


# ##############################################################
# helpers
# ##############################################################

def stationary_ar1(n: int = 2000, phi: float = 0.5) -> numpy.ndarray:
    """AR(1) with |phi| < 1: stationary, so the ADF test should reject a unit root."""
    x = numpy.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + numpy.random.normal()
    return x


def random_walk(n: int = 2000, drift: float = 0.0) -> numpy.ndarray:
    """Unit root process: the ADF test should fail to reject."""
    return numpy.cumsum(drift + numpy.random.normal(0.0, 1.0, n))


def lag_order_results(**selected) -> LagOrderResults:
    """A statsmodels LagOrderResults with hand-picked minima.

    ``ics[metric][k]`` is the criterion value at lag ``k``; the value the report
    should surface for a metric is ``ics[metric][selected[metric]]``, which is
    made unmistakable here by giving every (metric, lag) cell a distinct value.
    """
    ics = {
        "aic": [100.0, 101.0, 102.0, 103.0],
        "bic": [200.0, 201.0, 202.0, 203.0],
        "fpe": [300.0, 301.0, 302.0, 303.0],
        "hqic": [400.0, 401.0, 402.0, 403.0],
    }
    return LagOrderResults(ics, dict(selected), vecm=False)


def a_param(test_id: str = "test-id", label: str = "$t$", value: float = 1.5) -> StatisticalTestParam:
    return StatisticalTestParam(test_id=test_id, label=label, value=value)


def a_test_data(upper: StatisticalTestParam | None = None) -> StatisticalTestData:
    return StatisticalTestData(
        test_id="test-id",
        status=HypothesisTestStatus.PASSED,
        stat=a_param(label="$t$", value=-3.5),
        pval=a_param(label="$p-value$", value=0.02),
        params=[a_param(label="$s$", value=4)],
        sig=a_param(label="10%", value=0.1),
        lower=a_param(label="$t_L$", value=-2.57),
        upper=upper,
    )


def compact_test_data() -> StatisticalTestData:
    """A minimal StatisticalTestData whose full ``repr`` is short enough to pin."""
    param = StatisticalTestParam(test_id="tid", label="L", value=1.0)
    return StatisticalTestData(test_id="tid", status=HypothesisTestStatus.PASSED,
                               stat=param, pval=param, params=[], sig=None,
                               lower=None, upper=None)


# ##############################################################
# HypothesisTestStatus
# ##############################################################

class TestHypothesisTestStatus:

    def test_members(self):
        assert HypothesisTestStatus.PASSED.value == "PASSED"
        assert HypothesisTestStatus.FAILED.value == "FAILED"

    def test_to_bool(self):
        assert HypothesisTestStatus.PASSED.to_bool() is True
        assert HypothesisTestStatus.FAILED.to_bool() is False

    @pytest.mark.parametrize("flag,expected", [(True, HypothesisTestStatus.PASSED),
                                               (False, HypothesisTestStatus.FAILED)])
    def test_from_bool(self, flag, expected):
        assert HypothesisTestStatus.from_bool(flag) is expected

    @pytest.mark.parametrize("flag", [True, False])
    def test_bool_round_trip(self, flag):
        assert HypothesisTestStatus.from_bool(flag).to_bool() is flag

    def test_from_bool_accepts_numpy_bool(self):
        # ADFTestReport.status_vals holds numpy.bool_, not builtin bool.
        assert HypothesisTestStatus.from_bool(numpy.bool_(True)) is HypothesisTestStatus.PASSED
        assert HypothesisTestStatus.from_bool(numpy.bool_(False)) is HypothesisTestStatus.FAILED

    @pytest.mark.parametrize("value,expected", [
        # from_bool is a plain truthiness test, and the estimator layer feeds it
        # numpy scalars of every dtype (and, when a report field was never set,
        # None). Zero of any numeric type is FAILED, anything else is PASSED.
        (numpy.float64(0.0), HypothesisTestStatus.FAILED),
        (numpy.float64(1.5), HypothesisTestStatus.PASSED),
        (numpy.float64(-1.5), HypothesisTestStatus.PASSED),
        (numpy.int64(0), HypothesisTestStatus.FAILED),
        (numpy.int64(1), HypothesisTestStatus.PASSED),
        (None, HypothesisTestStatus.FAILED),
        (0, HypothesisTestStatus.FAILED),
        (1, HypothesisTestStatus.PASSED),
    ])
    def test_from_bool_is_a_truthiness_test(self, value, expected):
        assert HypothesisTestStatus.from_bool(value) is expected

    def test_is_a_str_enum_so_it_serialises_as_its_value(self):
        assert HypothesisTestStatus.PASSED == "PASSED"
        assert json.dumps({"status": HypothesisTestStatus.FAILED}) == '{"status": "FAILED"}'


# ##############################################################
# HypothesisTestType
# ##############################################################

class TestHypothesisTestType:

    def test_values(self):
        # The two FBM members deliberately serialise under shortened values.
        assert HypothesisTestType.STATIONARITY.value == "STATIONARITY"
        assert HypothesisTestType.STATIONARITY_OFFSET.value == "STATIONARITY_OFFSET"
        assert HypothesisTestType.STATIONARITY_DRIFT.value == "STATIONARITY_DRIFT"
        assert HypothesisTestType.BM.value == "BM"
        assert HypothesisTestType.FBM_AUTO_CORR.value == "AUTO_CORR"
        assert HypothesisTestType.FBM_NEG_AUTO_CORR.value == "NEG_AUTO_CORR"

    @pytest.mark.parametrize("test_type,expected", [
        (HypothesisTestType.STATIONARITY, "Stationarity Test"),
        (HypothesisTestType.STATIONARITY_OFFSET, "Stationarity Test with Constant Offset."),
        (HypothesisTestType.STATIONARITY_DRIFT, "Stationarity Test with Drift."),
        (HypothesisTestType.BM, "Brownian Motion Test"),
        (HypothesisTestType.FBM_AUTO_CORR, "Autocorrelation Test"),
        (HypothesisTestType.FBM_NEG_AUTO_CORR, "Negative Autocorrelation Test"),
    ])
    def test_desc(self, test_type, expected):
        assert test_type.desc() == expected

    @pytest.mark.parametrize("test_type", ADF_TYPES)
    @pytest.mark.parametrize("status_vals,expected", [
        # ADFTestReport.status_vals is [1%, 5%, 10%] and True means "statistic
        # above the critical value", i.e. the unit root was not rejected. Only
        # the 10% slot matters and the report status is its negation: the first
        # two entries are varied here to prove they are ignored.
        ([True, True, True], HypothesisTestStatus.FAILED),
        ([False, False, True], HypothesisTestStatus.FAILED),
        ([True, True, False], HypothesisTestStatus.PASSED),
        ([False, False, False], HypothesisTestStatus.PASSED),
    ])
    def test_stationarity_status_negates_the_ten_percent_slot(self, test_type, status_vals, expected):
        assert test_type.status(status_vals) is expected

    @pytest.mark.parametrize("test_type", ADF_TYPES)
    def test_stationarity_status_requires_three_significance_levels(self, test_type):
        # The if-chain indexes status[2] unconditionally.
        with pytest.raises(IndexError):
            test_type.status([False, False])

    @pytest.mark.parametrize("status_vals,expected", [
        ([False, False, False], HypothesisTestStatus.FAILED),
        ([False, True, False], HypothesisTestStatus.PASSED),
        ([True, True, True], HypothesisTestStatus.PASSED),
        ([True], HypothesisTestStatus.PASSED),
        ([], HypothesisTestStatus.FAILED),
    ])
    def test_bm_status_passes_when_any_lag_passes(self, status_vals, expected):
        assert HypothesisTestType.BM.status(status_vals) is expected

    @pytest.mark.parametrize("test_type", FBM_TYPES)
    @pytest.mark.parametrize("status_vals,expected", [
        # These are the negation of BM: the variance ratio test is run one
        # tailed and correlation is claimed when a lag *fails* the test.
        ([True, True, True], HypothesisTestStatus.FAILED),
        ([True, False, True], HypothesisTestStatus.PASSED),
        ([False, False, False], HypothesisTestStatus.PASSED),
        ([], HypothesisTestStatus.FAILED),
    ])
    def test_fbm_status_passes_when_any_lag_fails(self, test_type, status_vals, expected):
        assert test_type.status(status_vals) is expected

    def test_status_rejects_unknown_test_type(self):
        # status() dispatches on self.value, so a stand-in with a foreign value
        # reaches the else branch that no real member can.
        bogus = types.SimpleNamespace(value="NOT_A_TEST")
        with pytest.raises(Exception, match="Test type is invalid"):
            HypothesisTestType.status(bogus, [True, True, True])

    def test_desc_rejects_unknown_test_type(self):
        bogus = types.SimpleNamespace(value="NOT_A_TEST")
        with pytest.raises(Exception, match="Test type is invalid"):
            HypothesisTestType.desc(bogus)


def test_hypothesis_type_members():
    assert [t.value for t in HypothesisType] == ["TWO_TAIL", "LOWER_TAIL", "UPPER_TAIL"]
    assert HypothesisType.LOWER_TAIL == "LOWER_TAIL"


# ##############################################################
# StatisticalTestParam
# ##############################################################

class TestStatisticalTestParam:

    def test_attributes(self):
        param = StatisticalTestParam(test_id="abc", label="$t$", value=-2.5)
        assert (param.test_id, param.label, param.value) == ("abc", "$t$", -2.5)

    def test_repr_and_str(self):
        param = a_param(test_id="abc", label="$t$", value=-2.5)
        assert repr(param) == "TestParam(label=($t$), value=(-2.5), test_id=(abc))"
        assert str(param) == "label=($t$), value=(-2.5), test_id=(abc)"

    def test_to_json(self):
        assert json.loads(a_param(test_id="abc", label="L", value=0.25).to_json()) == {
            "test_id": "abc", "label": "L", "value": 0.25,
        }

    def test_to_json_pretty_indents_by_four(self):
        pretty = a_param().to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('    "')
        assert json.loads(pretty) == json.loads(a_param().to_json())

    def test_from_dict_round_trip(self):
        param = a_param(test_id="abc", label=r"$\tau$", value=3.0)
        restored = StatisticalTestParam.from_dict(json.loads(param.to_json()))
        assert isinstance(restored, StatisticalTestParam)
        assert (restored.test_id, restored.label, restored.value) == ("abc", r"$\tau$", 3.0)

    @pytest.mark.parametrize("missing", ["label", "value", "test_id"])
    def test_from_dict_requires_every_key(self, missing):
        data = {"label": "L", "value": 1.0, "test_id": "abc"}
        del data[missing]
        with pytest.raises(KeyError):
            StatisticalTestParam.from_dict(data)

    def test_numpy_float_value_serialises(self):
        # Estimator output is numpy.float64; json handles it because it derives
        # from float, and the round trip lands on a builtin float.
        param = a_param(value=numpy.float64(1.25))
        assert StatisticalTestParam.from_dict(json.loads(param.to_json())).value == 1.25


# ##############################################################
# StatisticalTestData
# ##############################################################

class TestStatisticalTestData:

    def test_attributes(self):
        data = a_test_data(upper=a_param(label="$t_U$", value=2.57))
        assert data.test_id == "test-id"
        assert data.status is HypothesisTestStatus.PASSED
        assert data.stat.value == -3.5
        assert data.pval.value == 0.02
        assert [p.label for p in data.params] == ["$s$"]
        assert data.sig.label == "10%"
        assert data.lower.value == -2.57
        assert data.upper.value == 2.57

    def test_to_json_structure(self):
        raw = json.loads(a_test_data(upper=a_param(label="$t_U$", value=2.57)).to_json())
        assert set(raw) == {"test_id", "status", "stat", "pval", "params", "sig", "lower", "upper"}
        assert raw["status"] == "PASSED"
        assert raw["stat"] == {"test_id": "test-id", "label": "$t$", "value": -3.5}
        assert raw["params"] == [{"test_id": "test-id", "label": "$s$", "value": 4}]

    def test_to_json_writes_null_for_absent_optional_params(self):
        # The ADF pipeline always leaves `upper` unset (lower tail test only).
        assert json.loads(a_test_data().to_json())["upper"] is None

    def test_to_json_pretty_indents_by_four(self):
        pretty = a_test_data().to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('    "')
        assert json.loads(pretty) == json.loads(a_test_data().to_json())

    def test_from_dict_rebuilds_nested_params(self):
        raw = {
            "test_id": "abc",
            "status": HypothesisTestStatus.PASSED,
            "stat": {"test_id": "abc", "label": "$t$", "value": -3.5},
            "pval": {"test_id": "abc", "label": "$p$", "value": 0.02},
            "params": [{"test_id": "abc", "label": "$s$", "value": 4}],
            "sig": {"test_id": "abc", "label": "10%", "value": 0.1},
            "lower": {"test_id": "abc", "label": "$t_L$", "value": -2.57},
            "upper": {"test_id": "abc", "label": "$t_U$", "value": 2.57},
        }
        data = StatisticalTestData.from_dict(raw)
        assert isinstance(data.stat, StatisticalTestParam)
        assert data.stat.value == -3.5
        assert isinstance(data.params[0], StatisticalTestParam)
        assert data.params[0].value == 4
        assert data.upper.label == "$t_U$"
        assert data.test_id == "abc"

    def test_from_dict_defaults_when_optional_keys_are_absent(self):
        data = StatisticalTestData.from_dict({"test_id": "abc", "params": []})
        assert data.status is HypothesisTestStatus.FAILED
        assert (data.stat, data.pval, data.sig, data.lower, data.upper) == (None,) * 5
        assert data.params == []

    @pytest.mark.parametrize("missing", ["params", "test_id"])
    def test_from_dict_requires_params_and_test_id(self, missing):
        data = {"params": [], "test_id": "abc"}
        del data[missing]
        with pytest.raises(KeyError):
            StatisticalTestData.from_dict(data)

    @pytest.mark.xfail(strict=True, reason=(
        "StatisticalTestData.from_dict (hyp_test.py:239) does `status = data['status'] if "
        "'status' in data else HypothesisTestStatus.FAILED` with no enum coercion, so a "
        "decoded JSON status stays a bare str and `.to_bool()` is unavailable after a round "
        "trip. Identical to the hyp_type/hyp_test_type/status defect in "
        "StatisticalTestReport.from_dict (hyp_test.py:303-305)."))
    def test_from_dict_rehydrates_the_status_enum(self):
        raw = json.loads(a_test_data(upper=a_param()).to_json())
        data = StatisticalTestData.from_dict(raw)
        assert data.status is HypothesisTestStatus.PASSED
        assert data.status.to_bool() is True

    @pytest.mark.xfail(strict=True, reason=(
        "StatisticalTestData.from_dict guards optional params with `if 'stat' in data`, "
        "but to_json emits absent params as JSON null, so the key is present with a None "
        "value and StatisticalTestParam.from_dict(None) raises TypeError. Every ADF report "
        "hits this: __adf_report_from_result always passes upper=None."))
    def test_json_round_trip_with_an_absent_optional_param(self):
        raw = json.loads(a_test_data(upper=None).to_json())
        restored = StatisticalTestData.from_dict(raw)
        assert restored.upper is None

    def test_repr_wraps_str_under_the_class_name(self):
        data = a_test_data()
        assert repr(data) == f"StatisticalTestData({data})"
        assert "status=(HypothesisTestStatus.PASSED)" in repr(data)

    def test_repr_is_balanced(self):
        assert repr(compact_test_data()) == (
            "StatisticalTestData(test_id=(tid), "
            "status=(HypothesisTestStatus.PASSED), "
            "stat=(label=(L), value=(1.0), test_id=(tid)), "
            "pval=(label=(L), value=(1.0), test_id=(tid)), "
            "params=([]), sig=(None), lower=(None), upper=(None))")


# ##############################################################
# StatisticalTestReport
# ##############################################################

class TestStatisticalTestReport:

    def report(self, hyp_test_type=HypothesisTestType.STATIONARITY, upper=None):
        return StatisticalTestReport(
            test_id="test-id",
            status=HypothesisTestStatus.PASSED,
            hyp_type=HypothesisType.LOWER_TAIL,
            hyp_test_type=hyp_test_type,
            test_data=[a_test_data(upper=upper)],
        )

    @pytest.mark.parametrize("test_type,expected", [
        (HypothesisTestType.STATIONARITY, "Stationarity Test"),
        (HypothesisTestType.BM, "Brownian Motion Test"),
        (HypothesisTestType.FBM_NEG_AUTO_CORR, "Negative Autocorrelation Test"),
    ])
    def test_desc_is_derived_from_the_test_type(self, test_type, expected):
        # desc is not a constructor argument: the report fills it in itself.
        assert self.report(hyp_test_type=test_type).desc == expected

    def test_attributes(self):
        report = self.report()
        assert report.test_id == "test-id"
        assert report.status is HypothesisTestStatus.PASSED
        assert report.hyp_type is HypothesisType.LOWER_TAIL
        assert report.hyp_test_type is HypothesisTestType.STATIONARITY
        assert len(report.test_data) == 1

    def test_to_json_structure(self):
        raw = json.loads(self.report(hyp_test_type=HypothesisTestType.FBM_AUTO_CORR).to_json())
        assert set(raw) == {"test_id", "status", "hyp_type", "hyp_test_type", "test_data", "desc"}
        assert raw["status"] == "PASSED"
        assert raw["hyp_type"] == "LOWER_TAIL"
        # Persisted under the enum *value*, not the member name.
        assert raw["hyp_test_type"] == "AUTO_CORR"
        assert raw["desc"] == "Autocorrelation Test"
        assert raw["test_data"][0]["stat"]["value"] == -3.5

    def test_to_json_pretty_indents_by_four(self):
        pretty = self.report().to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('    "')

    def test_from_dict_with_enum_members(self):
        raw = json.loads(self.report(upper=a_param(label="$t_U$", value=2.57)).to_json())
        raw["hyp_test_type"] = HypothesisTestType.STATIONARITY
        report = StatisticalTestReport.from_dict(raw)
        assert report.desc == "Stationarity Test"
        assert report.test_id == "test-id"
        assert isinstance(report.test_data[0], StatisticalTestData)
        assert report.test_data[0].stat.value == -3.5

    def test_from_dict_status_defaults_to_failed(self):
        report = StatisticalTestReport.from_dict({
            "test_id": "abc",
            "hyp_type": HypothesisType.TWO_TAIL,
            "hyp_test_type": HypothesisTestType.BM,
            "test_data": [],
        })
        assert report.status is HypothesisTestStatus.FAILED
        assert report.desc == "Brownian Motion Test"

    @pytest.mark.parametrize("missing", ["hyp_type", "hyp_test_type", "test_data", "test_id"])
    def test_from_dict_requires_every_key_but_status(self, missing):
        data = {
            "test_id": "abc",
            "status": HypothesisTestStatus.PASSED,
            "hyp_type": HypothesisType.TWO_TAIL,
            "hyp_test_type": HypothesisTestType.BM,
            "test_data": [],
        }
        del data[missing]
        with pytest.raises(KeyError):
            StatisticalTestReport.from_dict(data)

    def test_from_dict_recomputes_desc_and_ignores_the_persisted_one(self):
        # to_json emits `desc`, but the constructor derives it from the test type
        # and from_dict never reads the key, so a stale value cannot leak back in.
        raw = json.loads(self.report(hyp_test_type=HypothesisTestType.BM,
                                     upper=a_param(label="$t_U$", value=2.57)).to_json())
        assert raw["desc"] == "Brownian Motion Test"
        raw["desc"] = "STALE DESCRIPTION"
        raw["hyp_test_type"] = HypothesisTestType.STATIONARITY_DRIFT
        assert StatisticalTestReport.from_dict(raw).desc == "Stationarity Test with Drift."

    def test_from_dict_does_not_need_a_desc_key_at_all(self):
        raw = json.loads(self.report(hyp_test_type=HypothesisTestType.BM,
                                     upper=a_param(label="$t_U$", value=2.57)).to_json())
        del raw["desc"]
        raw["hyp_test_type"] = HypothesisTestType.BM
        assert StatisticalTestReport.from_dict(raw).desc == "Brownian Motion Test"

    @pytest.mark.xfail(strict=True, reason=(
        "StatisticalTestReport.from_dict (hyp_test.py:303-304) copies data['status'] and "
        "data['hyp_type'] straight through with no enum coercion, so after a JSON round trip "
        "both are bare str: `report.status` is 'PASSED' rather than HypothesisTestStatus.PASSED "
        "and `report.hyp_type` is 'LOWER_TAIL' rather than HypothesisType.LOWER_TAIL. "
        "hyp_test_type is patched back to an enum member here so that only these two are "
        "under test -- the hyp_test_type third of the same defect is pinned separately by "
        "test_json_round_trip."))
    def test_from_dict_rehydrates_the_status_and_hyp_type_enums(self):
        raw = json.loads(self.report().to_json())
        assert raw["status"] == "PASSED" and raw["hyp_type"] == "LOWER_TAIL"
        raw["hyp_test_type"] = HypothesisTestType.STATIONARITY
        report = StatisticalTestReport.from_dict(raw)
        assert report.status is HypothesisTestStatus.PASSED
        assert report.hyp_type is HypothesisType.LOWER_TAIL

    @pytest.mark.xfail(strict=True, reason=(
        "StatisticalTestReport.from_dict passes data['hyp_test_type'] straight to the "
        "constructor, which calls hyp_test_type.desc(). Decoded JSON supplies a plain str, "
        "so the round trip raises AttributeError: 'str' object has no attribute 'desc'. "
        "The fix belongs in from_dict (coerce the value back to HypothesisTestType) or in "
        "__init__ (coerce before calling desc()); nothing in this file constrains which."))
    def test_json_round_trip(self):
        report = self.report(upper=a_param(label="$t_U$", value=2.57))
        restored = StatisticalTestReport.from_dict(json.loads(report.to_json()))
        assert restored.desc == "Stationarity Test"

    def test_repr_wraps_str_under_the_class_name(self):
        report = self.report()
        assert repr(report) == f"TestReport({report})"
        assert "hyp_test_type=(HypothesisTestType.STATIONARITY)" in repr(report)

    def test_repr_is_balanced(self):
        report = StatisticalTestReport(test_id="tid",
                                       status=HypothesisTestStatus.PASSED,
                                       hyp_type=HypothesisType.LOWER_TAIL,
                                       hyp_test_type=HypothesisTestType.STATIONARITY,
                                       test_data=[])
        assert repr(report) == (
            "TestReport(test_id=(tid), "
            "status=(HypothesisTestStatus.PASSED), "
            "hyp_type=(HypothesisType.LOWER_TAIL), "
            "hyp_test_type=(HypothesisTestType.STATIONARITY), "
            "desc=(Stationarity Test), "
            "test_data=([]))")


# ##############################################################
# ErrorMetric and LagOrderTestResult
# ##############################################################

def test_error_metric():
    assert [m.value for m in ErrorMetric] == ["AIC", "BIC", "FPE", "HQIC"]
    assert str(ErrorMetric.AIC) == "AIC"
    assert repr(ErrorMetric.HQIC) == "ErrorMetric(HQIC)"
    assert json.dumps({"m": ErrorMetric.FPE}) == '{"m": "FPE"}'


def test_lag_order_test_result():
    order = a_param(test_id="abc", label=r"$\tau_{AIC}$", value=2)
    value = a_param(test_id="abc", label=r"$\varepsilon_{AIC}$", value=-0.5)
    result = LagOrderTestResult(test_id="abc", order=order, error_metric=ErrorMetric.AIC, value=value)

    assert result.test_id == "abc"
    assert result.order.value == 2
    assert result.error_metric is ErrorMetric.AIC
    assert result.value.value == -0.5
    assert repr(result) == (r"LagOrderTestResult(test_id=abc, "
                            r"order=(label=($\tau_{AIC}$), value=(2), test_id=(abc)), "
                            r"error_metric=(AIC), "
                            r"value=(label=($\varepsilon_{AIC}$), value=(-0.5), test_id=(abc)))")
    assert repr(result) == f"LagOrderTestResult({result})"


# ##############################################################
# VAROrderTestReport and its builder
# ##############################################################

PREFIX = "_VAROrderTestReport__"


class TestVAROrderTestReport:
    """The report exposes only mangled private attributes, so its contents are
    read back through to_json()."""

    def test_each_metric_reports_its_own_minimum(self):
        raw = json.loads(var_order_report_from_result(
            lag_order_results(aic=2, bic=1, fpe=3, hqic=0)).to_json())

        expected = {"aic": (2, 102.0), "bic": (1, 201.0), "fpe": (3, 303.0), "hqic": (0, 400.0)}
        for metric, (order, value) in expected.items():
            entry = raw[PREFIX + metric]
            assert entry["error_metric"] == metric.upper()
            assert entry["order"]["value"] == order
            assert entry["order"]["label"] == "$\\tau_{" + metric.upper() + "}$"
            assert entry["value"]["value"] == value
            assert entry["value"]["label"] == "$\\varepsilon_{" + metric.upper() + "}$"

    @pytest.mark.parametrize("selected,expected", [
        # numpy.bincount([aic, bic, hqic]) then argmax: the modal order over the
        # three criteria, ties broken toward the smallest order. FPE never votes.
        (dict(aic=2, bic=2, fpe=0, hqic=2), 2),
        (dict(aic=2, bic=1, fpe=3, hqic=2), 2),
        (dict(aic=1, bic=1, fpe=3, hqic=3), 1),
        (dict(aic=3, bic=1, fpe=1, hqic=2), 1),
        (dict(aic=0, bic=0, fpe=3, hqic=3), 0),
        (dict(aic=1, bic=1, fpe=1, hqic=1), 1),
    ])
    def test_consensus_order_is_the_mode_of_aic_bic_hqic(self, selected, expected):
        raw = json.loads(var_order_report_from_result(lag_order_results(**selected)).to_json())
        assert raw[PREFIX + "order"]["value"] == expected
        assert raw[PREFIX + "order"]["label"] == "$\\tau_{min}$"

    def test_fpe_is_excluded_from_the_consensus(self):
        # AIC/BIC/HQIC unanimously pick 1 while FPE picks 3: FPE is still
        # reported, but the consensus ignores it.
        raw = json.loads(var_order_report_from_result(
            lag_order_results(aic=1, bic=1, fpe=3, hqic=1)).to_json())
        assert raw[PREFIX + "fpe"]["order"]["value"] == 3
        assert raw[PREFIX + "order"]["value"] == 1

    def test_orders_are_coerced_to_builtin_int(self):
        # statsmodels hands back numpy integers, which json.dumps cannot encode;
        # the builder wraps every order in int(). Checked on the objects rather
        # than on decoded JSON, where the distinction no longer exists.
        report = var_order_report_from_result(lag_order_results(
            aic=numpy.int64(2), bic=numpy.int64(2), fpe=numpy.int64(1), hqic=numpy.int64(2)))
        for metric in ["aic", "bic", "fpe", "hqic"]:
            assert type(getattr(report, PREFIX + metric).order.value) is int
        assert type(getattr(report, PREFIX + "order").value) is int
        assert json.loads(report.to_json())[PREFIX + "order"]["value"] == 2

    def test_one_uuid4_is_shared_by_every_part_of_the_report(self):
        raw = json.loads(var_order_report_from_result(
            lag_order_results(aic=1, bic=1, fpe=1, hqic=1)).to_json())
        test_id = raw[PREFIX + "test_id"]
        assert uuid.UUID(test_id).version == 4
        for metric in ["aic", "bic", "fpe", "hqic"]:
            assert raw[PREFIX + metric]["test_id"] == test_id
            assert raw[PREFIX + metric]["order"]["test_id"] == test_id
            assert raw[PREFIX + metric]["value"]["test_id"] == test_id
        assert raw[PREFIX + "order"]["test_id"] == test_id

    def test_distinct_reports_get_distinct_ids(self):
        first = json.loads(var_order_report_from_result(lag_order_results(aic=1, bic=1, fpe=1, hqic=1)).to_json())
        second = json.loads(var_order_report_from_result(lag_order_results(aic=1, bic=1, fpe=1, hqic=1)).to_json())
        assert first[PREFIX + "test_id"] != second[PREFIX + "test_id"]

    def test_to_json_pretty_indents_by_three(self):
        report = var_order_report_from_result(lag_order_results(aic=1, bic=1, fpe=1, hqic=1))
        pretty = report.to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('   "')
        assert json.loads(pretty) == json.loads(report.to_json())

    def test_repr_and_str(self):
        report = var_order_report_from_result(lag_order_results(aic=2, bic=1, fpe=3, hqic=0))
        assert repr(report).startswith("VAROrderTestReport(test_id=")
        assert r"value=(2)" in repr(report)
        assert "error_metric=(HQIC)" in repr(report)
        assert repr(report) == f"VAROrderTestReport({report})"

    def test_constructed_directly(self):
        order = a_param(label=r"$\tau_{min}$", value=1)
        aic = LagOrderTestResult("abc", order, ErrorMetric.AIC, a_param(value=1.0))
        report = VAROrderTestReport(test_id="abc", order=order, aic=aic, bic=aic, fpe=aic, hqic=aic)
        raw = json.loads(report.to_json())
        assert raw[PREFIX + "test_id"] == "abc"
        assert raw[PREFIX + "order"]["value"] == 1

    def test_var2_order_round_trip(self):
        # A diagonal, comfortably stationary VAR(2) with lag 2 coefficients (0.2,
        # 0.3) far from zero. Only BIC -- the strongly consistent criterion -- is
        # pinned: over 401 seeds it selected 2 every time, whereas HQIC selected 3
        # on one of them (which also dragged the AIC/BIC/HQIC consensus to 3) and
        # AIC over-selected on 10% of them. The consensus rule itself is pinned
        # against independent constants by
        # test_consensus_order_is_the_mode_of_aic_bic_hqic.
        phi = numpy.array([[[0.5, 0.0], [0.0, 0.4]],
                           [[0.2, 0.0], [0.0, 0.3]]])
        _, samples = var_data.create_source(phi, npts=2000)
        result, report = var_data.compute_order(samples, maxlags=6)

        assert int(result.bic) == 2
        raw = json.loads(report.to_json())
        # The true order is 2, so the reported BIC entry must carry lag 2 and the
        # BIC criterion evaluated there -- index 2 is the known truth here, not a
        # read-back of result.bic.
        assert raw[PREFIX + "bic"]["error_metric"] == "BIC"
        assert raw[PREFIX + "bic"]["order"]["value"] == 2
        assert raw[PREFIX + "bic"]["value"]["value"] == pytest.approx(
            float(result.ics["bic"][2]), rel=1.0e-12)
        # The BIC criterion really is minimised at lag 2 over the whole scan.
        assert int(numpy.argmin(result.ics["bic"])) == 2
        # Every metric is reported, and the consensus is one of the orders voted for.
        assert {raw[PREFIX + m]["error_metric"] for m in ["aic", "bic", "fpe", "hqic"]} == {
            "AIC", "BIC", "FPE", "HQIC"}
        assert raw[PREFIX + "order"]["value"] in {
            raw[PREFIX + m]["order"]["value"] for m in ["aic", "bic", "hqic"]}


# ##############################################################
# Granger causality models
# ##############################################################

class TestGrangerCausality:

    def records(self):
        # Shape produced by lib.stats.causality_matrix(...).to_dict('records').
        return [
            {"pvalue": 0.9, "critical_value": 0.05, "result": False, "dependent_var": 1, "causal_var": 1},
            {"pvalue": 0.01, "critical_value": 0.05, "result": True, "dependent_var": 1, "causal_var": 2},
            {"pvalue": 0.4, "critical_value": 0.05, "result": False, "dependent_var": 2, "causal_var": 1},
            {"pvalue": 0.8, "critical_value": 0.05, "result": False, "dependent_var": 2, "causal_var": 2},
        ]

    def test_result_from_dict_and_repr(self):
        result = GrangerCausalityTestResult.from_dict(self.records()[1], "est-id")
        text = repr(result)
        assert text.startswith("GrangerCausalityTestResult(")
        assert "causal_var=(2)" in text
        assert "dependent_var=(1)" in text
        assert "pvalue=(0.01)" in text
        assert "critical_value=(0.05)" in text
        assert "result=(True)" in text
        assert "est_id=(est-id)" in text
        assert text == f"GrangerCausalityTestResult({result})"

    def test_result_repr_has_no_trailing_separator(self):
        result = GrangerCausalityTestResult.from_dict(self.records()[1], "est-id")
        assert repr(result) == (
            "GrangerCausalityTestResult(causal_var=(2), dependent_var=(1), pvalue=(0.01), "
            "est_id=(est-id), critical_value=(0.05), result=(True))")

    @pytest.mark.parametrize("missing", ["dependent_var", "causal_var", "pvalue", "critical_value", "result"])
    def test_result_from_dict_requires_every_key(self, missing):
        data = dict(self.records()[0])
        del data[missing]
        with pytest.raises(KeyError):
            GrangerCausalityTestResult.from_dict(data, "est-id")

    def test_report_attributes_and_json(self):
        results = [GrangerCausalityTestResult.from_dict(r, "est-id") for r in self.records()]
        report = GrangerCausalityTestReport(test_id="est-id", rank=1, results=results)

        assert report.test_id == "est-id"
        assert report.rank == 1
        assert len(report.results) == 4

        raw = json.loads(report.to_json())
        assert set(raw) == {"test_id", "rank", "results"}
        prefix = "_GrangerCausalityTestResult__"
        assert raw["results"][1][prefix + "pvalue"] == 0.01
        assert raw["results"][1][prefix + "causal_var"] == 2
        assert raw["results"][1][prefix + "result"] is True
        assert raw["results"][1][prefix + "est_id"] == "est-id"

    def test_report_pretty_json_and_repr(self):
        report = GrangerCausalityTestReport(test_id="est-id", rank=2, results=[])
        pretty = report.to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('   "')
        assert json.loads(pretty) == {"test_id": "est-id", "rank": 2, "results": []}
        assert repr(report) == "GrangerCausalityTestReport(test_id=(est-id), rank = (2), results=([]))"
        assert repr(report) == f"GrangerCausalityTestReport({report})"


class TestGrangerCausalityFromTheRealProducer:
    """``GrangerCausalityTestReport.rank`` is filled in by
    ``lib/data/impl/stats.py:__granger_causality_model_from_result`` as
    ``len(numpy.unique(dependent_var[result]))`` -- the number of *distinct*
    dependent variables with at least one incoming causal relation, which is not
    the number of True cells. A three variable system in which one variable is
    driven by both of the others separates the two readings."""

    def one_driven_variable(self, n: int = 300) -> numpy.ndarray:
        # x and w are independent white noise, y is driven by the previous value
        # of both. Series order is [x, y, w], so the true causal cells are
        # (dependent 2, causal 1) and (dependent 2, causal 3): two True results
        # but a single distinct dependent variable.
        x = numpy.random.normal(size=n)
        w = numpy.random.normal(size=n)
        y = numpy.zeros(n)
        for i in range(1, n):
            y[i] = 0.8 * x[i - 1] + 0.8 * w[i - 1] + 0.3 * numpy.random.normal()
        return numpy.array([x, y, w])

    def matrix_and_report(self):
        # critical_value 1e-5 rather than the 0.05 default: the two true
        # relations have F statistics in the hundreds so their p-values are 0 to
        # four decimals either way, while nine cells tested at 5% produced a
        # spurious extra relation on three seeds in eight. At 1e-5 the recovered
        # causal set was exactly {(2,1), (2,3)} and the rank exactly 1 for all
        # 150 seeds swept.
        samples = self.one_driven_variable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats_data.compute_causality_matrix(samples, nlags=2, critical_value=1.0e-5)

    def test_rank_counts_distinct_dependent_variables_not_causal_pairs(self):
        matrix, report = self.matrix_and_report()

        detected = sorted((int(row["dependent_var"]), int(row["causal_var"]))
                          for _, row in matrix.iterrows() if bool(row["result"]))
        assert detected == [(2, 1), (2, 3)]
        # Two causal relations, both pointing at variable 2, so the rank is 1.
        assert report.rank == 1

    def test_report_wiring(self):
        matrix, report = self.matrix_and_report()

        assert isinstance(report, GrangerCausalityTestReport)
        assert len(report.results) == len(matrix) == 9
        assert all(isinstance(r, GrangerCausalityTestResult) for r in report.results)
        # One uuid4 shared by the report and every result it holds.
        assert uuid.UUID(report.test_id).version == 4
        prefix = "_GrangerCausalityTestResult__"
        raw = json.loads(report.to_json())
        assert raw["rank"] == 1
        assert {r[prefix + "est_id"] for r in raw["results"]} == {report.test_id}
        assert {r[prefix + "critical_value"] for r in raw["results"]} == {1.0e-5}
        assert sorted((r[prefix + "dependent_var"], r[prefix + "causal_var"])
                      for r in raw["results"]) == [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
        # The persisted flag is the p-value comparison the matrix reports.
        assert [r[prefix + "result"] for r in raw["results"]] == [
            bool(v) for v in matrix["result"]]

    def test_no_causality_means_rank_zero(self):
        n = 300
        samples = numpy.array([numpy.random.normal(size=n) for _ in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix, report = stats_data.compute_causality_matrix(samples, nlags=2,
                                                                 critical_value=1.0e-5)
        assert not matrix["result"].any()
        assert report.rank == 0


# ##############################################################
# Johansen cointegration models
# ##############################################################

class TestJohansenCointTestStatistic:

    def test_compares_the_statistic_against_every_critical_value(self):
        critical_values = numpy.array([13.4294, 15.4943, 19.9349])
        stat = JohansenCointTestStatistic(test_id="abc", test_rank=0, test_stat=16.0,
                                          critical_values=critical_values)
        assert stat.test_result == [True, True, False]
        assert all(isinstance(flag, bool) for flag in stat.test_result)
        assert stat.null_hypothesis == "r<=0"
        assert stat.critical_values == [13.4294, 15.4943, 19.9349]
        assert isinstance(stat.critical_values, list)
        assert stat.significance_levels == ["Critical Value 90%",
                                            "Critical Value 95%",
                                            "Critical Value 99%"]

    def test_statistic_equal_to_the_critical_value_does_not_reject(self):
        stat = JohansenCointTestStatistic(test_id="abc", test_rank=1, test_stat=10.0,
                                          critical_values=numpy.array([10.0, 20.0, 30.0]))
        assert stat.test_result == [False, False, False]
        assert stat.null_hypothesis == "r<=1"

    def test_json_serialisable_with_numpy_input(self):
        stat = JohansenCointTestStatistic(test_id="abc", test_rank=0,
                                          test_stat=numpy.float64(30.0),
                                          critical_values=numpy.array([13.4, 15.5, 19.9]))
        raw = json.loads(json.dumps(stat, default=lambda o: o.__dict__))
        assert raw["test_stat"] == 30.0
        assert raw["test_result"] == [True, True, True]

    def test_critical_values_must_be_an_ndarray(self):
        # __init__ calls critical_values.tolist(), so the plain Python list a
        # hand-written caller would reach for is rejected. Every construction
        # inside navi comes from statsmodels' result.cvt/result.cvm rows.
        with pytest.raises(AttributeError, match="tolist"):
            JohansenCointTestStatistic(test_id="abc", test_rank=0, test_stat=16.0,
                                       critical_values=[13.4, 15.5, 19.9])

    @pytest.mark.xfail(strict=True, reason=(
        "JohansenCointTestStatistic hard-codes three significance_levels (hyp_test.py:575) "
        "but derives test_result by zipping over whatever critical_values it was handed "
        "(hyp_test.py:576), with no length check. Two critical values therefore yield a two "
        "entry test_result labelled by three levels, so 'Critical Value 99%' has no result. "
        "Same defect class as the truncated rank list in "
        "test_rank_has_one_entry_per_significance_level."))
    def test_a_short_critical_value_row_is_rejected_or_labelled_consistently(self):
        stat = JohansenCointTestStatistic(test_id="abc", test_rank=0, test_stat=16.0,
                                          critical_values=numpy.array([1.0, 2.0]))
        assert len(stat.test_result) == len(stat.significance_levels)

    def test_repr_and_str(self):
        stat = JohansenCointTestStatistic(test_id="abc", test_rank=0, test_stat=16.0,
                                          critical_values=numpy.array([1.0, 2.0, 3.0]))
        assert repr(stat).startswith("JohansenCointTestStatistic(test_id=(abc)")
        assert "null_hypothesis=(r<=0)" in repr(stat)
        assert "test_stat=(16.0)" in repr(stat)
        assert repr(stat) == f"JohansenCointTestStatistic({stat})"


class TestJohansenCointTestRank:

    def test_attributes(self):
        ranks = JohansenCointTestRank(test_id="abc", test_ranks=[2, 1, 1])
        assert ranks.test_ranks == [2, 1, 1]
        assert ranks.significance_levels == ["Critical Value 90%",
                                             "Critical Value 95%",
                                             "Critical Value 99%"]
        assert repr(ranks) == "JohansenCointTestRank(test_id=(abc), test_ranks=([2, 1, 1]))"
        assert repr(ranks) == f"JohansenCointTestRank({ranks})"


class TestJohansenCointTestEigenVector:

    def test_discards_zero_imaginary_parts(self):
        # numpy>=2 returns complex128 from linalg.eig even for a real spectrum
        # and json cannot encode complex, so the model takes the real part.
        vector = JohansenCointTestEigenVector(test_id="abc",
                                              eigen_value=numpy.complex128(0.2 + 0.0j),
                                              eigen_vector=numpy.array([1.0 + 0.0j, -2.0 + 0.0j]))
        assert type(vector.eigen_value) is float
        assert vector.eigen_value == pytest.approx(0.2)
        assert vector.eigen_vector == [1.0, -2.0]
        assert json.loads(json.dumps(vector, default=lambda o: o.__dict__))["eigen_vector"] == [1.0, -2.0]

    def test_drops_any_imaginary_component(self):
        vector = JohansenCointTestEigenVector(test_id="abc",
                                              eigen_value=0.5 + 0.25j,
                                              eigen_vector=numpy.array([1.0 + 3.0j]))
        assert vector.eigen_value == pytest.approx(0.5)
        assert vector.eigen_vector == [1.0]

    def test_accepts_real_input(self):
        vector = JohansenCointTestEigenVector(test_id="abc", eigen_value=0.3,
                                              eigen_vector=numpy.array([0.5, 0.25]))
        assert vector.eigen_value == pytest.approx(0.3)
        assert vector.eigen_vector == [0.5, 0.25]
        assert repr(vector).startswith("JohansenCointTestEigenVector(test_id=(abc)")
        assert "eigen_vector=([0.5, 0.25])" in repr(vector)
        assert repr(vector) == f"JohansenCointTestEigenVector({vector})"


class TestJohansenCointTestReport:

    def report(self, ranks=(2, 1, 1)):
        trace = [JohansenCointTestStatistic("abc", i, 30.0 - 20.0 * i, numpy.array([13.4, 15.5, 19.9]))
                 for i in range(2)]
        eigen = [JohansenCointTestStatistic("abc", i, 25.0 - 20.0 * i, numpy.array([12.3, 14.3, 18.5]))
                 for i in range(2)]
        vectors = [JohansenCointTestEigenVector("abc", 0.2, numpy.array([1.0, -2.0])),
                   JohansenCointTestEigenVector("abc", 0.01, numpy.array([-1.0, 0.5]))]
        return JohansenCointTestReport(test_id="abc", trace_test=trace, eigen_test=eigen,
                                       ranks=JohansenCointTestRank("abc", list(ranks)),
                                       eigen_vectors=vectors)

    @pytest.mark.parametrize("ranks,expected", [((2, 1, 1), 1), ((2, 2, 2), 2), ((0, 1, 2), 0)])
    def test_rank_is_the_most_conservative_over_significance_levels(self, ranks, expected):
        assert self.report(ranks=ranks).rank == expected

    def test_rank_of_an_empty_rank_list_is_not_defined(self):
        # __init__ takes min() over ranks.test_ranks unconditionally. An empty
        # list is exactly what the truncated rank builder in
        # lib/data/impl/vecm.py can produce for a degenerate system.
        with pytest.raises(ValueError, match="empty"):
            JohansenCointTestReport(test_id="abc", trace_test=[], eigen_test=[],
                                    ranks=JohansenCointTestRank("abc", []),
                                    eigen_vectors=[])

    def test_json(self):
        raw = json.loads(self.report().to_json())
        assert set(raw) == {"test_id", "trace_test", "eigen_test", "ranks", "eigen_vectors", "rank"}
        assert raw["rank"] == 1
        assert raw["ranks"]["test_ranks"] == [2, 1, 1]
        assert raw["trace_test"][0]["test_result"] == [True, True, True]
        assert raw["trace_test"][1]["test_result"] == [False, False, False]
        assert raw["eigen_test"][0]["test_stat"] == 25.0
        assert raw["eigen_vectors"][0]["eigen_vector"] == [1.0, -2.0]

    def test_pretty_json_indents_by_three(self):
        report = self.report()
        pretty = report.to_json(pretty=True)
        assert pretty.splitlines()[1].startswith('   "')
        assert json.loads(pretty) == json.loads(report.to_json())

    def test_repr_and_str(self):
        report = self.report()
        assert repr(report).startswith("JohansenTestResult(test_id=(abc)")
        assert "rank=(1)" in repr(report)
        assert repr(report) == f"JohansenTestResult({report})"


class TestJohansenRoundTrip:
    """The Johansen models are only ever built from a statsmodels test result, so
    a cointegrated pair with a known rank is pushed through the real pipeline."""

    def cointegrated_pair(self, n: int = 2000) -> numpy.ndarray:
        # y - 2x is a stationary AR(1), x is a random walk: one cointegrating
        # relation, so the Johansen trace test must reject r<=0.
        x = numpy.cumsum(numpy.random.normal(0.0, 1.0, n))
        spread = numpy.zeros(n)
        for i in range(1, n):
            spread[i] = 0.5 * spread[i - 1] + numpy.random.normal(0.0, 0.5)
        return numpy.array([x, 2.0 * x + spread])

    def test_cointegrated_pair_rejects_the_no_relation_null(self):
        _, model, _ = vecm_data.compute_johansen_coint_test(self.cointegrated_pair(), 2)

        # Power against a spread this tight is effectively 1, and by an enormous
        # margin: over 161 seeds the r<=0 trace statistic was never below 15x the
        # 99% critical value, so a fixed 5x margin is a real claim about power
        # rather than a restatement of test_result[2].
        assert model.trace_test[0].null_hypothesis == "r<=0"
        assert model.trace_test[0].test_result == [True, True, True]
        assert model.trace_test[0].test_stat > 5.0 * model.trace_test[0].critical_values[2]
        assert model.rank >= 1
        # One eigenvalue dominates: the single cointegrating relation. The gap
        # was more than a factor of ten for every one of those seeds.
        assert model.eigen_vectors[0].eigen_value > 10.0 * model.eigen_vectors[1].eigen_value
        assert model.trace_test[1].null_hypothesis == "r<=1"
        assert len(model.eigen_vectors) == 2

    def test_json_carries_every_reported_number(self):
        _, model, _ = vecm_data.compute_johansen_coint_test(self.cointegrated_pair(), 2)
        raw = json.loads(model.to_json())

        assert set(raw) == {"test_id", "trace_test", "eigen_test", "ranks",
                            "eigen_vectors", "rank"}
        assert raw["test_id"] == model.test_id
        assert raw["rank"] == model.rank
        assert raw["ranks"]["test_ranks"] == model.ranks.test_ranks
        assert len(raw["trace_test"]) == len(raw["eigen_test"]) == 2
        assert raw["trace_test"][0]["test_stat"] == pytest.approx(model.trace_test[0].test_stat)
        assert raw["trace_test"][0]["test_result"] == model.trace_test[0].test_result
        assert raw["trace_test"][0]["critical_values"] == pytest.approx(
            model.trace_test[0].critical_values)
        assert raw["eigen_test"][1]["null_hypothesis"] == "r<=1"
        assert raw["eigen_vectors"][0]["eigen_vector"] == pytest.approx(
            model.eigen_vectors[0].eigen_vector)
        assert raw["eigen_vectors"][0]["eigen_value"] == pytest.approx(
            model.eigen_vectors[0].eigen_value)
        # Nothing survived as a numpy scalar or a complex number.
        assert all(type(v) is float for v in raw["eigen_vectors"][0]["eigen_vector"])

    @pytest.mark.xfail(strict=True, reason=(
        "lib/data/impl/vecm.py __vecm_johansen_coint_test_report_from_result fills "
        "JohansenCointTestEigenVector from result.evec[i], a ROW of the eigenvector matrix. "
        "coint_johansen returns eigenvectors as COLUMNS (evec[:, i]), so the reported vector "
        "is not the cointegrating vector: for y = 2x + stationary the model reports a ratio "
        "near -0.027 whose combination of the series is still a random walk, instead of the "
        "-0.5 column whose combination is stationary."))
    def test_dominant_eigen_vector_is_the_cointegrating_vector(self):
        samples = self.cointegrated_pair()
        _, model, _ = vecm_data.compute_johansen_coint_test(samples, 2)
        vector = model.eigen_vectors[0].eigen_vector
        # y = 2x + stationary spread, so the cointegrating vector is (1, -1/2)
        # up to scale and the combination it forms is stationary.
        assert vector[1] / vector[0] == pytest.approx(-0.5, abs=0.05)
        combination = vector[0] * samples[0] + vector[1] * samples[1]
        assert numpy.std(combination) < 0.5 * numpy.std(samples[0])

    def test_independent_random_walks_report_a_well_formed_trace_test(self):
        # Under the null the trace test only has its nominal size, so no boolean
        # outcome is pinned on a single draw here -- see the ensemble test below.
        # Only the structure of the report is a deterministic contract.
        samples = numpy.array([random_walk(2000), random_walk(2000)])
        _, model, _ = vecm_data.compute_johansen_coint_test(samples, 2)

        stat = model.trace_test[0]
        assert stat.null_hypothesis == "r<=0"
        assert model.trace_test[1].null_hypothesis == "r<=1"
        assert len(stat.test_result) == len(stat.critical_values) == 3
        assert all(isinstance(flag, bool) for flag in stat.test_result)
        assert numpy.isfinite(stat.test_stat)
        # Critical values are ordered 90% < 95% < 99%, so rejections are nested.
        assert stat.critical_values == sorted(stat.critical_values)
        assert stat.test_result == sorted(stat.test_result, reverse=True)
        assert 0 <= model.rank <= 2

    def test_the_trace_test_rejects_independence_at_roughly_its_nominal_size(self):
        # The size of the r<=0 trace test under independence is a *rate*, not a
        # per draw outcome: pinning `test_result[2] is False` on one seed flakes
        # on ~8% of seeds. Measured over 200 independent pairs the 99% rejection
        # rate sat between 0.02 and 0.05 for every outer seed tried, and the 90%
        # column -- where the det_order=0 size distortion shows -- rejected an
        # order of magnitude more often.
        trials = 200
        rejected_99 = 0
        rejected_90 = 0
        for _ in range(trials):
            samples = numpy.array([random_walk(500), random_walk(500)])
            _, model, _ = vecm_data.compute_johansen_coint_test(samples, 2)
            rejected_99 += int(model.trace_test[0].test_result[2])
            rejected_90 += int(model.trace_test[0].test_result[0])

        assert rejected_99 / trials < 0.15
        assert rejected_90 > rejected_99

    @pytest.mark.xfail(strict=True, reason=(
        "JohansenCointTestRank labels its ranks with three significance levels, but "
        "lib/data/impl/vecm.py __vecm_johansen_coint_test_report_from_result builds the list "
        "with `for i in range(n)` where n is the number of series rather than the number of "
        "critical value columns, so a bivariate system yields only 2 ranks for 3 levels "
        "(and JohansenCointTestReport.rank then takes the min over a truncated list)."))
    def test_rank_has_one_entry_per_significance_level(self):
        _, model, _ = vecm_data.compute_johansen_coint_test(self.cointegrated_pair(), 2)
        assert len(model.ranks.test_ranks) == len(model.ranks.significance_levels)


# ##############################################################
# End to end: the status if-chains driven by real test statistics
# ##############################################################

class TestStatusFromRealPipelines:
    """HypothesisTestType.status is what turns raw per-lag/per-significance
    booleans into the report status, so it is also checked against the ADF and
    variance ratio pipelines that call it."""

    @pytest.mark.parametrize("compute,test_type,desc", [
        (adf_data.compute_adf_test, HypothesisTestType.STATIONARITY,
         "Stationarity Test"),
        (adf_data.compute_adf_offset_test, HypothesisTestType.STATIONARITY_OFFSET,
         "Stationarity Test with Constant Offset."),
        (adf_data.compute_adf_drift_test, HypothesisTestType.STATIONARITY_DRIFT,
         "Stationarity Test with Drift."),
    ])
    def test_stationary_ar1_passes_the_stationarity_test(self, compute, test_type, desc):
        # phi = 0.5 over 2000 points is emphatically stationary: all 40 seeds
        # tried rejected the unit root at the 1% level, for all three variants.
        result, report = compute(stationary_ar1())

        assert report.hyp_test_type is test_type
        assert report.desc == desc
        assert report.hyp_type is HypothesisType.LOWER_TAIL
        assert report.status is HypothesisTestStatus.PASSED
        # Per significance level the ADF null is rejected at all three levels,
        # so each StatisticalTestData is FAILED while the report is PASSED.
        assert list(result.status_vals) == [False, False, False]
        assert [d.status for d in report.test_data] == [HypothesisTestStatus.FAILED] * 3
        assert [d.sig.value for d in report.test_data] == [0.01, 0.05, 0.1]
        assert len({d.test_id for d in report.test_data} | {report.test_id}) == 1

    def test_random_walk_fails_the_stationarity_test(self):
        # A driftless random walk is only rejected at the test's nominal size,
        # which puts a bare unit root at a ~10% seed failure rate against the
        # 10% slot the status if-chain reads. Adding a drift the no-constant ADF
        # variant cannot absorb pushes the statistic firmly positive: no
        # rejection at any level for any of 40 seeds.
        result, report = adf_data.compute_adf_test(random_walk(drift=0.05))
        assert list(result.status_vals) == [True, True, True]
        assert report.status is HypothesisTestStatus.FAILED
        assert [d.status for d in report.test_data] == [HypothesisTestStatus.PASSED] * 3

    def test_brownian_motion_passes_the_bm_variance_ratio_test(self):
        # sig_level 0.01 rather than the 0.1 default: BM.status only needs one
        # of the five lags inside the two tail interval, and a 1% per lag size
        # makes a sweep of all five vanishingly unlikely.
        result, report = fbm_data.compute_vr_test(random_walk(2048), HypothesisTestType.BM,
                                                  sig_level=0.01, s=S_VALS)
        assert report.hyp_type is HypothesisType.TWO_TAIL
        assert report.desc == "Brownian Motion Test"
        assert report.status is HypothesisTestStatus.PASSED
        assert len(report.test_data) == len(S_VALS)
        assert [d.params[0].value for d in report.test_data] == S_VALS
        # Two tail test, so both critical values are attached to every row.
        assert all(d.lower is not None and d.upper is not None for d in report.test_data)
        assert all(d.sig.value == 0.01 for d in report.test_data)

    def test_persistent_fbm_fails_the_bm_variance_ratio_test(self):
        # The FAILED direction of BM.status: it reports FAILED only when *every*
        # lag falls outside the two tail interval. H = 0.8 over 2048 points puts
        # the variance ratio statistic in the tens at every lag, so all five
        # reject; over 201 seeds at sig_level 0.01 the report was FAILED every
        # time.
        samples = fbm_model.generate_fft(0.8, 2048)
        result, report = fbm_data.compute_vr_test(samples, HypothesisTestType.BM,
                                                  sig_level=0.01, s=S_VALS)
        assert report.hyp_type is HypothesisType.TWO_TAIL
        assert report.desc == "Brownian Motion Test"
        assert not any(result.status_vals)
        assert report.status is HypothesisTestStatus.FAILED
        assert all(d.status is HypothesisTestStatus.FAILED for d in report.test_data)

    def test_persistent_fbm_passes_the_autocorrelation_test(self):
        # H = 0.8 over 2048 points: the variance ratio statistic clears the
        # upper tail critical value at every lag by a wide margin, so every
        # per-lag test fails and FBM_AUTO_CORR reports PASSED.
        samples = fbm_model.generate_fft(0.8, 2048)
        result, report = fbm_data.compute_vr_test(samples, HypothesisTestType.FBM_AUTO_CORR,
                                                  sig_level=0.05, s=S_VALS)
        assert report.hyp_type is HypothesisType.UPPER_TAIL
        assert report.desc == "Autocorrelation Test"
        assert not any(result.status_vals)
        assert report.status is HypothesisTestStatus.PASSED
        assert all(d.status is HypothesisTestStatus.FAILED for d in report.test_data)

    def test_antipersistent_fbm_passes_the_negative_autocorrelation_test(self):
        samples = fbm_model.generate_fft(0.2, 2048)
        result, report = fbm_data.compute_vr_test(samples, HypothesisTestType.FBM_NEG_AUTO_CORR,
                                                  sig_level=0.05, s=S_VALS)
        assert report.hyp_type is HypothesisType.LOWER_TAIL
        assert report.desc == "Negative Autocorrelation Test"
        assert not any(result.status_vals)
        assert report.status is HypothesisTestStatus.PASSED

    @pytest.mark.parametrize("test_type,hyp_type,critical_value_slot", [
        (HypothesisTestType.FBM_AUTO_CORR, HypothesisType.UPPER_TAIL, 1),
        (HypothesisTestType.FBM_NEG_AUTO_CORR, HypothesisType.LOWER_TAIL, 0),
    ])
    def test_brownian_motion_fails_both_one_tailed_correlation_tests(
            self, test_type, hyp_type, critical_value_slot):
        # The FAILED direction of the FBM if-chains: FAILED requires *every* lag
        # to stay on the null side of the one tailed critical value. Five
        # correlated lags at a 1% per lag size gives a family-wise false alarm
        # rate around 5%, which is what made the 0.01 version of this test flake;
        # at sig_level 1e-4 (critical value ~3.72) a driftless random walk
        # reported FAILED for all 201 seeds swept, for both tails.
        result, report = fbm_data.compute_vr_test(random_walk(2048), test_type,
                                                  sig_level=0.0001, s=S_VALS)
        assert report.hyp_type is hyp_type
        assert all(result.status_vals)
        assert report.status is HypothesisTestStatus.FAILED
        assert all(d.status is HypothesisTestStatus.PASSED for d in report.test_data)
        # One tailed, so exactly one critical value is populated and it is the
        # one the tail calls for.
        assert result.critical_values[1 - critical_value_slot] is None
        assert result.critical_values[critical_value_slot] is not None
        unused = "upper" if critical_value_slot == 0 else "lower"
        used = "lower" if critical_value_slot == 0 else "upper"
        assert all(getattr(d, unused) is None for d in report.test_data)
        assert all(getattr(d, used).value == pytest.approx(
            result.critical_values[critical_value_slot]) for d in report.test_data)
        assert len(report.test_data) == len(S_VALS)

    @pytest.mark.parametrize("samples_factory,test_type,sig_level,expected", [
        # The heteroscedasticity consistent statistic is a different code path
        # (lib/models/fbm.py vr_stat_hetero_scan) feeding the same if-chains.
        # Each row was swept over 201 seeds with zero failures.
        (lambda: random_walk(2048), HypothesisTestType.BM, 0.01,
         HypothesisTestStatus.PASSED),
        (lambda: fbm_model.generate_fft(0.8, 2048), HypothesisTestType.BM, 0.01,
         HypothesisTestStatus.FAILED),
        (lambda: fbm_model.generate_fft(0.8, 2048), HypothesisTestType.FBM_AUTO_CORR, 0.05,
         HypothesisTestStatus.PASSED),
        (lambda: fbm_model.generate_fft(0.2, 2048), HypothesisTestType.FBM_NEG_AUTO_CORR, 0.05,
         HypothesisTestStatus.PASSED),
    ])
    def test_hetero_variance_ratio_drives_the_same_status_chain(
            self, samples_factory, test_type, sig_level, expected):
        samples = samples_factory()
        result, report = fbm_data.compute_hetero_vr_test(samples, test_type,
                                                         sig_level=sig_level, s=S_VALS)
        assert report.hyp_test_type is test_type
        assert report.desc == test_type.desc()
        assert report.status is expected
        assert len(report.test_data) == len(S_VALS)
        assert [d.params[0].value for d in report.test_data] == S_VALS
        assert all(d.sig.value == sig_level for d in report.test_data)

    def test_the_hetero_statistic_is_not_the_homo_statistic(self):
        # Guards the parametrisation above against silently exercising the
        # homoscedastic path: the heteroscedasticity consistent denominator is
        # strictly larger for a persistent fBm, so every lag's statistic shrinks.
        # True at all five lags for all 120 seeds swept.
        samples = fbm_model.generate_fft(0.8, 2048)
        homo, _ = fbm_data.compute_vr_test(samples, HypothesisTestType.FBM_AUTO_CORR,
                                           sig_level=0.05, s=S_VALS)
        hetero, _ = fbm_data.compute_hetero_vr_test(samples, HypothesisTestType.FBM_AUTO_CORR,
                                                    sig_level=0.05, s=S_VALS)
        assert homo.s_vals == hetero.s_vals == S_VALS
        assert numpy.all(numpy.array(hetero.stats) < numpy.array(homo.stats))
        assert homo.critical_values == hetero.critical_values
