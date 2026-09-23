"""Prose verification must detect changed results, precision and inequalities."""

import json
import pytest

from studentbench.paper_results import compare, verify_metrics


def test_printed_precision_is_not_an_arbitrary_numeric_tolerance():
    rule = dict(operation="round", expected=.015, digits=3)
    assert compare(.0151977032, rule)
    assert not compare(.0156, rule)


def test_strict_p_value_inequality_excludes_its_boundary():
    rule = dict(operation="less_than", expected=.001)
    assert compare(.0009999, rule)
    assert not compare(.001, rule)
    assert not compare(float("nan"), rule)


def test_interval_shape_and_both_printed_endpoints_are_checked():
    rule = dict(operation="round", expected=[-2.18, 1.03], digits=2)
    assert compare([-2.180125, 1.026328], rule)
    assert not compare([-2.180125, 1.036328], rule)
    assert not compare([[-2.180125, 1.026328]], rule)


def test_changed_metric_invalidates_a_previous_pass(tmp_path):
    rule = dict(id="age", metric="age", locations=["appendix_study.tex:4"],
                operation="round", expected=83.3, digits=1)
    assert verify_metrics({"age": 83.340327}, [rule], tmp_path, "same-run")["passed"]
    with pytest.raises(ValueError, match="age"):
        verify_metrics({"age": 82.340327}, [rule], tmp_path, "same-run")
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert not saved["passed"]
    assert saved["failures"][0]["actual"] == 82.340327


def test_missing_claim_fails_after_recording_all_checks(tmp_path):
    rules = [dict(id=name, metric=name, locations=["main.tex"], operation="equal", expected=1)
             for name in ["present", "missing"]]
    with pytest.raises(ValueError, match="missing"):
        verify_metrics({"present": 1}, rules, tmp_path, "run")
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["checks"] == 2
    assert len((tmp_path / "checks.jsonl").read_text().splitlines()) == 2


def test_rankings_require_exact_membership_and_order():
    rule = dict(operation="equal", expected=["first", "second"])
    assert compare(["first", "second"], rule)
    assert not compare(["second", "first"], rule)
    assert not compare(["first"], rule)
