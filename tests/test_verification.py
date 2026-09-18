"""A complete verification report cannot silently accept missing results."""

from studentbench.verification import Comparison


def test_missing_result_fails():
    check = Comparison()
    check.compare({"estimate": 1.25, "p": 0.02}, {"estimate": 1.25})
    assert not check.result()["passed"]
    assert check.failures[0]["location"] == "result.p"


def test_scalar_tolerance_does_not_hide_a_changed_conclusion():
    check = Comparison()
    check.compare(
        {"estimate": 1.25, "equivalent": True},
        {"estimate": 1.25 + 1e-12, "equivalent": False},
    )
    assert not check.result()["passed"]
    assert len(check.failures) == 1


def test_tiny_probabilities_require_relative_accuracy():
    check = Comparison()
    check.compare({"p_tost": 1e-12}, {"p_tost": 0.0})
    assert not check.result()["passed"]
