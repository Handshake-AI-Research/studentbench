"""Incomplete or changed individual fits must fail release verification."""

import copy
import json

import pytest

from studentbench import core_verification as verification


def test_individual_verification_rejects_missing_models_and_changed_pvalues(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "verification").mkdir(parents=True)
    monkeypatch.setattr(verification, "__file__", str(repo / "studentbench/core_verification.py"))
    expected = {"paper_commit": "test", "models": {
        "original:model_a": {"p_tost": .04, "estimate_pp": 1.},
        "exclude_repeat:model_a": {"p_tost": .03, "estimate_pp": .8},
    }}
    (repo / "verification/individual_equivalence_expected.json").write_text(json.dumps(expected))
    analysis = tmp_path / "analysis"
    paths = [analysis / stage / "individual_cr2/results.json" for stage in ("costs", "repeat_main")]
    good = {"complete": True, "models": expected["models"]}
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(good))
    assert verification.verify_individual_equivalence(analysis, tmp_path / "checks")["models_checked"] == 2

    bad = copy.deepcopy(good)
    del bad["models"]["exclude_repeat:model_a"]
    paths[1].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="coverage differs"):
        verification.verify_individual_equivalence(analysis, tmp_path / "checks")

    bad = copy.deepcopy(good)
    bad["models"]["original:model_a"]["p_tost"] = .06
    paths[1].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="p_tost"):
        verification.verify_individual_equivalence(analysis, tmp_path / "checks")
