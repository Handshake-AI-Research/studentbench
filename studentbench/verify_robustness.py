"""Compare fresh engagement and dependence results with the pinned paper targets.

Expected values are used only here, after computation. They never enter model
fitting, feature extraction, candidate selection, or figure generation.
"""

from pathlib import Path
import json
import numpy as np
from .data import sha256
from .journal import Journal, write_json


def run(analysis_dir, output_dir, expected_path=None):
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    expected_path = (
        Path(expected_path)
        if expected_path
        else Path(__file__).resolve().parents[1]
        / "verification/robustness_expected.json"
    )
    expected = json.loads(expected_path.read_text())
    files = {
        "engagement": analysis_dir / "engagement/results.json",
        "repeat": analysis_dir / "sensitivity/exclude_repeat/results.json",
        "robustness": analysis_dir / "engagement_robustness/results.json",
        "reviewer": analysis_dir / "reviewer/results.json",
        "tutor": analysis_dir / "tutor_dependence/results.json",
    }
    signature = (
        sha256(expected_path)
        + sha256(Path(__file__))
        + "".join(sha256(p) for p in files.values())
    )
    actual = {k: json.loads(p.read_text()) for k, p in files.items()}
    journal = Journal(output_dir / "checks.jsonl")
    checks = []

    def compare(key, a, b):
        cached = journal.get(key, signature)
        if cached is not None:
            assert cached["passed"], key
            checks.append(cached)
            return
        x, y = np.asarray(a, float), np.asarray(b, float)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), key
        probability = key.endswith((":p_raw", ":p_holm276", ":p"))
        passed = bool(
            np.allclose(x, y, rtol=3e-7, atol=1e-300 if probability else 2e-9)
        )
        record = dict(check=key, passed=passed, actual=a, expected=b)
        journal.save(key, signature, record)
        checks.append(record)
        if not passed:
            raise AssertionError(f"{key}: {a} differs from expected {b}")

    def rows_check(label, rows, targets):
        lookup = {r["record_id"]: r for r in rows}
        assert len(lookup) == len(rows) == len(targets), label
        assert set(lookup) == {r["record_id"] for r in targets}, label
        for target in targets:
            for field, value in target.items():
                if field != "record_id":
                    compare(
                        label + ":" + target["record_id"] + ":" + field,
                        lookup[target["record_id"]][field],
                        value,
                    )

    rows_check("engagement", actual["engagement"]["tests"], expected["engagement"])
    rows_check("repeat", actual["repeat"]["tests"], expected["repeat_excluded"])
    rows_check(
        "diagnostics", actual["robustness"]["diagnostic_fits"], expected["diagnostics"]
    )
    lookup = {r["record_id"]: r for r in actual["robustness"]["repeat_participation"]}
    assert len(lookup) == len(expected["repeat_diagnostics"]) == 7
    for target in expected["repeat_diagnostics"]:
        for label in ["original", "exclude_repeat"]:
            for field, value in target[label].items():
                compare(
                    "repeat_diagnostics:"
                    + target["record_id"]
                    + ":"
                    + label
                    + ":"
                    + field,
                    lookup[target["record_id"]][label][field],
                    value,
                )
    for name, target in expected["reviewer"].items():
        got = actual["reviewer"]["results"][name]
        compare(name + ":n_decisive", got["n_decisive"], target["n_decisive"])
        for cluster, value in target["cluster_counts"].items():
            compare(name + ":" + cluster, got["cluster_counts"][cluster], value)
        for label in ["original", "sensitivity", "sensitivity_normal"]:
            compare(
                name + ":" + label + ":leader_count",
                got[label]["leader_ahead_count_holm"],
                target[label]["leader_ahead_count_holm"],
            )
            models = {r["model"]: r for r in got[label]["models"]}
            for model in target[label]["models"]:
                for field, value in model.items():
                    if field != "model":
                        compare(
                            name + ":" + label + ":" + model["model"] + ":" + field,
                            models[model["model"]][field],
                            value,
                        )
    lookup = {r["section"]: r for r in actual["tutor"]["results"]}
    for target in expected["tutor_dependence"]:
        for field, value in target.items():
            if field != "section":
                compare(
                    "tutor:" + target["section"] + ":" + field,
                    lookup[target["section"]][field],
                    value,
                )
    receipt = dict(
        status="PASS",
        complete=True,
        checks=len(checks),
        paper_commit=expected["paper_commit"],
        expected_sha256=sha256(expected_path),
        input_sha256={k: sha256(p) for k, p in files.items()},
    )
    write_json(output_dir / "summary.json", receipt)
    return receipt
