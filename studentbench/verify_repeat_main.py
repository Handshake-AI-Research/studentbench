"""Comparison-only checks of the freshly recomputed repeat-cohort analyses."""

from pathlib import Path
import json
import math
from .data import sha256
from .journal import Journal, write_json


def run(analysis_dir, output_dir, expected_path=None):
    expected_path = (
        Path(expected_path)
        if expected_path
        else Path(__file__).resolve().parents[1]
        / "verification/expected_repeat_main.json"
    )
    actual_path = Path(analysis_dir) / "repeat_main/results.json"
    return verify_json(actual_path, output_dir, expected_path)


def verify_json(actual_path, output_dir, expected_path):
    actual_path, expected_path = Path(actual_path), Path(expected_path)
    expected = json.loads(expected_path.read_text())
    actual = json.loads(actual_path.read_text())
    signature = sha256(expected_path) + sha256(actual_path) + sha256(Path(__file__))
    journal = Journal(Path(output_dir) / "checks.jsonl")
    count = 0
    for target in expected["targets"]:
        key = "/".join(map(str, target["path"]))
        prior = journal.get(key, signature)
        if prior is not None:
            assert prior["passed"], key
            count += 1
            continue
        value = actual
        for part in target["path"]:
            value = value[part]
        wanted = target["expected"]
        if isinstance(wanted, bool) or not isinstance(wanted, (int, float)):
            passed = value == wanted
        else:
            field = str(target["path"][-1])
            probability = (
                field == "p"
                or field.startswith("p_")
                or field.endswith("_p")
                or "_p_" in field
            )
            absolute = target.get("atol", 1e-300 if probability else 2e-9)
            passed = math.isclose(
                value, wanted, rel_tol=target.get("rtol", 3e-7), abs_tol=absolute
            )
        journal.save(key, signature, dict(passed=passed, actual=value, expected=wanted))
        assert passed, f"{key}: {value} differs from {wanted}"
        count += 1
    receipt = dict(
        status="PASS",
        complete=True,
        checks=count,
        expected_sha256=sha256(expected_path),
        input_sha256=sha256(actual_path),
        paper_commit=expected["paper_commit"],
    )
    write_json(Path(output_dir) / "summary.json", receipt)
    return receipt
