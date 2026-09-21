"""Compare freshly computed aggregates with approved paper results.

The expectation archive is read only here, after analysis. It is never used to
fit a model, fill missing data, select a cohort or draw a scientific figure.
"""

from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd
from .journal import Journal, write_json


def _has_number(value):
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, dict):
        return any(_has_number(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_number(v) for v in value)
    return False


def _numeric_compare(actual, expected, location, atol, rtol, probability_rtol=5e-7):
    """Check every expected numerical leaf; strings are provenance or labels."""
    if isinstance(expected, dict):
        count = 0
        for key, value in expected.items():
            if not _has_number(value):
                continue
            if key not in actual:
                raise ValueError(f"Missing output statistic: {location}/{key}")
            count += _numeric_compare(
                actual[key], value, f"{location}/{key}", atol, rtol, probability_rtol
            )
        return count
    if isinstance(expected, list):
        if not _has_number(expected):
            return 0
        if len(actual) != len(expected):
            raise ValueError(f"List coverage differs: {location}")
        return sum(
            _numeric_compare(a, e, f"{location}/{i}", atol, rtol, probability_rtol)
            for i, (a, e) in enumerate(zip(actual, expected))
        )
    if not isinstance(expected, (int, float, bool)):
        return 0
    if not isinstance(actual, (int, float, bool)) or not math.isfinite(float(actual)):
        raise ValueError(f"Nonfinite or missing statistic: {location}")
    probability = any(
        re.search(r"(?:^|_)(?:p|pvalue|pvalues)(?:_|$)", part)
        for part in location.split("/")
    )
    # Do not let a blanket absolute tolerance conceal wrong very small p-values.
    # A zero result must never match a positive probability, regardless of the
    # tolerance used for other outputs from an optimization routine.
    if probability and not 0 <= float(actual) <= 1:
        raise ValueError(f"Invalid probability: {location}")
    use_atol = 0.0 if probability else atol
    use_rtol = max(rtol, probability_rtol) if probability else rtol
    if not math.isclose(
        float(actual), float(expected), abs_tol=use_atol, rel_tol=use_rtol
    ):
        raise ValueError(f"{location}: {actual} != approved {expected}")
    return 1


def verify_core(
    analysis_dir: Path, output_dir: Path, expected_path: Path | None = None
):
    """Return PASS only after all core aggregate expectations have been checked."""
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    if expected_path is None:
        expected_path = (
            Path(__file__).resolve().parents[1] / "verification/core_expected.jsonl"
        )
    expected_path = Path(expected_path)
    signature = hashlib.sha256(
        expected_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    journal = Journal(output_dir / "core_checks.jsonl")
    checks = []
    for index, line in enumerate(expected_path.read_text().splitlines()):
        spec = json.loads(line)
        path = analysis_dir / spec["output"]
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        key = str(index) + ":" + spec["output"]
        result = journal.get(key, signature + pin)
        if result is None:
            count = 0
            if spec["type"] == "table":
                expected = (
                    pd.DataFrame(spec["expected"]).set_index(spec["keys"]).sort_index()
                )
                actual = pd.read_csv(path).set_index(spec["keys"]).sort_index()
                if not actual.index.equals(expected.index):
                    raise ValueError(
                        f"{spec['output']}: missing, extra or different result rows"
                    )
                for column in spec["columns"]:
                    a = actual[column].to_numpy(float)
                    e = expected[column].to_numpy(float)
                    if not np.array_equal(np.isnan(a), np.isnan(e)):
                        raise ValueError(
                            f"{spec['output']}/{column}: missingness differs"
                        )
                    for i in np.flatnonzero(np.isfinite(e)):
                        count += _numeric_compare(
                            float(a[i]),
                            float(e[i]),
                            f"{spec['output']}/{i}/{column}",
                            spec["atol"],
                            spec["rtol"],
                            spec.get("probability_rtol", 5e-7),
                        )
            else:
                actual = (
                    [json.loads(row) for row in path.read_text().splitlines()]
                    if spec["type"] == "jsonl"
                    else json.loads(path.read_text())
                )
                count = _numeric_compare(
                    actual, spec["expected"], spec["output"], spec["atol"], spec["rtol"],
                    spec.get("probability_rtol", 5e-7)
                )
            result = {
                "output": spec["output"],
                "statistics_checked": count,
                "status": "PASS",
                "output_sha256": pin,
                "comparison_source": spec["source"],
            }
            journal.save(key, signature + pin, result)
        checks.append(result)
    summary = {
        "complete": True,
        "status": "PASS",
        "result_groups": len(checks),
        "statistics_checked": sum(row["statistics_checked"] for row in checks),
        "expectations_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
    }
    write_json(output_dir / "core_verification.json", summary)
    return summary
