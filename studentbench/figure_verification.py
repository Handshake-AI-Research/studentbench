"""Verify the numerical inputs actually consumed by all 18 statistical figures.

Expected aggregates are checks only. Neither analysis nor figure code reads them.
Figure 6 additionally recomputes its plotted curves, uncertainty bands and bins.
Fixed illustrations are checked by file hash, separately from statistical results.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re

import pandas as pd

from .journal import Journal, plain, write_json

REPOSITORY = Path(__file__).resolve().parents[1]
IDENTIFIERS = {
    "record_id",
    "arm_id",
    "model",
    "scope",
    "section",
    "topic",
    "x",
    "y",
    "coefficient_names",
    "leader_arm_id",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity_fields(records):
    """Bind estimates to their subjects, allowing harmless input-row reordering."""
    if not records or not isinstance(records[0], dict):
        return []
    for fields in (
        ("record_id",),
        ("scope", "arm_id"),
        ("section", "topic", "arm_id"),
        ("section", "topic"),
        ("arm_id",),
        ("model",),
        ("section",),
    ):
        if all(all(k in r for k in fields) for r in records):
            identities = [tuple(r[k] for k in fields) for r in records]
            if len(identities) == len(set(identities)):
                return fields
    return []


def _compare(actual, expected, location, atol, rtol, check_text=False):
    if isinstance(expected, dict):
        count = 0
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Missing plotted input: {location}/{key}")
            count += _compare(
                actual[key], value, f"{location}/{key}", atol, rtol, key in IDENTIFIERS
            )
        return count
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Wrong number of plotted inputs: {location}")
        keys = _identity_fields(expected)
        if keys:
            index = {tuple(row[k] for k in keys): row for row in actual}
            if len(index) != len(actual):
                raise ValueError(f"Duplicate plotted input: {location}")
            actual = [index[tuple(row[k] for k in keys)] for row in expected]
        return sum(
            _compare(a, e, f"{location}/{i}", atol, rtol, check_text)
            for i, (a, e) in enumerate(zip(actual, expected))
        )
    if expected is None:
        if actual is not None:
            raise ValueError(f"Plotted missingness differs: {location}")
        return 0
    if isinstance(expected, (int, float, bool)):
        if not isinstance(actual, (int, float, bool)) or not math.isfinite(
            float(actual)
        ):
            raise ValueError(f"Nonfinite plotted value: {location}")
        is_probability = any(
            re.search(r"(?:^|_)(?:p|pvalue|pvalues)(?:_|$)", part)
            for part in location.split("/")
        )
        if is_probability and not 0 <= actual <= 1:
            raise ValueError(f"Invalid plotted probability: {location}")
        absolute = 0.0 if is_probability else atol
        if not math.isclose(actual, expected, abs_tol=absolute, rel_tol=rtol):
            raise ValueError(f"{location}: computed {actual}; paper {expected}")
        return 1
    if check_text and actual != expected:
        raise ValueError(
            f"Plotted identity differs: {location}: {actual} != {expected}"
        )
    return 0


def _engagement_curves(path):
    script = REPOSITORY / "figures/figure_06_engagement_practice.py"
    spec = importlib.util.spec_from_file_location("figure06", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = json.loads(path.read_text())
    records = [r for r in package["session_inputs"] if r.get("common_primary", True)]
    fits = []
    for selected in package["selected_figure"]:
        fit = module.fit_model(
            records, {**selected, "record_id": selected["source_test_id"]}
        )
        fit["holm_p"] = selected["p_holm276"]
        fits.append(fit)
    return fits


def verify_figures(analysis_root, output_dir):
    """Check all plotted data and write an incremental, input-pinned receipt."""
    analysis_root, output_dir = Path(analysis_root), Path(output_dir)
    reference_path = REPOSITORY / "verification/figure_expected.json"
    reference = json.loads(reference_path.read_text())
    signature = _sha256(reference_path) + _sha256(__file__)
    journal = Journal(output_dir / "figure_checks.jsonl")
    results = []
    for check in reference["checks"]:
        path = analysis_root / check["input"]
        input_hash = _sha256(path)
        script_hashes = "".join(
            _sha256(next((REPOSITORY / "figures").glob(f"figure_{n:02}_*.py")))
            for n in check["figures"]
        )
        fingerprint = signature + input_hash + script_hashes
        result = journal.get(check["input"], fingerprint)
        if result is None:
            expected = check["expected"]
            if check["kind"] == "table":
                actual = plain(pd.read_csv(path).to_dict("records"))
                keys = check["keys"]
                index = {tuple(row[k] for k in keys): row for row in actual}
                if len(index) != len(expected) or len(actual) != len(expected):
                    raise ValueError(f"Wrong table row coverage: {check['input']}")
                actual = [index[tuple(row[k] for k in keys)] for row in expected]
            elif check["kind"] == "engagement_curves":
                actual = _engagement_curves(path)
            elif check["kind"] == "jsonl":
                index = {
                    r["record_id"]: r
                    for r in map(json.loads, path.read_text().splitlines())
                }
                actual = [index[r["record_id"]] for r in expected]
            else:
                actual = json.loads(path.read_text())
            count = _compare(
                actual, expected, check["input"], check["atol"], check["rtol"]
            )
            result = dict(
                input=check["input"],
                input_sha256=input_hash,
                figures=check["figures"],
                statistics_checked=count,
                status="PASS",
            )
            journal.save(check["input"], fingerprint, result)
        results.append(result)
    manifest = json.loads((REPOSITORY / "figures/manifest.json").read_text())
    fixed = []
    for figure in manifest["figures"]:
        if figure["kind"] != "fixed_artwork":
            continue
        path = REPOSITORY / figure["static_asset"]
        if _sha256(path) != figure["static_sha256"]:
            raise ValueError(f"Fixed Figure {figure['number']} differs from the paper")
        fixed.append(figure["number"])
    summary = dict(
        complete=True,
        status="PASS",
        statistical_figures=18,
        fixed_figures=len(fixed),
        plot_input_groups=len(results),
        statistics_checked=sum(r["statistics_checked"] for r in results),
        paper_commit=reference["paper_commit"],
        paper_pdf_sha256=reference["paper_pdf_sha256"],
        expectations_sha256=_sha256(reference_path),
        scope="Scientific plotted values and fixed artwork; not PDF byte identity or typography.",
    )
    write_json(output_dir / "figure_verification.json", summary)
    return summary


def write_figure_manifest(analysis_root, figure_dir):
    """Record input and output hashes for all 24 completed paper figures."""
    analysis_root, figure_dir = Path(analysis_root), Path(figure_dir)
    rows = json.loads((REPOSITORY / "figures/manifest.json").read_text())["figures"]
    completed = []
    for row in rows:
        row = dict(row)
        row["inputs"] = [
            {"path": p, "sha256": _sha256(analysis_root / p)}
            for p in row["analysis_inputs"]
        ]
        paths = [figure_dir / p for p in row["outputs"]]
        if row["kind"] == "fixed_artwork" and _sha256(paths[0]) != row["static_sha256"]:
            raise ValueError(f"Fixed illustration differs: {paths[0].name}")
        row["artifacts"] = [{"path": str(p), "sha256": _sha256(p)} for p in paths]
        if row["script"]:
            row["script_sha256"] = _sha256(REPOSITORY / row["script"])
        completed.append(row)
    write_json(figure_dir / "manifest.json", {"complete": True, "figures": completed})
    return completed
