"""Table 2: count the recorded primary exclusion reason in each screening stage."""

from collections import Counter
from pathlib import Path
import json
import pandas as pd
from .data import sha256
from .journal import Journal, write_csv, write_json

LABELS = {
    "pretest_score_too_low": "Pre-test score below eligibility range",
    "pretest_score_too_high": "Pre-test score above eligibility range",
    "low_effort": "Insufficient assessment effort",
    "incomplete": "Incomplete study",
    "no_score_row": "Assessment score unavailable",
    "repeat_attempt": "Repeated attempt",
    "substantive_additional_treatment": "Substantive additional treatment",
    "unexpected_ai_tutoring_exposure": "Unexpected AI tutoring",
    "plan_fallback_contamination": "Lesson plan from another AI tutor",
    "insufficient_ai_tutoring_engagement": "Insufficient AI tutoring participation",
    "incomplete_assessment": "Incomplete assessment",
    "rapid_assessment_submission": "Rapid assessment submissions",
}


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    journal = Journal(output_dir / "profiles.jsonl")
    records = []
    groups = [
        (
            "excluded_leaderboard_data_due_to_failures",
            "Eligibility and study-protocol checks",
            1222,
        ),
        (
            "excluded_from_final_leaderboard_by_protocol_adherence",
            "Final adherence checks",
            219,
        ),
    ]
    for folder, stage, expected in groups:
        paths = sorted(
            (data_dir / "4_supplementary_and_excluded_research_data" / folder).rglob(
                "student.json"
            )
        )
        assert len(paths) == expected
        for path in paths:
            key = str(path.relative_to(data_dir))
            signature = sha256(path) + sha256(Path(__file__))
            row = journal.get(key, signature)
            if row is None:
                profile = json.loads(path.read_text())
                row = dict(
                    stage=stage,
                    section=profile["instrument_id"].removeprefix("gre_"),
                    reason=profile["primary_main_leaderboard_exclusion_reason"],
                )
                assert row["reason"] in LABELS
                journal.save(key, signature, row)
            records.append(row)
    rows = []
    for _, stage, _ in groups:
        counts = Counter(
            (r["reason"], r["section"]) for r in records if r["stage"] == stage
        )
        for reason, label in LABELS.items():
            q, v = counts[reason, "quant"], counts[reason, "verbal"]
            if q + v:
                rows.append(
                    dict(stage=stage, reason=label, quant=q, verbal=v, total=q + v)
                )
    write_csv(output_dir / "exclusion_counts.csv", pd.DataFrame(rows))
    write_json(
        output_dir / "summary.json",
        dict(complete=True, profiles=len(records), exclusions=[1222, 219]),
    )
    return rows
