"""Read the public StudentBench tables and validate assessment-derived outcomes.

No database, account, private participant mapping or original research folder is
needed. Table-wide constants live in collection_metadata.json rather than being
repeated in every CSV row. Joins below expose those constants explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd

from .journal import Journal, write_csv


class Dataset:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.parameters = self.read_json("studentbench_overall_parameters.json")
        self.metadata = self.read_json("collection_metadata.json")

    def path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("A dataset input must lie inside the dataset directory.")
        return path

    def read_json(self, relative: str | Path):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def read_csv(self, relative: str | Path) -> list[dict[str, str]]:
        path = self.path(relative)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        key = path.relative_to(self.root).as_posix()
        table = self.metadata.get("tables", {}).get(key, {})
        constants = table.get("constants", table)
        grouped = self.metadata.get("grouped_tables", {}).get(key)
        for row in rows:
            values = dict(constants)
            if grouped:
                matches = [
                    item
                    for item in grouped["rows"]
                    if all(str(item[k]) == row[k] for k in grouped["key_fields"])
                ]
                if len(matches) != 1:
                    raise ValueError(f"{key}: ambiguous grouped metadata")
                values.update(matches[0])
            for field, value in values.items():
                value = str(value)
                if field in row and row[field] != value:
                    raise ValueError(f"{key}: conflicting metadata for {field}")
                row.setdefault(field, value)
        if path.name == "human_grading_sessions.csv":
            plans = self.read_json(path.with_name("reviewed_lesson_plans.json"))[
                "plans"
            ]
            plans = {p["lesson_plan_id"]: p for p in plans}
            for row in rows:
                for side in ("a", "b"):
                    plan = plans[row[f"lesson_plan_{side}_id"]]
                    row[f"lesson_plan_{side}_ai_model_preset_id"] = plan[
                        "ai_model_preset_id"
                    ]
                    row[f"lesson_plan_{side}_kind"] = plan["lesson_plan_kind"]
        return rows

    def frame(self, relative: str | Path, **kwargs) -> pd.DataFrame:
        """Use pandas' normal numeric/missing-value parsing after metadata joins."""
        rows = self.read_csv(relative)
        if not rows:
            return pd.read_csv(self.path(relative), **kwargs)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        buffer.seek(0)
        return pd.read_csv(buffer, **kwargs)

    def profiles(self):
        for path in sorted(
            (self.root / "1_main_leaderboard_data").glob(
                "gre_*/*/task_id_*/student_data/student.json"
            )
        ):
            yield path, self.read_json(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized boolean: {value!r}")


def assessment(path: Path, expected_n: int = 27) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} questions")
    positions = [
        int(r["presented_position"]) for r in rows if r["presented_position"].strip()
    ]
    if len(positions) != len(set(positions)) or any(
        p < 1 or p > expected_n for p in positions
    ):
        raise ValueError(f"{path}: invalid presented positions")
    attempted = correct = 0
    for row in rows:
        skipped = boolean(row["skipped"])
        answered = bool(row["student_answer"].strip())
        credited = int(row["is_correct"])
        if credited not in (0, 1) or skipped == answered or (skipped and credited):
            raise ValueError(f"{path}: inconsistent answer/credit fields")
        attempted += answered and not skipped
        correct += credited
    return {"rows": len(rows), "attempted": attempted, "correct": correct}


def load_sessions(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """One row per released main-study session, checked against all test answers."""
    data = Dataset(data_dir)
    eligibility = data.parameters["analysis_cohort_eligibility"]
    expected = int(eligibility["assessment_question_count"])
    minimum = int(eligibility["minimum_attempted_questions_each_test_inclusive"])
    journal = Journal(output_dir / "sessions.jsonl")
    records = []
    for path, profile in data.profiles():
        pre_path = path.with_name("pretest_responses.csv")
        post_path = path.with_name("posttest_responses.csv")
        signature = (
            sha256(path) + sha256(pre_path) + sha256(post_path) + sha256(Path(__file__))
        )
        sid = str(profile["student_id"])
        row = journal.get(sid, signature)
        if row is None:
            section = path.parents[3].name.removeprefix("gre_")
            folder = path.parents[2].name
            kind = "ai" if folder.startswith("ai_arm_") else folder.removesuffix("_arm")
            arm = folder.removeprefix("ai_arm_") if kind == "ai" else kind
            if (
                profile["instrument_id"] != f"gre_{section}"
                or profile["tutoring_arm"] != kind
            ):
                raise ValueError(f"{sid}: profile and folder identities differ")
            if profile["assessment_max_score_points"] != expected:
                raise ValueError(f"{sid}: assessment maximum differs")
            pre, post = assessment(pre_path, expected), assessment(post_path, expected)
            if min(pre["attempted"], post["attempted"]) < minimum:
                raise ValueError(
                    f"{sid}: released session fails attempted-question rule"
                )
            if (pre["correct"], post["correct"]) != (
                profile["pretest_score_points"],
                profile["posttest_score_points"],
            ):
                raise ValueError(f"{sid}: assessment and profile scores differ")
            gain = post["correct"] - pre["correct"]
            if gain != profile["score_lift_points"]:
                raise ValueError(f"{sid}: score lift differs")
            if not math.isclose(
                profile["posttest_irt_theta"] - profile["pretest_irt_theta"],
                profile["irt_theta_lift"],
                rel_tol=0,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{sid}: theta lift differs")
            row = dict(
                student_id=sid,
                section=section,
                instrument=section,
                kind=kind,
                arm_id=arm,
                pre_score_points=pre["correct"],
                post_score_points=post["correct"],
                gain_points=gain,
                pre_pct=100 * pre["correct"] / expected,
                post_pct=100 * post["correct"] / expected,
                gain_pp=100 * gain / expected,
                form_order=profile["assessment_form_order"],
                relative_profile_path=path.relative_to(data.root).as_posix(),
                human_tutor_id=profile.get("human_tutor_id"),
                pretest_rows=pre["rows"],
                posttest_rows=post["rows"],
                pretest_attempted=pre["attempted"],
                posttest_attempted=post["attempted"],
                profile_sha256=sha256(path),
                pretest_sha256=sha256(pre_path),
                posttest_sha256=sha256(post_path),
                input_signature=signature,
                **{
                    k: profile[k]
                    for k in (
                        "pretest_irt_theta",
                        "posttest_irt_theta",
                        "irt_theta_lift",
                    )
                },
            )
            journal.save(sid, signature, row)
        records.append(row)
    frame = (
        pd.DataFrame(records)
        .sort_values(["section", "kind", "arm_id", "student_id"])
        .reset_index(drop=True)
    )
    expected_counts = {
        ("quant", "ai"): 1139,
        ("quant", "human"): 61,
        ("quant", "control"): 91,
        ("verbal", "ai"): 1000,
        ("verbal", "human"): 79,
        ("verbal", "control"): 99,
    }
    if (
        not frame.student_id.is_unique
        or frame.groupby(["section", "kind"]).size().to_dict() != expected_counts
    ):
        raise ValueError("Released main-study cohort differs from the paper cohort")
    write_csv(output_dir / "participant_rows.csv", frame)
    return frame
