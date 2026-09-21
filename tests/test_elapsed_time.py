"""A shared relative clock preserves timing windows, including zero offsets."""

import csv
import json
from datetime import timedelta
from decimal import Decimal

import pytest

from studentbench.engagement import elapsed_time, extract_session


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_elapsed_parser_preserves_signed_microseconds():
    for value in (-86_400_000_001, -1, 0, 1, 86_400_000_001):
        seconds = Decimal(value) / 1_000_000
        assert elapsed_time(str(seconds)) == timedelta(microseconds=value)
        assert elapsed_time(float(seconds)) == timedelta(microseconds=value)
    assert elapsed_time(0) == timedelta(0)
    assert elapsed_time("1.5") == timedelta(seconds=1.5)
    for value in (None, "", "not-a-time", "NaN", "Infinity", False):
        assert elapsed_time(value) is None
    with pytest.raises(ValueError, match="microsecond precision"):
        elapsed_time("0.0000001")


def test_zero_times_remain_observed_and_windows_are_half_open(tmp_path):
    task = tmp_path / "task"
    student = task / "student_data"
    student.mkdir(parents=True)
    profile = student / "student.json"
    profile.write_text(
        json.dumps(
            {
                "student_id": "test-session",
                "instrument_id": "gre_quant",
                "ai_model_preset_id": "test-model",
                "assessment_form_order": "AB",
                "pretest_score_points": 5,
                "posttest_score_points": 10,
                "assessment_max_score_points": 27,
            }
        )
    )
    write_rows(
        student / "interaction_events.csv",
        ["event_type", "payload_json"],
        [
            {
                "event_type": "treatment_start",
                "payload_json": json.dumps(
                    {"assigned_intervention_started_elapsed_seconds": 0}
                ),
            }
        ],
    )
    tutoring = task / "ai_tutoring_data"
    write_rows(
        tutoring / "student_ai_tutor_interaction_trajectory.csv",
        [
            "event_type",
            "content",
            "timestamp_elapsed_seconds",
            "latency_ms",
            "practice_problem_id",
            "answer_attempt_number",
            "recorded_elapsed_seconds",
            "is_correct",
        ],
        [
            {
                "event_type": "student_message",
                "content": "First message",
                "timestamp_elapsed_seconds": 0,
            },
            {
                "event_type": "student_message",
                "content": "Later message",
                "timestamp_elapsed_seconds": 1800,
            },
            {
                "event_type": "student_message",
                "content": "Outside window",
                "timestamp_elapsed_seconds": 3600,
            },
            {
                "event_type": "ai_tutor_message",
                "content": "First reply",
                "timestamp_elapsed_seconds": 1,
                "latency_ms": 1000,
            },
            {
                "event_type": "ai_tutor_message",
                "content": "Boundary crossing reply",
                "timestamp_elapsed_seconds": 1800,
                "latency_ms": 10_000,
            },
            {
                "event_type": "ai_tutor_message",
                "content": "Later reply",
                "timestamp_elapsed_seconds": 1801,
                "latency_ms": 1000,
            },
            {
                "event_type": "practice_answer_submitted",
                "timestamp_elapsed_seconds": 1800,
                "recorded_elapsed_seconds": 0,
                "practice_problem_id": "p1",
                "answer_attempt_number": 1,
                "is_correct": "True",
            },
        ],
    )
    write_rows(
        tutoring / "student_practice_problem_attempts.csv",
        [
            "practice_problem_id",
            "answer_submission_count",
            "was_answered",
            "final_is_correct",
        ],
        [
            {
                "practice_problem_id": "p1",
                "answer_submission_count": 1,
                "was_answered": "True",
                "final_is_correct": "True",
            }
        ],
    )
    write_rows(
        tutoring / "practice_problems_created_during_tutoring.csv",
        ["practice_problem_id", "format"],
        [],
    )
    plans = task / "ai_lesson_plan_data"
    plans.mkdir()
    (plans / "lesson_plan_including_practice_problems.json").write_text(
        json.dumps(
            {
                "concepts": [
                    {
                        "practice_problems": [
                            {"practice_problem_id": "p1", "format": "MCQ"}
                        ]
                    }
                ],
            }
        )
    )
    row = extract_session(profile)
    assert row["common_temporal"]
    assert row["student_chat_messages_first30"] == 1
    assert row["student_chat_messages_minutes30to60"] == 1
    assert row["latency_mean_first30_request_complete_s"] == 1
    assert row["latency_mean_minutes30to60_request_complete_s"] == 1
    assert row["practice_first_credit_closed_first30"] == 1
    assert row["practice_first_credit_closed_minutes30to60"] == 0
