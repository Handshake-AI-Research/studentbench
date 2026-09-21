"""Figure 6 and Table 12: reply time, student chat, practice, and learning.

Read the released trajectories and practice records directly. Blank first-attempt
history remains missing; it is never treated as an incorrect answer. Temporal
windows use the unique server treatment-start event, not the profile timestamp.
"""

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics

import pandas as pd
from . import engagement_models as models

CLOSED = {
    "MCQ",
    "QC",
    "MS",
    "MSR",
    "SE",
    "TC",
    "SIP",
    "SPR",
    "NE",
    "NEF",
    "SENTENCE EQUIVALENCE",
}
PRACTICE = models.P
PRIMARY = (
    models.L
    + models.E
    + PRACTICE
    + ["gain_pp", "pre_pct", "instrument", "arm_id", "form_order"]
)
TEMPORAL = (
    models.TL
    + [x + suffix for x in models.E for suffix in ["_first30", "_minutes30to60"]]
    + [x + "_minutes30to60" for x in PRACTICE]
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, row):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def elapsed_time(value):
    """Read shared-clock seconds, retaining exact microseconds including zero."""
    try:
        microseconds = Decimal(str(value)) * 1_000_000
    except InvalidOperation:
        return None
    if not microseconds.is_finite():
        return None
    if microseconds != microseconds.to_integral_value():
        raise ValueError("Elapsed seconds must be exact to microsecond precision")
    try:
        return timedelta(microseconds=int(microseconds))
    except OverflowError:
        return None


def positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def yes(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def source_paths(profile_path):
    task = profile_path.parent.parent
    paths = [
        profile_path,
        profile_path.with_name("interaction_events.csv"),
        task / "ai_tutoring_data/student_ai_tutor_interaction_trajectory.csv",
        task / "ai_tutoring_data/student_practice_problem_attempts.csv",
        task / "ai_tutoring_data/practice_problems_created_during_tutoring.csv",
        task / "ai_lesson_plan_data/lesson_plan_including_practice_problems.json",
    ]
    other = task / "ai_lesson_plan_data/other_lesson_plans_used_during_tutoring.json"
    if other.exists():
        paths.append(other)
    return paths


def extract_session(profile_path):
    profile = json.loads(profile_path.read_text())
    task = profile_path.parent.parent
    trajectory = csv_rows(
        task / "ai_tutoring_data/student_ai_tutor_interaction_trajectory.csv"
    )
    attempts = csv_rows(task / "ai_tutoring_data/student_practice_problem_attempts.csv")
    created = csv_rows(
        task / "ai_tutoring_data/practice_problems_created_during_tutoring.csv"
    )
    events = csv_rows(profile_path.with_name("interaction_events.csv"))
    row = dict(
        student_id=profile["student_id"],
        instrument=profile["instrument_id"].removeprefix("gre_"),
        arm_id=profile["ai_model_preset_id"],
        form_order=profile["assessment_form_order"],
        pre_pct=100
        * profile["pretest_score_points"]
        / profile["assessment_max_score_points"],
        gain_pp=100
        * (profile["posttest_score_points"] - profile["pretest_score_points"])
        / profile["assessment_max_score_points"],
    )

    # Nonblank typed chat is distinct from practice-answer submission events.
    chat = [
        e
        for e in trajectory
        if e["event_type"] == "student_message" and e["content"].strip()
    ]
    ai = [e for e in trajectory if e["event_type"] == "ai_tutor_message"]
    word_counts = [len(e["content"].split()) for e in chat]
    latencies = [
        positive(e["latency_ms"]) / 1000
        for e in ai
        if positive(e["latency_ms"]) is not None
    ]
    row.update(
        student_chat_messages=len(chat),
        student_messages_ge5_words=sum(n >= 5 for n in word_counts),
        student_messages_ge10_words=sum(n >= 10 for n in word_counts),
        student_words=sum(word_counts),
        student_nonspace_characters=sum(
            not c.isspace() for e in chat for c in e["content"]
        ),
        latency_mean_s=mean(latencies),
        latency_median_s=median(latencies),
    )

    # A reset can create more than one start event; exclude only its time windows.
    starts = set()
    client_fields = False
    for event in events:
        if event["event_type"] != "treatment_start":
            continue
        payload = json.loads(event["payload_json"] or "{}")
        value = elapsed_time(
            payload.get(
                "assigned_intervention_started_elapsed_seconds",
                payload.get("treatment_started_elapsed_seconds"),
            )
        )
        if value is not None:
            starts.add(value)
        client_fields |= bool(
            event.get("client_seq") or event.get("client_elapsed_seconds")
        )
    anchor = next(iter(starts)) if len(starts) == 1 and not client_fields else None
    chat_timed = [(e, elapsed_time(e["timestamp_elapsed_seconds"])) for e in chat]
    valid_chat = anchor is not None and all(time is not None for _, time in chat_timed)
    for label, lower, upper in [("first30", 0, 1800), ("minutes30to60", 1800, 3600)]:
        window = [
            e
            for e, time in chat_timed
            if time is not None
            and anchor is not None
            and lower <= (time - anchor).total_seconds() < upper
        ]
        counts = [len(e["content"].split()) for e in window]
        values = [len(window), sum(n >= 5 for n in counts), sum(counts)]
        for name, value in zip(models.E, values):
            row[name + "_" + label] = value if valid_chat else None
        contained = []
        for event in ai:
            finish, duration = (
                elapsed_time(event["timestamp_elapsed_seconds"]),
                positive(event["latency_ms"]),
            )
            if anchor is not None and finish is not None and duration is not None:
                finish_s = (finish - anchor).total_seconds()
                if lower <= finish_s - duration / 1000 and finish_s < upper:
                    contained.append(duration / 1000)
        row["latency_mean_" + label + "_request_complete_s"] = mean(contained)
        row["latency_median_" + label + "_request_complete_s"] = median(contained)

    # Recover problem formats across original, revised, and on-demand plans.
    formats = {}

    def add_problem(problem):
        value = (problem.get("format") or "OPEN").strip().upper() or "OPEN"
        key = problem["practice_problem_id"]
        assert key not in formats or formats[key] == value, (
            "Conflicting problem formats"
        )
        formats[key] = value

    def add_plan(plan):
        for concept in plan["concepts"]:
            for problem in concept.get("practice_problems", []):
                add_problem(problem)

    add_plan(
        json.loads(
            (
                task
                / "ai_lesson_plan_data/lesson_plan_including_practice_problems.json"
            ).read_text()
        )
    )
    other = task / "ai_lesson_plan_data/other_lesson_plans_used_during_tutoring.json"
    if other.exists():
        for plan in json.loads(other.read_text())[
            "other_lesson_plans_used_during_tutoring"
        ]:
            add_plan(plan["lesson_plan"])
    for problem in created:
        add_problem(problem)
    by_problem = defaultdict(list)
    for event in trajectory:
        if event["event_type"] == "practice_answer_submitted":
            by_problem[event["practice_problem_id"]].append(event)
    assert len({p["practice_problem_id"] for p in attempts}) == len(attempts)
    problems = []
    for problem in attempts:
        key = problem["practice_problem_id"]
        fmt = formats[key]
        assert fmt in CLOSED | {"OPEN"}, fmt
        count = int(problem["answer_submission_count"] or 0)
        history = sorted(by_problem[key], key=lambda e: int(e["answer_attempt_number"]))
        ordinals = [int(e["answer_attempt_number"]) for e in history]
        assert len(set(ordinals)) == len(ordinals)
        assert yes(problem["was_answered"]) == (count > 0)
        first = next((e for e in history if int(e["answer_attempt_number"]) == 1), None)
        final = (
            next((e for e in history if int(e["answer_attempt_number"]) == count), None)
            if count
            else None
        )

        def recorded(event):
            if event is None:
                return None
            # Detailed answers carry the recorded clock; recovery rows use the
            # event clock. Zero is a valid time, not a missing observation.
            value = elapsed_time(event["recorded_elapsed_seconds"])
            return (
                value
                if value is not None
                else elapsed_time(event["timestamp_elapsed_seconds"])
            )

        problems.append(
            dict(
                answered=count > 0,
                submissions=count,
                closed=fmt in CLOSED,
                final_credit=yes(problem["final_is_correct"]),
                first_credit=yes(first["is_correct"]) if first else None,
                first_time=recorded(first),
                final_time=recorded(final),
                times=[recorded(e) for e in history],
                complete=ordinals == list(range(1, count + 1)),
            )
        )
    answered = [p for p in problems if p["answered"]]
    closed = [p for p in answered if p["closed"]]
    row.update(
        practice_answered_distinct=len(answered),
        practice_submissions=sum(p["submissions"] for p in problems),
        practice_final_credit_all=sum(p["final_credit"] for p in problems),
        practice_final_credit_closed=sum(p["final_credit"] for p in closed),
        practice_first_credit_closed=sum(p["first_credit"] for p in closed)
        if all(p["first_credit"] is not None for p in closed)
        else None,
    )
    available = [
        all(p["first_time"] is not None for p in answered),
        all(p["complete"] and all(t is not None for t in p["times"]) for p in answered),
        all(p["final_time"] is not None for p in answered),
        all(p["final_time"] is not None for p in closed),
        all(p["first_time"] is not None for p in closed),
    ]
    selected_times = [
        [p["first_time"] for p in answered],
        [t for p in answered for t in p["times"]],
        [p["final_time"] for p in answered if p["final_credit"]],
        [p["final_time"] for p in closed if p["final_credit"]],
        [p["first_time"] for p in closed if p["first_credit"]],
    ]
    for name, valid, times in zip(PRACTICE, available, selected_times):
        for label, lower, upper in [
            ("first30", 0, 1800),
            ("minutes30to60", 1800, 3600),
        ]:
            row[name + "_" + label] = (
                sum(lower <= (t - anchor).total_seconds() < upper for t in times)
                if anchor is not None and valid
                else None
            )
    row["common_primary"] = all(row[k] is not None for k in PRIMARY)
    row["common_temporal"] = row["common_primary"] and all(
        row[k] is not None for k in TEMPORAL
    )
    row["first_closed_credit"] = row["practice_first_credit_closed"]
    return row


def extract(data_dir, output_dir):
    """Extract each session once, resuming only when its inputs and code match."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / "features.jsonl"
    cache = (
        {r["student_id"]: r for r in map(json.loads, journal.open())}
        if journal.exists()
        else {}
    )
    code_hash = digest(__file__)
    rows = []
    paths = sorted(
        (data_dir / "1_main_leaderboard_data").glob(
            "gre_*/ai_arm*/task_id_*/student_data/student.json"
        )
    )
    assert len(paths) == 2139
    for i, path in enumerate(paths, 1):
        pins = {str(p.relative_to(data_dir)): digest(p) for p in source_paths(path)}
        identity = hashlib.sha256(
            json.dumps([code_hash, pins], sort_keys=True).encode()
        ).hexdigest()
        student_id = json.loads(path.read_text())["student_id"]
        cached = cache.get(student_id)
        if cached and cached["identity"] == identity:
            row = cached["features"]
        else:
            row = extract_session(path)
            append(
                journal,
                {
                    "student_id": student_id,
                    "identity": identity,
                    "inputs": pins,
                    "features": row,
                },
            )
        rows.append(row)
        if i % 250 == 0:
            print(f"Extracted {i}/{len(paths)} AI sessions", flush=True)
    with (output_dir / "features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            handle.flush()
    # The paper's extraction materialized CSV before fitting. Preserve that
    # numeric parsing boundary: last-bit latency differences can split ties in
    # the partial-rank robustness checks even though OLS fits are unchanged.
    materialized = pd.read_csv(output_dir / "features.csv").set_index("student_id")
    for row in rows:
        for field in [k for k in row if k.startswith("latency_")]:
            value = materialized.loc[row["student_id"], field]
            row[field] = None if pd.isna(value) else float(value)
    return rows


def fit_tests(rows, output_dir):
    """Fit the finite registry; append every fit before beginning the next."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame.common_primary].copy()
    identity = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        + Path(models.__file__).read_bytes()
    ).hexdigest()
    journal = output_dir / "tests.jsonl"
    cached = (
        {
            r["record_id"]: r
            for r in map(json.loads, journal.open())
            if r["identity"] == identity
        }
        if journal.exists()
        else {}
    )
    results = []
    for record_id, spec in models.registry().items():
        family, scope, x, y, edge = spec
        test = dict(
            record_id=record_id, family=family, scope=scope, x=x, y=y, edge=edge
        )
        if record_id in cached:
            fitted = cached[record_id]["fit"]
        else:
            selected = (
                frame.loc[frame.common_temporal] if family == "temporal_raw" else frame
            )
            if scope != "combined":
                selected = selected.loc[selected.instrument == scope]
            fitted = models.scaled_fit(selected, test)
            independent = models.independent_qr_fit(selected, test)
            for field in ["estimate_per_unit", "ci95_per_unit", "p_raw"]:
                models.assert_close(
                    independent[field],
                    fitted[field],
                    record_id + ":" + field,
                    probability=field == "p_raw",
                )
            append(journal, dict(record_id=record_id, identity=identity, fit=fitted))
        results.append({**test, **fitted})
    corrections = models.adjustments([r["p_raw"] for r in results])
    for row, p, q in zip(results, corrections["holm"], corrections["bh"]):
        row.update(p_holm276=p, p_bh276=q, status="complete")
    selected = [
        r
        for r in results
        if r["record_id"]
        in {
            f"primary_raw:{scope}:{x}:{y}"
            for scope in ["quant", "verbal"]
            for x, y in [
                ("latency_mean_s", "student_chat_messages"),
                ("student_chat_messages", "practice_first_credit_closed"),
                ("practice_first_credit_closed", "gain_pp"),
            ]
        }
    ]
    summary = dict(
        status="complete",
        complete=True,
        run_identity=identity,
        sessions=len(frame),
        temporal_sessions=int(frame.common_temporal.sum()),
        tests=results,
        selected_figure=selected,
    )
    write_json(output_dir / "results.json", summary)
    return summary


def run(data_dir, output_dir):
    rows = extract(data_dir, output_dir)
    result = fit_tests(rows, output_dir)
    figure_rows = []
    for fit in result["selected_figure"]:
        row = dict(fit)
        row.update(
            source_test_id=fit["record_id"],
            estimate=fit["estimate_per_x_sd"],
            se=fit["se_per_x_sd"],
            exposure_scale=fit["x_sd"],
            estimate_per10=fit["estimate_per_unit"] * 10,
            ci95_per10=[v * 10 for v in fit["ci95_per_unit"]],
        )
        for field in ["x", "y"]:
            if row[field] == "practice_first_credit_closed":
                row[field] = "first_closed_credit"
        figure_rows.append(row)
    package = dict(
        schema_version="1.0",
        status="complete",
        session_inputs=[r for r in rows if r["common_primary"]],
        tests=result["tests"],
        selected_figure=figure_rows,
        aliases={"first_closed_credit": "practice_first_credit_closed"},
    )
    write_json(Path(output_dir) / "first_attempt_practice_learning.json", package)
    return result
