"""Reconstruct session costs from request measurements and fixed replay rates.

Recorded charges take precedence when every metered request has a charge.
Otherwise use the measured cache multiplier and the study's fixed token prices.
Unmetered requests remain missing; affected totals are lower bounds.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import math
from pathlib import Path

from .data import Dataset, sha256
from .journal import Journal, write_json


def reconstruct_session(records, rate, multiplier):
    """Compute a session without reading any precomputed session estimate."""
    if not records:
        raise ValueError("A cost session must have request records")
    arm = records[0]["ai_model_preset_id"]
    sid = records[0]["student_id"]
    if any(r["student_id"] != sid or r["ai_model_preset_id"] != arm for r in records):
        raise ValueError("Mixed identities in cost session")
    if any(r["token_usage_recorded"].lower() not in {"true", "false"} for r in records):
        raise ValueError("Invalid token-usage flag")

    def usage(record):
        return record["token_usage_recorded"].lower() == "true"

    token_fields = ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    for record in records:
        if usage(record) and all(record[field] == "" for field in token_fields):
            raise ValueError("Usage flag is true but every token count is missing")
    totals = {
        field: (
            sum(int(r[field] or 0) for r in records)
            if any(r[field] != "" for r in records) else None
        )
        for field in token_fields
    }
    charges = [
        float(r["recorded_cost_usd"])
        for r in records if r["recorded_cost_usd"] != ""
    ]
    recorded = sum(charges)
    metered = sum(usage(r) for r in records)
    unmetered = sum(not usage(r) for r in records)
    if any(not math.isfinite(x) or x < 0 for x in charges):
        raise ValueError("Invalid recorded charge")
    complete_charges = metered > 0 and all(
        r["recorded_cost_usd"] != "" for r in records if usage(r)
    )
    replay = None
    reasoning_additional = False
    if rate is not None:
        reasoning_flag = rate["reasoning_tokens_additional"].strip().lower()
        if reasoning_flag not in {"true", "false"}:
            raise ValueError("Invalid reasoning-token billing convention")
        reasoning_additional = reasoning_flag == "true"
    if multiplier != "" and metered and rate is None and not complete_charges:
        raise ValueError(f"Missing replay price for {arm}")
    if multiplier != "" and metered and rate is not None:
        output = totals["completion_tokens"] or 0
        if reasoning_additional:
            output += totals["reasoning_tokens"] or 0
        replay = (
            (totals["prompt_tokens"] or 0)
            * float(rate["input_usd_per_million_tokens"])
            * float(multiplier)
            + output * float(rate["output_usd_per_million_tokens"])
        ) / 1e6
    best = (recorded if charges else None) if complete_charges or replay is None else replay
    if best is None:
        method = "no_cost_measurement"
    elif unmetered or (replay is None and not complete_charges):
        method = "partial_evidence_lower_bound"
    elif complete_charges:
        method = "complete_recorded_production_sum"
    else:
        method = "replay_measured_session_estimate"
    billed_fields = token_fields if reasoning_additional else token_fields[:2]
    partial_requests = sum(
        usage(r) and any(r[field] == "" for field in billed_fields)
        for r in records
    )

    def dollars(value):
        # Released summaries use 12 significant digits, not 12 decimal places.
        return None if value is None else float(format(value, ".12g"))

    return dict(
        student_id=sid, ai_model_preset_id=arm, **totals,
        total_cost_records=len(records), records_with_recorded_cost=len(charges),
        partially_recorded_token_requests=partial_requests,
        recorded_cost_usd_sum=dollars(recorded if charges else None),
        replay_estimated_total_cost_usd=dollars(replay),
        best_available_total_cost_usd=dollars(best),
        best_available_cost_method=method,
    )


def verify_summary(actual, expected):
    for field, value in actual.items():
        if field == "partially_recorded_token_requests":
            continue  # Additional measurement-quality diagnostic, not a saved estimate.
        reference = expected[field]
        if isinstance(value, (int, float)):
            matched = reference != "" and math.isclose(value, float(reference), rel_tol=0, abs_tol=1e-10)
        elif value is None:
            matched = reference == ""
        else:
            matched = value == reference
        if not matched:
            raise ValueError(f"Cost reconstruction mismatch: {actual['student_id']} / {field}")


def reconstruct_all(data_dir, output_dir):
    """Check every main and supplementary cost session, resuming by input hash."""
    data = Dataset(data_dir)
    output_dir = Path(output_dir)
    journal = Journal(output_dir / "sessions.jsonl")
    price_path = data.path("3_cost_data/replay_price_schedule.csv")
    rates = {r["ai_model_preset_id"]: r for r in data.read_csv(price_path)}
    common = sha256(price_path) + sha256(data.path("collection_metadata.json")) + sha256(Path(__file__))
    results, sources = {}, []
    for summary_path in sorted(data.root.rglob("student_cost_summary.csv")):
        expected = {r["student_id"]: r for r in data.read_csv(summary_path)}
        request_path = summary_path.with_name("llm_cost_records.csv")
        signature = hashlib.sha256((common + sha256(summary_path) + sha256(request_path)).encode()).hexdigest()
        for path in (summary_path, request_path):
            sources.append(dict(path=str(path), ssot_relative_path=path.relative_to(data.root).as_posix(),
                                bytes=path.stat().st_size, sha256=sha256(path)))
        seen = set()
        relative = summary_path.relative_to(data.root).as_posix()
        rows = []
        with request_path.open(newline="") as stream:
            for sid, group in itertools.groupby(csv.DictReader(stream), key=lambda r: r["student_id"]):
                if sid in seen or sid not in expected:
                    raise ValueError(f"Unexpected or noncontiguous cost session in {relative}")
                seen.add(sid)
                key = relative + ":" + sid
                actual = journal.get(key, signature)
                if actual is None:
                    reference = expected[sid]
                    actual = reconstruct_session(list(group), rates.get(reference["ai_model_preset_id"]),
                                                 reference["cache_input_cost_multiplier"])
                    verify_summary(actual, reference)
                    journal.save(key, signature, actual)
                else:
                    # Exhaust group before itertools advances to the next session.
                    for _ in group: pass
                    verify_summary(actual, expected[sid])
                rows.append(actual)
        if seen != set(expected):
            raise ValueError(f"Missing request records in {relative}")
        results[relative] = rows
    for path in (price_path, data.path("collection_metadata.json")):
        sources.append(dict(path=str(path), ssot_relative_path=path.relative_to(data.root).as_posix(),
                            bytes=path.stat().st_size, sha256=sha256(path)))
    write_json(output_dir / "verification.json", dict(complete=True, tables=len(results),
               sessions=sum(map(len, results.values())), source_files=sources,
               partially_recorded_token_requests=sum(r["partially_recorded_token_requests"] for rows in results.values() for r in rows),
               sessions_with_partial_token_fields=sum(r["partially_recorded_token_requests"] > 0 for rows in results.values() for r in rows),
               token_accounting="The usage flag means at least one token count was recorded. Replay estimates price observed token counts; missing categories are not observations of zero and may leave costs unmeasured.",
               replay_calibration="Released measured multipliers are fixed inputs; historical provider calls are not rerun."))
    return results, sources
