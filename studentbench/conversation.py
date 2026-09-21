"""Six fixed conversation indicators and student-reported answer disputes."""

from __future__ import annotations
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

import re
from concurrent.futures import ThreadPoolExecutor, as_completed


_DETECTOR_VERSION = "paper-six-deterministic-v2"


_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "because",
    "before",
    "being",
    "could",
    "does",
    "from",
    "have",
    "here",
    "into",
    "just",
    "like",
    "more",
    "okay",
    "problem",
    "question",
    "really",
    "right",
    "should",
    "some",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "think",
    "this",
    "those",
    "very",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
    "youre",
}


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", _norm_text(text).lower())


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, _norm_text(text), flags=re.I) is not None


def _tutor_turns(turns: list[tuple[str, str]]) -> list[str]:
    return [
        content for role, content in turns if role == "tutor" and _norm_text(content)
    ]


def _contingent_scaffolding(turns: list[tuple[str, str]]) -> bool:
    evaluative = (
        r"\b(?:correct|exactly|good (?:job|work|attempt|thinking|instinct)|well done|"
        r"not quite|almost|close|that works|doesn'?t work|you (?:got|chose|said|noticed)|"
        r"the issue|the mistake|common trap|let'?s (?:fix|check|break|look)|yes[,!]|no[,!])\b"
    )
    scaffold = r"\b(?:try|step|look|check|because|means|notice|compare|rewrite|explain|reason|why|how)\b"
    last_student = ""
    for role, content in turns:
        if role == "student":
            last_student = _norm_text(content)
            continue
        if role != "tutor" or len(_words(last_student)) < 2:
            continue
        if _has(content, evaluative):
            return True
        student_tokens = {
            w for w in _words(last_student) if len(w) >= 4 and w not in _STOPWORDS
        }
        tutor_tokens = {
            w for w in _words(content) if len(w) >= 4 and w not in _STOPWORDS
        }
        if len(student_tokens & tutor_tokens) >= 2 and _has(content, scaffold):
            return True
    return False


def _self_explanation(turns: list[tuple[str, str]]) -> bool:
    pattern = (
        r"\bwhy do you think\b|\bin your own words\b|\bexplain (?:your|the|why|how)\b|"
        r"\bwalk me through\b|\bhow did you (?:get|arrive|figure)\b|"
        r"\bwhat'?s your reasoning\b|\btell me why\b|\bjustify\b|"
        r"\breasoning behind\b|\bwhy (?:is|does|would|do|did)\b"
    )
    return any("?" in text and _has(text, pattern) for text in _tutor_turns(turns))


def _interactive_questioning(turns: list[tuple[str, str]]) -> bool:
    pattern = (
        r"\b(?:what do you think|tell me|can you|could you|try|your turn|"
        r"what would|which (?:choice|answer)|how would you|talk me through)\b"
    )
    return (
        sum(("?" in text or _has(text, pattern)) for text in _tutor_turns(turns)) >= 4
    )


def _productive_struggle(turns: list[tuple[str, str]]) -> bool:
    pattern = (
        r"\b(?:try (?:it|this|that|the first)|give it a (?:shot|try)|take a stab|"
        r"your turn|before I|first,? you|start by|attempt (?:it|this)|"
        r"work (?:it|this) out|solve (?:it|this)|submit (?:an|your) answer)\b"
    )
    return any(_has(text, pattern) for text in _tutor_turns(turns)[:5])


def _worked_example_after_attempt(turns: list[tuple[str, str]]) -> bool:
    seen_student = False
    pattern = (
        r"\b(?:solution|step 1|first,|then,|therefore|so the answer|we get|calculate)\b"
    )
    for role, content in turns:
        if role == "student" and len(_words(content)) >= 2:
            seen_student = True
        elif (
            role == "tutor"
            and seen_student
            and len(_words(content)) >= 140
            and _has(content, pattern)
        ):
            return True
    return False


def _metacognitive_prompting(turns: list[tuple[str, str]]) -> bool:
    pattern = (
        r"\b(?:why (?:does|would|is|are|did)|how do you know|what tells you|"
        r"does that make sense|check your reasoning|sanity check|"
        r"how (?:would|could) you check|what strategy|why is that)\b"
    )
    return (
        sum(("?" in text and _has(text, pattern)) for text in _tutor_turns(turns)) >= 2
    )


_DETECTORS = {
    "contingent_scaffolding": _contingent_scaffolding,
    "self_explanation_prompts": _self_explanation,
    "interactive_questioning": _interactive_questioning,
    "productive_struggle": _productive_struggle,
    "worked_example_after_attempt": _worked_example_after_attempt,
    "metacognitive_prompting": _metacognitive_prompting,
}


def _parse_ai_turns(path: Path) -> list[tuple[str, str]]:
    turns = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            event, content = (
                str(row.get("event_type") or ""),
                _norm_text(row.get("content") or ""),
            )
            if event == "student_message" and content:
                turns.append(("student", content))
            elif event == "ai_tutor_message" and content:
                turns.append(("tutor", content))
    return turns


def _parse_human_turns(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    role, buffer = "", []

    def flush() -> None:
        nonlocal role, buffer
        if role and buffer:
            turns.append(
                (role, " ".join(value.strip() for value in buffer if value.strip()))
            )
        role, buffer = "", []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Tutor:"):
            flush()
            role, buffer = "tutor", [line.split(":", 1)[1].strip()]
        elif line.startswith("Student:"):
            flush()
            role, buffer = "student", [line.split(":", 1)[1].strip()]
        elif role and line.strip() and not set(line.strip()) <= {"="}:
            buffer.append(line.strip())
    flush()
    return turns


def _crop_human_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    finished = r"\b(?:i(?:'m| am)? (?:done|finished)|i finish|finished (?:the )?(?:pre-?)?test|completed (?:the )?test)\b"
    start_positive = (
        r"\b(?:got your plan|plan (?:is )?(?:ready|put together)|start (?:the )?(?:lesson|tutoring)|"
        r"get started (?:with )?(?:the lesson|tutoring)|pull up your test|go over (?:the|your) test|"
        r"take a look at|based on (?:your|the) (?:test|results)|share my screen|first question|talk me through)\b"
    )
    start_negative = r"\b(?:give me|need|take) (?:just )?(?:a |about )?(?:couple|few|5|five|10|ten) minutes\b|\bcreate (?:the|your|a) lesson plan\b"
    start = 0
    for complete_index in [
        i
        for i, (role, content) in enumerate(turns)
        if role == "student" and _has(content, finished)
    ]:
        for j in range(complete_index + 1, min(len(turns), complete_index + 18)):
            role, content = turns[j]
            if (
                role == "tutor"
                and _has(content, start_positive)
                and not _has(content, start_negative)
            ):
                start = j
                break
        if start:
            break
    if not start:
        for j, (role, content) in enumerate(turns[6:], start=6):
            if (
                role == "tutor"
                and _has(content, start_positive)
                and not _has(content, start_negative)
            ):
                start = j
                break
    end, tutor_seen = len(turns), 0
    end_pattern = (
        r"\b(?:start|take|begin|move (?:on|over)) (?:the |your )?post-?test\b|"
        r"\bpost-?test (?:link|section|now|next)\b"
    )
    for j in range(start, len(turns)):
        role, content = turns[j]
        tutor_seen += int(role == "tutor")
        if tutor_seen >= 8 and role == "tutor" and _has(content, end_pattern):
            end = j
            break
    cropped = turns[start:end]
    return cropped if cropped else turns


def _hash_input_paths(ssot: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ssot).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _conversation_jobs(ssot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = []
    coverage: dict[str, Any] = {
        "ai_profiles_total": 0,
        "ai_trajectories_observed": 0,
        "human_profiles_total": 0,
        "human_transcripts_observed": 0,
        "human_transcript_missing_task_ids": [],
        "human_by_section": {},
    }
    for section in ("quant", "verbal"):
        root = ssot / "1_main_leaderboard_data" / f"gre_{section}"
        section_coverage = {
            "profiles_total": 0,
            "transcripts_observed": 0,
            "missing_task_ids": [],
        }
        for profile in sorted(root.glob("ai_arm_*/*/student_data/student.json")):
            coverage["ai_profiles_total"] += 1
            task = profile.parents[1]
            trajectory = (
                task
                / "ai_tutoring_data"
                / "student_ai_tutor_interaction_trajectory.csv"
            )
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            coverage["ai_trajectories_observed"] += 1
            paths = [profile, trajectory]
            jobs.append(
                {
                    "student_id": task.name,
                    "instrument": section,
                    "arm": profile.parents[2].name.removeprefix("ai_arm_"),
                    "kind": "ai",
                    "paths": paths,
                    "source_hash": _hash_input_paths(ssot, paths),
                }
            )
        for profile in sorted((root / "human_arm").glob("*/student_data/student.json")):
            coverage["human_profiles_total"] += 1
            section_coverage["profiles_total"] += 1
            task = profile.parents[1]
            transcripts = sorted(
                task.glob("zoom_tutoring_conversation_with_human_data/*.txt")
            )
            if not transcripts:
                coverage["human_transcript_missing_task_ids"].append(task.name)
                section_coverage["missing_task_ids"].append(task.name)
                continue
            coverage["human_transcripts_observed"] += 1
            section_coverage["transcripts_observed"] += 1
            paths = [profile, *transcripts]
            jobs.append(
                {
                    "student_id": task.name,
                    "instrument": section,
                    "arm": "human",
                    "kind": "human",
                    "paths": paths,
                    "source_hash": _hash_input_paths(ssot, paths),
                }
            )
        coverage["human_by_section"][section] = section_coverage
    coverage["human_transcript_missing_task_ids"].sort()
    coverage["human_transcripts_missing"] = (
        coverage["human_profiles_total"] - coverage["human_transcripts_observed"]
    )
    coverage["human_transcript_coverage"] = (
        f"{coverage['human_transcripts_observed']}/{coverage['human_profiles_total']}"
    )
    return jobs, coverage


def _process_conversation_job(job: dict[str, Any]) -> dict[str, Any]:
    dialogue_paths = job["paths"][1:]
    if job["kind"] == "ai":
        turns = _parse_ai_turns(dialogue_paths[0])
    else:
        turns = _crop_human_turns(
            [turn for path in dialogue_paths for turn in _parse_human_turns(path)]
        )
    return {
        "detector_version": _DETECTOR_VERSION,
        "source_hash": job["source_hash"],
        "student_id": job["student_id"],
        "instrument": job["instrument"],
        "arm": job["arm"],
        "n_tutoring_turns": len(turns),
        "n_tutor_turns": sum(role == "tutor" for role, _ in turns),
        "n_student_turns": sum(role == "student" for role, _ in turns),
        **{key: int(function(turns)) for key, function in _DETECTORS.items()},
    }


def _compute_conversation_sessions(
    ssot: Path, cache_path: Path, *, force: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    jobs, coverage = _conversation_jobs(ssot)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[(row["instrument"], row["student_id"])] = row
    pending = (
        jobs
        if force
        else [
            job
            for job in jobs
            if not (
                (row := latest.get((job["instrument"], job["student_id"])))
                and row.get("source_hash") == job["source_hash"]
                and row.get("detector_version") == _DETECTOR_VERSION
            )
        ]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        with cache_path.open("a", encoding="utf-8") as sink:
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                futures = {
                    pool.submit(_process_conversation_job, job): job for job in pending
                }
                for future in as_completed(futures):
                    row = future.result()
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
                    latest[(row["instrument"], row["student_id"])] = row
    return (
        pd.DataFrame([latest[(job["instrument"], job["student_id"])] for job in jobs]),
        coverage,
    )


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return None


def _problem_flag_jobs(ssot: Path) -> list[dict[str, Any]]:
    jobs = []
    for section in ("quant", "verbal"):
        root = ssot / "1_main_leaderboard_data" / f"gre_{section}"
        for task in sorted(root.glob("ai_arm_*/task_id_*")):
            paths = [
                task / "student_data/student.json",
                task / "student_data/interaction_events.csv",
                task / "ai_tutoring_data/student_practice_problem_attempts.csv",
            ]
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError(f"incomplete practice-flag inputs: {task}")
            profile = json.loads(paths[0].read_text(encoding="utf-8"))
            jobs.append(
                {
                    "student_id": str(profile["student_id"]),
                    "section": section,
                    "model_preset": str(profile["ai_model_preset_id"]),
                    "paths": paths,
                    "source_hash": _hash_input_paths(ssot, paths),
                }
            )
    return jobs


def _process_problem_flag_job(job: dict[str, Any]) -> dict[str, Any]:
    _, events_path, attempts_path = job["paths"]
    all_sources: set[str] = set()
    answered_sources: list[str] = []
    with attempts_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            source = str(
                row.get("source_problem_id") or row.get("practice_problem_id") or ""
            ).rsplit(":", 1)[-1]
            if not source:
                continue
            all_sources.add(source)
            if _parse_bool(row.get("was_answered")) is True:
                answered_sources.append(source)
    clicks: Counter[str] = Counter()
    with events_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_type") != "student_disagreed":
                continue
            payload = json.loads(row.get("payload_json") or "{}")
            if payload.get("problem_id") is not None:
                clicks[str(payload["problem_id"]).rsplit(":", 1)[-1]] += 1
    return {
        "student_id": job["student_id"],
        "section": job["section"],
        "model_preset": job["model_preset"],
        "source_hash": job["source_hash"],
        "n_answered": int(len(answered_sources)),
        "n_flagged": int(sum(source in clicks for source in answered_sources)),
        "n_dispute_clicks": int(sum(clicks.values())),
        "n_matched_dispute_clicks": int(
            sum(count for source, count in clicks.items() if source in all_sources)
        ),
    }


def _compute_problem_flag_students(
    ssot: Path, cache_path: Path, *, force: bool = False
) -> pd.DataFrame:
    jobs = _problem_flag_jobs(ssot)
    latest: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["student_id"]] = row
    pending = (
        jobs
        if force
        else [
            job
            for job in jobs
            if not (
                (row := latest.get(job["student_id"]))
                and row.get("source_hash") == job["source_hash"]
            )
        ]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        with cache_path.open("a", encoding="utf-8") as sink:
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                futures = {
                    pool.submit(_process_problem_flag_job, job): job for job in pending
                }
                for future in as_completed(futures):
                    row = future.result()
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
                    sink.flush()
                    latest[row["student_id"]] = row
    return pd.DataFrame([latest[job["student_id"]] for job in jobs])[
        [
            "student_id",
            "section",
            "model_preset",
            "source_hash",
            "n_answered",
            "n_flagged",
            "n_dispute_clicks",
            "n_matched_dispute_clicks",
        ]
    ]


def _flag_scope_rows(
    students: pd.DataFrame,
    scope: str,
    *,
    draws: int,
    seed: int,
    normalize_sonnet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    data = students.copy()
    if normalize_sonnet:
        data["configuration"] = data.model_preset.replace(
            {
                "sonnet-4.6-low": "sonnet-section-specific-low",
                "sonnet-5-low": "sonnet-section-specific-low",
            }
        )
    else:
        data["configuration"] = data.model_preset
    rows = []
    for model, block in data.groupby("configuration"):
        num, den = block.n_flagged.to_numpy(float), block.n_answered.to_numpy(float)
        rng = np.random.default_rng(seed + sum(ord(c) for c in str(model)))
        samples = np.empty(draws)
        for b in range(draws):
            take = rng.integers(0, len(block), len(block))
            samples[b] = num[take].sum() / den[take].sum()
        rows.append(
            {
                "scope": scope,
                "model_preset": model,
                "n_students": int(len(block)),
                "n_answered": int(den.sum()),
                "n_flagged": int(num.sum()),
                "flags_per_100_answered": float(100 * num.sum() / den.sum()),
                "ci_low_per_100": float(100 * np.quantile(samples, 0.025)),
                "ci_high_per_100": float(100 * np.quantile(samples, 0.975)),
            }
        )
    data["rate"] = data.n_flagged / data.n_answered
    kw = stats.kruskal(
        *(g.rate.to_numpy(float) for _, g in data.groupby("configuration"))
    )
    return pd.DataFrame(rows), {
        "scope": scope,
        "n_students": int(len(data)),
        "n_models": int(data.configuration.nunique()),
        "kruskal_H": float(kw.statistic),
        "p_omnibus": float(kw.pvalue),
    }
