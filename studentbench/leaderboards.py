"""Descriptive non-overlap of model intervals across five paper leaderboards.

Each unordered model pair is counted once. Interval non-overlap is a descriptive
criterion, separate from the paper's Holm-corrected hypothesis tests.
"""

from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import pandas as pd
from scipy import stats
from .journal import Journal, write_json, write_csv


def run(data_dir: Path, output_dir: Path):
    """Use fresh neighboring learning/cost/teaching outputs, never target tables."""
    output_dir = Path(output_dir).resolve()
    root = output_dir.parent
    paths = [
        root / "teaching" / f"fit_{name}_combined.json"
        for name in ["planning", "practice"]
    ]
    paths += [
        root / "teaching/conversation_strategy_prevalence.csv",
        root / "costs/cost_efficiency_by_arm_scope.csv",
        root / "costs/resource_session_rows.csv",
    ]
    signature = hashlib.sha256(
        b"".join(path.read_bytes() for path in paths) + Path(__file__).read_bytes()
    ).hexdigest()
    journal = Journal(output_dir / "pairwise_intervals.jsonl")
    boards = {}
    for name, path in zip(["planning", "practice"], paths[:2]):
        fit = json.loads(path.read_text())
        boards[name] = [
            {"model": r["model"], "mean": r["ability"], "lo": r["lo"], "hi": r["hi"]}
            for r in fit["models"]
        ]
    conversation = pd.read_csv(paths[2])
    boards["conversation"] = [
        {
            "model": r.arm,
            "mean": r.mean_strategy_score,
            "lo": r.mean_strategy_score_ci_low,
            "hi": r.mean_strategy_score_ci_high,
        }
        for r in conversation.itertuples()
        if r.scope == "combined"
        and not r.is_human
        and r.arm in {m["model"] for m in boards["planning"]}
    ]
    costs = pd.read_csv(paths[3])
    boards["cost_per_gain"] = [
        {
            "model": r.arm_id,
            "mean": r.cost_per_gain_pp,
            "lo": r.cost_per_gain_pp_ci_low,
            "hi": r.cost_per_gain_pp_ci_high,
        }
        for r in costs.itertuples()
        if r.scope == "combined" and r.kind == "ai"
    ]
    sessions = pd.read_csv(paths[4])
    shared = set(r["model"] for r in boards["cost_per_gain"])
    boards["student_engagement"] = []
    for model, block in sessions[sessions.arm_id.isin(shared)].groupby("arm_id"):
        values = block.n_nonblank_student_messages
        mean = float(values.mean())
        half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
        boards["student_engagement"].append(
            {"model": model, "mean": mean, "lo": mean - half, "hi": mean + half}
        )
    pairs, counts = {}, {}
    interval_rows = []
    for name, rows in boards.items():
        pairs[name] = {}
        interval_rows.extend({"leaderboard": name, **row} for row in rows)
        for left, right in combinations(sorted(rows, key=lambda r: r["model"]), 2):
            pair = (left["model"], right["model"])
            disjoint = left["hi"] < right["lo"] or right["hi"] < left["lo"]
            result = {
                "leaderboard": name,
                "model_a": pair[0],
                "model_b": pair[1],
                "disjoint_95_ci": disjoint,
            }
            key = name + ":" + ":".join(pair)
            if journal.get(key, signature) is None:
                journal.save(key, signature, result)
            pairs[name][pair] = disjoint
        counts[name] = {
            "models": len(rows),
            "pairs": len(pairs[name]),
            "disjoint": sum(pairs[name].values()),
            "overlapping": len(pairs[name]) - sum(pairs[name].values()),
        }
    common = sorted(
        set.intersection(*(set(r["model"] for r in rows) for rows in boards.values()))
    )
    unresolved = []
    separated = without_engagement = 0
    for pair in combinations(common, 2):
        disjoint = {name: rows[pair] for name, rows in pairs.items()}
        separated += any(disjoint.values())
        without_engagement += any(
            v for name, v in disjoint.items() if name != "student_engagement"
        )
        if not any(disjoint.values()):
            unresolved.append(pair)
    n = math.comb(len(common), 2)
    write_csv(output_dir / "model_intervals.csv", pd.DataFrame(interval_rows))
    result = {
        "complete": True,
        "per_leaderboard": counts,
        "common_models": common,
        "common_comparison": {
            "models": len(common),
            "pairs": n,
            "disjoint_at_least_one": separated,
            "fraction": separated / n,
            "percent": 100 * separated / n,
            "disjoint_without_engagement": without_engagement,
            "pairs_overlapping_on_every_dimension": unresolved,
        },
    }
    write_json(output_dir / "summary.json", result)
    return result
