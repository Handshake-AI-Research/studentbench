"""Table 8: repeat-participation sensitivity using released exclusion lists.

The public list supplies study-session membership only; no private participant
mapping or cross-section identity pairs are needed or reconstructed.
"""

from pathlib import Path
import numpy as np
from scipy import stats
from .engagement import csv_rows, extract, fit_tests, write_json
from .data import load_sessions


def welch_tost(ai, human, margin):
    ai, human = np.asarray(ai, float), np.asarray(human, float)
    variance_ai, variance_human = (
        ai.var(ddof=1) / len(ai),
        human.var(ddof=1) / len(human),
    )
    se = np.sqrt(variance_ai + variance_human)
    df = (variance_ai + variance_human) ** 2 / (
        variance_ai**2 / (len(ai) - 1) + variance_human**2 / (len(human) - 1)
    )
    estimate = ai.mean() - human.mean()
    p = max(
        stats.t.sf((estimate + margin) / se, df),
        stats.t.cdf((estimate - margin) / se, df),
    )
    return dict(
        estimate=float(estimate),
        p=float(p),
        margin=float(margin),
        ci90=(estimate + np.array([-1, 1]) * stats.t.ppf(0.95, df) * se).tolist(),
        n_ai=len(ai),
        n_human=len(human),
    )


def run(data_dir, output_dir, engagement_dir=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = {
        r["student_id"]
        for r in csv_rows(
            data_dir / "6_population_demographics/repeat_participant_sessions.csv"
        )
    }
    assert len(excluded) == 172
    engagement_dir = (
        Path(engagement_dir) if engagement_dir else output_dir / "engagement"
    )
    # extract() verifies current source hashes even when reusing its own journal.
    rows = extract(data_dir, engagement_dir)
    original = fit_tests(rows, engagement_dir)
    restricted = fit_tests(
        [r for r in rows if r["student_id"] not in excluded],
        output_dir / "exclude_repeat",
    )
    assert original["sessions"] == 2135 and restricted["sessions"] == 1985
    sessions = (
        load_sessions(data_dir, output_dir / "sessions")
        .rename(columns={"gain_pp": "gain"})
        .to_dict("records")
    )
    assert len(sessions) == 2469
    equivalence = {}
    for scope in ["quant", "verbal", "combined"]:
        full = [r for r in sessions if scope == "combined" or r["section"] == scope]
        ai = np.array([r["gain"] for r in full if r["kind"] == "ai"])
        human = np.array([r["gain"] for r in full if r["kind"] == "human"])
        margin = 0.25 * np.sqrt(
            ((len(ai) - 1) * ai.var(ddof=1) + (len(human) - 1) * human.var(ddof=1))
            / (len(ai) + len(human) - 2)
        )
        equivalence[scope] = {}
        for label, selected in [
            ("original", full),
            ("exclude_repeat", [r for r in full if r["student_id"] not in excluded]),
        ]:
            equivalence[scope][label] = welch_tost(
                [r["gain"] for r in selected if r["kind"] == "ai"],
                [r["gain"] for r in selected if r["kind"] == "human"],
                margin,
            )
    result = dict(
        status="complete",
        complete=True,
        excluded_sessions=172,
        remaining_sessions=2297,
        fits_verified=552,
        equivalence=equivalence,
        original_selected=original["selected_figure"],
        exclude_repeat_selected=restricted["selected_figure"],
    )
    write_json(output_dir / "results.json", result)
    return result
