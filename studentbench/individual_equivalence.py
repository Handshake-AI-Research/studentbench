"""Individual AI–human comparisons using the primary combined CR2 model.

Public session observations refit the coefficients and the full-cohort margin.
Global moment sums retain the private cross-section dependence structure without
publishing participant links or cluster membership (assets/analysis/README.md).
"""

from pathlib import Path
import hashlib
import json

from scipy import stats

from .data import Dataset, load_sessions, sha256
from .journal import Journal, write_json
from . import primary_equivalence as primary

INPUT = Path(__file__).resolve().parents[1] / "assets/analysis/individual_equivalence_cr2.json"


def fit_model(frame, source, weights, margin):
    if source["session_fingerprint"] != primary.session_fingerprint(frame):
        raise ValueError("Public observations differ from individual CR2 moment inputs")
    x, contrast = primary.design(frame, "combined", weights)
    result = primary.fit(x, frame.post_pct.to_numpy(float), contrast, source["moments"])
    lower = float(stats.t.sf((result["estimate_pp"] + margin) / result["se_pp"], result["df"]))
    upper = float(stats.t.cdf((result["estimate_pp"] - margin) / result["se_pp"], result["df"]))
    result.update(margin_pp=margin, p_lower=lower, p_upper=upper,
                  p_tost=max(lower, upper), equivalent_at_05=max(lower, upper) < .05,
                  n_ai=int((frame.kind == "ai").sum()), n_human=int((frame.kind == "human").sum()))
    return result


def run(data_dir, output_dir, sessions=None):
    output_dir = Path(output_dir)
    if sessions is None:
        sessions = load_sessions(Path(data_dir), output_dir / "sessions")
    sessions = sessions[sessions.kind.isin(["ai", "human"])].copy()
    excluded = {r["student_id"] for r in Dataset(data_dir).read_csv(
        "6_population_demographics/repeat_participant_sessions.csv")}
    weights = {section: float((sessions.section == section).mean()) for section in ["quant", "verbal"]}
    margin = primary.fixed_margins(sessions)["combined"]
    inputs = json.loads(INPUT.read_text())
    shared = sorted(set(sessions.loc[(sessions.kind == "ai") & (sessions.section == "quant"), "arm_id"])
                    & set(sessions.loc[(sessions.kind == "ai") & (sessions.section == "verbal"), "arm_id"]))
    signature = sha256(Path(__file__)) + sha256(Path(primary.__file__)) + sha256(INPUT)
    signature += primary.session_fingerprint(sessions)
    signature += hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
    journal = Journal(output_dir / "models.jsonl")
    results = {}
    for cohort in ["original", "exclude_repeat"]:
        retained = sessions if cohort == "original" else sessions[~sessions.student_id.isin(excluded)]
        for arm in shared:
            key = cohort + ":" + arm
            value = journal.get(key, signature)
            if value is None:
                frame = retained[(retained.kind == "human") | (retained.arm_id == arm)]
                value = fit_model(frame, inputs["models"][key], weights, margin)
                value.update(arm_id=arm, cohort=cohort)
                journal.save(key, signature, value)
            results[key] = value
    result = dict(complete=True, models=results, section_weights=weights, fixed_margin_pp=margin)
    write_json(output_dir / "results.json", result)
    return result
