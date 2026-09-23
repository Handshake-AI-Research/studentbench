"""The adjusted human–AI section interaction in the statistical appendix."""

from pathlib import Path
import hashlib
import json

import numpy as np
import statsmodels.formula.api as smf

from .data import load_sessions, sha256
from .journal import Journal, write_json


def section_interaction(sessions):
    """Compare human-minus-pooled-AI gain differences across the two sections."""
    frame = sessions[sessions.kind.isin(["ai", "human"])].copy()
    frame["is_human"] = (frame.kind == "human").astype(int)
    frame["is_verbal"] = (frame.section == "verbal").astype(int)
    frame["section_form"] = frame.section + ":" + frame.form_order
    formula = (
        "post_pct ~ is_human + is_human:is_verbal + pre_pct + "
        "pre_pct:is_verbal + C(section_form)"
    )
    fit = smf.ols(formula, frame).fit(cov_type="HC3")
    rank = int(np.linalg.matrix_rank(fit.model.exog))
    if rank != fit.model.exog.shape[1]:
        raise ValueError("Section-interaction model is not full rank")
    term = "is_human:is_verbal"
    adjusted = dict(
        formula=formula,
        n=int(fit.nobs),
        rank=rank,
        columns=fit.model.exog.shape[1],
        estimate_pp=float(fit.params[term]),
        se=float(fit.bse[term]),
        ci95=fit.conf_int().loc[term].tolist(),
        p_two_sided=float(fit.pvalues[term]),
        quant_human_minus_ai_pp=float(fit.params["is_human"]),
        verbal_human_minus_ai_pp=float(fit.params["is_human"] + fit.params[term]),
    )
    return dict(adjusted_sensitivity=adjusted)


def run(data_dir: Path, output_dir: Path):
    """Estimate the section interaction from released assessment responses."""
    output_dir = Path(output_dir)
    sessions = load_sessions(data_dir, output_dir / "inputs")
    signature = hashlib.sha256(
        (sha256(Path(__file__)) + "".join(sessions.input_signature)).encode()
    ).hexdigest()
    journal = Journal(output_dir / "results.jsonl")
    interaction = journal.get("section_interaction", signature)
    if interaction is None:
        interaction = journal.save(
            "section_interaction", signature, section_interaction(sessions)
        )
    summary = dict(complete=True, **interaction)
    write_json(output_dir / "summary.json", summary)
    return summary


def verify(analysis_dir: Path, output_dir: Path):
    """Compare new estimates with pinned published analysis; never feed inputs."""
    from .verification import Comparison

    reference = (
        Path(__file__).resolve().parents[1]
        / "verification/condition_interaction_expected.json"
    )
    expected = json.loads(reference.read_text())
    actual = json.loads(
        (Path(analysis_dir) / "condition_interaction/summary.json").read_text()
    )
    comparison = Comparison(absolute_tolerance=1e-10, relative_tolerance=1e-8)
    comparison.compare(expected, actual, "section_interaction_and_confidence")
    result = comparison.result()
    write_json(Path(output_dir) / "condition_interaction.json", result)
    if not result["passed"]:
        raise ValueError("Section-interaction or confidence-level verification failed")
    return result
