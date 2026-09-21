"""Appendix checks of section differences and equivalence confidence levels.

The primary contrast asks whether the human-minus-AI learning gap differs
between Verbal and Quantitative. Its adjusted sensitivity uses a separate
linear baseline adjustment and section-specific assessment-form effects.
The confidence-level check holds each original equivalence margin fixed.
"""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .data import load_sessions, sha256
from .journal import Journal, write_json


def section_interaction(sessions):
    """Compare human-minus-pooled-AI gain differences across the two sections."""
    frame = sessions[sessions.kind.isin(["ai", "human"])].copy()
    groups = {}
    for section in ["quant", "verbal"]:
        for kind in ["ai", "human"]:
            gain = frame.loc[
                (frame.section == section) & (frame.kind == kind), "gain_pp"
            ]
            groups[section + ":" + kind] = dict(
                n=len(gain), mean=float(gain.mean()), variance=float(gain.var(ddof=1))
            )
    gaps = {
        section: groups[section + ":human"]["mean"] - groups[section + ":ai"]["mean"]
        for section in ["quant", "verbal"]
    }
    variances = [row["variance"] / row["n"] for row in groups.values()]
    variance = sum(variances)
    se = math.sqrt(variance)
    df = variance**2 / sum(
        value**2 / (row["n"] - 1) for value, row in zip(variances, groups.values())
    )
    estimate = gaps["verbal"] - gaps["quant"]
    half = float(stats.t.ppf(0.975, df)) * se
    raw = dict(
        estimate_pp=estimate,
        se=se,
        df=df,
        ci95=[estimate - half, estimate + half],
        p_two_sided=float(2 * stats.t.sf(abs(estimate / se), df)),
        normal_reference_p=float(2 * stats.norm.sf(abs(estimate / se))),
        quant_human_minus_ai_pp=gaps["quant"],
        verbal_human_minus_ai_pp=gaps["verbal"],
        group_means=groups,
    )
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
    corrected = multipletests(
        [raw["p_two_sided"], adjusted["p_two_sided"]], method="holm"
    )[1]
    for result, pvalue in zip([raw, adjusted], corrected):
        result["holm_p_two_test_family"] = float(pvalue)
    return dict(raw_interaction=raw, adjusted_sensitivity=adjusted)


def confidence_level_sensitivity(sessions):
    """Recompute 90% and 95% Welch intervals with identical quarter-SD margins."""
    results = []
    for scope in ["quant", "verbal", "combined"]:
        block = sessions if scope == "combined" else sessions[sessions.section == scope]
        ai = block.loc[block.kind == "ai", "gain_pp"].to_numpy()
        human = block.loc[block.kind == "human", "gain_pp"].to_numpy()
        na, nh = len(ai), len(human)
        va, vh = ai.var(ddof=1), human.var(ddof=1)
        gap = ai.mean() - human.mean()
        variance = va / na + vh / nh
        se = math.sqrt(variance)
        df = variance**2 / ((va / na) ** 2 / (na - 1) + (vh / nh) ** 2 / (nh - 1))
        margin = 0.25 * math.sqrt(((na - 1) * va + (nh - 1) * vh) / (na + nh - 2))
        pvalue = max(
            stats.t.sf((gap + margin) / se, df), stats.t.cdf((gap - margin) / se, df)
        )
        for confidence in [0.90, 0.95]:
            alpha = (1 - confidence) / 2
            half = float(stats.t.ppf(1 - alpha, df)) * se
            results.append(
                dict(
                    scope=scope,
                    confidence=confidence,
                    tost_alpha=alpha,
                    n_ai=na,
                    n_human=nh,
                    difference_pp=float(gap),
                    se=se,
                    df=float(df),
                    margin_pp=margin,
                    ci_low_pp=float(gap - half),
                    ci_high_pp=float(gap + half),
                    p_tost=float(pvalue),
                    equivalent=bool(pvalue < alpha),
                )
            )
    return results


def run(data_dir: Path, output_dir: Path):
    """Read released response files and save both small appendix checks."""
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
    confidence = journal.get("confidence_level_sensitivity", signature)
    if confidence is None:
        confidence = journal.save(
            "confidence_level_sensitivity",
            signature,
            confidence_level_sensitivity(sessions),
        )
    summary = dict(
        complete=True, **interaction, confidence_level_sensitivity=confidence
    )
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
