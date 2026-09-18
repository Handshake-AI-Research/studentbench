"""High-baseline learning sensitivity using the released IRT measurements.

This module recomputes 24 specifications in each section and Combined. It
uses the recorded pre/post theta values; it does not refit item calibration.
"""

from __future__ import annotations
import math
import hashlib
from pathlib import Path
from itertools import combinations
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from .proficiency import MODEL_LABELS, _eta_squared, _family, _hc3, _holm, _wald
from .journal import Journal


def high_baseline_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recompute the 72 declared high-baseline sensitivity specifications.

    The earlier positive subgroup result used pre-treatment IRT ability to define
    the upper quartile and raw pre-to-post gain as the outcome.  It did not use
    the integer-score quartiles or model-minus-control estimand shown elsewhere
    in the main learning figures.  This function reproduces the full declared 24-specification
    family per instrument on the final 13-arm AI cohort.  It also applies the
    same family to a scale-safe Combined scope: upper-tail thresholds are
    resolved separately inside Quant and Verbal, only the 12 exact shared AI
    identities are retained, raw score gain is compared with a section fixed
    effect, and IRT lift is standardized within instrument before pooling.
    """
    journal = Journal(Path(output_dir) / "fits.jsonl")
    signature = hashlib.sha256(
        frame.to_csv(index=False).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("proficiency.py").read_bytes()
    ).hexdigest()
    quantiles = (0.50, 2.0 / 3.0, 0.75, 0.80)
    stratifiers = ("raw_pre_score", "within_form_raw_percentile", "pretest_irt_theta")
    outcomes = ("gain_pp", "irt_theta_lift")
    variants: list[dict[str, Any]] = []
    selected_cells: dict[tuple[str, str, float, str], pd.DataFrame] = {}

    for section in ("quant", "verbal"):
        base = frame[(frame.section == section) & (frame.kind == "ai")].copy()
        arms = sorted(base.arm_id.unique())
        if len(arms) != 13:
            raise ValueError(
                f"{section}: expected 13 AI configurations, found {len(arms)}"
            )
        base["within_form_raw_percentile"] = base.groupby("form_order")[
            "pre_score_points"
        ].rank(method="max", pct=True)
        for stratifier in stratifiers:
            for quantile in quantiles:
                if stratifier == "raw_pre_score":
                    threshold = float(np.quantile(base.pre_score_points, quantile))
                    selected = base[base.pre_score_points >= threshold].copy()
                    threshold_label = f"pre-test score ≥ {threshold:g}/27"
                    baseline_field = "pre_pct"
                elif stratifier == "pretest_irt_theta":
                    threshold = float(np.quantile(base.pretest_irt_theta, quantile))
                    selected = base[base.pretest_irt_theta >= threshold].copy()
                    threshold_label = f"pre-test IRT theta ≥ {threshold:.3f}"
                    baseline_field = "pretest_irt_theta"
                else:
                    threshold = float(quantile)
                    selected = base[base.within_form_raw_percentile >= quantile].copy()
                    threshold_label = f"within-form score percentile ≥ {quantile:.3f}"
                    baseline_field = "pre_pct"
                counts = selected.groupby("arm_id").size()
                if set(counts.index) != set(arms):
                    raise ValueError(
                        f"{section}/{stratifier}/{quantile}: an AI arm was lost"
                    )
                for outcome in outcomes:
                    key = f"{section}:{stratifier}:{quantile}:{outcome}"
                    cached = journal.get(key, signature)
                    if cached is not None:
                        variants.append(cached)
                        selected_cells[
                            (section, stratifier, float(quantile), outcome)
                        ] = selected
                        continue
                    groups = [
                        selected.loc[selected.arm_id == arm, outcome].to_numpy(float)
                        for arm in arms
                    ]
                    classical = stats.f_oneway(*groups)
                    reference = arms[0]
                    baseline = selected[baseline_field].to_numpy(float)
                    baseline = (baseline - baseline.mean()) / (
                        baseline.std(ddof=0) or 1.0
                    )
                    names = (
                        ["Intercept"]
                        + [f"arm:{arm}" for arm in arms[1:]]
                        + ["baseline", "baseline_squared"]
                    )
                    matrix = np.column_stack(
                        [
                            np.ones(len(selected)),
                            *[
                                (selected.arm_id == arm).to_numpy(float)
                                for arm in arms[1:]
                            ],
                            baseline,
                            baseline**2,
                        ]
                    )
                    fit = _hc3(matrix, selected[outcome].to_numpy(float))
                    contrast = np.zeros((len(arms) - 1, len(names)), dtype=float)
                    for index, arm in enumerate(arms[1:]):
                        contrast[index, names.index(f"arm:{arm}")] = 1.0
                    robust = _wald(fit, contrast)
                    row = {
                        "section": section,
                        "stratifier": stratifier,
                        "upper_tail_quantile": float(quantile),
                        "outcome": outcome,
                        "threshold": threshold,
                        "threshold_label": threshold_label,
                        "n": int(len(selected)),
                        "min_arm_n": int(counts.min()),
                        "max_arm_n": int(counts.max()),
                        "n_ai_arms": len(arms),
                        "anova_f": float(classical.statistic),
                        "anova_df_between": len(arms) - 1,
                        "anova_df_within": int(len(selected) - len(arms)),
                        "anova_p_raw": float(classical.pvalue),
                        "eta_squared": _eta_squared(groups),
                        "hc3_chi2": float(robust["wald_chi2"]),
                        "hc3_df": int(robust["df"]),
                        "hc3_p_raw": float(robust["p_raw"]),
                        "hc3_reference_arm": reference,
                    }
                    journal.save(key, signature, row)
                    variants.append(row)
                    selected_cells[(section, stratifier, float(quantile), outcome)] = (
                        selected
                    )

    # A raw theta value has no cross-instrument meaning because the two IRT
    # models were fitted separately.  Combined therefore resolves every cut
    # within instrument first.  The score-gain outcome is already on the common
    # percentage-point scale; theta lift is z-standardized within instrument.
    ai_by_section = {
        section: frame[(frame.section == section) & (frame.kind == "ai")].copy()
        for section in ("quant", "verbal")
    }
    shared_arms = sorted(
        set(ai_by_section["quant"].arm_id.unique())
        & set(ai_by_section["verbal"].arm_id.unique())
    )
    if len(shared_arms) != 12:
        raise ValueError(
            f"combined IRT audit expected 12 shared AI identities, found {len(shared_arms)}"
        )
    for section, base in ai_by_section.items():
        base["within_form_raw_percentile"] = base.groupby("form_order")[
            "pre_score_points"
        ].rank(method="max", pct=True)
        theta_lift_sd = float(base.irt_theta_lift.std(ddof=0)) or 1.0
        base["irt_theta_lift_standardized"] = (
            base.irt_theta_lift - float(base.irt_theta_lift.mean())
        ) / theta_lift_sd
        ai_by_section[section] = base

    for stratifier in stratifiers:
        for quantile in quantiles:
            parts: list[pd.DataFrame] = []
            threshold_parts: list[str] = []
            for section, base in ai_by_section.items():
                if stratifier == "raw_pre_score":
                    threshold = float(np.quantile(base.pre_score_points, quantile))
                    selected = base[base.pre_score_points >= threshold].copy()
                    baseline_field = "pre_pct"
                    threshold_parts.append(f"{section} score ≥ {threshold:g}/27")
                elif stratifier == "pretest_irt_theta":
                    threshold = float(np.quantile(base.pretest_irt_theta, quantile))
                    selected = base[base.pretest_irt_theta >= threshold].copy()
                    baseline_field = "pretest_irt_theta"
                    threshold_parts.append(f"{section} theta ≥ {threshold:.3f}")
                else:
                    threshold = float(quantile)
                    selected = base[base.within_form_raw_percentile >= quantile].copy()
                    baseline_field = "pre_pct"
                    threshold_parts.append(
                        f"{section} within-form percentile ≥ {quantile:.3f}"
                    )
                parts.append(selected[selected.arm_id.isin(shared_arms)].copy())
            selected = pd.concat(parts, ignore_index=True)
            counts = selected.groupby("arm_id").size()
            if set(counts.index) != set(shared_arms):
                raise ValueError(
                    f"combined/{stratifier}/{quantile}: a shared AI arm was lost"
                )

            for outcome in outcomes:
                outcome_column = (
                    "gain_pp" if outcome == "gain_pp" else "irt_theta_lift_standardized"
                )
                y = selected[outcome_column].to_numpy(float)
                key = f"combined:{stratifier}:{quantile}:{outcome}"
                cached = journal.get(key, signature)
                if cached is not None:
                    variants.append(cached)
                    chosen = selected.copy()
                    chosen["combined_outcome"] = y
                    chosen["combined_baseline_z"] = selected.groupby("section")[
                        baseline_field
                    ].transform(
                        lambda values: (
                            (values - values.mean()) / (values.std(ddof=0) or 1.0)
                        )
                    )
                    selected_cells[
                        ("combined", stratifier, float(quantile), outcome)
                    ] = chosen
                    continue
                section_verbal = (selected.section == "verbal").to_numpy(float)
                arm_columns = [
                    (selected.arm_id == arm).to_numpy(float) for arm in shared_arms[1:]
                ]
                reduced = np.column_stack([np.ones(len(selected)), section_verbal])
                full = np.column_stack(
                    [np.ones(len(selected)), *arm_columns, section_verbal]
                )
                beta_reduced = np.linalg.lstsq(reduced, y, rcond=None)[0]
                beta_full = np.linalg.lstsq(full, y, rcond=None)[0]
                sse_reduced = float(np.square(y - reduced @ beta_reduced).sum())
                sse_full = float(np.square(y - full @ beta_full).sum())
                df_between = len(shared_arms) - 1
                df_within = int(len(selected) - full.shape[1])
                classical_f = ((sse_reduced - sse_full) / df_between) / (
                    sse_full / df_within
                )
                classical_p = float(stats.f.sf(classical_f, df_between, df_within))

                baseline = (
                    selected.groupby("section")[baseline_field]
                    .transform(
                        lambda values: (
                            (values - values.mean()) / (values.std(ddof=0) or 1.0)
                        )
                    )
                    .to_numpy(float)
                )
                names = (
                    ["Intercept"]
                    + [f"arm:{arm}" for arm in shared_arms[1:]]
                    + ["section:verbal", "baseline", "baseline_squared"]
                )
                robust_matrix = np.column_stack(
                    [
                        np.ones(len(selected)),
                        *arm_columns,
                        section_verbal,
                        baseline,
                        baseline**2,
                    ]
                )
                fit = _hc3(robust_matrix, y)
                contrast = np.zeros((len(shared_arms) - 1, len(names)), dtype=float)
                for index, arm in enumerate(shared_arms[1:]):
                    contrast[index, names.index(f"arm:{arm}")] = 1.0
                robust = _wald(fit, contrast)
                row = {
                    "section": "combined",
                    "stratifier": stratifier,
                    "upper_tail_quantile": float(quantile),
                    "outcome": outcome,
                    "threshold": math.nan,
                    "threshold_label": "; ".join(threshold_parts),
                    "n": int(len(selected)),
                    "min_arm_n": int(counts.min()),
                    "max_arm_n": int(counts.max()),
                    "n_ai_arms": len(shared_arms),
                    "anova_f": float(classical_f),
                    "anova_df_between": df_between,
                    "anova_df_within": df_within,
                    "anova_p_raw": classical_p,
                    "eta_squared": float((sse_reduced - sse_full) / sse_reduced),
                    "hc3_chi2": float(robust["wald_chi2"]),
                    "hc3_df": int(robust["df"]),
                    "hc3_p_raw": float(robust["p_raw"]),
                    "hc3_reference_arm": shared_arms[0],
                }
                journal.save(key, signature, row)
                variants.append(row)
                chosen = selected.copy()
                chosen["combined_outcome"] = y
                chosen["combined_baseline_z"] = baseline
                selected_cells[("combined", stratifier, float(quantile), outcome)] = (
                    chosen
                )

    variant_frame = pd.DataFrame(variants)
    for section in ("quant", "verbal", "combined"):
        mask = variant_frame.section == section
        variant_frame.loc[mask, "anova_p_holm_24"] = _holm(
            variant_frame.loc[mask, "anova_p_raw"].tolist()
        )
        variant_frame.loc[mask, "hc3_p_holm_24"] = _holm(
            variant_frame.loc[mask, "hc3_p_raw"].tolist()
        )

    chosen_key = ("quant", "pretest_irt_theta", 0.75, "gain_pp")
    chosen = selected_cells[chosen_key].copy()
    chosen_variant = variant_frame[
        (variant_frame.section == chosen_key[0])
        & (variant_frame.stratifier == chosen_key[1])
        & np.isclose(variant_frame.upper_tail_quantile, chosen_key[2])
        & (variant_frame.outcome == chosen_key[3])
    ].iloc[0]

    arm_rows: list[dict[str, Any]] = []
    for arm, block in chosen.groupby("arm_id"):
        values = block.gain_pp.to_numpy(float)
        mean = float(values.mean())
        half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
        arm_rows.append(
            {
                "section": "quant",
                "subgroup": "top pre-test IRT ability quartile",
                "arm_id": str(arm),
                "arm_label": MODEL_LABELS[str(arm)],
                "family": _family(str(arm)),
                "n": int(len(values)),
                "mean_gain_pp": mean,
                "ci_low_pp": mean - half,
                "ci_high_pp": mean + half,
            }
        )
    arm_outcomes = (
        pd.DataFrame(arm_rows)
        .sort_values("mean_gain_pp", ascending=False)
        .reset_index(drop=True)
    )
    arm_outcomes["descriptive_rank"] = np.arange(1, len(arm_outcomes) + 1)

    grouped = {
        arm: chosen.loc[chosen.arm_id == arm, "gain_pp"].to_numpy(float)
        for arm in sorted(chosen.arm_id.unique())
    }
    df_error = int(len(chosen) - len(grouped))
    mse = (
        sum(
            float(np.square(values - values.mean()).sum())
            for values in grouped.values()
        )
        / df_error
    )
    pair_rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(grouped), 2):
        key = f"quant:irt_q75:tukey:{left}:{right}"
        cached = journal.get(key, signature)
        if cached is not None:
            pair_rows.append(cached)
            continue
        left_values, right_values = grouped[left], grouped[right]
        estimate = float(left_values.mean() - right_values.mean())
        se_q = math.sqrt(mse / 2.0 * (1.0 / len(left_values) + 1.0 / len(right_values)))
        q_value = abs(estimate) / se_q
        p_value = float(stats.studentized_range.sf(q_value, len(grouped), df_error))
        row = {
            "left_arm_id": left,
            "left_arm_label": MODEL_LABELS[left],
            "right_arm_id": right,
            "right_arm_label": MODEL_LABELS[right],
            "left_minus_right_gain_pp": estimate,
            "p_tukey_kramer": p_value,
            "difference_detected_0_05": bool(p_value < 0.05),
        }
        journal.save(key, signature, row)
        pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)

    verbal_variant = variant_frame[
        (variant_frame.section == "verbal")
        & (variant_frame.stratifier == "pretest_irt_theta")
        & np.isclose(variant_frame.upper_tail_quantile, 0.75)
        & (variant_frame.outcome == "gain_pp")
    ].iloc[0]
    combined_variant = variant_frame[
        (variant_frame.section == "combined")
        & (variant_frame.stratifier == "pretest_irt_theta")
        & np.isclose(variant_frame.upper_tail_quantile, 0.75)
        & (variant_frame.outcome == "gain_pp")
    ].iloc[0]

    def _pairwise_counts(scope: str) -> dict[str, Any]:
        """All-pairs resolution inside the chosen IRT-Q75 cell."""
        cell = selected_cells[(scope, "pretest_irt_theta", 0.75, "gain_pp")].copy()
        arms_here = sorted(cell.arm_id.unique())
        if scope == "combined":
            baseline = cell["combined_baseline_z"].to_numpy(float)
            section_verbal = (cell.section == "verbal").to_numpy(float)
            names = (
                ["Intercept"]
                + [f"arm:{arm}" for arm in arms_here[1:]]
                + ["section:verbal", "baseline", "baseline_squared"]
            )
            matrix = np.column_stack(
                [
                    np.ones(len(cell)),
                    *[(cell.arm_id == arm).to_numpy(float) for arm in arms_here[1:]],
                    section_verbal,
                    baseline,
                    baseline**2,
                ]
            )
        else:
            baseline = cell.pretest_irt_theta.to_numpy(float)
            baseline = (baseline - baseline.mean()) / (baseline.std(ddof=0) or 1.0)
            names = (
                ["Intercept"]
                + [f"arm:{arm}" for arm in arms_here[1:]]
                + ["baseline", "baseline_squared"]
            )
            matrix = np.column_stack(
                [
                    np.ones(len(cell)),
                    *[(cell.arm_id == arm).to_numpy(float) for arm in arms_here[1:]],
                    baseline,
                    baseline**2,
                ]
            )
        fit = _hc3(matrix, cell.gain_pp.to_numpy(float))
        contrast_rows: list[np.ndarray] = []
        for left, right in combinations(arms_here, 2):
            row = np.zeros(len(names), dtype=float)
            if left != arms_here[0]:
                row[names.index(f"arm:{left}")] += 1.0
            if right != arms_here[0]:
                row[names.index(f"arm:{right}")] -= 1.0
            contrast_rows.append(row)
        raw_p: list[float] = []
        for contrast_row in contrast_rows:
            estimate = float(contrast_row @ fit["beta"])
            variance = float(contrast_row @ fit["covariance"] @ contrast_row)
            z = estimate / math.sqrt(max(variance, 1e-15))
            raw_p.append(float(2.0 * stats.norm.sf(abs(z))))
        adjusted_p = _holm(raw_p)
        return {
            "pairs": len(raw_p),
            "hc3_holm_differences_detected_0_05": int(
                sum(value < 0.05 for value in adjusted_p)
            ),
            "minimum_hc3_holm_p": float(min(adjusted_p)),
        }

    scope_pairwise = {
        scope: _pairwise_counts(scope) for scope in ("quant", "verbal", "combined")
    }
    scope_pairwise["quant"]["raw_familywise_differences_detected_0_05"] = int(
        pairwise.difference_detected_0_05.sum()
    )
    # The old analysis only saved the two zero counts below. Recompute them:
    # Verbal mirrors the Quant Tukey-Kramer family. Combined uses an explicit
    # section-adjusted OLS contrast family with Holm correction. The latter is
    # an additional direct verification of the reported absence of resolved pairs.
    for scope in ("verbal", "combined"):
        cell = selected_cells[(scope, "pretest_irt_theta", 0.75, "gain_pp")]
        arms = sorted(cell.arm_id.unique())
        matrix = np.column_stack(
            [
                np.ones(len(cell)),
                *[(cell.arm_id == arm).to_numpy(float) for arm in arms[1:]],
                *(
                    [(cell.section == "verbal").to_numpy(float)]
                    if scope == "combined"
                    else []
                ),
            ]
        )
        values = cell.gain_pp.to_numpy(float)
        beta = np.linalg.lstsq(matrix, values, rcond=None)[0]
        residual_df = len(cell) - int(np.linalg.matrix_rank(matrix))
        mse = float(np.square(values - matrix @ beta).sum() / residual_df)
        covariance = mse * np.linalg.inv(matrix.T @ matrix)
        raw_pairs = []
        for left, right in combinations(arms, 2):
            key = f"{scope}:irt_q75:raw_pair:{left}:{right}"
            row = journal.get(key, signature)
            if row is None:
                vector = np.zeros(matrix.shape[1])
                if left != arms[0]:
                    vector[arms.index(left)] += 1
                if right != arms[0]:
                    vector[arms.index(right)] -= 1
                estimate = float(vector @ beta)
                se = math.sqrt(float(vector @ covariance @ vector))
                pvalue = (
                    float(
                        stats.studentized_range.sf(
                            abs(estimate) / (se / math.sqrt(2)), len(arms), residual_df
                        )
                    )
                    if scope == "verbal"
                    else float(2 * stats.t.sf(abs(estimate / se), residual_df))
                )
                row = dict(
                    scope=scope,
                    left_arm_id=left,
                    right_arm_id=right,
                    estimate_pp=estimate,
                    se=se,
                    residual_df=residual_df,
                    p=pvalue,
                )
                journal.save(key, signature, row)
            raw_pairs.append(row)
        corrected = (
            [row["p"] for row in raw_pairs]
            if scope == "verbal"
            else _holm([row["p"] for row in raw_pairs])
        )
        for row, adjusted_p in zip(raw_pairs, corrected):
            row["p_familywise"] = float(adjusted_p)
        from .journal import write_csv

        write_csv(
            Path(output_dir) / (scope + "_raw_pairwise.csv"), pd.DataFrame(raw_pairs)
        )
        scope_pairwise[scope].update(
            raw_familywise_differences_detected_0_05=int(
                sum(value < 0.05 for value in corrected)
            ),
            raw_pairwise_method=(
                "Tukey-Kramer"
                if scope == "verbal"
                else "Additional verification: section-adjusted OLS Student-t contrasts, Holm family"
            ),
            minimum_raw_familywise_p=float(min(corrected)),
        )

    scope_audit = []
    for scope, row in (
        ("quant", chosen_variant),
        ("verbal", verbal_variant),
        ("combined", combined_variant),
    ):
        scope_audit.append(
            {
                "scope": scope,
                "n": int(row["n"]),
                "n_ai_arms": int(row["n_ai_arms"]),
                "threshold_label": str(row["threshold_label"]),
                "eta_squared": float(row["eta_squared"]),
                "anova_f": float(row["anova_f"]),
                "anova_df_between": int(row["anova_df_between"]),
                "anova_df_within": int(row["anova_df_within"]),
                "anova_p_raw": float(row["anova_p_raw"]),
                "anova_p_holm_24": float(row["anova_p_holm_24"]),
                "hc3_chi2": float(row["hc3_chi2"]),
                "hc3_df": int(row["hc3_df"]),
                "hc3_p_raw": float(row["hc3_p_raw"]),
                "hc3_p_holm_24": float(row["hc3_p_holm_24"]),
                "pairwise": scope_pairwise[scope],
                "separates_after_declared_family": bool(
                    float(row["anova_p_holm_24"]) < 0.05
                    or float(row["hc3_p_holm_24"]) < 0.05
                ),
            }
        )
    summary = {
        "analysis_status": "exploratory reproduction of a previously declared high-baseline specification family",
        "scientific_source": "final frozen studentbench_data_ssot only",
        "chosen_quant_specification": chosen_variant.to_dict(),
        "matched_verbal_specification": verbal_variant.to_dict(),
        "combined_specification": combined_variant.to_dict(),
        "scope_audit": scope_audit,
        "family_definition": {
            "stratifiers": list(stratifiers),
            "upper_tail_quantiles": list(quantiles),
            "outcomes": list(outcomes),
            "tests_per_scope": 24,
            "multiplicity": "Holm family-wise correction across all 24 AI-configuration omnibus tests separately within Quant, Verbal, and scale-safe Combined",
            "combined_scale_rule": "upper-tail thresholds resolved separately within instrument; gain is in percentage points; IRT lift is standardized within instrument; all Combined models include a section fixed effect",
        },
        "pairwise_method": "Tukey-Kramer across all 78 AI-configuration pairs in the chosen Quant subgroup",
        "pairwise_rows": int(len(pairwise)),
        "pairwise_differences_detected_0_05": int(
            pairwise.difference_detected_0_05.sum()
        ),
        "top_arm_id": str(arm_outcomes.iloc[0].arm_id),
        "bottom_arm_id": str(arm_outcomes.iloc[-1].arm_id),
        "interpretation": (
            "The chosen Quant specification is nominally positive before correction but does not survive "
            "Holm correction across the reproduced 24-specification family. Verbal and scale-safe Combined "
            "are null before and after correction. None supports a stable AI-tutor ranking in the final frozen cohort."
        ),
    }
    return variant_frame, arm_outcomes, pairwise, summary
